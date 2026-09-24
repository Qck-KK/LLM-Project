"""LoRA fine-tuning of the step encoder, as the control for the frozen-encoder premise.

Every other experiment in this project keeps Qwen frozen and trains only a
lightweight head. That premise is this project's own simplification -- the
anchor paper (PQM, Li & Li) trains a 7B backbone on 8 GPUs -- so nothing here
shows what freezing costs. This script supplies the missing arm: the same PQM
loss and the same linear value head, but with LoRA adapters on the encoder's
attention projections.

To keep the comparison honest on a single 8GB GPU, both arms train on the SAME
10% subset of Math-Shepherd. The result is a scaled-down but controlled answer
to "does unfreezing the encoder help?", not a compute-matched replication of PQM.

The measured operating point on an RTX 4060 is batch 4 at max_length 512: batch
8 already spills out of 8GB and runs ~7x slower.
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model

from dataset import MathShepherdStepDataset, collate_fn
from pqm_loss import pqm_loss


def build_model(model_name, step_token, lora_r, lora_alpha, lora_dropout, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    added = tokenizer.add_special_tokens(
        {"additional_special_tokens": [step_token]}
    )
    step_token_id = tokenizer.convert_tokens_to_ids(step_token)

    base = AutoModel.from_pretrained(model_name, dtype=torch.float32)
    if added > 0:
        base.resize_token_embeddings(len(tokenizer))

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(base, config).to(device)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    head = nn.Linear(base.config.hidden_size, 1).to(device)
    return tokenizer, model, head, step_token_id, base.config.hidden_size


def gather_step_states(hidden, input_ids, step_token_id):
    """One vector per step marker, padded to the batch's longest step count."""
    is_step = input_ids.eq(step_token_id)
    counts = is_step.sum(dim=1)
    max_steps = max(int(counts.max().item()), 1)
    batch = hidden.shape[0]
    step_hidden = hidden.new_zeros(batch, max_steps, hidden.shape[-1])
    step_mask = torch.zeros(batch, max_steps, dtype=torch.bool, device=hidden.device)
    for row in range(batch):
        idx = is_step[row].nonzero(as_tuple=True)[0][:max_steps]
        if idx.numel():
            step_hidden[row, :idx.numel()] = hidden[row, idx]
            step_mask[row, :idx.numel()] = True
    return step_hidden, step_mask


def align_labels(labels, step_mask):
    """collate_fn already trims labels to surviving markers; pad/crop to match."""
    n_steps = step_mask.shape[1]
    if labels.shape[1] < n_steps:
        labels = torch.nn.functional.pad(
            labels, (0, n_steps - labels.shape[1]), value=-100
        )
    elif labels.shape[1] > n_steps:
        labels = labels[:, :n_steps]
    return labels.masked_fill(~step_mask, -100)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--val_cache_dir", default=None,
                        help="Unused during training; kept for protocol symmetry.")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--step_token", default="ки")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--zeta", type=float, default=4.0)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_records", type=int, default=0,
                        help="Smoke-test cap on trajectories; 0 uses all.")
    parser.add_argument("--log_every", type=int, default=200)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    tokenizer, model, head, step_token_id, hidden_size = build_model(
        args.model_name, args.step_token, args.lora_r, args.lora_alpha,
        args.lora_dropout, args.device
    )
    n_lora = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("[lora] trainable {0:,} adapter params + {1:,} head params".format(
        n_lora, sum(p.numel() for p in head.parameters())))

    dataset = MathShepherdStepDataset(args.train_file, step_token=args.step_token)
    if args.max_records:
        dataset.records = dataset.records[:args.max_records]
    print("[lora] {0:,} trajectories from {1}".format(len(dataset), args.train_file))

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
        collate_fn=lambda b: collate_fn(b, tokenizer, max_length=args.max_length),
    )
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()), lr=args.lr
    )
    scaler = torch.amp.GradScaler(args.device)

    history = {"epoch": [], "train_loss": [], "step": [], "step_loss": []}
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.time()

    for epoch in range(args.epochs):
        model.train()
        head.train()
        running, seen, window, window_n = 0.0, 0, 0.0, 0
        for batch_index, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)
            labels = batch["labels"].to(args.device)
            with torch.amp.autocast(args.device, dtype=torch.float16):
                hidden = model(input_ids=input_ids,
                               attention_mask=attention_mask).last_hidden_state
                step_hidden, step_mask = gather_step_states(
                    hidden, input_ids, step_token_id
                )
                q_values = head(step_hidden).squeeze(-1)
            loss = pqm_loss(q_values.float(),
                            align_labels(labels, step_mask), zeta=args.zeta)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            size = input_ids.shape[0]
            running += loss.item() * size
            seen += size
            window += loss.item() * size
            window_n += size
            if (batch_index + 1) % args.log_every == 0:
                print("[epoch {0}/{1}] batch {2}  loss={3:.4f}  "
                      "elapsed={4:.1f}min".format(
                          epoch + 1, args.epochs, batch_index + 1,
                          window / max(window_n, 1), (time.time() - start) / 60),
                      flush=True)
                history["step"].append(epoch + (batch_index + 1) / len(loader))
                history["step_loss"].append(window / max(window_n, 1))
                window, window_n = 0.0, 0
        epoch_loss = running / max(seen, 1)
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(epoch_loss)
        print("[epoch {0}/{1}] train_loss={2:.4f}  elapsed={3:.1f}min".format(
            epoch + 1, args.epochs, epoch_loss, (time.time() - start) / 60), flush=True)

        model.save_pretrained(os.path.join(args.save_dir, "adapter"))
        torch.save(head.state_dict(), os.path.join(args.save_dir, "value_head.pt"))

    elapsed = time.time() - start
    summary = {
        "train_file": args.train_file,
        "n_trajectories": len(dataset),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "lr": args.lr,
        "zeta": args.zeta,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "seed": args.seed,
        "n_lora_params": n_lora,
        "total_train_time_sec": elapsed,
        "peak_mem_mb": (torch.cuda.max_memory_allocated() / 2 ** 20
                        if args.device == "cuda" else None),
        "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
        "history": history,
    }
    with open(os.path.join(args.results_dir, "lora_training.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("[lora] done in {0:.1f}min -> {1}".format(elapsed / 60, args.save_dir))


if __name__ == "__main__":
    main()
