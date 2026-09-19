"""
benchmark.py
=============
Measures throughput on the current machine before a full precompute run.
"""

import argparse
import time

import torch

from encoder import FrozenStepEncoder
from eval.eval_utils import HEAD_CHOICES, get_device
from pqm_loss import pqm_loss
from reward_heads import build_reward_head


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--head", default="mlp", choices=HEAD_CHOICES)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--n_steps_per_example", type=int, default=6)
    parser.add_argument("--n_batches", type=int, default=20)
    parser.add_argument("--warmup_batches", type=int, default=3)
    parser.add_argument("--dataset_size", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"[bench] device={device}  model={args.model_name}  batch_size={args.batch_size}  seq_len={args.seq_len}")

    encoder = FrozenStepEncoder(model_name=args.model_name).to(device)
    encoder.eval()
    head = build_reward_head(args.head, hidden_size=encoder.hidden_size).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3)

    vocab_size = encoder.model.config.vocab_size
    input_ids = torch.randint(0, vocab_size, (args.batch_size, args.seq_len), device=device)
    stride = args.seq_len // (args.n_steps_per_example + 1)
    for i in range(1, args.n_steps_per_example + 1):
        input_ids[:, i * stride] = encoder.step_token_id
    attention_mask = torch.ones_like(input_ids)

    labels = torch.randint(0, 2, (args.batch_size, args.n_steps_per_example), device=device)

    def run_step():
        with torch.no_grad():
            step_hidden, step_mask = encoder(input_ids, attention_mask)
        n_steps = step_hidden.shape[1]
        lab = labels[:, :n_steps] if labels.shape[1] >= n_steps else torch.nn.functional.pad(
            labels, (0, n_steps - labels.shape[1]), value=-100
        )
        q_values = head(step_hidden, step_mask)
        loss = pqm_loss(q_values, lab)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()

    for _ in range(args.warmup_batches):
        run_step()

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    for _ in range(args.n_batches):
        run_step()
    dt = time.time() - t0

    steps_per_sec = args.n_batches / dt
    examples_per_sec = steps_per_sec * args.batch_size
    tokens_per_sec = examples_per_sec * args.seq_len

    print(f"\n[result] {dt:.1f}s for {args.n_batches} batches")
    print(f"[result] {steps_per_sec:.2f} train-steps/sec  |  {examples_per_sec:.1f} examples/sec  |  {tokens_per_sec:.0f} tokens/sec")

    if device == "cuda":
        print(f"[result] peak GPU memory: {torch.cuda.max_memory_allocated()/1e6:.0f} MB")
    elif device == "mps":
        try:
            print(f"[result] current MPS memory: {torch.mps.current_allocated_memory()/1e6:.0f} MB")
        except Exception:
            pass

    if args.dataset_size:
        n_batches_per_epoch = args.dataset_size / args.batch_size
        epoch_time_sec = n_batches_per_epoch / steps_per_sec
        print(f"\n[estimate] for a {args.dataset_size:,}-example dataset:")
        print(f"[estimate]   ~{epoch_time_sec/60:.1f} min/epoch")


if __name__ == "__main__":
    main()
