"""
benchmark.py
=============
Measures REAL throughput on YOUR hardware instead of relying on guessed
numbers. Run this first on both machines (M4 Air / RTX 4060) before
committing to a full training run.

Usage:
    python benchmark.py --model_name Qwen/Qwen2.5-0.5B --batch_size 8 \
        --seq_len 512 --n_batches 20 --head mlp

It will print:
    - tokens/sec (encoder forward pass, the expensive part)
    - steps/sec (full train step: encoder fwd + head fwd/bwd + optimizer)
    - estimated time for a full epoch, given --dataset_size
    - peak memory used
"""

import argparse
import time

import torch

from encoder import FrozenStepEncoder
from reward_heads import build_reward_head
from pqm_loss import pqm_loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--head", default="mlp", choices=["linear", "mlp", "cnn", "gru", "attention"])
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--n_steps_per_example", type=int, default=6)
    p.add_argument("--n_batches", type=int, default=20)
    p.add_argument("--warmup_batches", type=int, default=3)
    p.add_argument("--dataset_size", type=int, default=None,
                    help="If set, prints an estimated full-epoch time.")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = args.device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"[bench] device={device}  model={args.model_name}  "
          f"batch_size={args.batch_size}  seq_len={args.seq_len}")

    encoder = FrozenStepEncoder(model_name=args.model_name).to(device)
    encoder.eval()
    head = build_reward_head(args.head, hidden_size=encoder.hidden_size).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3)

    # synthetic batch: random token ids of the right shape, with `n_steps_per_example`
    # step-marker tokens sprinkled in so the encoder's step-extraction logic
    # actually has something to find. This measures REAL compute cost --
    # random ids don't change FLOPs vs real text of the same length/shape.
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
        S = step_hidden.shape[1]
        lab = labels[:, :S] if labels.shape[1] >= S else torch.nn.functional.pad(
            labels, (0, S - labels.shape[1]), value=-100
        )
        q = head(step_hidden, step_mask)
        loss = pqm_loss(q, lab)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()

    # warmup (first calls pay compilation/allocator overhead, don't count them)
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
    print(f"[result] {steps_per_sec:.2f} train-steps/sec  |  "
          f"{examples_per_sec:.1f} examples/sec  |  {tokens_per_sec:.0f} tokens/sec")

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
        print(f"[estimate]   ~{epoch_time_sec/60:.1f} min/epoch  "
              f"(~{epoch_time_sec*3/3600:.2f} hours for 3 epochs)")


if __name__ == "__main__":
    main()
