"""
train_from_cache.py
=====================
Trains a reward head using embeddings precomputed by precompute_embeddings.py.
No LLM forward pass happens here at all -- this only touches the tiny head,
so it should be dramatically faster than train.py.

Usage:
    python train_from_cache.py --cache_dir cache/math_shepherd_qwen05b \
        --head mlp --epochs 10 --save_path checkpoints/mlp_head.pt

Repeat with --head cnn / gru / attention / linear to sweep all architectures
against the SAME cached embeddings -- this is the controlled comparison the
proposal calls for (identical encoder output, only the head differs).
"""

import argparse
import glob
import json
import os
import random
import time

import torch

from reward_heads import build_reward_head
from pqm_loss import pqm_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--head", required=True,
                         choices=["linear", "mlp", "cnn", "gru", "attention"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--zeta", type=float, default=4.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save_path", default="checkpoints/head.pt")
    parser.add_argument("--results_dir", default=None,
                         help="If set, writes training-efficiency JSON here for summarize_results.py.")
    args = parser.parse_args()

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )

    with open(os.path.join(args.cache_dir, "hidden_size.txt")) as f:
        hidden_size = int(f.read().strip())

    shard_paths = sorted(glob.glob(os.path.join(args.cache_dir, "shard_*.pt")))
    assert shard_paths, f"No cached shards found in {args.cache_dir}. Run precompute_embeddings.py first."
    print(f"[train] device={device}  head={args.head}  "
          f"{len(shard_paths)} cached shards  hidden_size={hidden_size}")

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

    head = build_reward_head(args.head, hidden_size=hidden_size).to(device)
    n_trainable = sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f"[train] trainable params: {n_trainable:,}")

    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    train_start = time.time()

    for epoch in range(args.epochs):
        random.shuffle(shard_paths)
        t0 = time.time()
        running_loss, n_batches = 0.0, 0

        for path in shard_paths:
            shard = torch.load(path, map_location=device)
            step_hidden = shard["step_hidden"].to(device).float()  # fp16 cache -> fp32 compute
            step_mask = shard["step_mask"].to(device)
            labels = shard["labels"].to(device)

            q_values = head(step_hidden, step_mask)
            loss = pqm_loss(q_values, labels, zeta=args.zeta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        dt = time.time() - t0
        print(f"[epoch {epoch+1}/{args.epochs}] loss={running_loss / max(n_batches,1):.4f}  "
              f"time={dt:.1f}s")

    total_train_time = time.time() - train_start
    torch.save(head.state_dict(), args.save_path)
    print(f"[done] saved head weights to {args.save_path}")

    if args.results_dir:
        os.makedirs(args.results_dir, exist_ok=True)
        peak_mem_mb = None
        if device == "cuda":
            peak_mem_mb = torch.cuda.max_memory_allocated() / 1e6
        out = {
            "head": args.head,
            "n_trainable_params": n_trainable,
            "epochs": args.epochs,
            "total_train_time_sec": total_train_time,
            "peak_mem_mb": peak_mem_mb,
            "device": device,
        }
        out_path = os.path.join(args.results_dir, f"{args.head}_efficiency.json")
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
