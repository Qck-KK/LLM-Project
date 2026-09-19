"""
precompute_embeddings.py
=========================
Runs the frozen encoder over a Math-Shepherd-style train/validation JSONL file
and caches each batch as step_hidden, step_mask, and labels tensors.

Run this once per split on the fastest available machine. The produced cache
can then be copied to the training machine and consumed by train_from_cache.py.
"""

import argparse
import functools
import os
import time

import torch
from torch.utils.data import DataLoader

from dataset import MathShepherdStepDataset, collate_fn
from encoder import FrozenStepEncoder
from eval.eval_utils import get_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = get_device(args.device)
    os.makedirs(args.cache_dir, exist_ok=True)
    print(f"[precompute] device={device}  cache_dir={args.cache_dir}")

    model_dtype = torch.float16 if args.dtype == "float16" and device != "cpu" else torch.float32
    encoder = FrozenStepEncoder(model_name=args.model_name, dtype=model_dtype).to(device)
    encoder.eval()
    with open(os.path.join(args.cache_dir, "hidden_size.txt"), "w") as f:
        f.write(str(encoder.hidden_size))

    dataset = MathShepherdStepDataset(args.train_file, max_length=args.max_length)
    collate = functools.partial(collate_fn, tokenizer=encoder.tokenizer, max_length=args.max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    save_dtype = torch.float16 if args.dtype == "float16" else torch.float32

    t0 = time.time()
    n_examples, shard_idx = 0, 0
    for shard_idx, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"]

        with torch.no_grad():
            step_hidden, step_mask = encoder(input_ids, attention_mask)

        n_steps = step_hidden.shape[1]
        if labels.shape[1] > n_steps:
            labels = labels[:, :n_steps]
        elif labels.shape[1] < n_steps:
            pad = labels.new_full((labels.shape[0], n_steps - labels.shape[1]), -100)
            labels = torch.cat([labels, pad], dim=1)

        torch.save(
            {
                "step_hidden": step_hidden.to(save_dtype).cpu(),
                "step_mask": step_mask.cpu(),
                "labels": labels.cpu(),
            },
            os.path.join(args.cache_dir, f"shard_{shard_idx:06d}.pt"),
        )

        n_examples += input_ids.shape[0]
        if shard_idx % 20 == 0:
            dt = time.time() - t0
            print(f"[precompute] shard {shard_idx}  examples_so_far={n_examples}  elapsed={dt/60:.1f}min")

    total_min = (time.time() - t0) / 60
    print(f"\n[precompute] done: {n_examples} examples -> {shard_idx + 1} shards in {args.cache_dir}")
    print(f"[precompute] total time: {total_min:.1f} min")


if __name__ == "__main__":
    main()
