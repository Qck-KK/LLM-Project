"""
precompute_embeddings.py
=========================
Runs the FROZEN encoder over the whole dataset ONCE and caches every batch's
(step_hidden, step_mask, labels) to disk. Since the encoder never changes
across heads or epochs, this is the only place the expensive 0.5B forward
pass needs to happen.

After this, use train_from_cache.py to train any of the 5 heads for as many
epochs as you want, reading cached tensors instead of re-running the encoder.
This turns "5 heads x 3 epochs x full encoder forward" into
"1 x full encoder forward" + "5 heads x 3 epochs x tiny head forward/backward".

Usage:
    python precompute_embeddings.py \
        --train_file data/math_shepherd_train.jsonl \
        --cache_dir cache/math_shepherd_qwen05b \
        --batch_size 16

Tip: run this on whichever machine is faster (likely the 4060). The cache
directory is just tensors on disk -- copy it over to the M4 Air afterwards
and train heads there just as fast, since head training barely uses compute.
"""

import argparse
import functools
import os
import time

import torch
from torch.utils.data import DataLoader

from encoder import FrozenStepEncoder
from dataset import MathShepherdStepDataset, collate_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=16,
                         help="Also becomes the effective training batch size later, "
                              "since each shard = one of these batches.")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    os.makedirs(args.cache_dir, exist_ok=True)
    print(f"[precompute] device={device}  cache_dir={args.cache_dir}")

    encoder = FrozenStepEncoder(model_name=args.model_name).to(device)
    encoder.eval()
    # Save hidden_size so train_from_cache.py doesn't need to reload the LLM at all.
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

        # align labels (from raw dataset) to however many steps survived truncation
        S = step_hidden.shape[1]
        if labels.shape[1] > S:
            labels = labels[:, :S]
        elif labels.shape[1] < S:
            pad = labels.new_full((labels.shape[0], S - labels.shape[1]), -100)
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
            print(f"[precompute] shard {shard_idx}  examples_so_far={n_examples}  "
                  f"elapsed={dt/60:.1f}min")

    total_min = (time.time() - t0) / 60
    print(f"\n[precompute] done: {n_examples} examples -> {shard_idx + 1} shards "
          f"in {args.cache_dir}")
    print(f"[precompute] total time: {total_min:.1f} min")


if __name__ == "__main__":
    main()
