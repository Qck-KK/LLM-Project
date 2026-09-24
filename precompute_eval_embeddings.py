"""
precompute_eval_embeddings.py
================================
Builds a cache for single-solution final-answer evaluation.

Input JSONL, one solution per line:
{"question": "...", "steps": ["step1", "step2", ...], "final_correct": 1}
"""

import argparse
import json
import os
import time

import torch

from encoder import FrozenStepEncoder
from eval.eval_utils import get_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--eval_file", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--lora_path", default=None,
                        help="PEFT adapter dir; exports features from the "
                             "LoRA-adapted encoder instead of the frozen one.")
    args = parser.parse_args()

    device = get_device(args.device)
    os.makedirs(args.cache_dir, exist_ok=True)
    print(f"[precompute_eval] device={device}  cache_dir={args.cache_dir}")

    model_dtype = torch.float16 if args.dtype == "float16" and device != "cpu" else torch.float32
    encoder = FrozenStepEncoder(model_name=args.model_name, dtype=model_dtype)
    if args.lora_path:
        # The eval caches must come from the same encoder the head was trained
        # with, otherwise the LoRA arm is scored on features it never saw.
        from peft import PeftModel
        encoder.model = PeftModel.from_pretrained(encoder.model, args.lora_path)
        encoder.model = encoder.model.merge_and_unload()
        print("[precompute] merged LoRA adapter from " + args.lora_path)
    encoder = encoder.to(device)
    encoder.eval()
    with open(os.path.join(args.cache_dir, "hidden_size.txt"), "w") as f:
        f.write(str(encoder.hidden_size))

    records = []
    # Windows defaults text reads to the locale codec (GBK here), which fails on
    # any non-ASCII character in the data, so the encoding must be explicit.
    with open(args.eval_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    texts, final_correct = [], []
    for rec in records:
        text = rec["question"].strip() + "\n"
        for step in rec["steps"]:
            text += step.strip() + f" {encoder.step_token}\n"
        texts.append(text)
        final_correct.append(int(rec["final_correct"]))

    save_dtype = torch.float16 if args.dtype == "float16" else torch.float32

    t0 = time.time()
    n_done, shard_idx = 0, 0
    with torch.no_grad():
        for shard_idx, start in enumerate(range(0, len(texts), args.batch_size)):
            batch_texts = texts[start:start + args.batch_size]
            batch_labels = torch.tensor(final_correct[start:start + args.batch_size])

            step_hidden, step_mask = encoder.encode_texts(
                batch_texts, device=device, max_length=args.max_length
            )

            torch.save(
                {
                    "step_hidden": step_hidden.to(save_dtype).cpu(),
                    "step_mask": step_mask.cpu(),
                    "final_correct": batch_labels,
                },
                os.path.join(args.cache_dir, f"shard_{shard_idx:06d}.pt"),
            )
            n_done += len(batch_texts)
            if shard_idx % 20 == 0:
                dt = time.time() - t0
                print(f"[precompute_eval] shard {shard_idx}  examples_so_far={n_done}  elapsed={dt/60:.1f}min")

    total_min = (time.time() - t0) / 60
    print(f"\n[precompute_eval] done: {n_done} examples -> {shard_idx + 1} shards in {args.cache_dir}")
    print(f"[precompute_eval] total time: {total_min:.1f} min")


if __name__ == "__main__":
    main()
