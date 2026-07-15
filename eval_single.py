"""
eval_single.py
================
NOTE: if you're evaluating MULTIPLE heads against the same eval set, use
precompute_eval_embeddings.py + eval_single_from_cache.py instead -- that
runs the frozen LLM encoder only ONCE total, instead of once per head as
this script does. Keep this one around for quick one-off checks where
building a cache isn't worth it.

Simplified verification evaluation: ONE candidate solution per question
(no Best-of-N selection). Useful when you just want to know "does the PRM's
score correctly predict whether this single solution's final answer is
right?" rather than "can it pick the best one out of many?".

Expected input JSONL, one line per (question, single candidate solution):
{
  "question": "...",
  "steps": ["step1", "step2", ...],
  "final_correct": 1
}

For each line we:
  1. Encode question + steps with the frozen encoder (batched).
  2. Get per-step Q-values from the trained head.
  3. Aggregate into one trajectory score (min / mean / last).
  4. Compare that score against `final_correct` to get:
       - best-threshold accuracy (does score > threshold predict correctness?)
       - AUC-style separation: P(score | correct) > P(score | incorrect)

Usage:
    python eval_single.py --eval_file data/single_eval.jsonl \
        --head mlp --checkpoint checkpoints/mlp_head.pt \
        --results_dir results
"""

import argparse
import json
import os
import time

import torch

from encoder import FrozenStepEncoder
from reward_heads import build_reward_head


def aggregate_trajectory_score(q_values: torch.Tensor, step_mask: torch.Tensor, mode: str = "min") -> float:
    valid_q = q_values[step_mask]
    if valid_q.numel() == 0:
        return float("-inf")
    if mode == "min":
        return valid_q.min().item()
    elif mode == "mean":
        return valid_q.mean().item()
    elif mode == "last":
        return valid_q[-1].item()
    raise ValueError(mode)


def best_threshold_accuracy(scores: torch.Tensor, labels: torch.Tensor, n_grid: int = 200):
    lo, hi = scores.min().item(), scores.max().item()
    best_acc, best_t = -1.0, 0.0
    for t in torch.linspace(lo, hi, n_grid):
        preds = (scores > t).long()
        acc = (preds == labels).float().mean().item()
        if acc > best_acc:
            best_acc, best_t = acc, t.item()
    return best_t, best_acc


def pairwise_separation(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """P(score_correct > score_incorrect) over all (correct, incorrect) pairs
    across the whole eval set -- an AUC-equivalent separation measure."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.numel() == 0 or neg.numel() == 0:
        return float("nan")
    comp = (pos.unsqueeze(1) > neg.unsqueeze(0)).float()
    return comp.mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_file", required=True)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--head", required=True,
                         choices=["linear", "mlp", "cnn", "gru", "attention"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--agg", default="min", choices=["min", "mean", "last"])
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--encode_batch_size", type=int, default=16)
    parser.add_argument("--results_dir", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"))

    encoder = FrozenStepEncoder(model_name=args.model_name).to(device)
    encoder.eval()

    head = build_reward_head(args.head, hidden_size=encoder.hidden_size).to(device)
    head.load_state_dict(torch.load(args.checkpoint, map_location=device))
    head.eval()

    records = []
    with open(args.eval_file) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    texts, labels = [], []
    for rec in records:
        text = rec["question"].strip() + "\n"
        for s in rec["steps"]:
            text += s.strip() + f" {encoder.step_token}\n"
        texts.append(text)
        labels.append(int(rec["final_correct"]))
    labels = torch.tensor(labels)

    print(f"[eval_single] scoring {len(texts)} solutions in batches of {args.encode_batch_size}...")
    t0 = time.time()

    scores = torch.empty(len(texts))
    with torch.no_grad():
        for start in range(0, len(texts), args.encode_batch_size):
            batch_texts = texts[start:start + args.encode_batch_size]
            step_hidden, step_mask = encoder.encode_texts(
                batch_texts, device=device, max_length=args.max_length
            )
            q_values = head(step_hidden, step_mask)
            for i in range(len(batch_texts)):
                scores[start + i] = aggregate_trajectory_score(q_values[i], step_mask[i], mode=args.agg)

    dt = time.time() - t0
    print(f"[eval_single] done in {dt:.1f}s ({len(texts)/max(dt,1e-9):.1f} solutions/sec)")

    thr, acc = best_threshold_accuracy(scores, labels)
    sep = pairwise_separation(scores, labels)

    print(f"\n=== {args.head} (agg={args.agg}) ===")
    print(f"Best-threshold accuracy (threshold={thr:.3f}): {acc:.4f}")
    print(f"Pairwise separation P(score_correct > score_incorrect): {sep:.4f}")
    print(f"n_solutions={len(texts)}  n_correct={int(labels.sum())}  n_incorrect={int((labels==0).sum())}")

    if args.results_dir:
        os.makedirs(args.results_dir, exist_ok=True)
        out = {
            "head": args.head,
            "checkpoint": args.checkpoint,
            "agg": args.agg,
            "single_eval_accuracy": acc,
            "single_eval_threshold": thr,
            "single_eval_separation": sep,
            "n_solutions": len(texts),
        }
        out_path = os.path.join(args.results_dir, f"{args.head}_single_metrics.json")
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
