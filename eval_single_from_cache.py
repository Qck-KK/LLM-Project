"""
eval_single_from_cache.py
============================
Scores a trained head against embeddings precomputed by
precompute_eval_embeddings.py. No LLM forward pass here -- only the tiny
head runs, so this is fast enough to re-run for every head in seconds.

Usage:
    python eval_single_from_cache.py --cache_dir cache/single_eval \
        --head mlp --checkpoint checkpoints/mlp_head.pt --results_dir results
"""

import argparse
import glob
import json
import os

import torch

from reward_heads import build_reward_head


def aggregate_trajectory_score(q_values: torch.Tensor, step_mask: torch.Tensor, mode: str = "min") -> torch.Tensor:
    """Batched version: q_values (B,S), step_mask (B,S) -> (B,) scores."""
    masked = q_values.masked_fill(~step_mask, float("inf") if mode == "min" else 0.0)
    if mode == "min":
        return masked.min(dim=1).values
    elif mode == "mean":
        return (q_values * step_mask).sum(dim=1) / step_mask.sum(dim=1).clamp(min=1)
    elif mode == "last":
        lengths = step_mask.sum(dim=1).clamp(min=1) - 1
        return q_values.gather(1, lengths.unsqueeze(1)).squeeze(1)
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
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.numel() == 0 or neg.numel() == 0:
        return float("nan")
    comp = (pos.unsqueeze(1) > neg.unsqueeze(0)).float()
    return comp.mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--head", required=True,
                         choices=["linear", "mlp", "cnn", "gru", "attention"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--agg", default="min", choices=["min", "mean", "last"])
    parser.add_argument("--results_dir", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"))

    with open(os.path.join(args.cache_dir, "hidden_size.txt")) as f:
        hidden_size = int(f.read().strip())

    head = build_reward_head(args.head, hidden_size=hidden_size).to(device)
    head.load_state_dict(torch.load(args.checkpoint, map_location=device))
    head.eval()

    shard_paths = sorted(glob.glob(os.path.join(args.cache_dir, "shard_*.pt")))
    assert shard_paths, f"No cached shards in {args.cache_dir}. Run precompute_eval_embeddings.py first."

    all_scores, all_labels = [], []
    with torch.no_grad():
        for path in shard_paths:
            shard = torch.load(path, map_location=device)
            step_hidden = shard["step_hidden"].to(device).float()
            step_mask = shard["step_mask"].to(device)
            # 自动适配不同数据集的标签键名
            if "final_correct" in shard:
                labels = shard["final_correct"].to(device)
            elif "label" in shard:
                labels = shard["label"].to(device)
            elif "labels" in shard:
                labels = shard["labels"].to(device)
            else:
                raise KeyError(f"在 shard 中找不到标签！当前可用的键有: {list(shard.keys())}")
            if labels.dim() == 2:
                # 找到每道题的最后一个有效步骤的索引
                lengths = step_mask.sum(dim=1).clamp(min=1).long() - 1
                # 提取最后一个有效步骤的标签作为整道题的最终标签
                labels = labels.gather(1, lengths.unsqueeze(1)).squeeze(1)


            q_values = head(step_hidden, step_mask)
            scores = aggregate_trajectory_score(q_values, step_mask, mode=args.agg)

            all_scores.append(scores.cpu())
            all_labels.append(labels.cpu())

    scores = torch.cat(all_scores)
    labels = torch.cat(all_labels)

    thr, acc = best_threshold_accuracy(scores, labels)
    sep = pairwise_separation(scores, labels)

    print(f"\n=== {args.head} (agg={args.agg}) ===")
    print(f"Best-threshold accuracy (threshold={thr:.3f}): {acc:.4f}")
    print(f"Pairwise separation P(score_correct > score_incorrect): {sep:.4f}")
    print(f"n_solutions={len(scores)}  n_correct={int(labels.sum())}  n_incorrect={int((labels==0).sum())}")

    if args.results_dir:
        os.makedirs(args.results_dir, exist_ok=True)
        out = {
            "head": args.head,
            "checkpoint": args.checkpoint,
            "agg": args.agg,
            "single_eval_accuracy": acc,
            "single_eval_threshold": thr,
            "single_eval_separation": sep,
            "n_solutions": len(scores),
        }
        out_path = os.path.join(args.results_dir, f"{args.head}_single_metrics.json")
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
