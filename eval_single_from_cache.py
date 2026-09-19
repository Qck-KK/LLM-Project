"""
eval_single_from_cache.py
============================
Scores a trained head against externally precomputed single-eval embeddings.
One trajectory-level half calibrates the threshold; the other half is used for
reported metrics. No LLM forward pass happens here.

Usage:
    python eval_single_from_cache.py --cache_dir cache/single_eval \
        --head mlp --checkpoint checkpoints/mlp_head.pt --results_dir results
"""

import argparse
import glob
import json
import os

import torch

from eval_utils import (
    HEAD_CHOICES,
    aggregate_trajectory_scores,
    average_precision,
    balanced_accuracy,
    best_threshold_accuracy,
    binary_accuracy,
    deterministic_example_split,
    get_device,
    pairwise_separation,
    roc_auc,
)
from reward_heads import build_reward_head


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--head", required=True, choices=HEAD_CHOICES)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--agg", default="min", choices=["min", "mean", "last"])
    parser.add_argument("--results_dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--calibration_fraction", type=float, default=0.5)
    parser.add_argument("--split_seed", type=int, default=42)
    args = parser.parse_args()

    device = get_device(args.device)

    with open(os.path.join(args.cache_dir, "hidden_size.txt")) as f:
        hidden_size = int(f.read().strip())

    head = build_reward_head(args.head, hidden_size=hidden_size).to(device)
    head.load_state_dict(torch.load(args.checkpoint, map_location=device))
    head.eval()

    shard_paths = sorted(glob.glob(os.path.join(args.cache_dir, "shard_*.pt")))
    assert shard_paths, f"No cached shards in {args.cache_dir}. Copy the precomputed single-eval cache here first."

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
            scores = aggregate_trajectory_scores(q_values, step_mask, mode=args.agg)

            all_scores.append(scores.cpu())
            all_labels.append(labels.cpu())

    scores = torch.cat(all_scores)
    labels = torch.cat(all_labels)

    calibration_mask, test_mask = deterministic_example_split(
        len(scores), args.calibration_fraction, args.split_seed
    )
    thr, calibration_acc = best_threshold_accuracy(
        scores[calibration_mask], labels[calibration_mask]
    )
    test_scores, test_labels = scores[test_mask], labels[test_mask]
    acc = binary_accuracy(test_scores, test_labels, thr)
    balanced_acc = balanced_accuracy(test_scores, test_labels, thr)
    auc = roc_auc(test_scores, test_labels)
    ap = average_precision(test_scores, test_labels)
    sep = pairwise_separation(test_scores, test_labels)

    print(f"\n=== {args.head} (agg={args.agg}) ===")
    print(f"Calibration accuracy (threshold={thr:.3f}): {calibration_acc:.4f}")
    print(f"Held-out accuracy: {acc:.4f}")
    print(f"Held-out balanced accuracy: {balanced_acc:.4f}")
    print(f"Held-out ROC-AUC / AP: {auc:.4f} / {ap:.4f}")
    print(f"Pairwise separation P(score_correct > score_incorrect): {sep:.4f}")
    print(f"n_test_solutions={len(test_scores)}  n_correct={int(test_labels.sum())}  "
          f"n_incorrect={int((test_labels == 0).sum())}")

    if args.results_dir:
        os.makedirs(args.results_dir, exist_ok=True)
        out = {
            "head": args.head,
            "checkpoint": args.checkpoint,
            "agg": args.agg,
            "single_eval_accuracy": acc,
            "single_eval_calibration_accuracy": calibration_acc,
            "single_eval_threshold": thr,
            "single_eval_balanced_accuracy": balanced_acc,
            "single_eval_roc_auc": auc,
            "single_eval_average_precision": ap,
            "single_eval_separation": sep,
            "calibration_fraction": args.calibration_fraction,
            "split_seed": args.split_seed,
            "n_solutions": int(test_mask.sum()),
        }
        out_path = os.path.join(args.results_dir, f"{args.head}_single_metrics.json")
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
