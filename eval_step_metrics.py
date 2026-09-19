"""
eval_step_metrics.py
======================
Held-out step-level evaluation for a trained reward head:
    - calibrated step accuracy and balanced accuracy
    - threshold-free ROC-AUC and average precision
    - within-trajectory Q-value ranking accuracy

Metrics are computed from cached validation embeddings and a trained head's
weights. The split is by trajectory: one half calibrates the classification
threshold and the other half is used for the reported metrics.

Usage:
    python eval_step_metrics.py --cache_dir cache/qwen05b_val \
        --head mlp --checkpoint checkpoints/mlp_head.pt

Run once per trained head, then compare the printed numbers across heads --
this reproduces the "Verification Performance" columns of the table.
"""

import argparse
import glob
import json
import os

import torch

from eval_utils import (
    HEAD_CHOICES,
    average_precision,
    balanced_accuracy,
    best_threshold_accuracy,
    binary_accuracy,
    deterministic_example_split,
    flatten_valid_steps,
    get_device,
    roc_auc,
)
from reward_heads import build_reward_head


@torch.no_grad()
def qvalue_ranking_accuracy(all_q, all_labels):
    """
    Pairwise ranking accuracy WITHIN each trajectory: for every (correct step,
    incorrect step) pair in the same example, check if Q(correct) > Q(incorrect).
    This is the metric PQM's training objective directly optimizes for, so it's
    the most informative single number for comparing architectures.
    """
    total, correct_pairs = 0, 0
    B, S = all_labels.shape
    for b in range(B):
        labels_b = all_labels[b]
        pos_idx = (labels_b == 1).nonzero(as_tuple=True)[0]
        neg_idx = (labels_b == 0).nonzero(as_tuple=True)[0]
        if pos_idx.numel() == 0 or neg_idx.numel() == 0:
            continue
        pos_q = all_q[b, pos_idx].unsqueeze(1)   # (P, 1)
        neg_q = all_q[b, neg_idx].unsqueeze(0)   # (1, N)
        correct_pairs += (pos_q > neg_q).sum().item()
        total += pos_idx.numel() * neg_idx.numel()
    return correct_pairs / max(total, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", required=True,
                         help="Validation/test cache copied from the external precompute step.")
    parser.add_argument("--head", required=True, choices=HEAD_CHOICES)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--results_dir", default=None,
                         help="If set, writes a JSON file here for summarize_results.py to pick up.")
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
    assert shard_paths, f"No cached shards in {args.cache_dir}"

    all_q_list, all_labels_list = [], []
    with torch.no_grad():
        for path in shard_paths:
            shard = torch.load(path, map_location=device)
            step_hidden = shard["step_hidden"].to(device).float()
            step_mask = shard["step_mask"].to(device)
            labels = shard["labels"].to(device)
            q = head(step_hidden, step_mask)
            all_q_list.append(q.cpu())
            all_labels_list.append(labels.cpu())

    # pad all shards to the same S before concatenating (shard sizes can differ
    # slightly if trajectories in different shards have different step counts)
    max_S = max(t.shape[1] for t in all_q_list)
    all_q = torch.cat([
        torch.nn.functional.pad(t, (0, max_S - t.shape[1]), value=0.0) for t in all_q_list
    ], dim=0)
    all_labels = torch.cat([
        torch.nn.functional.pad(t, (0, max_S - t.shape[1]), value=-100) for t in all_labels_list
    ], dim=0)

    calibration_mask, test_mask = deterministic_example_split(
        all_labels.shape[0], args.calibration_fraction, args.split_seed
    )
    calibration_q, calibration_labels = flatten_valid_steps(
        all_q, all_labels, calibration_mask
    )
    test_q, test_labels = flatten_valid_steps(all_q, all_labels, test_mask)
    thr, calibration_acc = best_threshold_accuracy(calibration_q, calibration_labels)
    acc_at_thr = binary_accuracy(test_q, test_labels, thr)
    balanced_acc = balanced_accuracy(test_q, test_labels, thr)
    auc = roc_auc(test_q, test_labels)
    ap = average_precision(test_q, test_labels)
    ranking_acc = qvalue_ranking_accuracy(all_q[test_mask], all_labels[test_mask])

    print(f"\n=== {args.head} ===")
    print(f"Calibration accuracy (threshold={thr:.3f}): {calibration_acc:.4f}")
    print(f"Held-out Step Reward Accuracy: {acc_at_thr:.4f}")
    print(f"Held-out Balanced Accuracy: {balanced_acc:.4f}")
    print(f"Held-out ROC-AUC / AP: {auc:.4f} / {ap:.4f}")
    print(f"Q-value Ranking Accuracy: {ranking_acc:.4f}")

    if args.results_dir:
        os.makedirs(args.results_dir, exist_ok=True)
        n_params = sum(p.numel() for p in head.parameters())
        out = {
            "head": args.head,
            "checkpoint": args.checkpoint,
            "step_reward_accuracy": acc_at_thr,
            "step_reward_calibration_accuracy": calibration_acc,
            "step_reward_threshold": thr,
            "step_balanced_accuracy": balanced_acc,
            "step_roc_auc": auc,
            "step_average_precision": ap,
            "qvalue_ranking_accuracy": ranking_acc,
            "calibration_fraction": args.calibration_fraction,
            "split_seed": args.split_seed,
            "n_calibration_trajectories": int(calibration_mask.sum()),
            "n_test_trajectories": int(test_mask.sum()),
            "n_trainable_params": n_params,
        }
        out_path = os.path.join(args.results_dir, f"{args.head}_step_metrics.json")
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
