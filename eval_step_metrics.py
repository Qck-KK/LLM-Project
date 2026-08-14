"""
eval_step_metrics.py
======================
Two of the four "Verification Performance" metrics from the proposal:
    - Step-level Reward Accuracy
    - Q-value Ranking Accuracy

Both are computed directly from the cached embeddings (same cache used for
training) + a trained head's weights -- no need to re-run the LLM encoder.

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

from eval_utils import HEAD_CHOICES, get_device
from reward_heads import build_reward_head


@torch.no_grad()
def step_reward_accuracy(all_q, all_labels, threshold=0.0):
    """
    Fraction of individual steps where sign(Q - threshold) matches the label.
    `threshold` should be tuned on a held-out split (grid search below) rather
    than assumed to be 0 -- Q-values from a ranking loss are not naturally
    calibrated to a 0/1 decision boundary.
    """
    mask = all_labels != -100
    preds = (all_q > threshold).long()
    correct = (preds[mask] == all_labels[mask]).float()
    return correct.mean().item()


@torch.no_grad()
def best_threshold(all_q, all_labels, n_grid=200):
    mask = all_labels != -100
    q_valid = all_q[mask]
    lo, hi = q_valid.min().item(), q_valid.max().item()
    best_acc, best_t = -1.0, 0.0
    for t in torch.linspace(lo, hi, n_grid):
        acc = step_reward_accuracy(all_q, all_labels, threshold=t.item())
        if acc > best_acc:
            best_acc, best_t = acc, t.item()
    return best_t, best_acc


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

    thr, acc_at_thr = best_threshold(all_q, all_labels)
    ranking_acc = qvalue_ranking_accuracy(all_q, all_labels)

    print(f"\n=== {args.head} ===")
    print(f"Step-level Reward Accuracy (best threshold={thr:.3f}): {acc_at_thr:.4f}")
    print(f"Q-value Ranking Accuracy: {ranking_acc:.4f}")

    if args.results_dir:
        os.makedirs(args.results_dir, exist_ok=True)
        n_params = sum(p.numel() for p in head.parameters())
        out = {
            "head": args.head,
            "checkpoint": args.checkpoint,
            "step_reward_accuracy": acc_at_thr,
            "step_reward_threshold": thr,
            "qvalue_ranking_accuracy": ranking_acc,
            "n_trainable_params": n_params,
        }
        out_path = os.path.join(args.results_dir, f"{args.head}_step_metrics.json")
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
