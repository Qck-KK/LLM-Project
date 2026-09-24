"""In-distribution single-solution evaluation derived from a step-level cache.

The optional single-solution experiment scores Qwen's own free-form GSM8K
generations, which differ from Math-Shepherd in step granularity and formatting.
A weak result there could mean either that the heads cannot judge a whole
solution, or merely that they do not transfer to another generator's output.

This module supplies the missing control: it reuses the *validation* step cache,
labels each trajectory correct only when every one of its steps is correct, and
aggregates the same per-step Q-values into a trajectory score. Same heads, same
split, same aggregation -- only the distribution changes.

It never runs the frozen encoder and never retrains.
"""

import argparse
import csv
import glob
import json
import os

import torch

from eval.eval_utils import (
    HEAD_CHOICES,
    aggregate_trajectory_scores,
    average_precision,
    deterministic_example_split,
    get_device,
    roc_auc,
)
from reward_heads import build_reward_head


def pad_and_concat(tensors, value):
    max_steps = max(tensor.shape[1] for tensor in tensors)
    return torch.cat([
        torch.nn.functional.pad(tensor, (0, max_steps - tensor.shape[1]), value=value)
        for tensor in tensors
    ])


@torch.no_grad()
def score_cache(head, shard_paths, device):
    head.eval()
    scores = []
    for path in shard_paths:
        shard = torch.load(path, map_location=device)
        step_hidden = shard["step_hidden"].to(device).float()
        step_mask = shard["step_mask"].to(device).bool()
        scores.append(head(step_hidden, step_mask).cpu())
    return pad_and_concat(scores, 0.0)


def percentile_ci(samples, alpha=0.05):
    tensor = torch.tensor(samples, dtype=torch.float64)
    tensor = tensor[torch.isfinite(tensor)]
    if tensor.numel() == 0:
        return float("nan"), float("nan")
    return (float(torch.quantile(tensor, alpha / 2)),
            float(torch.quantile(tensor, 1 - alpha / 2)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", required=True,
                        help="Step-level validation cache (labels + step_mask).")
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--checkpoint_pattern", default="{head}_head.pt")
    parser.add_argument("--heads", nargs="+", default=list(HEAD_CHOICES))
    parser.add_argument("--aggs", nargs="+", default=["min", "mean", "last"])
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--calibration_fraction", type=float, default=0.5)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = get_device(args.device)
    os.makedirs(args.results_dir, exist_ok=True)
    shard_paths = sorted(glob.glob(os.path.join(args.cache_dir, "shard_*.pt")))
    assert shard_paths, "No cached shards in " + args.cache_dir
    with open(os.path.join(args.cache_dir, "hidden_size.txt")) as f:
        hidden_size = int(f.read().strip())

    shards = [torch.load(p, map_location="cpu") for p in shard_paths]
    labels = pad_and_concat([s["labels"].long() for s in shards], -100)
    step_mask = pad_and_concat([s["step_mask"].bool() for s in shards], False)
    labels = labels.masked_fill(~step_mask, -100)
    del shards

    valid = (labels == 0) | (labels == 1)
    n_steps = valid.sum(dim=1)
    n_correct = (labels == 1).sum(dim=1)
    # A solution counts as correct only when every one of its steps is correct.
    final_correct = (n_steps > 0) & (n_correct == n_steps)
    usable = n_steps > 0

    _, test_mask = deterministic_example_split(
        labels.shape[0], args.calibration_fraction, args.split_seed
    )
    selected = test_mask & usable
    rows = selected.nonzero(as_tuple=True)[0]
    targets = final_correct[rows].long()

    print("[single-indist] trajectories={0}  positives={1} ({2:.4f})".format(
        int(rows.numel()), int(targets.sum()), float(targets.float().mean())))

    table = []
    generator = torch.Generator().manual_seed(args.seed)
    n_rows = rows.numel()
    resamples = [torch.randint(n_rows, (n_rows,), generator=generator)
                 for _ in range(args.bootstrap_samples)]

    for head_name in args.heads:
        checkpoint = os.path.join(
            args.checkpoint_dir, args.checkpoint_pattern.format(head=head_name)
        )
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(checkpoint)
        head = build_reward_head(head_name, hidden_size).to(device)
        head.load_state_dict(torch.load(checkpoint, map_location=device))
        q_values = score_cache(head, shard_paths, device)

        for agg in args.aggs:
            traj_scores = aggregate_trajectory_scores(q_values, valid, mode=agg)[rows]
            entry = {
                "head": head_name,
                "agg": agg,
                "n_solutions": int(n_rows),
                "positive_rate": float(targets.float().mean()),
                "roc_auc": roc_auc(traj_scores, targets),
                "average_precision": average_precision(traj_scores, targets),
            }
            auc_draws, ap_draws = [], []
            for pick in resamples:
                s, t = traj_scores[pick], targets[pick]
                auc_draws.append(roc_auc(s, t))
                ap_draws.append(average_precision(s, t))
            low, high = percentile_ci(auc_draws)
            entry["roc_auc_ci_low"], entry["roc_auc_ci_high"] = low, high
            low, high = percentile_ci(ap_draws)
            entry["average_precision_ci_low"] = low
            entry["average_precision_ci_high"] = high
            table.append(entry)
            print("  {0:<10} agg={1:<5} AUC={2:.4f} [{3:.4f}, {4:.4f}]".format(
                head_name, agg, entry["roc_auc"],
                entry["roc_auc_ci_low"], entry["roc_auc_ci_high"]))

    out_json = os.path.join(args.results_dir, "single_indist_metrics.json")
    with open(out_json, "w") as f:
        json.dump({"cache_dir": args.cache_dir,
                   "split_seed": args.split_seed,
                   "calibration_fraction": args.calibration_fraction,
                   "bootstrap_samples": args.bootstrap_samples,
                   "rows": table}, f, indent=2)
    out_csv = os.path.join(args.results_dir, "single_indist_metrics.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        writer.writeheader()
        writer.writerows(table)
    print("[saved] " + out_json + " / " + out_csv)


if __name__ == "__main__":
    main()
