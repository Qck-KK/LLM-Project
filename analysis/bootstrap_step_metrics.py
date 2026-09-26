"""Trajectory-level bootstrap confidence intervals for held-out step metrics.

Step-level ROC-AUC and Average Precision are this project's primary metrics, but
a point estimate alone cannot say whether a 0.005 gap between two heads is real.
This module resamples held-out *trajectories* -- never individual steps, which
are correlated within a solution -- and scores every head on the SAME resample,
so the paired difference between two heads gets a proper interval as well.

It runs one light forward pass per head; it never retrains and never runs the
frozen encoder.
"""

import argparse
import csv
import glob
import json
import os

import torch

from eval.eval_utils import (
    HEAD_CHOICES,
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
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--checkpoint_pattern", default="{head}_head.pt")
    parser.add_argument("--heads", nargs="+", default=list(HEAD_CHOICES))
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
    # A label whose step has no embedding cannot be scored; treat it as padding.
    labels = labels.masked_fill(~step_mask, -100)
    del shards

    _, test_mask = deterministic_example_split(
        labels.shape[0], args.calibration_fraction, args.split_seed
    )
    valid = ((labels == 0) | (labels == 1)) & test_mask[:, None]

    scores = {}
    for head_name in args.heads:
        checkpoint = os.path.join(
            args.checkpoint_dir, args.checkpoint_pattern.format(head=head_name)
        )
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(checkpoint)
        head = build_reward_head(head_name, hidden_size).to(device)
        head.load_state_dict(torch.load(checkpoint, map_location=device))
        scores[head_name] = score_cache(head, shard_paths, device)

    n_traj, max_steps = valid.shape
    flat_positions = torch.arange(n_traj * max_steps).reshape(n_traj, max_steps)
    flat_labels = labels.reshape(-1)
    flat_scores = {h: s.reshape(-1) for h, s in scores.items()}
    flat_valid = valid.reshape(-1)
    test_rows = test_mask.nonzero(as_tuple=True)[0]

    point = {}
    for head_name in args.heads:
        point[head_name] = {
            "roc_auc": roc_auc(flat_scores[head_name][flat_valid], flat_labels[flat_valid]),
            "average_precision": average_precision(
                flat_scores[head_name][flat_valid], flat_labels[flat_valid]
            ),
        }

    generator = torch.Generator().manual_seed(args.seed)
    draws = {h: {"roc_auc": [], "average_precision": []} for h in args.heads}
    n_test = test_rows.numel()
    for _ in range(args.bootstrap_samples):
        picked = test_rows[torch.randint(n_test, (n_test,), generator=generator)]
        rows_valid = valid[picked]
        positions = flat_positions[picked][rows_valid]
        target = flat_labels[positions]
        for head_name in args.heads:
            sample_scores = flat_scores[head_name][positions]
            draws[head_name]["roc_auc"].append(roc_auc(sample_scores, target))
            draws[head_name]["average_precision"].append(
                average_precision(sample_scores, target)
            )

    summary = {
        "cache_dir": args.cache_dir,
        "checkpoint_dir": args.checkpoint_dir,
        "n_test_trajectories": int(n_test),
        "n_test_steps": int(valid.sum()),
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "heads": {},
    }
    for head_name in args.heads:
        entry = {}
        for metric in ("roc_auc", "average_precision"):
            low, high = percentile_ci(draws[head_name][metric])
            entry[metric] = point[head_name][metric]
            entry[metric + "_ci_low"] = low
            entry[metric + "_ci_high"] = high
        summary["heads"][head_name] = entry

    pairs = []
    for i, a in enumerate(args.heads):
        for b in args.heads[i + 1:]:
            row = {"head_a": a, "head_b": b}
            for metric in ("roc_auc", "average_precision"):
                diffs = [x - y for x, y in zip(draws[a][metric], draws[b][metric])]
                finite = [d for d in diffs if d == d]
                low, high = percentile_ci(diffs)
                row[metric + "_diff"] = point[a][metric] - point[b][metric]
                row[metric + "_diff_ci_low"] = low
                row[metric + "_diff_ci_high"] = high
                row[metric + "_prob_a_better"] = (
                    sum(1 for d in finite if d > 0) / len(finite) if finite else float("nan")
                )
                row[metric + "_prob_a_worse"] = (
                    sum(1 for d in finite if d < 0) / len(finite) if finite else float("nan")
                )
                row[metric + "_significant"] = bool(low > 0 or high < 0)
            pairs.append(row)
    summary["pairwise"] = pairs

    with open(os.path.join(args.results_dir, "step_metrics_ci.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.results_dir, "step_metrics_ci.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["head", "roc_auc", "roc_auc_ci_low", "roc_auc_ci_high",
                         "average_precision", "average_precision_ci_low",
                         "average_precision_ci_high"])
        for head_name in args.heads:
            e = summary["heads"][head_name]
            writer.writerow([head_name, e["roc_auc"], e["roc_auc_ci_low"],
                             e["roc_auc_ci_high"], e["average_precision"],
                             e["average_precision_ci_low"], e["average_precision_ci_high"]])
    if pairs:  # a single head has nothing to pair against
        with open(os.path.join(args.results_dir, "step_metrics_ci_pairwise.csv"),
                  "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(pairs[0].keys()))
            writer.writeheader()
            writer.writerows(pairs)
    print("[saved] " + args.results_dir + "/step_metrics_ci.json / .csv / _pairwise.csv")


if __name__ == "__main__":
    main()
