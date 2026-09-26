"""Bootstrap confidence intervals for causal-prefix step metrics.

`analysis.bootstrap_step_metrics` covers the full-trajectory scores, but the
deployment-relevant question is which head ranks best when future steps are
hidden -- and there the gaps are small (attention leads mlp by ~0.005). A point
estimate cannot settle that.

This module reuses the `{head}_causal_predictions.pt` files that
`analysis.analyze_offline_pruning` already saved, so it runs no forward pass at
all. It resamples held-out trajectories and scores every head on the SAME
resample, giving paired intervals for both the causal AUC and the causal-minus-
full penalty.
"""

import argparse
import csv
import json
import os

import torch

from eval.eval_utils import (
    HEAD_CHOICES,
    average_precision,
    deterministic_example_split,
    roc_auc,
)


def percentile_ci(samples, alpha=0.05):
    tensor = torch.tensor(samples, dtype=torch.float64)
    tensor = tensor[torch.isfinite(tensor)]
    if tensor.numel() == 0:
        return float("nan"), float("nan")
    return (float(torch.quantile(tensor, alpha / 2)),
            float(torch.quantile(tensor, 1 - alpha / 2)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True,
                        help="Directory holding {head}_causal_predictions.pt")
    parser.add_argument("--heads", nargs="+", default=list(HEAD_CHOICES))
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    causal, full, labels, step_mask = {}, {}, None, None
    meta = None
    for head_name in args.heads:
        path = os.path.join(args.results_dir, head_name + "_causal_predictions.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location="cpu")
        causal[head_name] = payload["causal_q_values"]
        full[head_name] = payload["full_q_values"]
        if labels is None:
            labels, step_mask = payload["labels"], payload["step_mask"]
            meta = (payload["calibration_fraction"], payload["split_seed"])
        elif not torch.equal(payload["labels"], labels):
            raise RuntimeError("Label tensors differ between heads.")

    calibration_fraction, split_seed = meta
    _, test_mask = deterministic_example_split(
        labels.shape[0], calibration_fraction, split_seed
    )
    # A label without an embedding has no score to speak of.
    valid = ((labels == 0) | (labels == 1)) & step_mask & test_mask[:, None]

    n_traj, max_steps = valid.shape
    flat_positions = torch.arange(n_traj * max_steps).reshape(n_traj, max_steps)
    flat_labels = labels.reshape(-1)
    flat_causal = {h: s.reshape(-1) for h, s in causal.items()}
    flat_full = {h: s.reshape(-1) for h, s in full.items()}
    flat_valid = valid.reshape(-1)
    test_rows = test_mask.nonzero(as_tuple=True)[0]

    point = {}
    for head_name in args.heads:
        c = roc_auc(flat_causal[head_name][flat_valid], flat_labels[flat_valid])
        f = roc_auc(flat_full[head_name][flat_valid], flat_labels[flat_valid])
        point[head_name] = {
            "causal_roc_auc": c,
            "full_roc_auc": f,
            "causal_minus_full": c - f,
            "causal_average_precision": average_precision(
                flat_causal[head_name][flat_valid], flat_labels[flat_valid]
            ),
        }

    generator = torch.Generator().manual_seed(args.seed)
    draws = {h: {"causal_roc_auc": [], "causal_minus_full": []} for h in args.heads}
    n_test = test_rows.numel()
    for _ in range(args.bootstrap_samples):
        picked = test_rows[torch.randint(n_test, (n_test,), generator=generator)]
        rows_valid = valid[picked]
        positions = flat_positions[picked][rows_valid]
        target = flat_labels[positions]
        for head_name in args.heads:
            c = roc_auc(flat_causal[head_name][positions], target)
            f = roc_auc(flat_full[head_name][positions], target)
            draws[head_name]["causal_roc_auc"].append(c)
            draws[head_name]["causal_minus_full"].append(c - f)

    summary = {
        "results_dir": args.results_dir,
        "n_test_trajectories": int(n_test),
        "n_test_steps": int(valid.sum()),
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "heads": {},
    }
    for head_name in args.heads:
        entry = dict(point[head_name])
        for metric in ("causal_roc_auc", "causal_minus_full"):
            low, high = percentile_ci(draws[head_name][metric])
            entry[metric + "_ci_low"] = low
            entry[metric + "_ci_high"] = high
        summary["heads"][head_name] = entry

    pairs = []
    for i, a in enumerate(args.heads):
        for b in args.heads[i + 1:]:
            diffs = [x - y for x, y in zip(draws[a]["causal_roc_auc"],
                                          draws[b]["causal_roc_auc"])]
            finite = [d for d in diffs if d == d]
            low, high = percentile_ci(diffs)
            pairs.append({
                "head_a": a,
                "head_b": b,
                "causal_roc_auc_diff": point[a]["causal_roc_auc"] - point[b]["causal_roc_auc"],
                "causal_roc_auc_diff_ci_low": low,
                "causal_roc_auc_diff_ci_high": high,
                "causal_roc_auc_prob_a_better": (
                    sum(1 for d in finite if d > 0) / len(finite) if finite else float("nan")
                ),
                "causal_roc_auc_prob_a_worse": (
                    sum(1 for d in finite if d < 0) / len(finite) if finite else float("nan")
                ),
                "causal_roc_auc_significant": bool(low > 0 or high < 0),
            })
    summary["pairwise"] = pairs

    out_json = os.path.join(args.results_dir, "causal_metrics_ci.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    out_csv = os.path.join(args.results_dir, "causal_metrics_ci_pairwise.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(pairs[0].keys()))
        writer.writeheader()
        writer.writerows(pairs)
    print("[saved] " + out_json + " / " + out_csv)


if __name__ == "__main__":
    main()
