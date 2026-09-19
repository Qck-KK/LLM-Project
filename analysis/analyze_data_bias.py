"""Analyze label/position bias and deterministic no-model baselines.

This script reads only labels and masks from a validation cache. It does not
load an encoder, a reward-head checkpoint, or train a model.
"""

import argparse
import csv
import json
import math
import os

import torch

from eval.eval_utils import (
    average_precision,
    balanced_accuracy,
    binary_accuracy,
    deterministic_example_split,
    load_step_cache_labels,
    roc_auc,
)


def relative_position_bins(labels, example_mask, n_bins):
    rows = []
    for example_index in example_mask.nonzero(as_tuple=True)[0].tolist():
        valid = ((labels[example_index] == 0) | (labels[example_index] == 1)).nonzero(as_tuple=True)[0]
        n_steps = len(valid)
        for local_index, step_index in enumerate(valid.tolist()):
            relative_position = local_index / max(n_steps - 1, 1)
            position_bin = min(int(relative_position * n_bins), n_bins - 1)
            rows.append((example_index, step_index, relative_position, position_bin,
                         int(labels[example_index, step_index])))
    return rows


def baseline_metrics(scores, labels, threshold=0.5):
    return {
        "accuracy": binary_accuracy(scores, labels, threshold),
        "balanced_accuracy": balanced_accuracy(scores, labels, threshold),
        "roc_auc": roc_auc(scores, labels),
        "average_precision": average_precision(scores, labels),
        "n_steps": int(labels.numel()),
    }


def finite_or_none(value):
    if isinstance(value, dict):
        return {key: finite_or_none(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_or_none(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--position_bins", type=int, default=5)
    parser.add_argument("--calibration_fraction", type=float, default=0.5)
    parser.add_argument("--split_seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    labels, step_mask = load_step_cache_labels(args.cache_dir)
    calibration_mask, test_mask = deterministic_example_split(
        labels.shape[0], args.calibration_fraction, args.split_seed
    )
    all_examples = torch.ones(labels.shape[0], dtype=torch.bool)
    all_rows = relative_position_bins(labels, all_examples, args.position_bins)
    calibration_rows = relative_position_bins(labels, calibration_mask, args.position_bins)
    test_rows = relative_position_bins(labels, test_mask, args.position_bins)

    valid_labels = labels[(labels == 0) | (labels == 1)]
    lengths = ((labels == 0) | (labels == 1)).sum(dim=1)
    n_positive = int((valid_labels == 1).sum())
    n_negative = int((valid_labels == 0).sum())
    trajectories_with_positive = int((labels == 1).any(dim=1).sum())
    trajectories_with_negative = int((labels == 0).any(dim=1).sum())
    trajectories_with_both = int(((labels == 1).any(dim=1) & (labels == 0).any(dim=1)).sum())

    transitions = {"correct_to_error": 0, "error_to_correct": 0}
    first_error_positions = []
    for row, length in zip(labels, lengths.tolist()):
        sequence = row[:length]
        if length > 1:
            transitions["correct_to_error"] += int(((sequence[:-1] == 1) & (sequence[1:] == 0)).sum())
            transitions["error_to_correct"] += int(((sequence[:-1] == 0) & (sequence[1:] == 1)).sum())
        errors = (sequence == 0).nonzero(as_tuple=True)[0]
        if errors.numel():
            first_error_positions.append(errors[0].item() / max(length - 1, 1))

    position_rows = []
    for position_bin in range(args.position_bins):
        values = [row[4] for row in all_rows if row[3] == position_bin]
        position_rows.append({
            "position_bin": position_bin,
            "relative_position_start": position_bin / args.position_bins,
            "relative_position_end": (position_bin + 1) / args.position_bins,
            "n_steps": len(values),
            "correct_rate": sum(values) / len(values) if values else None,
            "error_rate": 1.0 - sum(values) / len(values) if values else None,
        })

    calibration_labels = torch.tensor([row[4] for row in calibration_rows], dtype=torch.long)
    test_labels = torch.tensor([row[4] for row in test_rows], dtype=torch.long)
    majority_probability = calibration_labels.float().mean().item()
    majority_score = torch.full((len(test_rows),), majority_probability)

    global_rate = majority_probability
    bin_rates = {}
    for position_bin in range(args.position_bins):
        values = [row[4] for row in calibration_rows if row[3] == position_bin]
        bin_rates[position_bin] = sum(values) / len(values) if values else global_rate
    position_scores = torch.tensor([bin_rates[row[3]] for row in test_rows], dtype=torch.float32)

    baselines = {
        "split_seed": args.split_seed,
        "calibration_fraction": args.calibration_fraction,
        "majority": baseline_metrics(majority_score, test_labels),
        "position_only": baseline_metrics(position_scores, test_labels),
        "position_bin_calibration_correct_rates": bin_rates,
    }

    summary = {
        "n_trajectories": int(labels.shape[0]),
        "n_calibration_trajectories": int(calibration_mask.sum()),
        "n_test_trajectories": int(test_mask.sum()),
        "n_steps": int(valid_labels.numel()),
        "n_correct_steps": n_positive,
        "n_error_steps": n_negative,
        "correct_rate": n_positive / max(n_positive + n_negative, 1),
        "trajectories_with_positive": trajectories_with_positive,
        "trajectories_with_negative": trajectories_with_negative,
        "trajectories_with_both": trajectories_with_both,
        "length_min": int(lengths.min()),
        "length_median": float(lengths.float().median()),
        "length_mean": float(lengths.float().mean()),
        "length_max": int(lengths.max()),
        "first_error_relative_position_mean": (
            sum(first_error_positions) / len(first_error_positions) if first_error_positions else None
        ),
        **transitions,
    }

    with open(os.path.join(args.results_dir, "data_bias.json"), "w") as f:
        json.dump(finite_or_none(summary), f, indent=2)
    with open(os.path.join(args.results_dir, "deterministic_baselines.json"), "w") as f:
        json.dump(finite_or_none(baselines), f, indent=2)
    with open(os.path.join(args.results_dir, "position_label_rates.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(position_rows[0]))
        writer.writeheader()
        writer.writerows(position_rows)

    try:
        import matplotlib.pyplot as plt
        centers = [(row["relative_position_start"] + row["relative_position_end"]) / 2
                   for row in position_rows]
        rates = [row["error_rate"] for row in position_rows]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(centers, rates, marker="o")
        axes[0].set(xlabel="Relative step position", ylabel="Error rate",
                    title="Label position bias", ylim=(0, 1))
        axes[0].grid(alpha=0.25)
        axes[1].hist(first_error_positions, bins=args.position_bins, range=(0, 1))
        axes[1].set(xlabel="Relative position of first error", ylabel="Trajectories",
                    title="First-error position")
        fig.tight_layout()
        fig.savefig(os.path.join(args.results_dir, "data_bias.png"), dpi=160)
        plt.close(fig)
    except ImportError:
        print("[warn] matplotlib is not installed; skipping data-bias plot.")

    print(json.dumps(summary, indent=2))
    print(json.dumps(baselines, indent=2))


if __name__ == "__main__":
    main()
