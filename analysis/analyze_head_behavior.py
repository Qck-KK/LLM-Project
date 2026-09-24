"""Post-hoc behavior analysis for trained reward heads.

The analysis reuses cached encoder embeddings. Original predictions are saved
once and then reused for length/error stratification and first-error boundary
analysis. Optional deterministic perturbations require head-only forward passes
but never run the encoder or retrain a model.
"""

import argparse
import csv
import glob
import json
import math
import os

import torch

from eval.eval_utils import (
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


PERTURBATIONS = ["reverse", "swap_adjacent", "mask_previous", "mask_first_error"]


def pad_and_concat(tensors, value):
    max_steps = max(tensor.shape[1] for tensor in tensors)
    return torch.cat([
        torch.nn.functional.pad(tensor, (0, max_steps - tensor.shape[1]), value=value)
        for tensor in tensors
    ])


def apply_perturbation(step_hidden, step_mask, labels, variant):
    if variant == "original":
        return step_hidden, step_mask, labels

    hidden = step_hidden.clone()
    changed_labels = labels.clone()
    for batch_index in range(hidden.shape[0]):
        length = int(step_mask[batch_index].sum())
        if length == 0:
            continue

        if variant in {"reverse", "swap_adjacent"}:
            order = torch.arange(length, device=hidden.device)
            if variant == "reverse":
                order = order.flip(0)
            else:
                for index in range(0, length - 1, 2):
                    order[index], order[index + 1] = order[index + 1].clone(), order[index].clone()
            hidden[batch_index, :length] = step_hidden[batch_index, order]
            changed_labels[batch_index, :length] = labels[batch_index, order]
        elif variant in {"mask_previous", "mask_first_error"}:
            errors = (labels[batch_index, :length] == 0).nonzero(as_tuple=True)[0]
            if errors.numel() == 0:
                continue
            first_error = int(errors[0])
            target = first_error if variant == "mask_first_error" else first_error - 1
            if target >= 0:
                hidden[batch_index, target] = 0
        else:
            raise ValueError(f"Unknown perturbation: {variant}")
    return hidden, step_mask, changed_labels


@torch.no_grad()
def score_cache(head, shard_paths, device, variant="original"):
    q_values, labels_list, masks = [], [], []
    head.eval()
    for path in shard_paths:
        shard = torch.load(path, map_location=device)
        step_hidden = shard["step_hidden"].to(device).float()
        step_mask = shard["step_mask"].to(device).bool()
        labels = shard["labels"].to(device).long()
        # Trajectories longer than the encoder's max_length lost their trailing
        # step markers, so the cache can carry labels for steps that have no
        # embedding. Scoring those positions feeds a zero vector to the head and
        # yields a constant, which silently corrupts every metric. A step without
        # an embedding is padding, so mark it as such.
        labels = labels.masked_fill(~step_mask, -100)
        step_hidden, step_mask, labels = apply_perturbation(
            step_hidden, step_mask, labels, variant
        )
        q_values.append(head(step_hidden, step_mask).cpu())
        labels_list.append(labels.cpu())
        masks.append(step_mask.cpu())
    return (
        pad_and_concat(q_values, 0.0),
        pad_and_concat(labels_list, -100),
        pad_and_concat(masks, False),
    )


def ranking_accuracy(q_values, labels, example_mask):
    total, correct = 0, 0
    for q_row, label_row in zip(q_values[example_mask], labels[example_mask]):
        positive = (label_row == 1).nonzero(as_tuple=True)[0]
        negative = (label_row == 0).nonzero(as_tuple=True)[0]
        if positive.numel() == 0 or negative.numel() == 0:
            continue
        comparisons = q_row[positive].unsqueeze(1) > q_row[negative].unsqueeze(0)
        correct += int(comparisons.sum())
        total += comparisons.numel()
    return correct / total if total else float("nan")


def metric_row(q_values, labels, example_mask, threshold):
    scores, targets = flatten_valid_steps(q_values, labels, example_mask)
    return {
        "n_trajectories": int(example_mask.sum()),
        "n_steps": int(targets.numel()),
        "accuracy": binary_accuracy(scores, targets, threshold),
        "balanced_accuracy": balanced_accuracy(scores, targets, threshold),
        "roc_auc": roc_auc(scores, targets),
        "average_precision": average_precision(scores, targets),
        "ranking_accuracy": ranking_accuracy(q_values, labels, example_mask),
    }


def build_group_masks(labels, test_mask):
    valid = (labels == 0) | (labels == 1)
    lengths = valid.sum(dim=1)
    error_counts = (labels == 0).sum(dim=1)
    test_lengths = lengths[test_mask].float()
    low = int(torch.quantile(test_lengths, 1 / 3).item())
    high = int(torch.quantile(test_lengths, 2 / 3).item())

    groups = {
        ("length", "short"): test_mask & (lengths <= low),
        ("length", "medium"): test_mask & (lengths > low) & (lengths <= high),
        ("length", "long"): test_mask & (lengths > high),
        ("error_count", "one_error"): test_mask & (error_counts == 1),
        ("error_count", "multiple_errors"): test_mask & (error_counts > 1),
    }

    early = torch.zeros_like(test_mask)
    middle = torch.zeros_like(test_mask)
    late = torch.zeros_like(test_mask)
    for index in test_mask.nonzero(as_tuple=True)[0].tolist():
        errors = (labels[index] == 0).nonzero(as_tuple=True)[0]
        if errors.numel() == 0:
            continue
        relative = errors[0].item() / max(int(lengths[index]) - 1, 1)
        target = early if relative < 1 / 3 else middle if relative < 2 / 3 else late
        target[index] = True
    groups.update({
        ("first_error_position", "early"): early,
        ("first_error_position", "middle"): middle,
        ("first_error_position", "late"): late,
    })
    return groups


def boundary_analysis(q_values, labels, calibration_mask, test_mask, max_offset, n_position_bins=5):
    calibration_scores, _ = flatten_valid_steps(q_values, labels, calibration_mask)
    center = calibration_scores.mean()
    scale = calibration_scores.std(unbiased=False).clamp(min=1e-6)
    standardized = (q_values - center) / scale

    pseudo_drops = {position_bin: [] for position_bin in range(n_position_bins)}
    for index in test_mask.nonzero(as_tuple=True)[0].tolist():
        length = int(((labels[index] == 0) | (labels[index] == 1)).sum())
        errors = (labels[index, :length] == 0).nonzero(as_tuple=True)[0]
        correct_prefix_end = int(errors[0]) if errors.numel() else length
        for step in range(1, correct_prefix_end):
            if labels[index, step - 1] == 1 and labels[index, step] == 1:
                relative = step / max(length - 1, 1)
                position_bin = min(int(relative * n_position_bins), n_position_bins - 1)
                pseudo_drops[position_bin].append(
                    float(standardized[index, step - 1] - standardized[index, step])
                )
    all_pseudo = [value for values in pseudo_drops.values() for value in values]
    global_pseudo = sum(all_pseudo) / len(all_pseudo) if all_pseudo else float("nan")

    curves = {offset: [] for offset in range(-max_offset, max_offset + 1)}
    real_drops, matched_pseudo = [], []
    for index in test_mask.nonzero(as_tuple=True)[0].tolist():
        length = int(((labels[index] == 0) | (labels[index] == 1)).sum())
        errors = (labels[index, :length] == 0).nonzero(as_tuple=True)[0]
        if errors.numel() == 0:
            continue
        first_error = int(errors[0])
        if first_error == 0:
            continue
        real_drops.append(float(standardized[index, first_error - 1] - standardized[index, first_error]))
        relative = first_error / max(length - 1, 1)
        position_bin = min(int(relative * n_position_bins), n_position_bins - 1)
        controls = pseudo_drops[position_bin]
        matched_pseudo.append(sum(controls) / len(controls) if controls else global_pseudo)
        for offset in curves:
            step = first_error + offset
            if 0 <= step < length:
                curves[offset].append(float(standardized[index, step]))

    curve_rows = []
    for offset, values in curves.items():
        tensor = torch.tensor(values)
        curve_rows.append({
            "offset": offset,
            "mean_standardized_q": float(tensor.mean()) if values else None,
            "std_standardized_q": float(tensor.std(unbiased=False)) if values else None,
            "n": len(values),
        })
    real_mean = sum(real_drops) / len(real_drops) if real_drops else float("nan")
    pseudo_mean = sum(matched_pseudo) / len(matched_pseudo) if matched_pseudo else float("nan")
    return {
        "first_error_boundary_drop": real_mean,
        "matched_correct_boundary_drop": pseudo_mean,
        "position_controlled_boundary_effect": real_mean - pseudo_mean,
        "n_error_boundaries": len(real_drops),
    }, curve_rows


def safe_json(value):
    if isinstance(value, dict):
        return {key: safe_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [safe_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, rows, columns):
    with open(path, "w") as f:
        f.write("| " + " | ".join(columns) + " |\n")
        f.write("|" + "---|" * len(columns) + "\n")
        for row in rows:
            values = []
            for column in columns:
                value = row.get(column, "")
                values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
            f.write("| " + " | ".join(values) + " |\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--heads", nargs="+", choices=HEAD_CHOICES, default=HEAD_CHOICES)
    parser.add_argument("--checkpoint_pattern", default="{head}_head.pt")
    parser.add_argument("--calibration_fraction", type=float, default=0.5)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--max_boundary_offset", type=int, default=2)
    parser.add_argument("--run_perturbations", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    device = get_device(args.device)
    shard_paths = sorted(glob.glob(os.path.join(args.cache_dir, "shard_*.pt")))
    if not shard_paths:
        raise FileNotFoundError(f"No cached shards in {args.cache_dir}")
    with open(os.path.join(args.cache_dir, "hidden_size.txt")) as f:
        hidden_size = int(f.read().strip())

    summary_rows, group_rows, curve_rows, perturbation_rows = [], [], [], []
    for head_name in args.heads:
        checkpoint = os.path.join(
            args.checkpoint_dir, args.checkpoint_pattern.format(head=head_name)
        )
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(checkpoint)
        head = build_reward_head(head_name, hidden_size).to(device)
        head.load_state_dict(torch.load(checkpoint, map_location=device))

        q_values, labels, step_mask = score_cache(head, shard_paths, device)
        calibration_mask, test_mask = deterministic_example_split(
            labels.shape[0], args.calibration_fraction, args.split_seed
        )
        calibration_q, calibration_labels = flatten_valid_steps(
            q_values, labels, calibration_mask
        )
        threshold, calibration_accuracy = best_threshold_accuracy(
            calibration_q, calibration_labels
        )
        overall = metric_row(q_values, labels, test_mask, threshold)
        boundary, head_curves = boundary_analysis(
            q_values, labels, calibration_mask, test_mask, args.max_boundary_offset
        )
        summary = {
            "head": head_name,
            "threshold": threshold,
            "calibration_accuracy": calibration_accuracy,
            **overall,
            **boundary,
        }
        summary_rows.append(summary)
        with open(os.path.join(args.results_dir, f"{head_name}_behavior_metrics.json"), "w") as f:
            json.dump(safe_json(summary), f, indent=2)
        torch.save(
            {"q_values": q_values, "labels": labels, "step_mask": step_mask,
             "split_seed": args.split_seed, "calibration_fraction": args.calibration_fraction},
            os.path.join(args.results_dir, f"{head_name}_step_predictions.pt"),
        )

        for row in head_curves:
            curve_rows.append({"head": head_name, **row})
        for (group_type, group_name), group_mask in build_group_masks(labels, test_mask).items():
            if group_mask.any():
                group_rows.append({
                    "head": head_name,
                    "group_type": group_type,
                    "group": group_name,
                    **metric_row(q_values, labels, group_mask, threshold),
                })

        original = overall
        perturbation_rows.append({
            "head": head_name, "variant": "original", **original,
            "roc_auc_delta": 0.0, "ranking_accuracy_delta": 0.0,
        })
        if args.run_perturbations:
            for variant in PERTURBATIONS:
                perturbed_q, perturbed_labels, _ = score_cache(
                    head, shard_paths, device, variant
                )
                metrics = metric_row(perturbed_q, perturbed_labels, test_mask, threshold)
                perturbation_rows.append({
                    "head": head_name,
                    "variant": variant,
                    **metrics,
                    "roc_auc_delta": metrics["roc_auc"] - original["roc_auc"],
                    "ranking_accuracy_delta": (
                        metrics["ranking_accuracy"] - original["ranking_accuracy"]
                    ),
                })

    write_csv(os.path.join(args.results_dir, "behavior_summary.csv"), summary_rows)
    write_csv(os.path.join(args.results_dir, "behavior_by_group.csv"), group_rows)
    write_csv(os.path.join(args.results_dir, "first_error_boundary_curves.csv"), curve_rows)
    write_csv(os.path.join(args.results_dir, "perturbation_results.csv"), perturbation_rows)
    write_markdown(
        os.path.join(args.results_dir, "behavior_summary.md"), summary_rows,
        ["head", "accuracy", "balanced_accuracy", "roc_auc", "average_precision",
         "ranking_accuracy", "first_error_boundary_drop",
         "matched_correct_boundary_drop", "position_controlled_boundary_effect"],
    )

    try:
        import matplotlib.pyplot as plt
        for head_name in args.heads:
            rows = [row for row in curve_rows if row["head"] == head_name]
            plt.plot([row["offset"] for row in rows],
                     [row["mean_standardized_q"] for row in rows], marker="o", label=head_name)
        plt.axvline(0, color="black", linestyle="--", alpha=0.5, label="first error")
        plt.xlabel("Step offset from first error")
        plt.ylabel("Mean standardized Q-value")
        plt.title("Reward behavior around the first error")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.results_dir, "first_error_boundary.png"), dpi=160)
        plt.close()

        group_orders = {
            "length": ["short", "medium", "long"],
            "first_error_position": ["early", "middle", "late"],
            "error_count": ["one_error", "multiple_errors"],
        }
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for axis, (group_type, order) in zip(axes, group_orders.items()):
            for head_name in args.heads:
                lookup = {
                    row["group"]: row["roc_auc"] for row in group_rows
                    if row["head"] == head_name and row["group_type"] == group_type
                }
                axis.plot(order, [lookup.get(name, float("nan")) for name in order],
                          marker="o", label=head_name)
            axis.set_title(group_type.replace("_", " "))
            axis.set_ylabel("Step ROC-AUC")
            axis.set_ylim(0, 1)
            axis.grid(alpha=0.25)
            axis.tick_params(axis="x", rotation=20)
        axes[-1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(args.results_dir, "behavior_by_group.png"), dpi=160)
        plt.close(fig)

        if args.run_perturbations:
            variants = PERTURBATIONS
            x = torch.arange(len(variants)).numpy()
            width = 0.8 / len(args.heads)
            fig, axis = plt.subplots(figsize=(10, 4.5))
            for head_index, head_name in enumerate(args.heads):
                lookup = {
                    row["variant"]: row["roc_auc_delta"] for row in perturbation_rows
                    if row["head"] == head_name
                }
                offsets = x - 0.4 + width / 2 + head_index * width
                axis.bar(offsets, [lookup.get(name, float("nan")) for name in variants],
                         width=width, label=head_name)
            axis.axhline(0, color="black", linewidth=1)
            axis.set_xticks(x, variants, rotation=20)
            axis.set_ylabel("ROC-AUC change from original")
            axis.set_title("Deterministic perturbation sensitivity")
            axis.grid(axis="y", alpha=0.25)
            axis.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(os.path.join(args.results_dir, "perturbation_sensitivity.png"), dpi=160)
            plt.close(fig)
    except ImportError:
        print("[warn] matplotlib is not installed; skipping behavior plots.")

    print(f"[saved] behavior analyses under {args.results_dir}")


if __name__ == "__main__":
    main()
