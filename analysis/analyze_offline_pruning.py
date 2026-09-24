"""Causal-prefix diagnostics and offline heuristic-pruning simulation.

The cached Qwen step representations are causal, but contextual reward heads
can read later step embeddings when they score a complete trajectory.  This
module therefore reruns each lightweight head on every available prefix and
uses only the final score of that prefix for pruning decisions.  It never runs
the frozen encoder and never generates new text.
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
    deterministic_example_split,
    get_device,
    roc_auc,
)
from reward_heads import build_reward_head


POLICIES = ("single_low", "two_consecutive")


def pad_and_concat(tensors, value):
    max_steps = max(tensor.shape[1] for tensor in tensors)
    return torch.cat([
        torch.nn.functional.pad(tensor, (0, max_steps - tensor.shape[1]), value=value)
        for tensor in tensors
    ])


@torch.no_grad()
def causal_prefix_scores(head, step_hidden, step_mask):
    """Score step t from H[:t+1], preventing a contextual head seeing the future."""
    head.eval()
    batch_size, max_steps, _ = step_hidden.shape
    scores = step_hidden.new_zeros(batch_size, max_steps)
    for step in range(max_steps):
        active = step_mask[:, step]
        if not active.any():
            continue
        prefix_scores = head(
            step_hidden[active, :step + 1],
            step_mask[active, :step + 1],
        )
        scores[active, step] = prefix_scores[:, -1]
    return scores


@torch.no_grad()
def score_cache_causally(head, shard_paths, device):
    causal_list, full_list, labels_list, mask_list = [], [], [], []
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
        causal_list.append(causal_prefix_scores(head, step_hidden, step_mask).cpu())
        full_list.append(head(step_hidden, step_mask).cpu())
        labels_list.append(labels.cpu())
        mask_list.append(step_mask.cpu())
    return (
        pad_and_concat(causal_list, 0.0),
        pad_and_concat(full_list, 0.0),
        pad_and_concat(labels_list, -100),
        pad_and_concat(mask_list, False),
    )


def classify_trajectory(label_row):
    valid = label_row[(label_row == 0) | (label_row == 1)]
    length = int(valid.numel())
    errors = (valid == 0).nonzero(as_tuple=True)[0]
    if errors.numel() == 0:
        return {"kind": "clean", "length": length, "first_error": None}
    first_error = int(errors[0])
    kind = "recovery" if (valid[first_error + 1:] == 1).any() else "monotone_error"
    return {"kind": kind, "length": length, "first_error": first_error}


def trajectory_metadata(labels):
    return [classify_trajectory(row) for row in labels]


def stopping_index(scores, threshold, policy):
    low = scores < threshold
    if policy == "single_low":
        hits = low.nonzero(as_tuple=True)[0]
        return int(hits[0]) if hits.numel() else None
    if policy == "two_consecutive":
        if scores.numel() < 2:
            return None
        hits = (low[:-1] & low[1:]).nonzero(as_tuple=True)[0]
        return int(hits[0]) + 1 if hits.numel() else None
    raise ValueError(f"Unknown pruning policy: {policy}")


def trigger_statistic(scores, policy):
    """Smallest threshold that can trigger a trajectory under a policy."""
    if policy == "single_low":
        return float(scores.min())
    if policy == "two_consecutive":
        if scores.numel() < 2:
            return float("inf")
        return float(torch.maximum(scores[:-1], scores[1:]).min())
    raise ValueError(f"Unknown pruning policy: {policy}")


def calibrate_threshold(scores, labels, calibration_mask, policy, false_prune_budget):
    """Choose the most aggressive threshold within a clean-trajectory risk budget."""
    metadata = trajectory_metadata(labels)
    statistics = []
    for index in calibration_mask.nonzero(as_tuple=True)[0].tolist():
        meta = metadata[index]
        if meta["kind"] != "clean" or meta["length"] == 0:
            continue
        statistics.append(trigger_statistic(scores[index, :meta["length"]], policy))
    if not statistics:
        raise ValueError("Calibration split contains no all-correct trajectories.")

    finite = sorted(set(value for value in statistics if math.isfinite(value)))
    if not finite:
        return float("-inf"), 0.0, len(statistics)
    score_dtype = scores.dtype if scores.dtype.is_floating_point else torch.float32
    negative_infinity = torch.tensor(float("-inf"), dtype=score_dtype)
    positive_infinity = torch.tensor(float("inf"), dtype=score_dtype)
    candidates = [
        float(torch.nextafter(torch.tensor(finite[0], dtype=score_dtype), negative_infinity))
    ]
    candidates.extend(
        float(torch.nextafter(torch.tensor(value, dtype=score_dtype), positive_infinity))
        for value in finite
    )
    best_threshold, best_rate = candidates[0], 0.0
    for candidate in candidates:
        false_prune_rate = sum(value < candidate for value in statistics) / len(statistics)
        if false_prune_rate <= false_prune_budget + 1e-12:
            best_threshold, best_rate = candidate, false_prune_rate
    return best_threshold, best_rate, len(statistics)


def build_pruning_records(scores, labels, example_indices, threshold, policy):
    metadata = trajectory_metadata(labels)
    records = []
    for index in example_indices:
        meta = metadata[index]
        length = meta["length"]
        if length == 0:
            continue
        stop = stopping_index(scores[index, :length], threshold, policy)
        first_error = meta["first_error"]
        saved = length - (stop + 1) if stop is not None else 0
        delay = None if stop is None or first_error is None else stop - first_error
        records.append({
            "index": index,
            "kind": meta["kind"],
            "length": length,
            "first_error": first_error,
            "stop": stop,
            "saved": saved,
            "delay": delay,
        })
    return records


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else float("nan")


def aggregate_pruning_records(records):
    clean = [record for record in records if record["kind"] == "clean"]
    errors = [record for record in records if record["kind"] == "monotone_error"]
    recovery = [record for record in records if record["kind"] == "recovery"]
    primary = clean + errors

    clean_pruned = sum(record["stop"] is not None for record in clean)
    pre_error = sum(
        record["stop"] is not None and record["stop"] < record["first_error"]
        for record in errors
    )
    detected = [
        record for record in errors
        if record["stop"] is not None and record["stop"] >= record["first_error"]
    ]
    delays = torch.tensor([record["delay"] for record in detected], dtype=torch.float32)
    total_steps = sum(record["length"] for record in primary)
    error_steps = sum(record["length"] for record in errors)
    total_saved = sum(record["saved"] for record in primary)
    safe_saved = sum(record["saved"] for record in detected)
    oracle_saved = sum(
        record["length"] - (record["first_error"] + 1) for record in errors
    )

    return {
        "n_trajectories": len(records),
        "n_primary_trajectories": len(primary),
        "n_clean": len(clean),
        "n_monotone_error": len(errors),
        "n_recovery": len(recovery),
        "clean_false_prune_rate": _ratio(clean_pruned, len(clean)),
        "pre_error_false_prune_rate": _ratio(pre_error, len(errors)),
        "error_coverage": _ratio(len(detected), len(errors)),
        "detection_at_0": _ratio(sum(record["delay"] == 0 for record in detected), len(errors)),
        "detection_at_1": _ratio(sum(record["delay"] <= 1 for record in detected), len(errors)),
        "detection_at_2": _ratio(sum(record["delay"] <= 2 for record in detected), len(errors)),
        "mean_detection_delay": float(delays.mean()) if delays.numel() else float("nan"),
        "median_detection_delay": float(delays.median()) if delays.numel() else float("nan"),
        "theoretical_step_saving_rate": _ratio(total_saved, total_steps),
        "safe_step_saving_rate": _ratio(safe_saved, total_steps),
        "error_only_safe_saving_rate": _ratio(safe_saved, error_steps),
        "oracle_step_saving_rate": _ratio(oracle_saved, total_steps),
        "oracle_efficiency_ratio": _ratio(safe_saved, oracle_saved),
        "recovery_prune_rate": _ratio(
            sum(record["stop"] is not None for record in recovery), len(recovery)
        ),
    }


def bootstrap_confidence_intervals(records, n_samples, seed):
    if n_samples <= 0 or not records:
        return {}
    generator = torch.Generator().manual_seed(seed)
    metric_names = [
        "clean_false_prune_rate",
        "pre_error_false_prune_rate",
        "error_coverage",
        "detection_at_0",
        "detection_at_1",
        "detection_at_2",
        "safe_step_saving_rate",
        "oracle_efficiency_ratio",
    ]
    samples = {name: [] for name in metric_names}
    strata = {
        kind: [record for record in records if record["kind"] == kind]
        for kind in ("clean", "monotone_error", "recovery")
    }
    for _ in range(n_samples):
        sample = []
        for stratum in strata.values():
            if not stratum:
                continue
            indices = torch.randint(len(stratum), (len(stratum),), generator=generator).tolist()
            sample.extend(stratum[index] for index in indices)
        metrics = aggregate_pruning_records(sample)
        for name in metric_names:
            value = metrics[name]
            if math.isfinite(value):
                samples[name].append(value)
    intervals = {}
    for name, values in samples.items():
        if not values:
            continue
        tensor = torch.tensor(values)
        intervals[f"{name}_ci_low"] = float(torch.quantile(tensor, 0.025))
        intervals[f"{name}_ci_high"] = float(torch.quantile(tensor, 0.975))
    return intervals


def causal_diagnostics(causal_scores, full_scores, labels, test_mask):
    selected_labels = labels[test_mask]
    valid = (selected_labels == 0) | (selected_labels == 1)
    causal = causal_scores[test_mask][valid].float()
    full = full_scores[test_mask][valid].float()
    targets = selected_labels[valid]
    if causal.numel() > 1 and causal.std(unbiased=False) > 0 and full.std(unbiased=False) > 0:
        correlation = float(torch.corrcoef(torch.stack([causal, full]))[0, 1])
    else:
        correlation = float("nan")
    causal_auc = roc_auc(causal, targets)
    full_auc = roc_auc(full, targets)
    return {
        "full_causal_mean_abs_difference": float((full - causal).abs().mean()),
        "full_causal_correlation": correlation,
        "causal_step_roc_auc": causal_auc,
        "full_step_roc_auc": full_auc,
        "causal_minus_full_roc_auc": causal_auc - full_auc,
    }


def group_indices(labels, test_mask):
    metadata = trajectory_metadata(labels)
    test_indices = test_mask.nonzero(as_tuple=True)[0].tolist()
    primary = [index for index in test_indices if metadata[index]["kind"] != "recovery"]
    lengths = torch.tensor([metadata[index]["length"] for index in primary], dtype=torch.float32)
    groups = []
    if lengths.numel():
        low = int(torch.quantile(lengths, 1 / 3))
        high = int(torch.quantile(lengths, 2 / 3))
        groups.extend([
            ("length", "short", [i for i in primary if metadata[i]["length"] <= low]),
            ("length", "medium", [i for i in primary if low < metadata[i]["length"] <= high]),
            ("length", "long", [i for i in primary if metadata[i]["length"] > high]),
        ])
    error_indices = [
        index for index in test_indices if metadata[index]["kind"] == "monotone_error"
    ]
    for group_name, lower, upper in (
        ("early", 0.0, 1 / 3),
        ("middle", 1 / 3, 2 / 3),
        ("late", 2 / 3, 1.000001),
    ):
        selected = []
        for index in error_indices:
            meta = metadata[index]
            relative = meta["first_error"] / max(meta["length"] - 1, 1)
            if lower <= relative < upper:
                selected.append(index)
        groups.append(("first_error_position", group_name, selected))
    return [(group_type, name, indices) for group_type, name, indices in groups if indices]


def finite_or_none(value):
    if isinstance(value, dict):
        return {key: finite_or_none(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_or_none(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_csv(path, rows):
    if not rows:
        return
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(finite_or_none(rows))


def write_markdown(path, rows, columns):
    with open(path, "w") as file:
        file.write("| " + " | ".join(columns) + " |\n")
        file.write("|" + "---|" * len(columns) + "\n")
        for row in rows:
            values = []
            for column in columns:
                value = row.get(column, "")
                if isinstance(value, float):
                    value = "" if not math.isfinite(value) else f"{value:.4f}"
                values.append(str(value))
            file.write("| " + " | ".join(values) + " |\n")


def add_reference_rows(labels, test_mask, rows):
    indices = test_mask.nonzero(as_tuple=True)[0].tolist()
    metadata = trajectory_metadata(labels)
    records = []
    for index in indices:
        meta = metadata[index]
        records.append({
            "index": index,
            "kind": meta["kind"],
            "length": meta["length"],
            "first_error": meta["first_error"],
            "stop": None,
            "saved": 0,
            "delay": None,
        })
    no_pruning = aggregate_pruning_records(records)
    rows.append({"head": "no_pruning", "policy": "none", "budget": 0.0,
                 "threshold": None, **no_pruning})

    oracle_records = []
    for record in records:
        oracle = dict(record)
        if oracle["kind"] == "monotone_error":
            oracle["stop"] = oracle["first_error"]
            oracle["saved"] = oracle["length"] - (oracle["stop"] + 1)
            oracle["delay"] = 0
        oracle_records.append(oracle)
    oracle = aggregate_pruning_records(oracle_records)
    rows.append({"head": "oracle_first_error", "policy": "oracle", "budget": 0.0,
                 "threshold": None, **oracle})


def save_plots(results_dir, rows, diagnostics, primary_budget, primary_policy):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib is not installed; skipping pruning plots.")
        return

    model_rows = [
        row for row in rows
        if row["head"] not in {"no_pruning", "oracle_first_error"}
    ]
    heads = sorted(set(row["head"] for row in model_rows))
    colors = {head: plt.cm.tab10(index % 10) for index, head in enumerate(heads)}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for head in heads:
        for policy, linestyle in (("single_low", "-"), ("two_consecutive", "--")):
            selected = sorted(
                [row for row in model_rows if row["head"] == head and row["policy"] == policy],
                key=lambda row: row["clean_false_prune_rate"],
            )
            if not selected:
                continue
            label = f"{head}/{policy}"
            x = [row["clean_false_prune_rate"] for row in selected]
            axes[0].plot(x, [row["safe_step_saving_rate"] for row in selected],
                         marker="o", linestyle=linestyle, color=colors[head], label=label)
            axes[1].plot(x, [row["error_coverage"] for row in selected],
                         marker="o", linestyle=linestyle, color=colors[head], label=label)
    axes[0].set(xlabel="Clean trajectory false-prune rate",
                ylabel="Safe step saving rate", title="Pruning risk vs. safe saving")
    axes[1].set(xlabel="Clean trajectory false-prune rate",
                ylabel="Error trajectory coverage", title="Pruning risk vs. error coverage")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.set_xlim(left=0)
        axis.set_ylim(bottom=0)
    axes[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "pruning_tradeoff.png"), dpi=160)
    plt.close(fig)

    selected = [
        row for row in model_rows
        if row["policy"] == primary_policy and abs(row["budget"] - primary_budget) < 1e-9
    ]
    if selected:
        fig, axis = plt.subplots(figsize=(8, 4.5))
        x = torch.arange(len(selected)).numpy()
        width = 0.25
        for offset, metric, label in (
            (-width, "detection_at_0", "Detection@0"),
            (0, "detection_at_1", "Detection@1"),
            (width, "detection_at_2", "Detection@2"),
        ):
            axis.bar(x + offset, [row[metric] for row in selected], width=width, label=label)
        axis.set_xticks(x, [row["head"] for row in selected], rotation=20)
        axis.set_ylabel("Fraction of monotone-error trajectories")
        axis.set_title(f"Detection delay at {primary_budget:.0%} clean-risk budget")
        axis.set_ylim(0, 1)
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(results_dir, "pruning_detection_delay.png"), dpi=160)
        plt.close(fig)

    if diagnostics:
        fig, axis = plt.subplots(figsize=(8, 4.5))
        names = [row["head"] for row in diagnostics]
        values = [row["full_causal_mean_abs_difference"] for row in diagnostics]
        axis.bar(names, values)
        axis.set_ylabel("Mean |full score - causal-prefix score|")
        axis.set_title("Future-context sensitivity of reward heads")
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(os.path.join(results_dir, "full_vs_causal_scores.png"), dpi=160)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--heads", nargs="+", choices=HEAD_CHOICES, default=HEAD_CHOICES)
    parser.add_argument("--checkpoint_pattern", default="{head}_head.pt")
    parser.add_argument("--calibration_fraction", type=float, default=0.5)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--budgets", nargs="+", type=float, default=[0.01, 0.05, 0.10])
    parser.add_argument("--primary_budget", type=float, default=0.05)
    parser.add_argument("--primary_policy", choices=POLICIES, default="single_low")
    parser.add_argument("--bootstrap_samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if any(budget < 0 or budget > 1 for budget in args.budgets):
        raise ValueError("All false-prune budgets must be between 0 and 1.")
    if not any(abs(args.primary_budget - budget) < 1e-12 for budget in args.budgets):
        raise ValueError("primary_budget must also appear in --budgets.")

    os.makedirs(args.results_dir, exist_ok=True)
    device = get_device(args.device)
    shard_paths = sorted(glob.glob(os.path.join(args.cache_dir, "shard_*.pt")))
    if not shard_paths:
        raise FileNotFoundError(f"No cached shards in {args.cache_dir}")
    with open(os.path.join(args.cache_dir, "hidden_size.txt")) as file:
        hidden_size = int(file.read().strip())

    rows, group_rows, diagnostic_rows = [], [], []
    thresholds = {}
    shared_labels = shared_mask = calibration_mask = test_mask = None
    per_head_primary = {}

    for head_index, head_name in enumerate(args.heads):
        checkpoint = os.path.join(
            args.checkpoint_dir, args.checkpoint_pattern.format(head=head_name)
        )
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(checkpoint)
        head = build_reward_head(head_name, hidden_size).to(device)
        head.load_state_dict(torch.load(checkpoint, map_location=device))
        causal, full, labels, step_mask = score_cache_causally(head, shard_paths, device)
        if shared_labels is None:
            shared_labels, shared_mask = labels, step_mask
            calibration_mask, test_mask = deterministic_example_split(
                labels.shape[0], args.calibration_fraction, args.split_seed
            )
        elif not torch.equal(labels, shared_labels):
            raise RuntimeError("Validation labels changed while scoring heads.")

        torch.save({
            "causal_q_values": causal,
            "full_q_values": full,
            "labels": labels,
            "step_mask": step_mask,
            "split_seed": args.split_seed,
            "calibration_fraction": args.calibration_fraction,
        }, os.path.join(args.results_dir, f"{head_name}_causal_predictions.pt"))

        diagnostic = {"head": head_name, **causal_diagnostics(causal, full, labels, test_mask)}
        diagnostic_rows.append(diagnostic)
        thresholds[head_name] = {}
        test_indices = test_mask.nonzero(as_tuple=True)[0].tolist()
        for policy_index, policy in enumerate(POLICIES):
            thresholds[head_name][policy] = {}
            for budget_index, budget in enumerate(args.budgets):
                threshold, calibration_rate, n_clean_calibration = calibrate_threshold(
                    causal, labels, calibration_mask, policy, budget
                )
                thresholds[head_name][policy][str(budget)] = {
                    "threshold": threshold,
                    "observed_clean_false_prune_rate": calibration_rate,
                    "n_clean_calibration": n_clean_calibration,
                }
                records = build_pruning_records(
                    causal, labels, test_indices, threshold, policy
                )
                metrics = aggregate_pruning_records(records)
                is_primary = (
                    policy == args.primary_policy
                    and abs(budget - args.primary_budget) < 1e-9
                )
                intervals = bootstrap_confidence_intervals(
                    records,
                    args.bootstrap_samples if is_primary else 0,
                    args.seed + head_index * 100 + policy_index * 10 + budget_index,
                )
                row = {
                    "head": head_name,
                    "policy": policy,
                    "budget": budget,
                    "threshold": threshold,
                    "calibration_clean_false_prune_rate": calibration_rate,
                    **metrics,
                    **intervals,
                }
                rows.append(row)

                if is_primary:
                    per_head_primary[head_name] = {**diagnostic, **row}
                    for group_type, group_name, indices in group_indices(labels, test_mask):
                        group_records = build_pruning_records(
                            causal, labels, indices, threshold, policy
                        )
                        group_rows.append({
                            "head": head_name,
                            "policy": policy,
                            "budget": budget,
                            "group_type": group_type,
                            "group": group_name,
                            **aggregate_pruning_records(group_records),
                        })

    # A fixed random-score comparator is calibrated with the same protocol.
    generator = torch.Generator().manual_seed(args.seed)
    random_scores = torch.rand(shared_labels.shape, generator=generator)
    random_scores = random_scores.masked_fill(~shared_mask, 0.0)
    thresholds["random_score"] = {}
    test_indices = test_mask.nonzero(as_tuple=True)[0].tolist()
    for policy in POLICIES:
        thresholds["random_score"][policy] = {}
        for budget in args.budgets:
            threshold, calibration_rate, n_clean_calibration = calibrate_threshold(
                random_scores, shared_labels, calibration_mask, policy, budget
            )
            thresholds["random_score"][policy][str(budget)] = {
                "threshold": threshold,
                "observed_clean_false_prune_rate": calibration_rate,
                "n_clean_calibration": n_clean_calibration,
            }
            records = build_pruning_records(
                random_scores, shared_labels, test_indices, threshold, policy
            )
            rows.append({
                "head": "random_score",
                "policy": policy,
                "budget": budget,
                "threshold": threshold,
                "calibration_clean_false_prune_rate": calibration_rate,
                **aggregate_pruning_records(records),
            })

    add_reference_rows(shared_labels, test_mask, rows)

    with open(os.path.join(args.results_dir, "pruning_thresholds.json"), "w") as file:
        json.dump(finite_or_none(thresholds), file, indent=2)
    for head_name, primary in per_head_primary.items():
        output = {
            "head": head_name,
            "primary_policy": args.primary_policy,
            "primary_budget": args.primary_budget,
            **primary,
        }
        with open(os.path.join(args.results_dir, f"{head_name}_pruning_metrics.json"), "w") as file:
            json.dump(finite_or_none(output), file, indent=2)

    write_csv(os.path.join(args.results_dir, "causal_diagnostics.csv"), diagnostic_rows)
    write_csv(os.path.join(args.results_dir, "pruning_results.csv"), rows)
    write_csv(os.path.join(args.results_dir, "pruning_by_group.csv"), group_rows)
    summary_rows = [
        row for row in rows
        if (row["policy"] == args.primary_policy
            and abs(row["budget"] - args.primary_budget) < 1e-9)
        or row["head"] in {"no_pruning", "oracle_first_error"}
    ]
    write_markdown(
        os.path.join(args.results_dir, "pruning_summary.md"),
        summary_rows,
        ["head", "policy", "budget", "clean_false_prune_rate",
         "pre_error_false_prune_rate", "error_coverage", "detection_at_0",
         "detection_at_1", "detection_at_2", "median_detection_delay",
         "safe_step_saving_rate", "oracle_efficiency_ratio"],
    )
    save_plots(
        args.results_dir, rows, diagnostic_rows, args.primary_budget, args.primary_policy
    )
    print(f"[saved] causal-prefix and offline-pruning analyses under {args.results_dir}")


if __name__ == "__main__":
    main()
