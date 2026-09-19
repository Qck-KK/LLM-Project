"""
summarize_results.py
======================
Scans a results directory for the JSON files produced by:
    - train_from_cache.py   (--results_dir ...)  -> {head}_efficiency.json
    - eval_step_metrics.py   (--results_dir ...)  -> {head}_step_metrics.json
    - eval_single_from_cache.py (--results_dir ...) -> {head}_single_metrics.json
    - analyze_head_behavior.py -> {head}_behavior_metrics.json
    - analyze_data_bias.py -> deterministic_baselines.json

and merges them (by "head" name) into ONE comparison table -- this is the
final deliverable table from the proposal (Section 5/6): one row per
architecture, all metrics side by side.

Usage:
    python summarize_results.py --results_dir results/ \
        --out_csv results/summary.csv --out_md results/summary.md

Run this any time after evaluating a new head -- it just re-scans the
directory, so partial results (e.g. only 3 of 5 heads done so far) are fine.
"""

import argparse
import glob
import json
import os


def load_all(results_dir, suffix):
    merged = {}
    for path in glob.glob(os.path.join(results_dir, f"*_{suffix}.json")):
        with open(path) as f:
            data = json.load(f)
        head = data["head"]
        merged[head] = data
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--out_csv", default=None)
    parser.add_argument("--out_md", default=None)
    args = parser.parse_args()

    efficiency = load_all(args.results_dir, "efficiency")
    step_metrics = load_all(args.results_dir, "step_metrics")
    bon_metrics = load_all(args.results_dir, "bon_metrics")
    single_metrics = load_all(args.results_dir, "single_metrics")
    behavior_metrics = load_all(args.results_dir, "behavior_metrics")

    deterministic_path = os.path.join(args.results_dir, "deterministic_baselines.json")
    if os.path.exists(deterministic_path):
        with open(deterministic_path) as f:
            deterministic = json.load(f)
        for name in ("majority", "position_only"):
            if name in deterministic:
                metrics = deterministic[name]
                behavior_metrics[name] = {
                    "head": name,
                    "n_trainable_params": 0,
                    **metrics,
                }

    heads = sorted(
        set(efficiency) | set(step_metrics) | set(bon_metrics)
        | set(single_metrics) | set(behavior_metrics)
    )
    if not heads:
        print(f"[summarize] no result JSON files found under {args.results_dir}")
        return

    # collect every BON@k key that appears across all bon_metrics files
    bon_keys = sorted({k for d in bon_metrics.values() for k in d if k.startswith("bon@")})

    columns = (
        ["head", "n_trainable_params", "epochs", "best_epoch", "total_train_time_sec", "peak_mem_mb"]
        + ["final_train_loss", "final_eval_loss"]
        + bon_keys
        + ["single_eval_accuracy", "single_eval_balanced_accuracy", "single_eval_roc_auc",
           "single_eval_average_precision", "single_eval_separation"]
        + ["step_reward_accuracy", "step_balanced_accuracy", "step_roc_auc",
           "step_average_precision", "qvalue_ranking_accuracy"]
        + ["first_error_boundary_drop", "matched_correct_boundary_drop",
           "position_controlled_boundary_effect"]
    )

    rows = []
    for head in heads:
        eff = efficiency.get(head, {})
        step = step_metrics.get(head, {})
        bon = bon_metrics.get(head, {})
        single = single_metrics.get(head, {})
        behavior = behavior_metrics.get(head, {})
        row = {
            "head": head,
            "n_trainable_params": eff.get(
                "n_trainable_params", step.get("n_trainable_params", behavior.get("n_trainable_params", ""))
            ),
            "epochs": eff.get("epochs", ""),
            "best_epoch": eff.get("best_epoch", ""),
            "total_train_time_sec": eff.get("total_train_time_sec", ""),
            "peak_mem_mb": eff.get("peak_mem_mb", ""),
            "final_train_loss": eff.get("final_train_loss", ""),
            "final_eval_loss": eff.get("final_eval_loss", ""),
            "step_reward_accuracy": step.get("step_reward_accuracy", behavior.get("accuracy", "")),
            "step_balanced_accuracy": step.get("step_balanced_accuracy", behavior.get("balanced_accuracy", "")),
            "step_roc_auc": step.get("step_roc_auc", behavior.get("roc_auc", "")),
            "step_average_precision": step.get("step_average_precision", behavior.get("average_precision", "")),
            "qvalue_ranking_accuracy": step.get("qvalue_ranking_accuracy", behavior.get("ranking_accuracy", "")),
            "single_eval_accuracy": single.get("single_eval_accuracy", ""),
            "single_eval_balanced_accuracy": single.get("single_eval_balanced_accuracy", ""),
            "single_eval_roc_auc": single.get("single_eval_roc_auc", ""),
            "single_eval_average_precision": single.get("single_eval_average_precision", ""),
            "single_eval_separation": single.get("single_eval_separation", ""),
            "first_error_boundary_drop": behavior.get("first_error_boundary_drop", ""),
            "matched_correct_boundary_drop": behavior.get("matched_correct_boundary_drop", ""),
            "position_controlled_boundary_effect": behavior.get("position_controlled_boundary_effect", ""),
        }
        for k in bon_keys:
            row[k] = bon.get(k, "")
        rows.append(row)

    # ---- CSV ----
    csv_path = args.out_csv or os.path.join(args.results_dir, "summary.csv")
    with open(csv_path, "w") as f:
        f.write(",".join(columns) + "\n")
        for row in rows:
            f.write(",".join(_fmt(row[c]) for c in columns) + "\n")
    print(f"[saved] {csv_path}")

    # ---- Markdown ----
    md_path = args.out_md or os.path.join(args.results_dir, "summary.md")
    with open(md_path, "w") as f:
        f.write("| " + " | ".join(columns) + " |\n")
        f.write("|" + "---|" * len(columns) + "\n")
        for row in rows:
            f.write("| " + " | ".join(_fmt(row[c]) for c in columns) + " |\n")
    print(f"[saved] {md_path}")

    # ---- also print to stdout ----
    print("\n" + open(md_path).read())


def _fmt(v):
    if v == "" or v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.4f}" if v < 1000 else f"{v:.1f}"
    return str(v)


if __name__ == "__main__":
    main()
