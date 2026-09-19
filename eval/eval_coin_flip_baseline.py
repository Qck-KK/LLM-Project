"""
eval_coin_flip_baseline.py
===========================
Random baseline for the final comparison table.

This baseline flips a fair coin to predict whether a step or trajectory is
correct. It does not train a model and does not load any checkpoint.
"""

import argparse
import glob
import json
import os

import torch

from eval.eval_utils import deterministic_example_split, pairwise_separation


def load_step_labels(cache_dir):
    paths = sorted(glob.glob(os.path.join(cache_dir, "shard_*.pt")))
    assert paths, f"No cached shards in {cache_dir}"

    labels_list = []
    for path in paths:
        shard = torch.load(path, map_location="cpu")
        labels_list.append(shard["labels"].long())

    max_steps = max(labels.shape[1] for labels in labels_list)
    return torch.cat([
        torch.nn.functional.pad(labels, (0, max_steps - labels.shape[1]), value=-100)
        for labels in labels_list
    ], dim=0)


def load_single_labels(cache_dir):
    paths = sorted(glob.glob(os.path.join(cache_dir, "shard_*.pt")))
    assert paths, f"No cached shards in {cache_dir}"

    all_labels = []
    for path in paths:
        shard = torch.load(path, map_location="cpu")
        if "final_correct" in shard:
            labels = shard["final_correct"].long()
        elif "label" in shard:
            labels = shard["label"].long()
        elif "labels" in shard:
            labels = shard["labels"].long()
            if labels.dim() == 2:
                step_mask = shard["step_mask"].bool()
                lengths = step_mask.sum(dim=1).clamp(min=1).long() - 1
                labels = labels.gather(1, lengths.unsqueeze(1)).squeeze(1)
        else:
            raise KeyError(f"No label key found in {path}. Keys: {list(shard.keys())}")
        all_labels.append(labels)
    return torch.cat(all_labels)


def random_step_accuracy(labels, generator):
    mask = labels != -100
    preds = torch.randint(0, 2, labels.shape, generator=generator)
    return (preds[mask] == labels[mask]).float().mean().item()


def random_ranking_accuracy(labels, generator):
    scores = torch.rand(labels.shape, generator=generator)
    total, correct_pairs = 0, 0
    for labels_b, scores_b in zip(labels, scores):
        pos_idx = (labels_b == 1).nonzero(as_tuple=True)[0]
        neg_idx = (labels_b == 0).nonzero(as_tuple=True)[0]
        if pos_idx.numel() == 0 or neg_idx.numel() == 0:
            continue
        pos_scores = scores_b[pos_idx].unsqueeze(1)
        neg_scores = scores_b[neg_idx].unsqueeze(0)
        correct_pairs += (pos_scores > neg_scores).sum().item()
        total += pos_idx.numel() * neg_idx.numel()
    return correct_pairs / max(total, 1)


def random_single_accuracy(labels, generator):
    preds = torch.randint(0, 2, labels.shape, generator=generator)
    return (preds == labels).float().mean().item()


def random_single_separation(labels, generator):
    scores = torch.rand(labels.shape, generator=generator)
    return pairwise_separation(scores, labels)


def mean_std(values):
    tensor = torch.tensor(values, dtype=torch.float32)
    std = tensor.std(unbiased=False).item() if tensor.numel() > 1 else 0.0
    return tensor.mean().item(), std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_cache_dir", required=True)
    parser.add_argument("--single_cache_dir", default=None)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--head_name", default="coin_flip")
    parser.add_argument("--calibration_fraction", type=float, default=0.5)
    parser.add_argument("--split_seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    generator = torch.Generator().manual_seed(args.seed)

    step_labels = load_step_labels(args.val_cache_dir)
    _, step_test_mask = deterministic_example_split(
        len(step_labels), args.calibration_fraction, args.split_seed
    )
    step_labels = step_labels[step_test_mask]
    step_accs, ranking_accs = [], []
    for _ in range(args.trials):
        step_accs.append(random_step_accuracy(step_labels, generator))
        ranking_accs.append(random_ranking_accuracy(step_labels, generator))

    step_acc, step_acc_std = mean_std(step_accs)
    ranking_acc, ranking_acc_std = mean_std(ranking_accs)

    efficiency = {
        "head": args.head_name,
        "baseline": "fair_coin",
        "n_trainable_params": 0,
        "epochs": 0,
        "total_train_time_sec": 0.0,
        "peak_mem_mb": None,
        "device": "none",
        "final_train_loss": None,
        "final_eval_loss": None,
    }
    with open(os.path.join(args.results_dir, f"{args.head_name}_efficiency.json"), "w") as f:
        json.dump(efficiency, f, indent=2)

    step_out = {
        "head": args.head_name,
        "baseline": "fair_coin",
        "trials": args.trials,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "calibration_fraction": args.calibration_fraction,
        "step_reward_accuracy": step_acc,
        "step_reward_accuracy_std": step_acc_std,
        "step_reward_threshold": None,
        "qvalue_ranking_accuracy": ranking_acc,
        "qvalue_ranking_accuracy_std": ranking_acc_std,
        "n_trainable_params": 0,
    }
    step_path = os.path.join(args.results_dir, f"{args.head_name}_step_metrics.json")
    with open(step_path, "w") as f:
        json.dump(step_out, f, indent=2)
    print(f"[saved] {step_path}")

    if args.single_cache_dir and os.path.exists(args.single_cache_dir):
        single_labels = load_single_labels(args.single_cache_dir)
        _, single_test_mask = deterministic_example_split(
            len(single_labels), args.calibration_fraction, args.split_seed
        )
        single_labels = single_labels[single_test_mask]
        single_accs, single_seps = [], []
        for _ in range(args.trials):
            single_accs.append(random_single_accuracy(single_labels, generator))
            single_seps.append(random_single_separation(single_labels, generator))
        single_acc, single_acc_std = mean_std(single_accs)
        single_sep, single_sep_std = mean_std(single_seps)

        single_out = {
            "head": args.head_name,
            "baseline": "fair_coin",
            "trials": args.trials,
            "seed": args.seed,
            "split_seed": args.split_seed,
            "calibration_fraction": args.calibration_fraction,
            "agg": "coin_flip",
            "single_eval_accuracy": single_acc,
            "single_eval_accuracy_std": single_acc_std,
            "single_eval_threshold": None,
            "single_eval_separation": single_sep,
            "single_eval_separation_std": single_sep_std,
            "n_solutions": len(single_labels),
        }
        single_path = os.path.join(args.results_dir, f"{args.head_name}_single_metrics.json")
        with open(single_path, "w") as f:
            json.dump(single_out, f, indent=2)
        print(f"[saved] {single_path}")

    print(f"\n=== {args.head_name} baseline ===")
    print(f"Step-level coin accuracy: {step_acc:.4f} +/- {step_acc_std:.4f}")
    print(f"Random ranking accuracy: {ranking_acc:.4f} +/- {ranking_acc_std:.4f}")


if __name__ == "__main__":
    main()
