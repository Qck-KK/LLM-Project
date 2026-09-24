"""Paired comparison of reward-head arms that may live in different caches.

Written for the LoRA arm against the frozen arms, and reused for the ablations
that differ only in how a head was trained (loss, zeta, learning rate, seed):
each arm is NAME:CACHE:HEAD:CKPT, and `--out_name` keeps the outputs apart.

The two arms live in different caches -- the LoRA encoder produces different step
features -- so `analysis.bootstrap_step_metrics` cannot pair them in one pass.
They do, however, score the SAME trajectories under the SAME split, so the
resampling can be shared: draw a set of held-out trajectories once and evaluate
every arm on that same draw. That gives a genuine paired interval for
"does unfreezing the encoder help?".

Bias to keep in mind when reading the result: the frozen arms trained for 30
epochs on the full data, the LoRA arm for 1. The handicap favours the frozen
arms, so a LoRA win is conservative and a LoRA loss is ambiguous.
"""

import argparse
import csv
import glob
import json
import os

import torch

from eval.eval_utils import (
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
def score(cache_dir, head_name, checkpoint, device):
    """Return (scores, labels, step_mask) for one arm, padded to its own width."""
    shard_paths = sorted(glob.glob(os.path.join(cache_dir, "shard_*.pt")))
    assert shard_paths, "No cached shards in " + cache_dir
    with open(os.path.join(cache_dir, "hidden_size.txt")) as f:
        hidden_size = int(f.read().strip())
    head = build_reward_head(head_name, hidden_size).to(device)
    head.load_state_dict(torch.load(checkpoint, map_location=device))
    head.eval()
    out, labels, masks = [], [], []
    for path in shard_paths:
        shard = torch.load(path, map_location=device)
        step_hidden = shard["step_hidden"].to(device).float()
        step_mask = shard["step_mask"].to(device).bool()
        out.append(head(step_hidden, step_mask).cpu())
        labels.append(shard["labels"].long().cpu())
        masks.append(step_mask.cpu())
    return (pad_and_concat(out, 0.0), pad_and_concat(labels, -100),
            pad_and_concat(masks, False))


def percentile_ci(samples, alpha=0.05):
    tensor = torch.tensor(samples, dtype=torch.float64)
    tensor = tensor[torch.isfinite(tensor)]
    if tensor.numel() == 0:
        return float("nan"), float("nan")
    return (float(torch.quantile(tensor, alpha / 2)),
            float(torch.quantile(tensor, 1 - alpha / 2)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", action="append", required=True, metavar="NAME:CACHE:HEAD:CKPT",
                        help="Repeatable. Example: "
                             "frozen_linear:cache/val_clean:linear:checkpoints/long30/linear_head.pt")
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--out_name", default="lora_vs_frozen",
                        help="Output file stem; any set of arms can be compared, not only LoRA.")
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--calibration_fraction", type=float, default=0.5)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = get_device(args.device)
    os.makedirs(args.results_dir, exist_ok=True)

    arms = {}
    reference_labels = None
    for spec in args.arm:
        name, cache_dir, head_name, checkpoint = spec.split(":", 3)
        scores, labels, step_mask = score(cache_dir, head_name, checkpoint, device)
        # Every arm must cover the same trajectories, or the pairing is invalid.
        if reference_labels is None:
            reference_labels = labels
            reference_mask = step_mask
        elif labels.shape[0] != reference_labels.shape[0]:
            raise RuntimeError("arm {0} has {1} trajectories, expected {2}".format(
                name, labels.shape[0], reference_labels.shape[0]))
        arms[name] = (scores, labels, step_mask)
        print("  loaded {0:<18} from {1}".format(name, cache_dir))

    n_traj = reference_labels.shape[0]
    _, test_mask = deterministic_example_split(
        n_traj, args.calibration_fraction, args.split_seed
    )
    test_rows = test_mask.nonzero(as_tuple=True)[0]
    n_test = test_rows.numel()

    # Each arm keeps its own valid mask: truncation can differ between encoders,
    # so a step present in one cache may be absent in another.
    prepared = {}
    for name, (scores, labels, step_mask) in arms.items():
        valid = ((labels == 0) | (labels == 1)) & step_mask & test_mask[:, None]
        width = valid.shape[1]
        flat_positions = torch.arange(n_traj * width).reshape(n_traj, width)
        prepared[name] = (scores.reshape(-1), labels.reshape(-1), valid, flat_positions)

    point = {}
    for name, (flat_scores, flat_labels, valid, _) in prepared.items():
        v = valid.reshape(-1)
        point[name] = {
            "roc_auc": roc_auc(flat_scores[v], flat_labels[v]),
            "average_precision": average_precision(flat_scores[v], flat_labels[v]),
            "n_steps": int(v.sum()),
        }

    generator = torch.Generator().manual_seed(args.seed)
    draws = {name: {"roc_auc": [], "average_precision": []} for name in arms}
    for _ in range(args.bootstrap_samples):
        picked = test_rows[torch.randint(n_test, (n_test,), generator=generator)]
        for name, (flat_scores, flat_labels, valid, flat_positions) in prepared.items():
            rows_valid = valid[picked]
            positions = flat_positions[picked][rows_valid]
            target = flat_labels[positions]
            draws[name]["roc_auc"].append(roc_auc(flat_scores[positions], target))
            draws[name]["average_precision"].append(
                average_precision(flat_scores[positions], target)
            )

    summary = {"n_test_trajectories": int(n_test),
               "bootstrap_samples": args.bootstrap_samples,
               "seed": args.seed, "split_seed": args.split_seed,
               "arm_specs": args.arm, "arms": {}}
    for name in arms:
        entry = dict(point[name])
        for metric in ("roc_auc", "average_precision"):
            low, high = percentile_ci(draws[name][metric])
            entry[metric + "_ci_low"] = low
            entry[metric + "_ci_high"] = high
        summary["arms"][name] = entry

    names = list(arms)
    pairs = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            row = {"arm_a": a, "arm_b": b}
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
                row[metric + "_significant"] = bool(low > 0 or high < 0)
            pairs.append(row)
    summary["pairwise"] = pairs

    out_json = os.path.join(args.results_dir, args.out_name + ".json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    if pairs:
        with open(os.path.join(args.results_dir, args.out_name + "_pairwise.csv"),
                  "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(pairs[0].keys()))
            writer.writeheader()
            writer.writerows(pairs)
    print("[saved] " + out_json)


if __name__ == "__main__":
    main()
