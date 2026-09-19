"""
train_from_cache.py
=====================
Trains a reward head using embeddings that were precomputed on another machine.
No LLM forward pass happens here at all -- this only touches the tiny head,
so it should be dramatically faster than training through the encoder.

Usage:
    python train_from_cache.py --cache_dir cache/math_shepherd_qwen05b \
        --head mlp --epochs 10 --save_path checkpoints/mlp_head.pt

Repeat with --head cnn / gru / attention / linear to sweep all architectures
against the SAME cached embeddings -- this is the controlled comparison the
proposal calls for (identical encoder output, only the head differs).
"""

import argparse
import glob
import json
import os
import random
import time

import torch

from eval_utils import HEAD_CHOICES, deterministic_example_split, get_device
from reward_heads import build_reward_head
from pqm_loss import pqm_loss


def load_shard(path, device):
    shard = torch.load(path, map_location=device)
    return (
        shard["step_hidden"].to(device).float(),
        shard["step_mask"].to(device),
        shard["labels"].to(device),
    )


@torch.no_grad()
def evaluate_loss(head, shard_paths, device, zeta, example_mask=None):
    if not shard_paths:
        return None
    head.eval()
    running_loss, n_examples = 0.0, 0
    example_offset = 0
    for path in shard_paths:
        step_hidden, step_mask, labels = load_shard(path, device)
        batch_size = step_hidden.shape[0]
        if example_mask is not None:
            selected = example_mask[example_offset:example_offset + batch_size].to(device)
            example_offset += batch_size
            if not selected.any():
                continue
            step_hidden = step_hidden[selected]
            step_mask = step_mask[selected]
            labels = labels[selected]
        q_values = head(step_hidden, step_mask)
        batch_size = step_hidden.shape[0]
        running_loss += pqm_loss(q_values, labels, zeta=zeta).item() * batch_size
        n_examples += batch_size
    return running_loss / max(n_examples, 1)


def save_loss_plot(history, out_path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib is not installed; skipping loss plot.")
        return

    epochs = history["epoch"]
    plt.figure(figsize=(7, 4.5))
    if history.get("train_step"):
        plt.plot(history["train_step"], history["train_step_loss"], alpha=0.45,
                 linewidth=1.0, label="train loss (within epoch)")
    plt.plot(epochs, history["train_loss"], marker="o", linewidth=2.0,
             label="train loss (epoch mean)")
    if any(v is not None for v in history["eval_loss"]):
        plt.plot(epochs, history["eval_loss"], marker="o", label="evaluation loss")
    plt.xlabel("Epoch")
    plt.ylabel("PQM loss")
    plt.title(f"{history['head']} reward head loss")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--val_cache_dir", default=None,
                         help="Optional validation cache for epoch-level evaluation loss.")
    parser.add_argument("--head", required=True, choices=HEAD_CHOICES)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--zeta", type=float, default=4.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save_path", default="checkpoints/head.pt")
    parser.add_argument("--results_dir", default=None,
                         help="If set, writes training-efficiency JSON here for summarize_results.py.")
    parser.add_argument("--loss_history_path", default=None)
    parser.add_argument("--loss_plot_path", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early_stopping_patience", type=int, default=2,
                        help="Stop after this many non-improving validation epochs; 0 disables it.")
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.0,
                        help="Minimum absolute validation-loss improvement.")
    parser.add_argument("--log_points_per_epoch", type=int, default=5,
                        help="Number of within-epoch train-loss points saved for plotting.")
    parser.add_argument("--calibration_fraction", type=float, default=0.5,
                        help="Fraction of validation trajectories used for early stopping.")
    parser.add_argument("--split_seed", type=int, default=42,
                        help="Fixed trajectory split shared with evaluation scripts.")
    args = parser.parse_args()

    device = get_device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    with open(os.path.join(args.cache_dir, "hidden_size.txt")) as f:
        hidden_size = int(f.read().strip())

    shard_paths = sorted(glob.glob(os.path.join(args.cache_dir, "shard_*.pt")))
    assert shard_paths, f"No cached shards found in {args.cache_dir}. Copy the precomputed cache here first."
    val_shard_paths = []
    validation_calibration_mask = None
    if args.val_cache_dir:
        val_shard_paths = sorted(glob.glob(os.path.join(args.val_cache_dir, "shard_*.pt")))
        assert val_shard_paths, f"No cached validation shards found in {args.val_cache_dir}."
        n_validation_examples = sum(
            torch.load(path, map_location="cpu")["step_hidden"].shape[0]
            for path in val_shard_paths
        )
        validation_calibration_mask, _ = deterministic_example_split(
            n_validation_examples, args.calibration_fraction, args.split_seed
        )
    print(f"[train] device={device}  head={args.head}  "
          f"{len(shard_paths)} cached shards  hidden_size={hidden_size}")
    if val_shard_paths:
        print(f"[train] validation shards: {len(val_shard_paths)}  "
              f"early-stop trajectories: {int(validation_calibration_mask.sum())}")

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

    head = build_reward_head(args.head, hidden_size=hidden_size).to(device)
    n_trainable = sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f"[train] trainable params: {n_trainable:,}")

    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    train_start = time.time()
    history = {
        "head": args.head,
        "seed": args.seed,
        "max_epochs": args.epochs,
        "epoch": [],
        "train_loss": [],
        "eval_loss": [],
        "train_step": [],
        "train_step_loss": [],
    }
    best_metric = float("inf")
    best_epoch = None
    epochs_without_improvement = 0

    for epoch in range(args.epochs):
        random.shuffle(shard_paths)
        t0 = time.time()
        running_loss, n_examples = 0.0, 0
        window_loss, window_examples = 0.0, 0
        log_every = max(1, len(shard_paths) // max(args.log_points_per_epoch, 1))
        head.train()

        for batch_idx, path in enumerate(shard_paths):
            step_hidden, step_mask, labels = load_shard(path, device)

            q_values = head(step_hidden, step_mask)
            loss = pqm_loss(q_values, labels, zeta=args.zeta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = step_hidden.shape[0]
            running_loss += loss.item() * batch_size
            n_examples += batch_size
            window_loss += loss.item() * batch_size
            window_examples += batch_size

            if (batch_idx + 1) % log_every == 0 or batch_idx + 1 == len(shard_paths):
                progress = epoch + (batch_idx + 1) / len(shard_paths)
                history["train_step"].append(progress)
                history["train_step_loss"].append(window_loss / max(window_examples, 1))
                window_loss, window_examples = 0.0, 0

        dt = time.time() - t0
        train_loss = running_loss / max(n_examples, 1)
        eval_loss = evaluate_loss(
            head, val_shard_paths, device, args.zeta, validation_calibration_mask
        )
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["eval_loss"].append(eval_loss)
        eval_text = "" if eval_loss is None else f"  eval_loss={eval_loss:.4f}"
        print(f"[epoch {epoch+1}/{args.epochs}] train_loss={train_loss:.4f}{eval_text}  time={dt:.1f}s")

        selection_metric = eval_loss if eval_loss is not None else train_loss
        improved = torch.isfinite(torch.tensor(selection_metric)).item() and (
            selection_metric < best_metric - args.early_stopping_min_delta
        )
        if improved:
            best_metric = selection_metric
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save(head.state_dict(), args.save_path)
            print(f"[checkpoint] new best epoch={best_epoch} metric={best_metric:.4f}")
        else:
            epochs_without_improvement += 1

        if (val_shard_paths and args.early_stopping_patience > 0
                and epochs_without_improvement >= args.early_stopping_patience):
            print(f"[early-stop] no validation improvement for "
                  f"{args.early_stopping_patience} epochs")
            break

    total_train_time = time.time() - train_start
    if best_epoch is None:
        best_epoch = len(history["epoch"])
        torch.save(head.state_dict(), args.save_path)
    history["best_epoch"] = best_epoch
    history["stopped_early"] = len(history["epoch"]) < args.epochs
    print(f"[done] best epoch {best_epoch}; saved head weights to {args.save_path}")

    if args.results_dir or args.loss_history_path or args.loss_plot_path:
        if args.results_dir:
            os.makedirs(args.results_dir, exist_ok=True)
        loss_dir = args.results_dir or os.path.dirname(args.save_path) or "."
        loss_history_path = args.loss_history_path or os.path.join(loss_dir, f"{args.head}_loss_history.json")
        loss_plot_path = args.loss_plot_path or os.path.join(loss_dir, f"{args.head}_loss_curve.png")
        with open(loss_history_path, "w") as f:
            json.dump(history, f, indent=2)
        save_loss_plot(history, loss_plot_path)
        print(f"[saved] {loss_history_path}")
        print(f"[saved] {loss_plot_path}")

    if args.results_dir:
        peak_mem_mb = None
        if device == "cuda":
            peak_mem_mb = torch.cuda.max_memory_allocated() / 1e6
        out = {
            "head": args.head,
            "n_trainable_params": n_trainable,
            "seed": args.seed,
            "split_seed": args.split_seed,
            "calibration_fraction": args.calibration_fraction,
            "epochs": len(history["epoch"]),
            "max_epochs": args.epochs,
            "best_epoch": best_epoch,
            "stopped_early": history["stopped_early"],
            "total_train_time_sec": total_train_time,
            "peak_mem_mb": peak_mem_mb,
            "device": device,
            "final_train_loss": history["train_loss"][best_epoch - 1] if history["train_loss"] else None,
            "final_eval_loss": history["eval_loss"][best_epoch - 1] if history["eval_loss"] else None,
            "last_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
            "last_eval_loss": history["eval_loss"][-1] if history["eval_loss"] else None,
        }
        out_path = os.path.join(args.results_dir, f"{args.head}_efficiency.json")
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
