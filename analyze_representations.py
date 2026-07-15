"""
analyze_representations.py
============================
The "Representation Analysis" section of the proposal -- explains WHY an
architecture scores the way it does, not just the accuracy number.

Produces, per head:
    1. Q-value distribution histogram, correct vs incorrect steps overlaid
    2. t-SNE / PCA scatter of step_hidden (the FROZEN encoder's output),
       colored by step correctness -- this is architecture-independent since
       step_hidden comes straight from the encoder, so run it ONCE and reuse
       across all heads.
    3. A simple separation score (AUC-style: P(Q_correct > Q_incorrect)),
       which is just qvalue_ranking_accuracy from eval_step_metrics.py,
       printed alongside the plots for convenience.

Usage:
    # encoder-only plot (run once):
    python analyze_representations.py --cache_dir cache/qwen05b_val --mode encoder_tsne

    # per-head Q-value distribution (run once per trained head):
    python analyze_representations.py --cache_dir cache/qwen05b_val \
        --mode qvalue_dist --head mlp --checkpoint checkpoints/mlp_head.pt
"""

import argparse
import glob
import os

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from reward_heads import build_reward_head


def load_cache(cache_dir, max_shards=None):
    shard_paths = sorted(glob.glob(os.path.join(cache_dir, "shard_*.pt")))
    if max_shards:
        shard_paths = shard_paths[:max_shards]
    hiddens, masks, labels = [], [], []
    for p in shard_paths:
        shard = torch.load(p, map_location="cpu")
        hiddens.append(shard["step_hidden"].float())
        masks.append(shard["step_mask"])
        labels.append(shard["labels"])
    max_S = max(h.shape[1] for h in hiddens)
    hiddens = torch.cat([torch.nn.functional.pad(h, (0, 0, 0, max_S - h.shape[1])) for h in hiddens])
    masks = torch.cat([torch.nn.functional.pad(m, (0, max_S - m.shape[1])) for m in masks])
    labels = torch.cat([torch.nn.functional.pad(l, (0, max_S - l.shape[1]), value=-100) for l in labels])
    return hiddens, masks, labels


def encoder_tsne(cache_dir, out_path="encoder_tsne.png", max_points=3000, method="tsne"):
    hiddens, masks, labels = load_cache(cache_dir)
    flat_h = hiddens[masks]
    flat_l = labels[masks]

    if flat_h.shape[0] > max_points:
        idx = torch.randperm(flat_h.shape[0])[:max_points]
        flat_h, flat_l = flat_h[idx], flat_l[idx]

    reducer = TSNE(n_components=2, init="pca", perplexity=30) if method == "tsne" else PCA(n_components=2)
    coords = reducer.fit_transform(flat_h.numpy())

    plt.figure(figsize=(7, 6))
    for lab, color, name in [(1, "tab:green", "correct"), (0, "tab:red", "incorrect")]:
        m = (flat_l == lab).numpy()
        plt.scatter(coords[m, 0], coords[m, 1], s=6, alpha=0.5, c=color, label=name)
    plt.legend()
    plt.title(f"Frozen encoder step representations ({method.upper()})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"[saved] {out_path}")


def qvalue_distribution(cache_dir, head_name, checkpoint, out_path=None, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"))
    with open(os.path.join(cache_dir, "hidden_size.txt")) as f:
        hidden_size = int(f.read().strip())

    head = build_reward_head(head_name, hidden_size=hidden_size).to(device)
    head.load_state_dict(torch.load(checkpoint, map_location=device))
    head.eval()

    hiddens, masks, labels = load_cache(cache_dir)
    with torch.no_grad():
        q = head(hiddens.to(device), masks.to(device)).cpu()

    q_correct = q[labels == 1]
    q_incorrect = q[labels == 0]

    plt.figure(figsize=(7, 5))
    plt.hist(q_correct.numpy(), bins=50, alpha=0.6, color="tab:green", label="correct steps", density=True)
    plt.hist(q_incorrect.numpy(), bins=50, alpha=0.6, color="tab:red", label="incorrect steps", density=True)
    plt.xlabel("Q-value")
    plt.ylabel("density")
    plt.legend()
    plt.title(f"Q-value distribution -- {head_name} head")
    plt.tight_layout()
    out_path = out_path or f"qvalue_dist_{head_name}.png"
    plt.savefig(out_path, dpi=150)
    print(f"[saved] {out_path}")
    print(f"mean Q correct={q_correct.mean().item():.3f}  mean Q incorrect={q_incorrect.mean().item():.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--mode", required=True, choices=["encoder_tsne", "encoder_pca", "qvalue_dist"])
    parser.add_argument("--head", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.mode == "encoder_tsne":
        encoder_tsne(args.cache_dir, out_path=args.out or "encoder_tsne.png", method="tsne")
    elif args.mode == "encoder_pca":
        encoder_tsne(args.cache_dir, out_path=args.out or "encoder_pca.png", method="pca")
    elif args.mode == "qvalue_dist":
        assert args.head and args.checkpoint, "--head and --checkpoint required for qvalue_dist"
        qvalue_distribution(args.cache_dir, args.head, args.checkpoint, out_path=args.out)


if __name__ == "__main__":
    main()
