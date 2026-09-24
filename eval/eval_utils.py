import glob
import os

import torch

from reward_heads import REWARD_HEADS


HEAD_CHOICES = sorted(REWARD_HEADS.keys())


def get_device(device=None):
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_step_cache_labels(cache_dir: str):
    """Load and pad step labels/masks without loading cached embeddings."""
    paths = sorted(glob.glob(os.path.join(cache_dir, "shard_*.pt")))
    if not paths:
        raise FileNotFoundError(f"No cached shards in {cache_dir}")

    labels_list, mask_list = [], []
    for path in paths:
        shard = torch.load(path, map_location="cpu")
        labels = shard["labels"].long()
        step_mask = shard.get("step_mask", labels != -100).bool()
        # Trajectories longer than the encoder's max_length lost their trailing
        # step markers, so the cache can carry labels for steps that have no
        # embedding. Scoring those positions feeds a zero vector to the head and
        # yields a constant, which silently corrupts every metric. A step without
        # an embedding is padding, so mark it as such.
        labels = labels.masked_fill(~step_mask, -100)
        labels_list.append(labels)
        mask_list.append(step_mask)

    max_steps = max(labels.shape[1] for labels in labels_list)
    labels = torch.cat([
        torch.nn.functional.pad(x, (0, max_steps - x.shape[1]), value=-100)
        for x in labels_list
    ])
    step_mask = torch.cat([
        torch.nn.functional.pad(x, (0, max_steps - x.shape[1]), value=False)
        for x in mask_list
    ])
    return labels, step_mask


def deterministic_example_split(n_examples: int, calibration_fraction: float = 0.5, seed: int = 42):
    """Return reproducible calibration/test example masks.

    The seed fixes the split; this is not a multi-seed experiment. Splitting by
    trajectory prevents steps from the same solution leaking into both sets.
    """
    if n_examples < 2:
        raise ValueError("At least two examples are required for a calibration/test split.")
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be between 0 and 1.")

    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(n_examples, generator=generator)
    n_calibration = min(max(int(round(n_examples * calibration_fraction)), 1), n_examples - 1)
    calibration = torch.zeros(n_examples, dtype=torch.bool)
    calibration[order[:n_calibration]] = True
    return calibration, ~calibration


def flatten_valid_steps(values: torch.Tensor, labels: torch.Tensor, example_mask=None):
    if example_mask is not None:
        values = values[example_mask]
        labels = labels[example_mask]
    valid = (labels == 0) | (labels == 1)
    return values[valid], labels[valid].long()


def binary_accuracy(scores: torch.Tensor, labels: torch.Tensor, threshold: float = 0.0) -> float:
    if labels.numel() == 0:
        return float("nan")
    return ((scores > threshold).long() == labels).float().mean().item()


def balanced_accuracy(scores: torch.Tensor, labels: torch.Tensor, threshold: float = 0.0) -> float:
    recalls = []
    predictions = (scores > threshold).long()
    for label in (0, 1):
        mask = labels == label
        if mask.any():
            recalls.append((predictions[mask] == label).float().mean())
    if not recalls:
        return float("nan")
    return torch.stack(recalls).mean().item()


def roc_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Rank-based ROC-AUC with average ranks for tied scores."""
    scores = scores.detach().float().cpu()
    labels = labels.detach().long().cpu()
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = torch.argsort(scores)
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    _, counts = torch.unique_consecutive(sorted_scores, return_counts=True)
    ends = counts.cumsum(0).float()
    starts = ends - counts.float() + 1.0
    average_ranks = (starts + ends) / 2.0
    ranks = torch.repeat_interleave(average_ranks, counts)
    positive_rank_sum = ranks[sorted_labels == 1].sum()
    auc = (positive_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return auc.item()


def average_precision(scores: torch.Tensor, labels: torch.Tensor) -> float:
    scores = scores.detach().float().cpu()
    labels = labels.detach().long().cpu()
    n_pos = int((labels == 1).sum())
    if n_pos == 0:
        return float("nan")
    order = torch.argsort(scores, descending=True)
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    _, inverse, counts = torch.unique_consecutive(
        sorted_scores, return_inverse=True, return_counts=True
    )
    positive_per_group = torch.zeros(len(counts), dtype=torch.float32)
    positive_per_group.scatter_add_(0, inverse, sorted_labels.float())
    cumulative_positive = positive_per_group.cumsum(0)
    cumulative_count = counts.cumsum(0).float()
    precision_at_threshold = cumulative_positive / cumulative_count
    recall_increment = positive_per_group / n_pos
    return (precision_at_threshold * recall_increment).sum().item()


def aggregate_trajectory_scores(q_values: torch.Tensor, step_mask: torch.Tensor, mode: str = "min") -> torch.Tensor:
    """Collapse per-step Q-values into one score per trajectory."""
    squeeze = q_values.dim() == 1
    if squeeze:
        q_values = q_values.unsqueeze(0)
        step_mask = step_mask.unsqueeze(0)

    if mode == "min":
        scores = q_values.masked_fill(~step_mask, float("inf")).min(dim=1).values
        scores = torch.where(step_mask.any(dim=1), scores, scores.new_full(scores.shape, float("-inf")))
    elif mode == "mean":
        scores = (q_values * step_mask).sum(dim=1) / step_mask.sum(dim=1).clamp(min=1)
    elif mode == "last":
        lengths = step_mask.sum(dim=1).clamp(min=1).long() - 1
        scores = q_values.gather(1, lengths.unsqueeze(1)).squeeze(1)
    elif mode == "sum":
        scores = (q_values * step_mask).sum(dim=1)
    elif mode == "prod":
        scores = q_values.masked_fill(~step_mask, 1.0).prod(dim=1)
    else:
        raise ValueError(mode)

    return scores.squeeze(0) if squeeze else scores


def best_threshold_accuracy(scores: torch.Tensor, labels: torch.Tensor, n_grid: int = 200):
    if scores.numel() == 0:
        return 0.0, float("nan")
    lo, hi = scores.min().item(), scores.max().item()
    epsilon = max(abs(hi - lo) * 1e-6, 1e-6)
    best_acc, best_t = -1.0, 0.0
    for t in torch.linspace(lo - epsilon, hi + epsilon, n_grid):
        preds = (scores > t).long()
        acc = (preds == labels).float().mean().item()
        if acc > best_acc:
            best_acc, best_t = acc, t.item()
    return best_t, best_acc


def pairwise_separation(scores: torch.Tensor, labels: torch.Tensor) -> float:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.numel() == 0 or neg.numel() == 0:
        return float("nan")
    return (pos.unsqueeze(1) > neg.unsqueeze(0)).float().mean().item()
