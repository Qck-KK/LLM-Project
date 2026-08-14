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
    lo, hi = scores.min().item(), scores.max().item()
    best_acc, best_t = -1.0, 0.0
    for t in torch.linspace(lo, hi, n_grid):
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
