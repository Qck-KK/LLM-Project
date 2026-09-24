"""
pqm_loss.py
============
Stage 3: PQM Comparative Ranking Loss (Eq. 10 of the PQM paper,
"Process Reward Model with Q-Value Rankings", Li & Li, ICLR 2025).

This is a re-typed, commented version of the OFFICIAL implementation
released by the authors at:
    https://github.com/WindyLee0822/Process_Q_Model

The ranking objective is the official one. Two numerical-safety changes were
made on top of it, and neither changes the loss in the normal regime:

1. Rewards are clamped to [-50, 50] before exp(), and reward + zeta is clamped
   at 50. The loss is therefore identical whenever |reward| <= 50 - zeta; beyond
   that the official expression overflows to inf.
2. An example with no supervised position (no correct step and no incorrect
   step) is excluded from the batch mean. The official per-example average
   divides 0 by 0 there and turns the whole batch loss into NaN. A batch with no
   supervised position at all returns a zero loss that keeps the graph attached.

Both behaviours are covered by tests/test_core.py.

Intuition: for each correct step i, this loss wants Q(step_i) to be ranked
above (a) all incorrect steps in the trajectory (with a margin `zeta`), and
(b) all correct steps that come BEFORE it (Q-values should be non-decreasing
along a correct trajectory). It's essentially a softmax/list-wise ranking
loss computed efficiently via cumulative sums instead of explicit pairwise
comparisons.
"""

import torch


def pqm_loss(rewards: torch.Tensor, labels: torch.Tensor, zeta: float = 4.0) -> torch.Tensor:
    """
    Args:
        rewards: (B, S) -- Q-values produced by a reward head, ALREADY masked
                  to whatever padded length S you used (padded positions
                  should just have any label != {0,1}, see `labels` below).
        labels:  (B, S) -- 1 = correct step, 0 = incorrect step, -100 = padding
        zeta:    float  -- margin hyperparameter (paper default: 4)

    Returns:
        scalar loss (mean over valid, correct steps in the batch)
    """
    has_neg = (labels == 0).sum(-1).bool()

    # The official expression is written with exp(reward). Clamping preserves
    # its ordering while preventing overflow from an unusually large Q-value.
    stable_rewards = rewards.clamp(min=-50.0, max=50.0)
    pos_rewards_exp = torch.where(labels == 1, stable_rewards.exp(), torch.zeros_like(rewards))
    neg_rewards_exp = torch.where(
        labels == 0, (stable_rewards + zeta).clamp(max=50.0).exp(), torch.zeros_like(rewards)
    ).flip(dims=[-1])
    neg_reward_sum = neg_rewards_exp.sum(-1)  # (B,)

    pos_rewards_cumsum = torch.cat(
        [torch.zeros(rewards.shape[0], 1, device=rewards.device).exp(), pos_rewards_exp], dim=1
    ).cumsum(-1)[:, :-1]
    pos_rewards_cumsum = torch.cat(
        [torch.zeros(rewards.shape[0], 1, device=rewards.device), pos_rewards_cumsum], dim=-1
    )

    reward_exp_cur = torch.where(labels == 1, pos_rewards_exp, torch.ones_like(rewards))
    reward_exp_cur = torch.cat(
        [torch.zeros(rewards.shape[0], 1, device=rewards.device).exp(), reward_exp_cur], dim=-1
    )

    loss = -torch.log(
        reward_exp_cur / (reward_exp_cur + pos_rewards_cumsum + neg_reward_sum[..., None] + 1e-5)
    )

    labels_padded = torch.cat([has_neg[..., None], labels], dim=-1)
    supervised = labels_padded == 1
    counts = supervised.sum(-1)
    per_example = torch.where(supervised, loss, torch.zeros_like(loss)).sum(-1) / counts.clamp(min=1)
    valid_examples = counts > 0
    if not valid_examples.any():
        return rewards.sum() * 0.0
    return per_example[valid_examples].mean()


def build_labels_from_mask(step_correctness: torch.Tensor, step_mask: torch.Tensor) -> torch.Tensor:
    """
    Helper to build the `labels` tensor pqm_loss expects, from:
        step_correctness: (B, S) 1/0, meaningful only where step_mask is True
        step_mask:        (B, S) bool, True = real step

    Padding positions are set to -100 so pqm_loss's `labels == 1` / `== 0`
    checks correctly skip them.
    """
    labels = step_correctness.clone().long()
    labels[~step_mask] = -100
    return labels
