

import torch


def pqm_loss(rewards: torch.Tensor, labels: torch.Tensor, zeta: float = 4.0) -> torch.Tensor:
   
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


def bce_step_loss(rewards: torch.Tensor, labels: torch.Tensor, zeta: float = 4.0) -> torch.Tensor:
    valid = (labels == 0) | (labels == 1)
    if not valid.any():
        return rewards.sum() * 0.0
    return torch.nn.functional.binary_cross_entropy_with_logits(
        rewards[valid], labels[valid].to(rewards.dtype)
    )


LOSSES = {"pqm": pqm_loss, "bce": bce_step_loss}


def build_labels_from_mask(step_correctness: torch.Tensor, step_mask: torch.Tensor) -> torch.Tensor:
    
    labels = step_correctness.clone().long()
    labels[~step_mask] = -100
    return labels
