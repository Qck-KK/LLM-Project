"""
reward_heads.py
================
Stage 2: Lightweight Reward Function Approximators.

Every head has the SAME interface so they are drop-in swappable:

    forward(step_hidden: (B, S, H), step_mask: (B, S) bool) -> q_values: (B, S)

step_hidden comes straight out of FrozenStepEncoder. Padded positions in
step_mask are ignored by the loss (see pqm_loss.py), but heads that mix
information ACROSS steps (CNN / GRU / Attention) must also respect the mask
internally so padding doesn't leak into real steps.
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Baseline: Linear reward head (original PRM-style head)
# ---------------------------------------------------------------------------
class LinearHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1)

    def forward(self, step_hidden, step_mask):
        q = self.proj(step_hidden).squeeze(-1)  # (B, S)
        return q


# ---------------------------------------------------------------------------
# MLP: nonlinear pointwise reward approximation
# ---------------------------------------------------------------------------
class MLPHead(nn.Module):
    def __init__(self, hidden_size: int, mlp_hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.GELU(),
            nn.Linear(mlp_hidden // 2, 1),
        )

    def forward(self, step_hidden, step_mask):
        return self.net(step_hidden).squeeze(-1)


# ---------------------------------------------------------------------------
# CNN: local structural patterns among consecutive steps (1D conv over steps)
# ---------------------------------------------------------------------------
class CNNHead(nn.Module):
    def __init__(self, hidden_size: int, channels: int = 128, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(hidden_size, channels, kernel_size, padding=pad)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.act = nn.GELU()
        self.out = nn.Linear(channels, 1)

    def forward(self, step_hidden, step_mask):
        # zero out padded steps before convolving so they don't pollute
        # neighboring real steps via the receptive field
        x = step_hidden * step_mask.unsqueeze(-1)
        x = x.transpose(1, 2)          # (B, H, S)
        x = self.act(self.conv1(x))
        x = x * step_mask.unsqueeze(1)
        x = self.act(self.conv2(x))
        x = x * step_mask.unsqueeze(1)
        x = x.transpose(1, 2)          # (B, S, channels)
        return self.out(x).squeeze(-1)


# ---------------------------------------------------------------------------
# GRU / BiGRU: explicit sequential dependency modeling between steps
# ---------------------------------------------------------------------------
class GRUHead(nn.Module):
    def __init__(self, hidden_size: int, gru_hidden: int = 128, bidirectional: bool = True):
        super().__init__()
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        out_dim = gru_hidden * (2 if bidirectional else 1)
        self.out = nn.Linear(out_dim, 1)

    def forward(self, step_hidden, step_mask):
        lengths = step_mask.sum(dim=1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            step_hidden, lengths, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=step_hidden.size(1)
        )
        return self.out(out).squeeze(-1)


# ---------------------------------------------------------------------------
# Attention Pooling: global trajectory-level interactions between all steps
# ---------------------------------------------------------------------------
class AttentionPoolingHead(nn.Module):
    def __init__(self, hidden_size: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.norm2 = nn.LayerNorm(hidden_size)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, step_hidden, step_mask):
        # key_padding_mask expects True = IGNORE this position
        key_padding_mask = ~step_mask
        attn_out, _ = self.self_attn(
            step_hidden, step_hidden, step_hidden, key_padding_mask=key_padding_mask
        )
        x = self.norm1(step_hidden + attn_out)
        x = self.norm2(x + self.ffn(x))
        return self.out(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
REWARD_HEADS = {
    "linear": LinearHead,
    "mlp": MLPHead,
    "cnn": CNNHead,
    "gru": GRUHead,
    "attention": AttentionPoolingHead,
}


def build_reward_head(name: str, hidden_size: int, **kwargs) -> nn.Module:
    if name not in REWARD_HEADS:
        raise ValueError(f"Unknown reward head '{name}'. Choose from {list(REWARD_HEADS)}")
    return REWARD_HEADS[name](hidden_size, **kwargs)
