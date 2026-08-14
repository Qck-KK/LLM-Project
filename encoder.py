"""
encoder.py
============
Stage 1: Frozen Semantic Encoder.

We wrap a small math-oriented LM (default: Qwen2.5-0.5B) and freeze ALL of its
parameters. It is used purely as a feature extractor.

Each reasoning step is terminated with a special step marker token. After a
forward pass, we gather the hidden state at every marker position and treat
those vectors as per-step representations.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class FrozenStepEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B",
        step_token: str = "ки",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        num_added = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": [step_token]}
        )
        self.step_token = step_token
        self.step_token_id = self.tokenizer.convert_tokens_to_ids(step_token)

        self.model = AutoModel.from_pretrained(model_name, torch_dtype=dtype)
        if num_added > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.hidden_size = self.model.config.hidden_size

        for param in self.model.parameters():
            param.requires_grad_(False)
        self.model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.eval()
        return self

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state

        is_step_pos = input_ids.eq(self.step_token_id)
        n_steps_per_example = is_step_pos.sum(dim=1)
        max_steps = int(n_steps_per_example.max().item())
        max_steps = max(max_steps, 1)

        batch_size, _, hidden_size = last_hidden.shape
        step_hidden = last_hidden.new_zeros(batch_size, max_steps, hidden_size)
        step_mask = torch.zeros(batch_size, max_steps, dtype=torch.bool, device=last_hidden.device)

        for b in range(batch_size):
            idx = is_step_pos[b].nonzero(as_tuple=True)[0]
            n = idx.numel()
            if n == 0:
                continue
            step_hidden[b, :n] = last_hidden[b, idx]
            step_mask[b, :n] = True

        return step_hidden, step_mask

    def encode_texts(self, texts, device="cuda", max_length=2048):
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        return self.forward(enc["input_ids"], enc["attention_mask"])
