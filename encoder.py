"""
encoder.py
============
Stage 1: Frozen Semantic Encoder.

We wrap a small math-oriented LM (default: Qwen2.5-0.5B) and freeze ALL of its
parameters. It is used purely as a feature extractor.

Design choice (follows Math-Shepherd / PQM convention):
  Each reasoning step in the input text is terminated with a special
  "step marker" token (default: "ки", following Math-Shepherd; you can swap
  this for any token that does not naturally occur in your corpus, e.g. "<STEP>").
  After a forward pass, we gather the hidden state AT the position of each step
  marker -> that vector represents "everything the model has read up to and
  including this step". This avoids having to mean/max-pool over variable-length
  step spans, and is exactly what Math-Shepherd / PQM do in practice.

Expected input format (produced by dataset.py):
  "<question> ... <step_1_text> ки <step_2_text> ки ... <step_N_text> ки"

Output:
  step_hidden : (B, S, H)  -- padded per-step embeddings
  step_mask   : (B, S)     -- True where a real step exists (False = padding)
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

        # Register the step marker as a dedicated special token so it is
        # ALWAYS a single, stable token id regardless of surrounding text.
        num_added = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": [step_token]}
        )
        self.step_token = step_token
        self.step_token_id = self.tokenizer.convert_tokens_to_ids(step_token)

        self.model = AutoModel.from_pretrained(model_name, torch_dtype=dtype)
        if num_added > 0:
            # embedding matrix must grow to fit the new special token
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.hidden_size = self.model.config.hidden_size

        # ---- freeze everything: encoder is never trained ----
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

    def train(self, mode: bool = True):
        # Prevent accidental unfreezing / dropout activation if someone calls
        # outer_module.train() -- the encoder always stays in eval mode.
        super().train(mode)
        self.model.eval()
        return self

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        input_ids, attention_mask: (B, T) already on the correct device.

        Returns:
            step_hidden: (B, S_max, H) float tensor, zero-padded
            step_mask:   (B, S_max) bool tensor, True = real step
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # (B, T, H)

        is_step_pos = input_ids.eq(self.step_token_id)  # (B, T) bool
        n_steps_per_example = is_step_pos.sum(dim=1)     # (B,)
        max_steps = int(n_steps_per_example.max().item())
        max_steps = max(max_steps, 1)  # guard against a pathological all-empty batch

        B, T, H = last_hidden.shape
        step_hidden = last_hidden.new_zeros(B, max_steps, H)
        step_mask = torch.zeros(B, max_steps, dtype=torch.bool, device=last_hidden.device)

        for b in range(B):
            idx = is_step_pos[b].nonzero(as_tuple=True)[0]
            n = idx.numel()
            if n == 0:
                continue
            step_hidden[b, :n] = last_hidden[b, idx]
            step_mask[b, :n] = True

        return step_hidden, step_mask

    def encode_texts(self, texts, device="cuda", max_length=2048):
        """Convenience helper: tokenize a batch of raw strings and encode them."""
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        return self.forward(enc["input_ids"], enc["attention_mask"])
