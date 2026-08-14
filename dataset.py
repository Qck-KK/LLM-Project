"""
dataset.py
===========
Loads Math-Shepherd-style data and formats it for FrozenStepEncoder.

Expected raw record (Math-Shepherd's public format, one JSON object per line):
{
  "question": "...",
  "steps": ["Step 1 text", "Step 2 text", ...],
  "labels": [1, 1, 0, 1, ...]   # 1 = step is correct, 0 = incorrect
}
Adjust `_load_raw` if your local copy uses different field names -- the rest
of the pipeline only depends on the (question, steps, labels) triple.
"""

import json

import torch
from torch.utils.data import Dataset


class MathShepherdStepDataset(Dataset):
    def __init__(self, path: str, step_token: str = "ки", max_length: int = 2048):
        self.step_token = step_token
        self.max_length = max_length
        self.records = self._load_raw(path)

    @staticmethod
    def _load_raw(path):
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    obj = json.loads(line)
                    if "question" in obj and "steps" in obj and "labels" in obj:
                        records.append(obj)
                        continue

                    raw_label_text = obj.get("label", "")
                    if "Step 1:" not in raw_label_text:
                        continue

                    q_part, steps_part = raw_label_text.split("Step 1:", 1)
                    question = q_part.strip()
                    step_lines = ("Step 1:" + steps_part).split("\n")

                    steps = []
                    labels = []
                    for s_line in step_lines:
                        s_line = s_line.strip()
                        if not s_line:
                            continue

                        if s_line.endswith("+"):
                            labels.append(1)
                            steps.append(s_line[:-1].strip())
                        elif s_line.endswith("-"):
                            labels.append(0)
                            steps.append(s_line[:-1].strip())
                        else:
                            labels.append(0)
                            steps.append(s_line)

                    records.append({
                        "question": question,
                        "steps": steps,
                        "labels": labels,
                    })
                except json.JSONDecodeError:
                    continue

        return records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        question = rec["question"]
        steps = rec["steps"]
        labels = rec["labels"]
        assert len(steps) == len(labels), "steps/labels length mismatch"

        text = question.strip() + "\n"
        for step in steps:
            text += step.strip() + f" {self.step_token}\n"

        return {
            "text": text,
            "labels": labels,
        }


def collate_fn(batch, tokenizer, max_length=2048):
    """
    Tokenizes a batch and pads the per-step labels to the same length the
    encoder will produce step embeddings for.
    """
    texts = [item["text"] for item in batch]
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )

    max_steps = max(len(item["labels"]) for item in batch)
    labels = torch.full((len(batch), max_steps), -100, dtype=torch.long)
    for i, item in enumerate(batch):
        n = len(item["labels"])
        labels[i, :n] = torch.tensor(item["labels"], dtype=torch.long)

    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": labels,
    }
