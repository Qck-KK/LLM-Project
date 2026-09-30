
import json

import torch
from torch.utils.data import Dataset

STEP_MARKER = "ки"


def parse_math_shepherd(input_text, label_text, marker=STEP_MARKER):
    segments = input_text.split(marker)
    if len(segments) < 2:
        return None
    labels, position = [], 0
    for segment in segments[:-1]:
        position += len(segment)
        if position >= len(label_text) or label_text[position] not in "+-":
            return None
        labels.append(1 if label_text[position] == "+" else 0)
        position += 1
    rebuilt = "".join(
        segment + ("+" if label == 1 else "-")
        for segment, label in zip(segments[:-1], labels)
    ) + segments[-1]
    if rebuilt != label_text:
        return None

    first = segments[0]
    split_at = first.find("Step 1:")
    if split_at < 0:
        return None
    question = first[:split_at].strip()
    steps = [first[split_at:].strip()] + [s.strip() for s in segments[1:-1]]
    return {"question": question, "steps": steps, "labels": labels}


class MathShepherdStepDataset(Dataset):
    def __init__(self, path: str, step_token: str = "ки", max_length: int = 2048):
        self.step_token = step_token
        self.max_length = max_length
        self.records = self._load_raw(path)

    @staticmethod
    def _load_raw(path):
        records = []
        skipped = 0
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

                    parsed = parse_math_shepherd(obj.get("input", ""), obj.get("label", ""))
                    if parsed is None:
                        skipped += 1
                        continue
                    records.append(parsed)
                except json.JSONDecodeError:
                    skipped += 1
                    continue

        if skipped:
            print(f"[dataset] skipped {skipped} unparseable records in {path}")
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
    
    texts = [item["text"] for item in batch]
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )

    step_token_id = tokenizer.convert_tokens_to_ids("ки")
    kept_counts = [int((row == step_token_id).sum()) for row in enc["input_ids"]]

    max_steps = max(max(kept_counts), 1)
    labels = torch.full((len(batch), max_steps), -100, dtype=torch.long)
    for i, item in enumerate(batch):
        n = min(len(item["labels"]), kept_counts[i])
        if n:
            labels[i, :n] = torch.tensor(item["labels"][:n], dtype=torch.long)

    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": labels,
    }
