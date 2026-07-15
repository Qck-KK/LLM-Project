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
                    # 如果已经是我们想要的清洗后格式，直接添加
                    if "question" in obj and "steps" in obj and "labels" in obj:
                        records.append(obj)
                        continue
                        
                    # 解析你提供的这种原始格式 (通过 "label" 字段解析)
                    raw_label_text = obj.get("label", "")
                    if "Step 1:" not in raw_label_text:
                        continue  # 格式异常，跳过
                        
                    # 分离题目和步骤
                    q_part, steps_part = raw_label_text.split("Step 1:", 1)
                    question = q_part.strip()
                    step_lines = ("Step 1:" + steps_part).split("\n")
                    
                    steps = []
                    labels = []
                    for s_line in step_lines:
                        s_line = s_line.strip()
                        if not s_line:
                            continue
                        
                        # 检查结尾的 + 或 - 来判断该步正确与否
                        if s_line.endswith("+"):
                            labels.append(1)
                            steps.append(s_line[:-1].strip()) # 去掉末尾的加号
                        elif s_line.endswith("-"):
                            labels.append(0)
                            steps.append(s_line[:-1].strip()) # 去掉末尾的减号
                        else:
                            # 容错处理
                            labels.append(0)
                            steps.append(s_line)
                            
                    records.append({
                        "question": question,
                        "steps": steps,
                        "labels": labels
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

        # Build "question + step1 <ки> step2 <ки> ... stepN <ки>"
        text = question.strip() + "\n"
        for s in steps:
            text += s.strip() + f" {self.step_token}\n"

        return {
            "text": text,
            "labels": labels,  # python list[int], length = num_steps
        }


def collate_fn(batch, tokenizer, max_length=2048):
    """
    Tokenizes a batch and pads the per-step `labels` to the same length
    the encoder will produce step embeddings for.
    """
    texts = [b["text"] for b in batch]
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )

    max_steps = max(len(b["labels"]) for b in batch)
    labels = torch.full((len(batch), max_steps), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["labels"])
        labels[i, :n] = torch.tensor(b["labels"], dtype=torch.long)

    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": labels,
    }
