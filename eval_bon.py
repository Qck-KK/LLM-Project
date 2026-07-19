"""
eval_bon.py
============
Best-of-N Verification Performance (BON@8 / BON@16 from the proposal).

This needs a DIFFERENT data format than training: for each question, several
candidate full solutions (e.g. sampled from a generator LLM), each with steps
+ whether its FINAL answer was correct. The trained PRM scores every
candidate; we pick the argmax-scored one and check whether ITS final answer
was correct. BON@k repeats this using k randomly-drawn candidates per
question (averaged over multiple draws) so results aren't just an artifact of
which k candidates happened to be sampled.

Expected input JSONL, one question per line:
{
  "question": "...",
  "candidates": [
     {"steps": ["step1", "step2", ...], "final_correct": 1},
     {"steps": [...], "final_correct": 0},
     ...   # need at least max(k) candidates per question
  ]
}

Usage:
    python eval_bon.py --eval_file data/bon_eval.jsonl \
        --head mlp --checkpoint checkpoints/mlp_head.pt \
        --ks 8 16 --n_draws 20
"""

import argparse
import json
import os
import random
import math

import torch

from encoder import FrozenStepEncoder
from reward_heads import build_reward_head


def aggregate_trajectory_score(q_values: torch.Tensor, step_mask: torch.Tensor, mode: str = "min") -> float:
    """
    Collapse a trajectory's per-step Q-values into one scalar score.
    `min` is the standard, conservative PRM aggregation (a solution is only as
    good as its weakest step) -- switch to 'mean' or 'last' if you want to
    compare aggregation strategies too, it's an easy ablation to add.
    """
    valid_q = q_values[step_mask]
    if valid_q.numel() == 0:
        return float("-inf")
    if mode == "min":
        return valid_q.min().item()
    elif mode == "mean":
        return valid_q.mean().item()
    elif mode == "last":
        return valid_q[-1].item()
    elif mode == "sum":
        return valid_q.sum().item()
    elif mode == "prod":
        return valid_q.prod().item()
    raise ValueError(mode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_file", required=True)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--head", required=True,
                         choices=["linear", "mlp", "cnn", "gru", "attention"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ks", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--n_draws", type=int, default=20,
                         help="How many random k-subsets to average over, per question.")
    parser.add_argument("--agg", default="min", choices=["min", "mean", "last", "sum", "prod"])
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--results_dir", default=None,
                         help="If set, writes a JSON file here for summarize_results.py to pick up.")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"))

    encoder = FrozenStepEncoder(model_name=args.model_name).to(device)
    encoder.eval()

    head = build_reward_head(args.head, hidden_size=encoder.hidden_size).to(device)
    head.load_state_dict(torch.load(args.checkpoint, map_location=device))
    head.eval()

    questions = []
    with open(args.eval_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))

    # Score every candidate of every question ONCE, then reuse for all k's/draws.
    all_scores, all_correct = [], []
    with torch.no_grad():
        for q_obj in questions:
            cand_scores, cand_correct = [], []
            for cand in q_obj["candidates"]:
                text = q_obj["question"].strip() + "\n"
                # --- 新增：兼容不同的候选数据格式 ---
                if "steps" in cand:
                    steps = cand["steps"]
                else:
                    # 如果找不到 steps，尝试提取长文本并按换行符切分
                    raw_text = cand.get("text", cand.get("response", cand.get("content", "")))
                    steps = [s for s in raw_text.split('\n') if s.strip()]
                for s in steps:
                    text += s.strip() + f" {encoder.step_token}\n"
                step_hidden, step_mask = encoder.encode_texts([text], device=device, max_length=args.max_length)
                q_values = head(step_hidden, step_mask)[0]
                score = aggregate_trajectory_score(q_values, step_mask[0], mode=args.agg)
                cand_scores.append(score)
                # 使用 .get() 方法依次尝试获取可能的正确性标签，如果都找不到则默认返回 0 (False)
                correct_val = cand.get("final_correct", cand.get("label", cand.get("is_correct", cand.get("correct", 0))))
                cand_correct.append(int(correct_val))
            all_scores.append(cand_scores)
            all_correct.append(cand_correct)

    print(f"\n=== BON results ({args.head}, agg={args.agg}) ===")
    bon_results = {}
    for k in args.ks:
        hits, trials = 0, 0
        for scores, corrects in zip(all_scores, all_correct):
            if len(scores) < k:
                continue  # not enough candidates for this question at this k
            for _ in range(args.n_draws):
                idx = random.sample(range(len(scores)), k)
                sub_scores = [scores[i] for i in idx]
                sub_correct = [corrects[i] for i in idx]
                best = max(range(k), key=lambda i: sub_scores[i])
                hits += sub_correct[best]
                trials += 1
        acc = hits / max(trials, 1)
        bon_results[f"bon@{k}"] = acc
        print(f"BON@{k}: {acc:.4f}  (over {trials} draws)")

    if args.results_dir:
        os.makedirs(args.results_dir, exist_ok=True)
        out = {"head": args.head, "checkpoint": args.checkpoint, "agg": args.agg, **bon_results}
        out_path = os.path.join(args.results_dir, f"{args.head}_bon_metrics.json")
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
