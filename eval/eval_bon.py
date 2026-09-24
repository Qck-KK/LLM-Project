"""Best-of-N reranking evaluation on Qwen's own GSM8K candidates.

This is the canonical way a process reward model gets used: sample N candidate
solutions, score each one, keep the highest. The workflow originally excluded
Best-of-N for compute reasons, but the expensive half is already paid -- the
21,104 candidates are encoded in `cache/single_eval` -- so scoring them with a
lightweight head costs seconds.

Two reference points frame every head:

* `majority_vote` (self-consistency): pick the most common final answer among the
  k candidates, no reward model involved. This is the baseline a PRM has to beat
  to be worth deploying; raw accuracy alone says nothing.
* `oracle` : correct if ANY of the k candidates is correct. The ceiling that
  reranking could reach with a perfect scorer.

For k < N each question is evaluated over several random subsets of its 16
candidates, so the curve does not depend on the arbitrary generation order.
Confidence intervals resample questions, which are the independent unit here --
the 16 candidates of one question are anything but independent.
"""

import argparse
import collections
import csv
import glob
import json
import os

import torch

from eval.eval_utils import (
    HEAD_CHOICES,
    aggregate_trajectory_scores,
    get_device,
)
from reward_heads import build_reward_head


def pad_and_concat(tensors, value):
    max_steps = max(tensor.shape[1] for tensor in tensors)
    return torch.cat([
        torch.nn.functional.pad(tensor, (0, max_steps - tensor.shape[1]), value=value)
        for tensor in tensors
    ])


@torch.no_grad()
def score_cache(head, shard_paths, device, agg):
    """One trajectory score per cached candidate, in cache order."""
    head.eval()
    scores = []
    for path in shard_paths:
        shard = torch.load(path, map_location=device)
        step_hidden = shard["step_hidden"].to(device).float()
        step_mask = shard["step_mask"].to(device).bool()
        q_values = head(step_hidden, step_mask)
        scores.append(aggregate_trajectory_scores(q_values, step_mask, mode=agg).cpu())
    return torch.cat(scores)


def percentile_ci(samples, alpha=0.05):
    tensor = torch.tensor(samples, dtype=torch.float64)
    tensor = tensor[torch.isfinite(tensor)]
    if tensor.numel() == 0:
        return float("nan"), float("nan")
    return (float(torch.quantile(tensor, alpha / 2)),
            float(torch.quantile(tensor, 1 - alpha / 2)))


def majority_choice(preds, subset):
    """Index within `subset` of the most common non-empty answer (ties: first)."""
    counts = collections.Counter(preds[i] for i in subset if preds[i] not in ("", None))
    if not counts:
        return subset[0]
    top = counts.most_common(1)[0][0]
    for i in subset:
        if preds[i] == top:
            return i
    return subset[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--eval_file", required=True,
                        help="single_eval.jsonl carrying question_id/candidate_id.")
    parser.add_argument("--source_file", default=None,
                        help="Original BoN file with pred_num, for majority voting.")
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--checkpoint_pattern", default="{head}_head.pt")
    parser.add_argument("--heads", nargs="+", default=list(HEAD_CHOICES))
    parser.add_argument("--aggs", nargs="+", default=["min", "mean", "last"])
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--subsets_per_question", type=int, default=20)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = get_device(args.device)
    os.makedirs(args.results_dir, exist_ok=True)

    records = [json.loads(line) for line in open(args.eval_file, encoding="utf-8")]
    correct = torch.tensor([int(r["final_correct"]) for r in records])
    question_of = [r["question_id"] for r in records]
    n_questions = max(question_of) + 1
    groups = [[] for _ in range(n_questions)]
    for index, qid in enumerate(question_of):
        groups[qid].append(index)

    preds = None
    if args.source_file:
        source = [json.loads(line) for line in open(args.source_file, encoding="utf-8")]
        preds = [c.get("pred_num", "") for d in source for c in d["candidates"]]
        if len(preds) != len(records):
            raise RuntimeError("source_file does not align with eval_file")

    shard_paths = sorted(glob.glob(os.path.join(args.cache_dir, "shard_*.pt")))
    assert shard_paths, "No cached shards in " + args.cache_dir
    with open(os.path.join(args.cache_dir, "hidden_size.txt")) as f:
        hidden_size = int(f.read().strip())
    n_cached = sum(torch.load(p, map_location="cpu")["final_correct"].shape[0]
                   for p in shard_paths)
    if n_cached != len(records):
        raise RuntimeError("cache holds {0} candidates, eval_file {1}".format(
            n_cached, len(records)))

    # Fixed subsets, shared by every head and baseline so the comparison is paired.
    generator = torch.Generator().manual_seed(args.seed)
    subsets = {}
    for k in args.ks:
        per_question = []
        for qid in range(n_questions):
            members = groups[qid]
            if k >= len(members):
                per_question.append([tuple(members)])
                continue
            draws = []
            for _ in range(args.subsets_per_question):
                order = torch.randperm(len(members), generator=generator)[:k]
                draws.append(tuple(members[int(i)] for i in order))
            per_question.append(draws)
        subsets[k] = per_question

    def evaluate(pick_fn):
        """Per-question accuracy, averaged over that question's subsets."""
        out = {}
        for k in args.ks:
            per_question = torch.zeros(n_questions)
            for qid in range(n_questions):
                hits = [int(correct[pick_fn(subset)]) for subset in subsets[k][qid]]
                per_question[qid] = sum(hits) / len(hits)
            out[k] = per_question
        return out

    curves = {}
    for head_name in args.heads:
        checkpoint = os.path.join(
            args.checkpoint_dir, args.checkpoint_pattern.format(head=head_name)
        )
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(checkpoint)
        head = build_reward_head(head_name, hidden_size).to(device)
        head.load_state_dict(torch.load(checkpoint, map_location=device))
        for agg in args.aggs:
            scores = score_cache(head, shard_paths, device, agg)
            curves[(head_name, agg)] = evaluate(
                lambda subset: max(subset, key=lambda i: float(scores[i]))
            )
            print("  scored {0:<14} agg={1}".format(head_name, agg))

    # References
    curves[("random", "-")] = evaluate(lambda subset: subset[0])
    curves[("oracle", "-")] = evaluate(
        lambda subset: max(subset, key=lambda i: int(correct[i]))
    )
    if preds is not None:
        curves[("majority_vote", "-")] = evaluate(
            lambda subset: majority_choice(preds, list(subset))
        )

    rows = []
    question_draws = [
        torch.randint(n_questions, (n_questions,), generator=generator)
        for _ in range(args.bootstrap_samples)
    ]
    for (head_name, agg), curve in curves.items():
        for k in args.ks:
            per_question = curve[k]
            point = float(per_question.mean())
            samples = [float(per_question[pick].mean()) for pick in question_draws]
            low, high = percentile_ci(samples)
            rows.append({
                "head": head_name,
                "agg": agg,
                "k": k,
                "accuracy": point,
                "ci_low": low,
                "ci_high": high,
                "n_questions": n_questions,
            })

    out_csv = os.path.join(args.results_dir, "bon_results.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    out_json = os.path.join(args.results_dir, "bon_results.json")
    with open(out_json, "w") as f:
        json.dump({
            "cache_dir": args.cache_dir,
            "checkpoint_dir": args.checkpoint_dir,
            "n_questions": n_questions,
            "n_candidates": len(records),
            "subsets_per_question": args.subsets_per_question,
            "bootstrap_samples": args.bootstrap_samples,
            "seed": args.seed,
            "rows": rows,
        }, f, indent=2)
    print("[saved] " + out_csv + " / " + out_json)


if __name__ == "__main__":
    main()
