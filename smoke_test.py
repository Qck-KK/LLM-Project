"""End-to-end smoke test: every stage of the pipeline on a few examples.

    python smoke_test.py            # a few minutes on a GPU, longer on CPU
    python smoke_test.py --no_lora  # skip the LoRA stage (needs peft)

It uses only files in the repository -- data/dummy_train.jsonl (150
Math-Shepherd trajectories) and the first questions of
data/gsm8k_qwen0.5b_bon16.jsonl -- plus Qwen2.5-0.5B, which Hugging Face
downloads on first use (about 1 GB). Everything is written under
runs/smoke/ and the script stops at the first failing stage.

The numbers it prints are meaningless (a head trained for one epoch on 120
trajectories); the point is that every script runs end to end with the same
arguments the full experiments use. run_experiments.py runs the real thing.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

PY = sys.executable
HEADS = ["linear", "mlp", "cnn", "gru", "attention", "attention_pe"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join("runs", "smoke"))
    parser.add_argument("--bon_questions", type=int, default=4)
    parser.add_argument("--no_lora", action="store_true")
    args = parser.parse_args()

    out = args.out
    if os.path.exists(out):
        shutil.rmtree(out)
    data, cache, ckpt, res = (os.path.join(out, d) for d in ("data", "cache", "checkpoints", "results"))
    for d in (data, cache, ckpt, res):
        os.makedirs(d)

    # 120 training and 30 validation trajectories from the bundled sample.
    with open(os.path.join("data", "dummy_train.jsonl"), encoding="utf-8") as f:
        rows = [line for line in f if line.strip()]
    with open(os.path.join(data, "train.jsonl"), "w", encoding="utf-8") as f:
        f.writelines(rows[:120])
    with open(os.path.join(data, "val.jsonl"), "w", encoding="utf-8") as f:
        f.writelines(rows[120:])
    with open(os.path.join("data", "gsm8k_qwen0.5b_bon16.jsonl"), encoding="utf-8") as f:
        bon = [next(f) for _ in range(args.bon_questions)]
    bon_source = os.path.join(data, "bon.jsonl")
    with open(bon_source, "w", encoding="utf-8") as f:
        f.writelines(bon)

    split = ["--calibration_fraction", "0.5", "--split_seed", "42"]
    boot = ["--bootstrap_samples", "50"]
    train_cache, val_cache, bon_cache = (os.path.join(cache, d) for d in ("train", "val", "bon"))
    single = os.path.join(data, "single_eval.jsonl")
    ckpt_of = lambda head: os.path.join(ckpt, head + "_head.pt")

    stages = [
        ("prepare: convert candidates",
         [PY, "prepare_data.py", "convert", "--source", bon_source, "--out", single]),
        ("encode: training set",
         [PY, "precompute_embeddings.py", "--train_file", os.path.join(data, "train.jsonl"),
          "--cache_dir", train_cache, "--batch_size", "8"]),
        ("encode: validation set",
         [PY, "precompute_embeddings.py", "--train_file", os.path.join(data, "val.jsonl"),
          "--cache_dir", val_cache, "--batch_size", "8"]),
        ("encode: Best-of-N candidates",
         [PY, "precompute_eval_embeddings.py", "--eval_file", single,
          "--cache_dir", bon_cache, "--batch_size", "8"]),
        ("audit: labels and baselines",
         [PY, "-m", "analysis.analyze_data_bias", "--cache_dir", val_cache,
          "--results_dir", res, *split]),
    ]
    for head in HEADS:
        stages.append(("train: " + head,
                       [PY, "train_from_cache.py", "--cache_dir", train_cache,
                        "--val_cache_dir", val_cache, "--head", head, "--epochs", "1",
                        *split, "--save_path", ckpt_of(head), "--results_dir", res]))
    stages.append(("train: linear with BCE loss",
                   [PY, "train_from_cache.py", "--cache_dir", train_cache,
                    "--val_cache_dir", val_cache, "--head", "linear", "--epochs", "1",
                    "--loss", "bce", *split, "--save_path", os.path.join(ckpt, "bce", "linear_head.pt"),
                    "--results_dir", os.path.join(res, "bce")]))
    for head in HEADS:
        stages.append(("evaluate: step metrics, " + head,
                       [PY, "-m", "eval.eval_step_metrics", "--cache_dir", val_cache,
                        "--head", head, "--checkpoint", ckpt_of(head), *split,
                        "--results_dir", res]))
    stages += [
        ("evaluate: single solutions (GSM8K)",
         [PY, "-m", "eval.eval_single_from_cache", "--cache_dir", bon_cache, "--head", "mlp",
          "--checkpoint", ckpt_of("mlp"), "--agg", "min", *split, "--results_dir", res]),
        ("evaluate: coin-flip baseline",
         [PY, "-m", "eval.eval_coin_flip_baseline", "--val_cache_dir", val_cache,
          "--single_cache_dir", bon_cache, "--results_dir", res, "--trials", "10", "--seed", "42",
          *split]),
        ("analyse: behaviour and perturbations",
         [PY, "-m", "analysis.analyze_head_behavior", "--cache_dir", val_cache,
          "--checkpoint_dir", ckpt, "--results_dir", res, "--heads", *HEADS, *split,
          "--run_perturbations"]),
        ("analyse: causal prefixes and pruning",
         [PY, "-m", "analysis.analyze_offline_pruning", "--cache_dir", val_cache,
          "--checkpoint_dir", ckpt, "--results_dir", res, "--heads", *HEADS,
          "--budgets", "0.05", "0.10", "--primary_budget", "0.10", *boot, *split, "--seed", "42"]),
        ("statistics: bootstrap step metrics",
         [PY, "-m", "analysis.bootstrap_step_metrics", "--cache_dir", val_cache,
          "--checkpoint_dir", ckpt, "--results_dir", res, "--heads", *HEADS, *boot, *split,
          "--seed", "42"]),
        ("statistics: bootstrap causal metrics",
         [PY, "-m", "analysis.bootstrap_causal_metrics", "--results_dir", res,
          "--heads", *HEADS, *boot, "--seed", "42"]),
        ("statistics: Holm correction",
         [PY, "-m", "analysis.holm_correction", os.path.join(res, "step_metrics_ci_pairwise.csv"),
          os.path.join(res, "causal_metrics_ci_pairwise.csv"), "--bootstrap_samples", "50"]),
        ("statistics: paired comparison of two runs",
         [PY, "-m", "analysis.compare_lora_frozen",
          "--arm", "pqm_linear:{0}:linear:{1}".format(val_cache, ckpt_of("linear")),
          "--arm", "bce_linear:{0}:linear:{1}".format(val_cache, os.path.join(ckpt, "bce", "linear_head.pt")),
          "--results_dir", res, "--out_name", "pqm_vs_bce", *boot, *split]),
        ("evaluate: Best-of-N vs majority voting",
         [PY, "-m", "eval.eval_bon", "--cache_dir", bon_cache, "--eval_file", single,
          "--source_file", bon_source, "--checkpoint_dir", ckpt, "--heads", *HEADS,
          "--aggs", "min", "mean", "last", "--ks", "1", "4", "16",
          "--subsets_per_question", "2", *boot, "--results_dir", res, "--seed", "42"]),
        ("evaluate: in-distribution single solutions",
         [PY, "-m", "eval.eval_single_from_step_cache", "--cache_dir", val_cache,
          "--source_file", os.path.join(data, "val.jsonl"), "--checkpoint_dir", ckpt,
          "--heads", *HEADS, "--aggs", "min", "--results_dir", res, *boot, *split, "--seed", "42"]),
        ("summarise: results table",
         [PY, "-m", "eval.summarize_results", "--results_dir", res,
          "--out_csv", os.path.join(res, "summary.csv"), "--out_md", os.path.join(res, "summary.md")]),
    ]
    if not args.no_lora:
        lora = os.path.join(ckpt, "lora")
        stages += [
            ("LoRA: train adapters on 16 trajectories",
             [PY, "train_lora.py", "--train_file", os.path.join(data, "train.jsonl"),
              "--epochs", "1", "--max_records", "16", "--batch_size", "4",
              "--log_every", "2", "--save_dir", lora, "--results_dir", os.path.join(res, "lora")]),
            ("LoRA: re-encode validation set",
             [PY, "precompute_embeddings.py", "--train_file", os.path.join(data, "val.jsonl"),
              "--cache_dir", os.path.join(cache, "val_lora"), "--batch_size", "8",
              "--lora_path", os.path.join(lora, "adapter")]),
            ("LoRA: compare with the frozen encoder",
             [PY, "-m", "analysis.compare_lora_frozen",
              "--arm", "lora_linear:{0}:linear:{1}".format(
                  os.path.join(cache, "val_lora"), os.path.join(lora, "linear_head.pt")),
              "--arm", "frozen_linear:{0}:linear:{1}".format(val_cache, ckpt_of("linear")),
              "--results_dir", res, *boot, *split]),
        ]

    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    log_path = os.path.join(out, "smoke.log")
    start = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        for index, (name, cmd) in enumerate(stages, 1):
            t0 = time.time()
            print("[{0:2d}/{1}] {2:<50}".format(index, len(stages), name), end="", flush=True)
            log.write("\n$ " + " ".join(cmd) + "\n")
            log.flush()
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
            if result.returncode:
                print("FAILED")
                print("\nStage failed; the last lines of {0}:\n".format(log_path))
                log.flush()
                with open(log_path, encoding="utf-8", errors="replace") as f:
                    print("".join(f.readlines()[-25:]))
                sys.exit(1)
            print("ok  {0:5.1f}s".format(time.time() - t0))

    with open(os.path.join(res, "linear_step_metrics.json")) as f:
        metrics = json.load(f)
    print("\nAll {0} stages passed in {1:.0f}s. Outputs: {2}".format(
        len(stages), time.time() - start, out))
    print("(Sanity only: linear head, 1 epoch on 120 trajectories -> held-out ROC-AUC "
          "{0:.3f})".format(metrics.get("step_roc_auc", float("nan"))))


if __name__ == "__main__":
    main()
