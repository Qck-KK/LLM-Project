"""Runs every cache-based experiment on the corrected Math-Shepherd caches.

The caches `cache/train_fixed` and `cache/val_fixed` were rebuilt after the
parser fix in dataset.py (multi-line steps had been split into fake error
steps). Everything trained or evaluated on the older `_clean` caches predates
that fix; this driver regenerates the results from scratch.

Groups, in the order they run (Qwen never runs; the caches must exist):

* main       -- the six heads under the final protocol (lr=1e-4, 30 epochs,
                patience 5, zeta 4, seed 42) -> checkpoints/final/
* main_eval  -- the full evaluation chain for those heads -> results_final/:
                data-bias audit and baselines, held-out step metrics,
                behaviour / first-error / perturbation analysis, causal-prefix
                pruning, trajectory-level bootstrap CIs with Holm correction,
                Best-of-N (argmax and PRM-weighted voting), out-of-distribution
                and in-distribution single-solution evaluation, summary table.
* bce        -- PQM ranking loss vs pointwise step BCE (linear / mlp / attention).
* zeta       -- PQM margin 2 and 8 against the default 4 (linear / attention).
* seeds      -- seeds 43 and 44 for attention / cnn / mlp.
* lr         -- 3e-5 and 3e-4 for all six heads, around the chosen 1e-4.
* long       -- mlp and attention_pe again with a 60-epoch cap: on the corrected
                data both picked epoch 30, the cap, so they had not converged.
* long_eval  -- paired comparison of the 60-epoch runs against the 30-epoch ones.
* main_eval_lr3e-5 -- the same evaluation chain for the six heads trained at
                lr=3e-5 (from the lr group). On the corrected data 3e-5 reaches a
                lower development loss than the protocol's 1e-4 for five heads,
                so the conclusions are re-checked there -> results_final_lr3e-5/
* lora       -- the frozen-encoder control: LoRA (r=16 on q/k/v/o) plus a linear
                value head, one pass over the corrected training set, then the
                validation and GSM8K caches re-encoded with the adapted encoder.
                Needs transformers and peft (--encoder_python). About 10 hours.
* lora_eval  -- LoRA vs the frozen linear and attention heads (paired), and
                Best-of-N for the LoRA head.
* ablation_eval -- paired bootstrap comparisons of every ablation against the
                final heads, Holm correction, Best-of-N for the BCE heads.

Training steps are skipped when their checkpoint and efficiency file already
exist, so an interrupted run resumes where it stopped. Evaluation always reruns.

    python run_experiments.py                          # everything
    python run_experiments.py --only main main_eval    # a subset
    python run_experiments.py --dry_run                # print the commands
"""

import argparse
import os
import shutil
import subprocess
import sys

TRAIN_CACHE = "cache/train_fixed"
VAL_CACHE = "cache/val_fixed"
BON_CACHE = "cache/single_eval"          # GSM8K candidates; unaffected by the parser fix
FINAL_CKPT = "checkpoints/final"
FINAL_OUT = "results_final"
ABL_CKPT = "checkpoints/ablations"
ABL_OUT = "results_ablations"
ALL_HEADS = ["linear", "mlp", "cnn", "gru", "attention", "attention_pe"]

SEED_HEADS = ["attention", "cnn", "mlp"]
SEEDS = [43, 44]
BCE_HEADS = ["linear", "mlp", "attention"]
ZETA_HEADS = ["linear", "attention"]
ZETAS = ["2", "8"]
OTHER_LRS = ["3e-5", "3e-4"]
LONG_HEADS = ["mlp", "attention_pe"]
LONG_EPOCHS = "60"

SPLIT = ["--calibration_fraction", "0.5", "--split_seed", "42"]
COMMON = ["--cache_dir", TRAIN_CACHE, "--val_cache_dir", VAL_CACHE,
          "--early_stopping_patience", "5", *SPLIT]
PY = sys.executable
ENCODER_PY = sys.executable  # overridden by --encoder_python for Qwen-running steps
LORA_CKPT = "checkpoints/ablations/lora"
VAL_LORA_CACHE = "cache/val_lora_fixed"
BON_LORA_CACHE = "cache/single_eval_lora_fixed"


def train_step(head, save_path, results_dir, lr="1e-4", zeta="4.0", seed="42", loss="pqm",
               epochs="30"):
    cmd = [PY, "train_from_cache.py", *COMMON, "--epochs", epochs, "--head", head,
           "--lr", lr, "--zeta", zeta, "--seed", str(seed), "--loss", loss,
           "--save_path", save_path, "--results_dir", results_dir]
    done = [save_path, os.path.join(results_dir, head + "_efficiency.json")]
    return cmd, done


def final(head):
    return "{0}/{1}_head.pt".format(FINAL_CKPT, head)


def ablation(kind, tag):
    return "{0}/{1}/{2}_head.pt".format(ABL_CKPT, kind, tag)


def training_steps(groups):
    steps = []
    if "main" in groups:
        for head in ALL_HEADS:
            steps.append(train_step(head, final(head), FINAL_OUT))
    train_log = os.path.join(ABL_OUT, "train")
    if "bce" in groups:
        for head in BCE_HEADS:
            steps.append(train_step(head, ablation("bce", head),
                                    os.path.join(train_log, "bce_" + head), loss="bce"))
    if "zeta" in groups:
        for head in ZETA_HEADS:
            for zeta in ZETAS:
                tag = "{0}_z{1}".format(head, zeta)
                steps.append(train_step(head, ablation("zeta", tag),
                                        os.path.join(train_log, tag), zeta=zeta))
    if "seeds" in groups:
        for head in SEED_HEADS:
            for seed in SEEDS:
                tag = "{0}_s{1}".format(head, seed)
                steps.append(train_step(head, ablation("seeds", tag),
                                        os.path.join(train_log, tag), seed=seed))
    if "lr" in groups:
        for lr in OTHER_LRS:
            for head in ALL_HEADS:
                tag = "{0}_lr{1}".format(head, lr)
                steps.append(train_step(head, ablation("lr", tag),
                                        os.path.join(train_log, tag), lr=lr))
    if "long" in groups:
        for head in LONG_HEADS:
            tag = "{0}_e{1}".format(head, LONG_EPOCHS)
            steps.append(train_step(head, ablation("long", tag),
                                    os.path.join(train_log, tag), epochs=LONG_EPOCHS))
    return steps


def bon(checkpoint_dir, heads, results_dir):
    return [PY, "-m", "eval.eval_bon", "--cache_dir", BON_CACHE,
            "--eval_file", "data/single_eval.jsonl",
            "--source_file", "data/gsm8k_qwen0.5b_bon16.jsonl",
            "--checkpoint_dir", checkpoint_dir, "--heads", *heads,
            "--aggs", "min", "mean", "last", "--ks", "1", "2", "4", "8", "16",
            "--subsets_per_question", "20", "--bootstrap_samples", "2000",
            "--results_dir", results_dir, "--seed", "42"]


def main_eval_steps(out=None, ckpt=None, pattern="{head}_head.pt"):
    """The full evaluation chain for six heads stored as ckpt/pattern."""
    out, ckpt = out or FINAL_OUT, ckpt or FINAL_CKPT
    pat = ["--checkpoint_pattern", pattern]

    def checkpoint(head):
        return os.path.join(ckpt, pattern.format(head=head))
    steps = [[PY, "-m", "analysis.analyze_data_bias", "--cache_dir", VAL_CACHE,
              "--results_dir", out, "--position_bins", "5", *SPLIT]]
    for head in ALL_HEADS:
        steps.append([PY, "-m", "eval.eval_step_metrics", "--cache_dir", VAL_CACHE,
                      "--head", head, "--checkpoint", checkpoint(head), *SPLIT,
                      "--results_dir", out])
    for head in ALL_HEADS:
        steps.append([PY, "-m", "eval.eval_single_from_cache", "--cache_dir", BON_CACHE,
                      "--head", head, "--checkpoint", checkpoint(head), "--agg", "min", *SPLIT,
                      "--results_dir", out])
    steps.append([PY, "-m", "eval.eval_coin_flip_baseline", "--val_cache_dir", VAL_CACHE,
                  "--single_cache_dir", BON_CACHE, "--results_dir", out,
                  "--trials", "100", "--seed", "42", *SPLIT])
    steps.append([PY, "-m", "analysis.analyze_head_behavior", "--cache_dir", VAL_CACHE,
                  "--checkpoint_dir", ckpt, *pat, "--results_dir", out, "--heads", *ALL_HEADS,
                  *SPLIT, "--run_perturbations"])
    steps.append([PY, "-m", "analysis.analyze_offline_pruning", "--cache_dir", VAL_CACHE,
                  "--checkpoint_dir", ckpt, *pat, "--results_dir", out, "--heads", *ALL_HEADS,
                  "--budgets", "0.01", "0.05", "0.10", "--primary_budget", "0.05",
                  "--primary_policy", "single_low", "--bootstrap_samples", "1000",
                  *SPLIT, "--seed", "42"])
    steps.append([PY, "-m", "analysis.bootstrap_step_metrics", "--cache_dir", VAL_CACHE,
                  "--checkpoint_dir", ckpt, *pat, "--results_dir", out, "--heads", *ALL_HEADS,
                  "--bootstrap_samples", "2000", *SPLIT, "--seed", "42"])
    steps.append([PY, "-m", "analysis.bootstrap_causal_metrics", "--results_dir", out,
                  "--heads", *ALL_HEADS, "--bootstrap_samples", "2000", "--seed", "42"])
    steps.append([PY, "-m", "analysis.holm_correction",
                  os.path.join(out, "step_metrics_ci_pairwise.csv"),
                  os.path.join(out, "causal_metrics_ci_pairwise.csv"),
                  "--bootstrap_samples", "2000"])
    steps.append(bon(ckpt, ALL_HEADS, out) + pat)
    steps.append([PY, "-m", "eval.eval_single_from_step_cache", "--cache_dir", VAL_CACHE,
                  "--source_file", "data/val.jsonl",
                  "--checkpoint_dir", ckpt, *pat, "--heads", *ALL_HEADS,
                  "--aggs", "min", "mean", "last", "--results_dir", out,
                  "--bootstrap_samples", "2000", *SPLIT, "--seed", "42"])
    steps.append([PY, "-m", "eval.summarize_results", "--results_dir", out,
                  "--out_csv", os.path.join(out, "summary.csv"),
                  "--out_md", os.path.join(out, "summary.md")])
    return steps


def holm(pairwise_csv, family):
    return [PY, "-m", "analysis.holm_correction", pairwise_csv,
            "--bootstrap_samples", "2000", "--family", family]


def arm(name, head, checkpoint):
    return ["--arm", "{0}:{1}:{2}:{3}".format(name, VAL_CACHE, head, checkpoint)]


def compare(out_name, arms):
    return [PY, "-m", "analysis.compare_lora_frozen", *arms, "--results_dir", ABL_OUT,
            "--out_name", out_name, "--bootstrap_samples", "2000", *SPLIT]


def ablation_eval_steps():
    comparisons = {}
    arms = []
    for head in SEED_HEADS:
        arms += arm(head + "_s42", head, final(head))
        for seed in SEEDS:
            tag = "{0}_s{1}".format(head, seed)
            arms += arm(tag, head, ablation("seeds", tag))
    comparisons["seeds"] = arms

    arms = []
    for head in BCE_HEADS:
        arms += arm("pqm_" + head, head, final(head))
        arms += arm("bce_" + head, head, ablation("bce", head))
    comparisons["loss_pqm_vs_bce"] = arms

    arms = []
    for head in ZETA_HEADS:
        arms += arm(head + "_z4", head, final(head))
        for zeta in ZETAS:
            tag = "{0}_z{1}".format(head, zeta)
            arms += arm(tag, head, ablation("zeta", tag))
    comparisons["zeta"] = arms

    arms = []
    for head in ALL_HEADS:
        arms += arm(head + "_lr1e-4", head, final(head))
        for lr in OTHER_LRS:
            tag = "{0}_lr{1}".format(head, lr)
            arms += arm(tag, head, ablation("lr", tag))
    comparisons["lr"] = arms

    steps = [compare(name, arms) for name, arms in comparisons.items()]
    # Only the comparisons each ablation is about: across heads under one seed or
    # one loss, within a head across zeta or lr values.
    families = {"seeds": "suffix", "loss_pqm_vs_bce": "suffix", "zeta": "prefix", "lr": "prefix"}
    for name in comparisons:
        steps.append(holm(os.path.join(ABL_OUT, name + "_pairwise.csv"), families[name]))
    steps.append(bon(os.path.join(ABL_CKPT, "bce"), BCE_HEADS, os.path.join(ABL_OUT, "bon_bce")))
    return steps


def long_eval_steps():
    arms = []
    for head in LONG_HEADS:
        tag = "{0}_e{1}".format(head, LONG_EPOCHS)
        arms += arm(head + "_e30", head, final(head))
        arms += arm(tag, head, ablation("long", tag))
    return [compare("long", arms), holm(os.path.join(ABL_OUT, "long_pairwise.csv"), "prefix")]


def lora_steps():
    train = [ENCODER_PY, "train_lora.py", "--train_file", "data/train.jsonl",
             "--epochs", "1", "--batch_size", "4", "--max_length", "512",
             "--lr", "1e-4", "--zeta", "4.0", "--seed", "42",
             "--save_dir", LORA_CKPT, "--results_dir", os.path.join(ABL_OUT, "lora")]
    adapter = os.path.join(LORA_CKPT, "adapter")
    return [
        (train, [os.path.join(LORA_CKPT, "linear_head.pt"),
                 os.path.join(ABL_OUT, "lora", "lora_training.json")]),
        ([ENCODER_PY, "precompute_embeddings.py", "--train_file", "data/val.jsonl",
          "--cache_dir", VAL_LORA_CACHE, "--lora_path", adapter,
          "--batch_size", "16", "--max_length", "512", "--dtype", "float16"], []),
        ([ENCODER_PY, "precompute_eval_embeddings.py", "--eval_file", "data/single_eval.jsonl",
          "--cache_dir", BON_LORA_CACHE, "--lora_path", adapter,
          "--batch_size", "16", "--max_length", "512", "--dtype", "float16"], []),
    ]


def lora_eval_steps():
    arms = (["--arm", "lora_linear:{0}:linear:{1}".format(
                VAL_LORA_CACHE, os.path.join(LORA_CKPT, "linear_head.pt"))]
            + arm("frozen_linear", "linear", final("linear"))
            + arm("frozen_attention", "attention", final("attention")))
    bon_lora = bon(LORA_CKPT, ["linear"], os.path.join(ABL_OUT, "bon_lora"))
    bon_lora[bon_lora.index("--cache_dir") + 1] = BON_LORA_CACHE
    return [compare("lora_vs_frozen", arms),
            holm(os.path.join(ABL_OUT, "lora_vs_frozen_pairwise.csv"), "all"),
            bon_lora]


ROBUST_OUT = "results_final_lr3e-5"


def copy_training_logs(lr, out):
    """summarize_results reads the efficiency files from the results directory."""
    os.makedirs(out, exist_ok=True)
    for head in ALL_HEADS:
        source = os.path.join(ABL_OUT, "train", "{0}_lr{1}".format(head, lr))
        for suffix in ("_efficiency.json", "_loss_history.json", "_loss_curve.png"):
            shutil.copy(os.path.join(source, head + suffix), out)


GROUPS = ["main", "main_eval", "bce", "zeta", "seeds", "lr", "ablation_eval",
          "long", "long_eval", "main_eval_lr3e-5", "lora", "lora_eval"]


def main():
    global OTHER_LRS, ENCODER_PY
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", choices=GROUPS, default=GROUPS)
    parser.add_argument("--lrs", nargs="+", default=OTHER_LRS,
                        help="Subset of the lr group, e.g. to split it across parallel workers.")
    parser.add_argument("--encoder_python", default=sys.executable,
                        help="Interpreter with transformers and peft, for the lora group.")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    OTHER_LRS = args.lrs
    ENCODER_PY = args.encoder_python

    steps = training_steps([g for g in ("main",) if g in args.only])
    if "main_eval" in args.only:
        steps += [(cmd, []) for cmd in main_eval_steps()]
    steps += training_steps([g for g in ("bce", "zeta", "seeds", "lr") if g in args.only])
    if "ablation_eval" in args.only:
        steps += [(cmd, []) for cmd in ablation_eval_steps()]
    steps += training_steps([g for g in ("long",) if g in args.only])
    if "long_eval" in args.only:
        steps += [(cmd, []) for cmd in long_eval_steps()]
    if "main_eval_lr3e-5" in args.only:
        if not args.dry_run:
            copy_training_logs("3e-5", ROBUST_OUT)
        steps += [(cmd, []) for cmd in main_eval_steps(
            ROBUST_OUT, os.path.join(ABL_CKPT, "lr"), "{head}_lr3e-5_head.pt")]
    if "lora" in args.only:
        steps += lora_steps()
    if "lora_eval" in args.only:
        steps += [(cmd, []) for cmd in lora_eval_steps()]

    for index, (cmd, done) in enumerate(steps, 1):
        label = "[{0}/{1}]".format(index, len(steps))
        if done and all(os.path.exists(p) for p in done):
            print(label + " skip, already done: " + done[0], flush=True)
            continue
        print(label + " $ " + " ".join(cmd), flush=True)
        if not args.dry_run:
            # Unbuffered children, so their progress reaches a redirected log.
            subprocess.run(cmd, check=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})


if __name__ == "__main__":
    main()
