"""Runs the follow-up experiments that close the remaining open questions.

Groups, in the order they run (all reuse the frozen-encoder caches; Qwen never
runs):

* bce      -- PQM ranking loss vs. pointwise step BCE, the comparison the
              anchor paper's central claim rests on.
* zeta     -- sensitivity to the PQM margin, fixed at 4.0 everywhere else.
* seeds30  -- seed robustness under the FINAL protocol (lr=1e-4, 30 epochs,
              patience 5). The earlier seed check used a 10-epoch cap.
* lr3e-5   -- extends the learning-rate grid below 1e-4, where the trend
              pointed but the grid stopped.
* eval     -- paired bootstrap comparisons for all of the above, Holm
              correction, Best-of-N with PRM-weighted voting, and the
              in-distribution single-solution control on the final caches.

Training steps are skipped when their checkpoint and efficiency file already
exist, so an interrupted run resumes where it stopped. Evaluation is cheap and
always reruns.

    python run_extra_experiments.py                    # everything
    python run_extra_experiments.py --only bce eval    # a subset
    python run_extra_experiments.py --dry_run          # print the commands
"""

import argparse
import os
import subprocess
import sys

TRAIN_CACHE = "cache/train_clean"
VAL_CACHE = "cache/val_clean"
FINAL_CKPT = "checkpoints/long30"
OUT = "results_ablations"
ALL_HEADS = ["linear", "mlp", "cnn", "gru", "attention", "attention_pe"]

SEED_HEADS = ["attention", "cnn", "mlp"]
SEEDS = [43, 44]
BCE_HEADS = ["linear", "mlp", "attention"]
ZETA_HEADS = ["linear", "attention"]
ZETAS = ["2", "8"]
LOW_LR = "3e-5"

COMMON = ["--cache_dir", TRAIN_CACHE, "--val_cache_dir", VAL_CACHE,
          "--epochs", "30", "--early_stopping_patience", "5",
          "--calibration_fraction", "0.5", "--split_seed", "42"]


def train_step(head, save_path, results_dir, lr="1e-4", zeta="4.0", seed="42", loss="pqm"):
    cmd = [sys.executable, "train_from_cache.py", *COMMON, "--head", head,
           "--lr", lr, "--zeta", zeta, "--seed", str(seed), "--loss", loss,
           "--save_path", save_path, "--results_dir", results_dir]
    done = [save_path, os.path.join(results_dir, head + "_efficiency.json")]
    return cmd, done


def training_steps(groups):
    steps = []
    if "bce" in groups:
        for head in BCE_HEADS:
            steps.append(train_step(head, "checkpoints/bce/{0}_head.pt".format(head),
                                    "results_bce/" + head, loss="bce"))
    if "zeta" in groups:
        for head in ZETA_HEADS:
            for zeta in ZETAS:
                tag = "{0}_z{1}".format(head, zeta)
                steps.append(train_step(head, "checkpoints/zeta/{0}_head.pt".format(tag),
                                        "results_zeta/" + tag, zeta=zeta))
    if "seeds30" in groups:
        for head in SEED_HEADS:
            for seed in SEEDS:
                tag = "{0}_s{1}".format(head, seed)
                steps.append(train_step(head, "checkpoints/seeds30/{0}_head.pt".format(tag),
                                        "results_seeds30/" + tag, seed=seed))
    if "lr3e-5" in groups:
        for head in ALL_HEADS:
            tag = "{0}_lr{1}".format(head, LOW_LR)
            steps.append(train_step(head, "checkpoints/lrsweep/{0}_head.pt".format(tag),
                                    "results_lrsweep/" + tag, lr=LOW_LR))
    return steps


def arm(name, head, checkpoint):
    return ["--arm", "{0}:{1}:{2}:{3}".format(name, VAL_CACHE, head, checkpoint)]


def final(head):
    return "{0}/{1}_head.pt".format(FINAL_CKPT, head)


def compare(out_name, arms):
    return [sys.executable, "-m", "analysis.compare_lora_frozen", *arms,
            "--results_dir", OUT, "--out_name", out_name, "--bootstrap_samples", "2000"]


def eval_steps():
    steps = []
    arms = []
    for head in SEED_HEADS:
        arms += arm(head + "_s42", head, final(head))
        for seed in SEEDS:
            arms += arm("{0}_s{1}".format(head, seed), head,
                        "checkpoints/seeds30/{0}_s{1}_head.pt".format(head, seed))
    steps.append(compare("seeds30", arms))

    arms = []
    for head in BCE_HEADS:
        arms += arm("pqm_" + head, head, final(head))
        arms += arm("bce_" + head, head, "checkpoints/bce/{0}_head.pt".format(head))
    steps.append(compare("loss_pqm_vs_bce", arms))

    arms = []
    for head in ZETA_HEADS:
        arms += arm(head + "_z4", head, final(head))
        for zeta in ZETAS:
            arms += arm("{0}_z{1}".format(head, zeta), head,
                        "checkpoints/zeta/{0}_z{1}_head.pt".format(head, zeta))
    steps.append(compare("zeta", arms))

    arms = []
    for head in ALL_HEADS:
        arms += arm(head + "_lr1e-4", head, final(head))
        arms += arm("{0}_lr{1}".format(head, LOW_LR), head,
                    "checkpoints/lrsweep/{0}_lr{1}_head.pt".format(head, LOW_LR))
    steps.append(compare("lr_3e-5_vs_1e-4", arms))

    pairwise = [os.path.join(OUT, name + "_pairwise.csv")
                for name in ("seeds30", "loss_pqm_vs_bce", "zeta", "lr_3e-5_vs_1e-4")]
    steps.append([sys.executable, "-m", "analysis.holm_correction", *pairwise,
                  "--bootstrap_samples", "2000"])

    bon = [sys.executable, "-m", "eval.eval_bon", "--cache_dir", "cache/single_eval",
           "--eval_file", "data/single_eval.jsonl",
           "--source_file", "data/gsm8k_qwen0.5b_bon16.jsonl",
           "--aggs", "min", "mean", "last", "--ks", "1", "2", "4", "8", "16",
           "--subsets_per_question", "20", "--bootstrap_samples", "2000", "--seed", "42"]
    # Final heads: same subsets and draws as before, so the argmax rows reproduce
    # the existing numbers and the weighted-vote rows are added alongside.
    steps.append(bon + ["--checkpoint_dir", FINAL_CKPT, "--heads", *ALL_HEADS,
                        "--results_dir", "results_conv"])
    steps.append(bon + ["--checkpoint_dir", "checkpoints/bce", "--heads", *BCE_HEADS,
                        "--results_dir", os.path.join(OUT, "bon_bce")])

    steps.append([sys.executable, "-m", "eval.eval_single_from_step_cache",
                  "--cache_dir", VAL_CACHE, "--checkpoint_dir", FINAL_CKPT,
                  "--heads", *ALL_HEADS, "--aggs", "min", "mean", "last",
                  "--results_dir", "results_conv", "--bootstrap_samples", "2000",
                  "--calibration_fraction", "0.5", "--split_seed", "42", "--seed", "42"])
    return [(cmd, []) for cmd in steps]


def main():
    groups = ["bce", "zeta", "seeds30", "lr3e-5", "eval"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", choices=groups, default=groups)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    steps = training_steps(args.only)
    if "eval" in args.only:
        steps += eval_steps()

    for index, (cmd, done) in enumerate(steps, 1):
        label = "[{0}/{1}]".format(index, len(steps))
        if done and all(os.path.exists(p) for p in done):
            print(label + " skip, already done: " + done[0])
            continue
        print(label + " $ " + " ".join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
