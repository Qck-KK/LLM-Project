# Lightweight PRM Complete Experiment Manual

This document is the single complete execution guide for the project. The workflow starts with environment setup, data checks, and encoder caching, then proceeds through data-bias auditing, Reward Head training, standard metric evaluation, confidence intervals, stratified and first-error-boundary analyses, perturbation experiments, causal-prefix checks, offline pruning, Best-of-N reranking, and the LoRA comparison.

## Protocol Revised: Conclusions from the Old Protocol Are Invalid

The initial version of this manual used `lr=1e-3`, at most 10 epochs, and `patience=2`, and compared five architectures under that setup. A later 6×3 learning-rate grid showed that **`1e-3` was the worst of the three candidate values for all six heads** (costing 0.056–0.445 in development loss), and that this effect was 4–8× larger than the architecture differences being compared. After tuning the learning rate, the architecture ranking reversed.

**Therefore, any architecture conclusion obtained under `lr=1e-3` is invalid.** The current protocol is:

| Item | Old | Current |
|---|---|---|
| Learning rate | `1e-3` | `1e-4` (best for 5 of 6 heads in the pre-fix grid; after the data fix, `3e-5` gives lower development loss for 5 heads, but the main results still use the pre-specified `1e-4`; all evaluations were also rerun at `3e-5`, see 14G) |
| Epoch cap | 10 | 30 |
| Early-stopping patience | 2 | 5 |
| Main metrics | Step-level metrics | Step-level metrics + **Best-of-N** |

The three items originally excluded for compute reasons have now all been completed and are no longer out of scope:

- **Best-of-N**: the encoder cache is the only expensive part, and it already exists; scoring takes only a few seconds. BoN is also the main metric in the anchor PQM paper, so excluding it was the most serious methodological omission in the initial version.
- **Multiple random seeds**: `attention` / `cnn` / `mlp` are each run with 3 seeds.
- **LoRA**: one full pass over all 440k examples is used as a comparison against the frozen-encoder assumption.

## Data Parsing Fixed: All Pre-Fix Results Are Invalid

`dataset.py` originally split steps by newline and treated every line not ending in `+` / `-` as an incorrect step. A Math-Shepherd step can span multiple lines (most commonly a final step such as `Step k: …\n\n# Answer\n\n42`), so one true step could be split into as many as three steps, with the extra ones labeled as errors. As a result, 13.5% of error labels were spurious, 17% of fully correct trajectories were labeled as containing errors, and false errors were concentrated near the final step.

The parser now splits by the `ки` marker and reads the `+` / `-` label from the aligned location in `label`. Records that do not match this format are rejected and counted (114 records, 0.03%).

`cache/train_clean`, `cache/val_clean`, `cache/val_lora`, and every checkpoint/result directory derived from them (`results_conv/`, `results_long30/`, `results_lrsweep/`, `results_seeds/`, `results_patience4/`, `results_loraeval/`, etc.) were built on incorrect labels and are retained only as historical records.

### Path Conventions

This manual uses the fixed paths:

| Purpose | Path |
|---|---|
| Training / validation cache | `cache/train_fixed`, `cache/val_fixed` |
| Final checkpoint | `checkpoints/final/{head}_head.pt` |
| Training logs and final evaluation results | `results_final/` |
| Ablations (loss, zeta, lr, seeds) | `checkpoints/ablations/`, `results_ablations/` |

`run_experiments.py` executes all cache-based experiments in the order described in this manual (see 14G). Sections 14A, 14E, and 14F retain commands that were actually run before the parser fix as historical records.

Training still uses fixed seed `42` (with 43 and 44 used for the multi-seed experiment). Except for embedding precomputation and LoRA training, all remaining experiments reuse cached representations and do not rerun Qwen.

## 0. Complete Experiment Order

The experiments should be run in the following order:

```text
Environment and data preparation
        ↓
Optional hardware benchmark
        ↓
Freeze Qwen encoder and generate train/val cache
        ↓
Data and positional bias audit
        ↓
Train Linear / MLP / CNN / BiGRU / Attention / Attention-PE
        ↓
Inspect train/development loss and best epoch
        ↓
Held-out step-level standard evaluation
        ↓
Optional single-solution evaluation
        ↓
Majority / Position-only / Coin-flip baselines
        ↓
Stratified analysis by trajectory length, first-error position, and error count
        ↓
First-error boundary vs. position-matched correct-boundary control
        ↓
Optional deterministic perturbation experiments
        ↓
Compare full-trajectory scores with causal-prefix scores
        ↓
Calibrate pruning thresholds and run held-out offline replay
        ↓
False-pruning–detection–safe-step-saving trade-off
        ↓
Learning-rate grid (determines whether the architecture conclusions above are valid)
        ↓
Trajectory-level paired bootstrap confidence intervals
        ↓
Best-of-N reranking vs. majority voting
        ↓
LoRA comparison (tests whether the frozen-encoder assumption is valid)
        ↓
Early-stopping and seed sensitivity
        ↓
Aggregate final tables, plots, and report conclusions
```

The learning-rate grid appears after architecture analysis because that is the historical order. **If rerunning from scratch, it should be moved before training**: determine a suitable learning rate for each head first, then compare architectures.

The complete experiment is designed to answer the following questions:

1. Can class balance or step position alone already predict correctness?
2. Does the frozen encoder's single-step representation contain correctness signal?
3. Do contextual models such as CNN, BiGRU, and Attention outperform Linear/MLP?
4. Where do architecture differences occur: short vs. long trajectories, local errors, or multiple consecutive errors?
5. Does the reward actually drop around the first incorrect step?
6. Does that boundary effect remain after controlling for step position?
7. When order or local information is perturbed, do different models exhibit sensitivities consistent with their architectures?
8. After removing future-step information, how much performance remains for each reward head?
9. Under a constrained false-pruning rate on correct trajectories, can reward signals translate into safe theoretical step savings?

## 1. Environment Setup

Enter the project directory and install dependencies:

```bash
pip install -r requirements.txt
```

Run the basic checks:

```bash
python -m unittest discover -v
python -m py_compile *.py analysis/*.py eval/*.py tests/*.py
```

Device selection order:

```text
CUDA → MPS → CPU
```

You can force a device with `--device cuda`, `--device mps`, or `--device cpu`.

## 2. Data Preparation

### 2.1 Training and Validation Sets

The training and validation sets use Math-Shepherd-style JSONL, with one reasoning trajectory per line:

```json
{"question": "problem", "steps": ["step 1", "step 2"], "labels": [1, 0]}
```

where:

- `question`: the math problem;
- `steps`: reasoning steps in order;
- `labels[i] = 1`: step `i` is correct;
- `labels[i] = 0`: step `i` is incorrect.

Recommended paths:

```text
data/train.jsonl
data/val.jsonl
```

### 2.2 Optional Single-Trajectory Evaluation Set

To evaluate whether an entire solution is correct, prepare:

```json
{"question": "problem", "steps": ["step 1", "step 2"], "final_correct": 1}
```

Recommended path:

```text
data/single_eval.jsonl
```

### 2.3 Data Isolation Principle

The validation cache is deterministically split by trajectory into two halves:

- calibration/development half: used for early stopping and classification-threshold selection;
- held-out test half: used only for final metrics and behavioral analyses.

All scripts use the same `calibration_fraction=0.5` and `split_seed=42`, ensuring an identical split. Steps from the same trajectory never cross between the two subsets.

## 3. Optional: Hardware Throughput Benchmark

If the embedding cache does not yet exist, first estimate precomputation time:

```bash
python benchmark.py \
  --model_name Qwen/Qwen2.5-0.5B \
  --head mlp \
  --batch_size 8 \
  --seq_len 512 \
  --n_batches 20 \
  --dataset_size 445000
```

Replace `dataset_size` with the actual number of training examples. The benchmark records:

- batches/second;
- examples/second;
- tokens/second;
- CUDA peak memory or current MPS memory;
- estimated time per epoch.

If train/val caches already exist, this experiment can be skipped.

## 4. Freeze the Encoder and Generate Caches

### 4.1 Training Cache

```bash
python precompute_embeddings.py \
  --train_file data/train.jsonl \
  --cache_dir cache/train_fixed \
  --batch_size 16 \
  --max_length 512 \
  --dtype float16
```

### 4.2 Validation Cache

```bash
python precompute_embeddings.py \
  --train_file data/val.jsonl \
  --cache_dir cache/val_fixed \
  --batch_size 16 \
  --max_length 512 \
  --dtype float16
```

### 4.3 Optional Single-Eval Cache

```bash
python precompute_eval_embeddings.py \
  --eval_file data/single_eval.jsonl \
  --cache_dir cache/single_eval \
  --batch_size 16 \
  --max_length 512 \
  --dtype float16
```

Expected directory structure:

```text
cache/train_fixed/hidden_size.txt
cache/train_fixed/shard_*.pt
cache/val_fixed/hidden_size.txt
cache/val_fixed/shard_*.pt
cache/single_eval/hidden_size.txt        # optional
cache/single_eval/shard_*.pt             # optional
```

All three splits must use the same:

- `model_name`;
- step token;
- `max_length`;
- tokenizer configuration.

If caches were copied from another machine, do not rerun the encoder; proceed directly to Experiment 1.

## 5. Experiment 1: Data Quality and Positional Bias Audit

Before comparing networks, first determine whether the data contain class-imbalance or step-position bias:

```bash
python -m analysis.analyze_data_bias \
  --cache_dir cache/val_fixed \
  --results_dir results_final \
  --position_bins 5 \
  --calibration_fraction 0.5 \
  --split_seed 42
```

This experiment measures:

- counts and proportions of correct/incorrect steps;
- trajectory-length distribution;
- fraction of trajectories containing both correct and incorrect steps;
- number of `correct→incorrect` and `incorrect→correct` transitions;
- relative position of the first incorrect step;
- error rate in different relative-position bins.

It also computes two deterministic baselines:

- `majority`: always predict the majority class in the calibration half;
- `position_only`: predict using only the step's relative-position bin, without reading the embedding.

Outputs:

```text
results_final/data_bias.json
results_final/data_bias.png
results_final/position_label_rates.csv
results_final/deterministic_baselines.json
```

This experiment first asks whether later models are merely exploiting a dataset pattern such as “later steps are more likely to be wrong.”

## 6. Experiment 2: Train Six Lightweight Reward Heads

Models:

- Linear;
- MLP;
- CNN;
- BiGRU;
- Attention;
- Attention + sinusoidal positional encoding (`attention_pe`, added after perturbation experiments showed that attention without positional encoding is permutation equivariant).

All six models must use exactly the same:

- train cache;
- development split;
- maximum 30 epochs;
- learning rate `1e-4`;
- PQM margin `zeta=4.0`;
- fixed seed `42`;
- early-stopping patience of 5.

Run:

```bash
for head in linear mlp cnn gru attention attention_pe; do
  python train_from_cache.py \
    --cache_dir cache/train_fixed \
    --val_cache_dir cache/val_fixed \
    --head "$head" \
    --epochs 30 \
    --early_stopping_patience 5 \
    --seed 42 \
    --calibration_fraction 0.5 \
    --split_seed 42 \
    --lr 1e-4 \
    --zeta 4.0 \
    --save_path "checkpoints/final/${head}_head.pt" \
    --results_dir results_final
done
```

Training protocol:

1. Each model trains for at most 30 epochs.
2. PQM loss is evaluated on the calibration/development half after each epoch.
3. Training stops early after five consecutive epochs without improvement.
4. The checkpoint with the lowest development loss is always retained.
5. The held-out test half is never used for checkpoint selection.
6. About five training-loss points are recorded per epoch so the learning curve is not overly sparse.

Each model outputs:

```text
checkpoints/final/{head}_head.pt
results_final/{head}_efficiency.json
results_final/{head}_loss_history.json
results_final/{head}_loss_curve.png
```

where:

- `epochs`: number of epochs actually run;
- `max_epochs`: maximum value, 30;
- `best_epoch`: epoch corresponding to the retained checkpoint;
- `stopped_early`: whether early stopping was triggered;
- `final_train_loss`: training loss at the best epoch;
- `final_eval_loss`: development loss at the best epoch.

Note: the CNN padding mask is now reapplied after every convolutional layer. CNN checkpoints trained with the old implementation should be retrained.

## 7. Experiment 3: Training Dynamics and Efficiency Comparison

After training, inspect:

```text
results_final/{head}_loss_curve.png
results_final/{head}_loss_history.json
results_final/{head}_efficiency.json
```

Compare:

- whether train loss keeps decreasing;
- whether development loss stabilizes or begins to rise;
- the best epoch for each head;
- number of trainable parameters;
- total training time;
- CUDA peak memory.

Do not compare only the final loss in the report. Prefer a parameter–performance or training-time–performance Pareto plot, or include efficiency metrics alongside performance in the final table.

## 8. Experiment 4: Held-Out Step-Level Standard Evaluation

```bash
for head in linear mlp cnn gru attention attention_pe; do
  python -m eval.eval_step_metrics \
    --cache_dir cache/val_fixed \
    --head "$head" \
    --checkpoint "checkpoints/final/${head}_head.pt" \
    --calibration_fraction 0.5 \
    --split_seed 42 \
    --results_dir results_final
done
```

Evaluation procedure:

1. Search for the classification threshold on the calibration half.
2. Fix that threshold and evaluate on the held-out test half.
3. Never re-select the threshold on the test half.

Reported metrics:

- Step Accuracy;
- Balanced Accuracy;
- ROC-AUC;
- Average Precision;
- within-trajectory Q-value Ranking Accuracy.

Recommended primary metrics:

- ROC-AUC;
- Average Precision;
- Q-value Ranking Accuracy.

These metrics do not depend on a threshold tuned on the test set. Ordinary Accuracy is only a supporting metric.

Output:

```text
results_final/{head}_step_metrics.json
```

## 9. Experiment 5: Optional Single-Solution Evaluation

If `cache/single_eval` exists:

```bash
for head in linear mlp cnn gru attention attention_pe; do
  python -m eval.eval_single_from_cache \
    --cache_dir cache/single_eval \
    --head "$head" \
    --checkpoint "checkpoints/final/${head}_head.pt" \
    --agg min \
    --calibration_fraction 0.5 \
    --split_seed 42 \
    --results_dir results_final
done
```

This experiment first aggregates per-step Q-values into a trajectory-level score. Supported aggregation methods:

- `min`: minimum step score;
- `mean`: mean step score;
- `last`: final-step score.

`min` is the default. To compare aggregation strategies, run all three separately and save them to different result directories to avoid overwriting.

Reported metrics:

- held-out Accuracy;
- Balanced Accuracy;
- ROC-AUC;
- Average Precision;
- Pairwise Separation.

Output:

```text
results_final/{head}_single_metrics.json
```

## 10. Experiment 6: Random Coin-Flip Baseline

The random baseline does not train a model and has negligible compute cost:

```bash
python -m eval.eval_coin_flip_baseline \
  --val_cache_dir cache/val_fixed \
  --single_cache_dir cache/single_eval \
  --results_dir results_final \
  --trials 100 \
  --seed 42 \
  --calibration_fraction 0.5 \
  --split_seed 42
```

If the single-eval cache does not exist, remove:

```text
--single_cache_dir cache/single_eval
```

This baseline runs only on the same held-out test half used by the other models. It does not replace the majority and position-only baselines from Experiment 1.

Outputs:

```text
results_final/coin_flip_efficiency.json
results_final/coin_flip_step_metrics.json
results_final/coin_flip_single_metrics.json       # when a single cache exists
```

## 11. Experiment 7: Architecture-Stratified and First-Error-Boundary Analysis

Run the core behavioral analysis:

```bash
python -m analysis.analyze_head_behavior \
  --cache_dir cache/val_fixed \
  --checkpoint_dir checkpoints/final \
  --results_dir results_final \
  --heads linear mlp cnn gru attention attention_pe \
  --calibration_fraction 0.5 \
  --split_seed 42
```

The script performs one lightweight forward pass for each head and saves step-wise Q-values. It does not rerun the encoder or retrain any model.

### 11.1 Pointwise vs. Contextual Heads

Model groups:

- Pointwise: Linear, MLP;
- Contextual: CNN, BiGRU, Attention.

First compare pointwise models against the position-only baseline. If Linear/MLP clearly outperform the positional baseline, the frozen encoder representation itself contains step-correctness information.

Then compare contextual heads against pointwise heads. If contextual heads improve further, interactions across steps may provide additional useful information.

### 11.2 Stratified Experiments

`behavior_by_group.csv` and the corresponding figure break down results by:

- trajectory length: short, medium, long;
- first-error position: early, middle, late;
- number of errors: one_error, multiple_errors.

These results test whether:

- CNN benefits mainly near local errors;
- BiGRU is more stable on long or multi-error trajectories;
- Attention gains increase with trajectory length;
- model advantages appear only for late errors.

### 11.3 First-Error Boundary

Align all trajectories containing errors by the first incorrect step:

```text
offset=-2    offset=-1    offset=0    offset=+1    offset=+2
two before  one before    first error one after    two after
```

Because different heads use different Q-value scales, standardize scores within each head using the calibration half.

Report three quantities:

- `first_error_boundary_drop`: standardized Q at the step before the error minus Q at the first error;
- `matched_correct_boundary_drop`: mean decrease for `correct→correct` transitions in the same relative-position bin;
- `position_controlled_boundary_effect`: difference between the two quantities above.

A positive value for the last metric means the score drop at the first error is larger than ordinary position-related changes can explain. This remains correlational evidence and should not be described as proof that the model causally “understands” reasoning errors.

Outputs:

```text
results_final/{head}_step_predictions.pt
results_final/{head}_behavior_metrics.json
results_final/behavior_summary.csv
results_final/behavior_summary.md
results_final/behavior_by_group.csv
results_final/behavior_by_group.png
results_final/first_error_boundary_curves.csv
results_final/first_error_boundary.png
```

## 12. Experiment 8: Optional Deterministic Perturbation Experiments

This experiment does not train models; it only adds a few reward-head forward passes:

```bash
python -m analysis.analyze_head_behavior \
  --cache_dir cache/val_fixed \
  --checkpoint_dir checkpoints/final \
  --results_dir results_final \
  --heads linear mlp cnn gru attention attention_pe \
  --calibration_fraction 0.5 \
  --split_seed 42 \
  --run_perturbations
```

Perturbations:

- `reverse`: reverse the valid step order;
- `swap_adjacent`: swap adjacent steps;
- `mask_previous`: zero out the embedding immediately before the first error;
- `mask_first_error`: zero out the first-error embedding.

Primary observations:

- change in ROC-AUC;
- change in Q-value Ranking Accuracy.

Expected use:

- Linear/MLP should be nearly insensitive to pure reordering;
- CNN should be more sensitive to disruption of local adjacency;
- BiGRU should be more sensitive to sequence reversal or adjacent swaps;
- Attention changes reflect the contribution of global interactions.

Limitation: cached encoder hidden states already encode the original context and position, so these are sensitivity analyses rather than strict causal interventions.

Outputs:

```text
results_final/perturbation_results.csv
results_final/perturbation_sensitivity.png
```

## 13. Experiment 9: Causal Prefixes and Offline Heuristic Pruning

CNN, BiGRU, and Attention scores computed on complete trajectories may use future step embeddings and therefore cannot directly simulate online early stopping. For step `t`, this experiment provides only the first `t` cached embeddings and takes the reward at the final step of the prefix. Frozen Qwen is not rerun.

Trajectories are divided into:

- `clean`: all steps correct;
- `monotone_error`: after the first error, the trajectory never recovers;
- `recovery`: contains a `correct→incorrect→correct` pattern.

The main pruning metrics use only the first two groups. Recovery trajectories are reported separately to avoid treating potentially recoverable paths as paths that should necessarily be pruned.

```bash
python -m analysis.analyze_offline_pruning \
  --cache_dir cache/val_fixed \
  --checkpoint_dir checkpoints/final \
  --results_dir results_final \
  --heads linear mlp cnn gru attention attention_pe \
  --budgets 0.01 0.05 0.10 \
  --primary_budget 0.05 \
  --primary_policy single_low \
  --bootstrap_samples 1000 \
  --calibration_fraction 0.5 \
  --split_seed 42 \
  --seed 42
```

The script compares two policies:

- `single_low`: stop when the current reward falls below the threshold;
- `two_consecutive`: stop after two consecutive rewards below the threshold.

Each threshold is calibrated only on the calibration half, with false-pruning budgets of 1%, 5%, and 10% on fully correct trajectories. The threshold is then fixed and evaluated on the held-out test half. The default primary comparison point is `single_low` at a 5% false-pruning budget.

Primary metrics:

- `clean_false_prune_rate`: false-pruning rate on fully correct trajectories;
- `pre_error_false_prune_rate`: fraction stopped incorrectly before the first error;
- `error_coverage`: fraction of erroneous trajectories successfully stopped after the first error;
- `detection_at_0/1/2`: fraction detected at the first-error step, within one step, or within two steps;
- `median_detection_delay`: median delay from the first error to stopping;
- `safe_step_saving_rate`: only remaining steps saved after a correctly triggered post-error stop count as savings;
- `oracle_efficiency_ratio`: fraction of the ideal first-error pruning opportunity actually achieved.

The 1,000 trajectory-level bootstrap samples estimate evaluation uncertainty only; they do not retrain models.

Outputs:

```text
results_final/{head}_causal_predictions.pt
results_final/{head}_pruning_metrics.json
results_final/causal_diagnostics.csv
results_final/pruning_thresholds.json
results_final/pruning_results.csv
results_final/pruning_by_group.csv
results_final/pruning_summary.md
results_final/full_vs_causal_scores.png
results_final/pruning_tradeoff.png
results_final/pruning_detection_delay.png
```

Both `theoretical_step_saving_rate` and `safe_step_saving_rate` are step-equivalent metrics based on cached trajectory lengths. They must not be described as real wall-clock or FLOPs speedups.

## 14. Experiment 10: Final Result Aggregation

After all experiments are complete, run:

```bash
python -m eval.summarize_results \
  --results_dir results_final \
  --out_csv results_final/summary.csv \
  --out_md results_final/summary.md
```

The final table includes:

- model parameter count;
- actual epoch count and best epoch;
- training time and peak memory;
- train/development loss;
- step-level Accuracy, Balanced Accuracy, ROC-AUC, and AP;
- Q-value Ranking Accuracy;
- optional single-solution metrics;
- first-error boundary drop;
- position-matched correct-boundary drop;
- position-controlled boundary effect;
- difference between full-trajectory and causal-prefix scores;
- detection rate, delay, and safe-step-saving rate at the 5% false-pruning budget;
- majority, position-only, and coin-flip baselines.

Outputs:

```text
results_final/summary.csv
results_final/summary.md
```

## 14A. Experiment 11: Learning-Rate Grid — The Most Important Step

The initial protocol fixed `lr=1e-3` and then compared architectures. This experiment tests whether that fixed value was defensible.

> Historical record: the grid below was run on the pre-fix `_clean` cache. After the parser fix, `run_experiments.py --only lr` rechecks the selected `1e-4` on `_fixed` caches using `{3e-5, 3e-4}`. Because `1e-3` was clearly worst for all six heads before the fix, it is not rerun.

```bash
for lr in 1e-3 3e-4 1e-4; do
  for head in linear mlp cnn gru attention attention_pe; do
    python train_from_cache.py \
      --cache_dir cache/train_clean \
      --val_cache_dir cache/val_clean \
      --head "$head" \
      --epochs 30 \
      --early_stopping_patience 5 \
      --seed 42 \
      --calibration_fraction 0.5 \
      --split_seed 42 \
      --lr "$lr" \
      --zeta 4.0 \
      --save_path "checkpoints/lrsweep/${head}_lr${lr}_head.pt" \
      --results_dir "results_lrsweep/${head}_lr${lr}"
  done
done
```

Observed result: **`1e-3` was worst for all six heads**, with improvements ranging from 0.056 (linear) to 0.445 (attention), while architecture differences were no larger than 0.05. The ranking therefore reversed: `cnn` led at `1e-3`, whereas `attention` led after tuning.

Two boundaries must be stated in the limitations:

- the grid did not probe the lower bound (`1e-4` was best for 5/6 heads and the trend was still monotonic);
- heads with different capacities require different numbers of epochs to reach their optimum (`linear` around 11, `gru` around 25).

## 14B. Experiment 12: Confidence Intervals for Primary Metrics

Architecture rankings depend on differences on the order of 0.005, so without confidence intervals it is impossible to know whether those differences are noise.

```bash
python -m analysis.bootstrap_step_metrics \
  --cache_dir cache/val_fixed \
  --checkpoint_dir checkpoints/final \
  --results_dir results_final \
  --heads linear mlp cnn gru attention attention_pe \
  --bootstrap_samples 2000 \
  --calibration_fraction 0.5 \
  --split_seed 42 \
  --seed 42

python -m analysis.bootstrap_causal_metrics \
  --results_dir results_final \
  --heads linear mlp cnn gru attention attention_pe \
  --bootstrap_samples 2000 \
  --seed 42
```

The resampling unit is the **trajectory**, not the step, because steps within the same solution are highly correlated. All heads are scored on the **same bootstrap resample** so head-to-head differences have proper paired intervals.

`bootstrap_causal_metrics` reuses causal predictions saved by `analyze_offline_pruning`, so it adds no forward-pass cost.

Pairwise comparison among six heads creates 15 tests per metric. Inspecting whether each interval crosses zero would inflate false positives. Use Holm correction to control family-wise error rate; this reads only the two pairwise CSV files above and does not require the model or cache:

```bash
python -m analysis.holm_correction \
  results_final/step_metrics_ci_pairwise.csv \
  results_final/causal_metrics_ci_pairwise.csv \
  --bootstrap_samples 2000
```

Outputs are written to same-named `*_pairwise_holm.csv` files; the originals are preserved. If a pair has exactly zero difference on every bootstrap resample, set `p=1`. When comparing more than one family of groups, use `--family prefix|suffix` so only meaningful comparisons are corrected together (see 14G).

Outputs: `step_metrics_ci.json` / `.csv` / `_pairwise.csv`, and `causal_metrics_ci.json`.

## 14C. Experiment 13: Best-of-N Reranking

This is the main metric in the anchor PQM paper and the PRM use case closest to practical deployment.

```bash
python -m eval.eval_bon \
  --cache_dir cache/single_eval \
  --eval_file data/single_eval.jsonl \
  --source_file data/gsm8k_qwen0.5b_bon16.jsonl \
  --checkpoint_dir checkpoints/final \
  --heads linear mlp cnn gru attention attention_pe \
  --aggs min mean last \
  --ks 1 2 4 8 16 \
  --subsets_per_question 20 \
  --bootstrap_samples 2000 \
  --results_dir results_final \
  --seed 42
```

Two reference lines are mandatory:

- `majority_vote` (self-consistency): count only final answers and do not use the reward model. **A PRM must outperform it to demonstrate deployment value**;
- `oracle`: count a question as correct if any of the 16 candidates is correct, giving the reranking upper bound.

For `k < 16`, sample multiple random subsets per question so results do not depend on candidate-generation order. Confidence intervals are bootstrapped **by question**, because the 16 candidates from the same question are not independent.

## 14D. Experiment 14: In-Distribution Single-Solution Control

If BoN / OOD single-solution performance is poor, distinguish between “the model cannot make trajectory-level judgments” and “the model does not transfer out of distribution.”

```bash
python -m eval.eval_single_from_step_cache \
  --cache_dir cache/val_fixed \
  --source_file data/val.jsonl \
  --checkpoint_dir checkpoints/final \
  --heads linear mlp cnn gru attention attention_pe \
  --aggs min mean last \
  --results_dir results_final \
  --bootstrap_samples 2000 \
  --calibration_fraction 0.5 \
  --split_seed 42 \
  --seed 42
```

This reuses the validation cache and uses “all steps correct” as the trajectory label, so the only difference from the OOD version is the data distribution. `--source_file` ensures this label comes from the full original step labels: the cache contains only steps not truncated by `max_length`, otherwise solutions whose only errors occur in the truncated portion would be mislabeled as correct (11 of 841 “correct” trajectories in the held-out validation half have this issue).

## 14E. Experiment 15: LoRA Comparison

Freezing the encoder is **a simplification introduced by this project**, not the PQM setup (which fully fine-tunes a 7B model on 8 GPUs). This experiment tests the cost of that assumption.

> Historical record: the commands below are the pre-fix version (`train_lora.py` also read labels through `dataset.py`, and `cache/val_lora` was generated by the old parser). The post-fix LoRA comparison is run by `run_experiments.py --only lora lora_eval` (see 14G). The post-fix conclusion is the opposite of the pre-fix result: LoRA is significantly better than the frozen version using the same linear head and is the only configuration that significantly outperforms majority voting on BoN.

```bash
python train_lora.py \
  --train_file data/train.jsonl \
  --epochs 1 \
  --batch_size 4 \
  --max_length 512 \
  --lr 1e-4 \
  --zeta 4.0 \
  --seed 42 \
  --save_dir checkpoints/lora_full \
  --results_dir results_lora_full

# LoRA changes the encoder, so the evaluation cache must be regenerated.
python precompute_embeddings.py \
  --train_file data/val.jsonl \
  --cache_dir cache/val_lora \
  --lora_path checkpoints/lora_full/adapter \
  --batch_size 16 \
  --max_length 512 \
  --dtype float16

python -m analysis.compare_lora_frozen \
  --arm "lora_linear:cache/val_lora:linear:checkpoints/lora_eval/linear_head.pt" \
  --arm "frozen_linear:cache/val_clean:linear:checkpoints/long30/linear_head.pt" \
  --arm "frozen_attention:cache/val_clean:attention:checkpoints/long30/attention_head.pt" \
  --results_dir results_loraeval \
  --bootstrap_samples 2000
```

On 8 GB VRAM, the only workable operating point is `batch_size=4, max_length=512`; `batch_size=8` runs out of memory and is 6.8× slower. One full epoch takes about 10 hours.

**Interpretation must explicitly state that the number of passes is asymmetric** (LoRA 1 epoch vs. frozen 30 epochs). This favors the frozen side. Therefore, “LoRA provides no benefit” would be a conservative conclusion, whereas “LoRA loses” cannot be separated from “one epoch was insufficient.”

## 14F. Experiment 16: Early-Stopping and Seed Sensitivity

```bash
# Does patience distort the ranking?
for head in linear mlp cnn gru attention; do
  python train_from_cache.py \
    --cache_dir cache/train_clean \
    --val_cache_dir cache/val_clean \
    --head "$head" \
    --epochs 10 \
    --early_stopping_patience 4 \
    --seed 42 \
    --calibration_fraction 0.5 \
    --split_seed 42 \
    --lr 1e-3 \
    --zeta 4.0 \
    --save_path "checkpoints/patience4/${head}_head.pt" \
    --results_dir "results_patience4"
done

# Is seed variance larger than architecture variance?
for seed in 43 44; do
  for head in attention cnn mlp; do
    python train_from_cache.py \
      --cache_dir cache/train_clean \
      --val_cache_dir cache/val_clean \
      --head "$head" \
      --epochs 10 \
      --early_stopping_patience 2 \
      --seed "$seed" \
      --calibration_fraction 0.5 \
      --split_seed 42 \
      --lr 1e-4 \
      --zeta 4.0 \
      --save_path "checkpoints/seeds/${head}_s${seed}_head.pt" \
      --results_dir "results_seeds/${head}_s${seed}"
  done
done
```

The criterion is **whether the intervals overlap**, not which point estimate is larger. Empirically, the AUC intervals of the three heads do not overlap pairwise, and the architecture gap is 3.5–5.5× the seed standard deviation.

Note that neither check was run under the current full protocol. The patience check used the invalidated `lr=1e-3` and therefore only shows that the old ranking was not caused by patience. The seed check used `lr=1e-4` but only 10 epochs; in 5 of 6 runs, the best checkpoint occurred at epoch 10, so training had not converged. These tests therefore support the ranking under a 10-epoch budget rather than the converged numbers in `results_conv/`. Both were also run on pre-fix data; the post-fix seed check is in the `seeds` group in 14G.

## 14G. One-Command Workflow: Post-Fix Main Experiments and Supplementary Ablations

`run_experiments.py` executes all cache-based experiments in this manual in order on `cache/train_fixed` / `cache/val_fixed`. Training steps are automatically skipped when both the checkpoint and efficiency file already exist, so an interrupted run can be resumed directly; evaluation steps rerun each time.

| Group | Content | Training Cost |
|---|---|---|
| `main` | Six heads under the current protocol (`lr=1e-4`, 30 epochs, patience 5, `zeta=4`, seed 42) | 6 runs |
| `main_eval` | All evaluations in Sections 5–14D: bias audit and baselines, held-out metrics, behavior / first-error / perturbation, causal-prefix pruning, bootstrap CIs and Holm correction, BoN (argmax and PRM-weighted voting), OOD and in-distribution single-solution, summary table | none |
| `bce` | Does PQM ranking loss outperform step-wise BCE? This is a central claim of the anchor paper | linear / mlp / attention |
| `zeta` | Is fixed `zeta=4.0` sensitive? | linear / attention × `zeta` ∈ {2, 8} |
| `seeds` | Is the ranking robust to seed under the current protocol? | attention / cnn / mlp × seeds 43, 44 |
| `lr` | Is `1e-4` still a suitable learning rate after the parser fix? | six heads × {`3e-5`, `3e-4`} |
| `ablation_eval` | Paired bootstrap + Holm correction for all groups above vs. final heads; BoN for BCE heads | none |
| `long` / `long_eval` | mlp and attention_pe both select epoch 30 on fixed data; retrain to 60 epochs and compare pairwise | 2 runs |
| `main_eval_lr3e-5` | After the fix, `3e-5` has lower development loss for 5/6 heads, so rerun the full evaluation chain on `3e-5` heads → `results_final_lr3e-5/` | none |
| `lora` / `lora_eval` | Comparison against the frozen-encoder assumption (see 14E), rerun on fixed data; requires a Python environment with `transformers` and `peft` (`--encoder_python`) | LoRA 1 epoch, about 10 hours |

```bash
python run_experiments.py --dry_run                  # inspect commands first
python run_experiments.py                            # run everything
python run_experiments.py --only main main_eval      # main experiment only
python run_experiments.py --only lr --lrs 3e-5       # split the lr group across parallel processes if desired
python run_experiments.py --only lora lora_eval --encoder_python <python with transformers/peft>
```

Outputs:

```text
checkpoints/final/{head}_head.pt                     # main experiment
results_final/                                       # training logs and all main evaluations
checkpoints/ablations/{bce,zeta,seeds,lr}/
results_ablations/train/                             # ablation training logs
results_ablations/{seeds,loss_pqm_vs_bce,zeta,lr}.json and *_pairwise(_holm).csv
results_ablations/bon_bce/bon_results.csv
results_ablations/{long,lora_vs_frozen}.json, results_ablations/bon_lora/
results_final_lr3e-5/                                # full evaluation of 3e-5 heads
```

Interpretation notes:

- Development-loss values from different `zeta` values or different loss functions are **not directly comparable** because the loss definitions differ. Compare held-out ROC-AUC / AP and BoN instead.
- Compare learning rates using both the best development loss and held-out AUC. If a learning rate achieves its best checkpoint only at epoch 30, the budget is insufficient, so superiority/inferiority should not be concluded from that run alone.
- Holm correction is applied only within comparisons that are meaningful for each ablation (`--family`): seed and loss compare different heads under the same setting (`suffix`), whereas zeta, learning rate, and 60-epoch comparisons compare different settings for the same head (`prefix`). Mixing cross-head and cross-setting pairs is not meaningful; moreover, with 2,000 resamples the minimum p-value is about 0.001, so families larger than about 50 make significance effectively impossible.
- Encoding uses `transformers`; the local `Training` environment does not have it installed, so caches were generated in the `node2` environment. For unaffected records, the two environments produce embedding cosine similarity ≥ 0.99996 (fp16 rounding differences).

## 15. Run the Full Workflow from the Notebook

Main entry point:

```text
complete_training.ipynb
```

The notebook follows the order in this manual:

1. set cache, checkpoint, and result paths;
2. check caches;
3. run the data and positional bias audit;
4. train six heads;
5. display loss curves;
6. run held-out step-level evaluation;
7. run stratified, first-error-boundary, and perturbation analyses;
8. run causal-prefix checks and offline heuristic pruning;
9. optionally run single-solution evaluation;
10. run the coin-flip baseline;
11. aggregate and display the final table, pruning trade-off plot, and primary operating point.

Default configuration:

```python
LR = 1e-4
EPOCHS = 30
SEED = 42
EARLY_STOPPING_PATIENCE = 5
CALIBRATION_FRACTION = 0.5
RUN_PERTURBATIONS = True
HEADS = ["linear", "mlp", "cnn", "gru", "attention", "attention_pe"]
```

By default, the notebook does not regenerate embedding caches. It reads `cache/train_fixed` and `cache/val_fixed`, writes checkpoints to `checkpoints/rerun/`, and writes results to `results_rerun/`, so it does not overwrite final results (`checkpoints/final/`, `results_final/`) or historical pre-fix records.

## 16. Recommended Structure for the Final Report

The report should follow the evidence chain rather than describing each network in isolation:

1. Project objective: can a frozen encoder plus a lightweight Reward Head perform process reward modeling?
2. Data overview: label proportions, trajectory lengths, first-error positions, and positional bias.
3. Baselines: majority, position-only, coin-flip.
4. Training process: common 30-epoch cap, patience=5 early stopping, and best checkpoint selection.
5. Overall performance: held-out ROC-AUC, AP, and ranking accuracy for all six heads.
6. Single-step semantics: do Linear/MLP outperform the position baseline?
7. Value of context: do CNN/BiGRU/Attention outperform pointwise heads?
8. Source of capability: stratification by length, first-error position, and error count.
9. Interpretable behavior: reward drop at the first-error boundary vs. correct-boundary control.
10. Structural sensitivity: ordering and local-mask perturbations.
11. Causal prefix: how much performance changes after removing future information.
12. Pruning value: detection, delay, and safe step savings under a fixed false-pruning budget.
13. Efficiency: trade-offs among parameter count, time, memory, and performance.
14. Learning-rate grid: explain why architecture conclusions from a fixed `lr` are invalid.
15. Best-of-N: compare against majority voting and oracle.
16. LoRA comparison: quantify the cost of freezing the encoder.
17. Limitations: each limitation should be paired with the experiment that quantifies it rather than stated vaguely:
   - interaction between epoch budget and model capacity (checked at 30 epochs; ranking unchanged),
   - learning-rate grid does not probe the lower bound,
   - LoRA is not compute-matched (1 epoch vs. 30 epochs, favoring the frozen side),
   - most configurations use one seed (three heads were checked with three seeds and their intervals did not overlap),
   - `max_length=512` drops about 9% of steps (2048-length re-encoding changes AUC by ±0.009),
   - offline step savings are not equivalent to real speedup.

Do not claim that a network “understands reasoning” based only on overall Accuracy. An architecture conclusion should be supported by at least:

- performance above deterministic bias baselines;
- better held-out threshold-free metrics;
- advantage on corresponding stratified subsets;
- reasonable first-error-boundary behavior;
- perturbation sensitivity consistent with the architecture;
- pruning signal that remains effective under causal-prefix evaluation.

## 17. Complete Post-Experiment Audit Checklist

### 17.1 Code Checks

```bash
python -m unittest discover -v
python -m py_compile *.py analysis/*.py eval/*.py tests/*.py
```

### 17.2 Configuration Consistency

Confirm that all heads use:

- the same train cache;
- the same validation cache;
- the same `seed=42`;
- the same `split_seed=42`;
- the same `calibration_fraction=0.5`;
- the same learning rate and PQM `zeta`;
- the same early-stopping rule.

### 17.3 Prevent Data Leakage

Confirm that:

- early stopping uses only the calibration/development half;
- classification thresholds are selected only on the calibration half;
- pruning thresholds are calibrated only on fully correct trajectories from the calibration split;
- the test half is never used for model, epoch, or threshold selection;
- every comparable model in the summary uses the same test half.

### 17.4 Required Outputs

At minimum, retain:

```text
checkpoints/*.pt
results_final/*_loss_history.json
results_final/*_loss_curve.png
results_final/*_efficiency.json
results_final/*_step_metrics.json
results_final/*_behavior_metrics.json
results_final/data_bias.json
results_final/deterministic_baselines.json
results_final/behavior_by_group.csv
results_final/first_error_boundary_curves.csv
results_final/perturbation_results.csv             # when perturbations are run
results_final/*_causal_predictions.pt
results_final/*_pruning_metrics.json
results_final/pruning_results.csv
results_final/pruning_by_group.csv
results_final/pruning_summary.md
results_final/summary.csv
results_final/summary.md
results_final/step_metrics_ci.json                  # confidence intervals for primary metrics
results_final/step_metrics_ci_pairwise.csv          # paired differences and significance
results_final/causal_metrics_ci.json                # causal-prefix confidence intervals
results_final/bon_results.csv                       # Best-of-N vs. majority / oracle
results_final/single_indist_metrics.csv             # in-distribution single-solution control
results_final/lora_vs_frozen.json                   # LoRA comparison
```

### 17.5 Conclusion Audit

Finishing the runs does not automatically validate the conclusions. Before delivery, verify each of the following:

1. **Every architecture claim has a confidence interval**, with either non-overlapping intervals or a significant paired difference.
2. **Pointwise-head causal-prefix scores must be exactly equal to full-trajectory scores** (difference should be on the order of `1e-6`). If not, the loading path is filtering steps by label rather than by `step_mask` — this bug previously inflated the causal drop by 40%.
3. **Attention without positional encoding must have exactly ΔAUC = 0 under reverse / swap perturbations** because it is permutation equivariant. Any non-zero value indicates an implementation error.
4. **The learning rate must not be blindly fixed**, or, if fixed, a grid must show that the chosen value is reasonable.
5. **Best-of-N must be compared with majority voting** — step-level metrics alone are insufficient to support deployment conclusions.
6. Checks 1–3 above are encoded as assertions in `tests/test_core.py`; `python -m unittest discover` verifies them.

See `RESULTS_MAP.md` for the purpose of each result directory. Final reported numbers should come from `results_final/`.
