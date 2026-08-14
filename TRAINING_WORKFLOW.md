# Training Workflow

This document describes the complete experiment flow for training lightweight
reward heads from frozen encoder embeddings, including the attention head
comparison and train/evaluation loss curves.

## 1. Required Files

Core model and loss:

- `encoder.py`: frozen LLM encoder that extracts one hidden vector per reasoning step
- `dataset.py`: Math-Shepherd-style JSONL loader and collator
- `reward_heads.py`: `linear`, `mlp`, `cnn`, `gru`, and `attention` reward heads
- `pqm_loss.py`: PQM comparative ranking loss

Cache creation:

- `precompute_embeddings.py`: builds train/validation caches from step-labeled JSONL
- `precompute_eval_embeddings.py`: builds optional single-solution evaluation cache
- `benchmark.py`: optional hardware throughput check before large precompute runs

Training and evaluation:

- `complete_training.ipynb`: notebook entry point for the final training workflow
- `train_from_cache.py`: trains reward heads and writes loss history/plots
- `eval_step_metrics.py`: evaluates step accuracy and Q-value ranking accuracy
- `eval_single_from_cache.py`: optional final-answer evaluation from single-eval cache
- `eval_coin_flip_baseline.py`: fair-coin random baseline for final comparison
- `summarize_results.py`: merges result JSON files into summary tables
- `eval_utils.py`: shared utilities used by training/evaluation scripts

## 2. Data Format

Training and validation JSONL should contain one sample per line:

```json
{"question": "...", "steps": ["step 1", "step 2"], "labels": [1, 0]}
```

Single-solution evaluation JSONL is optional:

```json
{"question": "...", "steps": ["step 1", "step 2"], "final_correct": 1}
```

## 3. Precompute Caches

If caches were already generated on another host, copy them into this repo and
skip to section 4. The final notebook expects this structure:

```text
cache/train/hidden_size.txt
cache/train/shard_*.pt
cache/val/hidden_size.txt
cache/val/shard_*.pt
```

To create those caches locally or on a faster machine:

```bash
python precompute_embeddings.py \
  --train_file data/train.jsonl \
  --cache_dir cache/train \
  --batch_size 16

python precompute_embeddings.py \
  --train_file data/val.jsonl \
  --cache_dir cache/val \
  --batch_size 16
```

Optional single-solution evaluation cache:

```bash
python precompute_eval_embeddings.py \
  --eval_file data/single_eval.jsonl \
  --cache_dir cache/single_eval \
  --batch_size 16
```

## 4. Train Reward Heads

Use the notebook:

```text
complete_training.ipynb
```

The first cell controls paths and head selection. By default it trains:

```python
HEADS = ["linear", "mlp", "cnn", "gru", "attention"]
```

For only the old reward head, attention, and the random baseline comparison:

```python
HEADS = ["linear", "attention"]
COMPARE_HEADS = ["linear", "attention", "coin_flip"]
```

Each trained head writes:

```text
checkpoints/{head}_head.pt
results/{head}_efficiency.json
results/{head}_loss_history.json
results/{head}_loss_curve.png
```

The loss curve contains both train loss and evaluation loss.

## 5. Evaluation And Comparison

The notebook runs:

```bash
python eval_step_metrics.py \
  --cache_dir cache/val \
  --head attention \
  --checkpoint checkpoints/attention_head.pt \
  --results_dir results
```

If `cache/single_eval` exists, it also runs:

```bash
python eval_single_from_cache.py \
  --cache_dir cache/single_eval \
  --head attention \
  --checkpoint checkpoints/attention_head.pt \
  --results_dir results
```

The notebook then runs the fair-coin baseline:

```bash
python eval_coin_flip_baseline.py \
  --val_cache_dir cache/val \
  --single_cache_dir cache/single_eval \
  --results_dir results \
  --trials 100
```

This baseline randomly predicts correctness with probability 0.5. It writes:

```text
results/coin_flip_efficiency.json
results/coin_flip_step_metrics.json
results/coin_flip_single_metrics.json
```

The baseline row has `n_trainable_params = 0` and no train/evaluation loss.

Finally it writes:

```text
results/summary.csv
results/summary.md
```

Use `final_eval_loss`, `qvalue_ranking_accuracy`, `step_reward_accuracy`, and
optional `single_eval_accuracy` / `single_eval_separation` to compare the
attention head against the previous reward head and the `coin_flip` baseline.

## 6. Outputs To Keep

Keep these after a full run:

- `checkpoints/*.pt`
- `results/*_loss_curve.png`
- `results/*_loss_history.json`
- `results/*_efficiency.json`
- `results/*_step_metrics.json`
- `results/coin_flip_*.json`
- `results/summary.csv`
- `results/summary.md`
