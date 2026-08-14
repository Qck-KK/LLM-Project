# Lightweight PRM Reward Head Training

This repo trains lightweight reward heads on top of frozen encoder embeddings.
The final workflow supports train/evaluation loss curves and direct comparison
between the existing reward head and an attention reward head.

For the full step-by-step experiment instructions, see:

```text
TRAINING_WORKFLOW.md
```

## Main Entry

Use:

```text
complete_training.ipynb
```

The notebook can consume caches copied from another host, or caches generated
locally with `precompute_embeddings.py`.

## File Layout

Core experiment files:

- `dataset.py`: loads Math-Shepherd-style JSONL data
- `encoder.py`: frozen step encoder used during precompute
- `precompute_embeddings.py`: creates train/validation embedding caches
- `precompute_eval_embeddings.py`: creates optional single-eval cache
- `benchmark.py`: optional hardware throughput check
- `complete_training.ipynb`: final notebook entry point
- `train_from_cache.py`: trains reward heads from cached embeddings
- `reward_heads.py`: defines `linear`, `mlp`, `cnn`, `gru`, and `attention`
- `pqm_loss.py`: PQM loss
- `eval_step_metrics.py`: validation-cache step metrics
- `eval_single_from_cache.py`: optional single-solution evaluation
- `eval_coin_flip_baseline.py`: fair-coin random baseline for final comparison
- `summarize_results.py`: writes `summary.csv` and `summary.md`
- `eval_utils.py`: shared utility functions

Old notebooks, historical result JSON files, and unused optional experiment
scripts were removed so the remaining project has one clear training path.
