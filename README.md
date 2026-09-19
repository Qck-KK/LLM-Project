# Lightweight PRM Reward Head Training

This repo trains lightweight reward heads on top of frozen encoder embeddings.
It compares pointwise heads (`linear`, `mlp`) with contextual heads (`cnn`,
`gru`, `attention`) and tests whether gains reflect semantic/error-boundary
behavior rather than majority-class or step-position bias.

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

The final protocol uses a maximum of 10 epochs, validation-loss early stopping
(patience 2), one fixed seed, a trajectory-level calibration/test split, and
threshold-free metrics. Best-of-N and multi-seed training are explicitly out
of scope because of compute constraints.

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

Evaluation and reporting (`eval/`):

- `eval/eval_utils.py`: shared evaluation utilities
- `eval/eval_step_metrics.py`: validation-cache step metrics
- `eval/eval_single_from_cache.py`: optional single-solution evaluation
- `eval/eval_coin_flip_baseline.py`: fair-coin random baseline
- `eval/summarize_results.py`: writes `summary.csv` and `summary.md`

Analysis (`analysis/`):

- `analysis/analyze_data_bias.py`: label audit plus majority and position-only baselines
- `analysis/analyze_head_behavior.py`: stratified, first-error-boundary, and optional
  deterministic perturbation analyses

Evaluation and analysis scripts are run as modules from the repository root,
for example `python -m eval.eval_step_metrics ...`.

Install the environment with `pip install -r requirements.txt`. Follow
`TRAINING_WORKFLOW.md` in order; it is the authoritative experiment manual.
Run `python -m unittest discover -v` for the lightweight audit checks.

Old notebooks, historical result JSON files, and unused optional experiment
scripts were removed so the remaining project has one clear training path.
