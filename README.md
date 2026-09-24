# Lightweight PRM Reward Heads on a Frozen Encoder

Trains five lightweight reward heads on top of a frozen Qwen2.5-0.5B step
encoder with the PQM comparative ranking loss, and asks which head architecture
is worth the parameters. The heads split into pointwise (`linear`, `mlp`) and
contextual (`cnn`, `gru`, `attention`), plus `attention_pe`, an order-aware
variant added after the perturbation analysis.

The anchor is *Process Reward Model with Q-Value Rankings* (PQM, Li & Li).
This repository reuses PQM's loss and its Math-Shepherd training corpus, but
deviates in two ways that matter and are discussed below: it freezes a 0.5B
encoder where PQM fine-tunes a 7B one, and it added an architecture axis that
PQM never ablates.

## What the experiments found

**Contextual heads win on held-out step scoring.** Held-out ROC-AUC over 2,223
trajectories, 2,000 trajectory-level bootstrap resamples, each head at 30
epochs and `lr=1e-4`:

| head | params | ROC-AUC [95% CI] |
|---|---|---|
| `attention` | 6.43M | **0.8834** [0.8710, 0.8959] |
| `attention_pe` | 6.43M | 0.8782 [0.8655, 0.8906] |
| `gru` | 788k | 0.8682 [0.8553, 0.8810] |
| `cnn` | 394k | 0.8627 [0.8502, 0.8759] |
| `mlp` | 263k | 0.8507 [0.8379, 0.8641] |
| `linear` | 897 | 0.7885 [0.7739, 0.8029] |
| *position-only baseline* | 0 | *0.6251* |
| *majority baseline* | 0 | *0.5000* |

13 of the 15 pairwise differences are significant, and all 13 survive a Holm-Bonferroni
correction across the 15 comparisons (Average Precision: likewise 13 of 15,
all surviving); `{attention, attention_pe}`
and `{gru, cnn}` are the two indistinguishable groups. Every head clears the
position-only baseline by a wide margin, so the frozen representation does carry
step-correctness signal rather than step position.

**That advantage comes from reading future steps, and does not survive.** The
encoder is causal, but a bidirectional head re-introduces access to later steps
above it. Scoring step *t* from the first *t* embeddings only -- the setting any
online early-exit policy actually faces -- removes most of the gap:

| head | causal ROC-AUC | penalty vs full trajectory |
|---|---|---|
| `mlp` | 0.8507 | 0.0000 |
| `attention` | 0.8500 | −0.0334 |
| `gru` | 0.8451 | −0.0230 |
| `attention_pe` | 0.8308 | −0.0474 |
| `linear` | 0.7885 | 0.0000 |
| `cnn` | **0.7737** | **−0.0890** |

Under causal scoring `attention`, `mlp` and `gru` are statistically
indistinguishable, and `cnn` -- the head that led under the original protocol --
falls significantly below the 897-parameter linear head. The exactly-zero
penalty for the pointwise heads is an invariant, asserted in the test suite. All
12 significant causal comparisons also survive Holm correction.

**None of it transfers to the task a PRM is for.** Reranking 16 Qwen2.5-0.5B
candidates per GSM8K question (1,319 questions, question-level bootstrap):

| selector | BoN@16 |
|---|---|
| oracle (any of 16 correct) | 0.7604 |
| **majority voting** (no reward model) | **0.4655** |
| best of 18 head/aggregation combinations | 0.3723 |
| random selection | 0.3283 |

The best configuration overlaps random selection and loses to free majority
voting by 0.093. This reproduces a known result -- step-level PRM metrics
correlate weakly with Best-of-N, and PRMs frequently fail to beat
self-consistency -- in a setting where all six heads share bit-identical frozen
features, which isolates the head's contribution.

**Freezing the encoder is not what breaks it.** LoRA (r=16, on q/k/v/o) over one
full pass of all 440k trajectories gives no significant gain over the frozen
encoder with the same linear head (+0.0106 AUC, 95% CI [−0.0005, +0.0224]), is
significantly worse than the frozen encoder with an attention head (−0.0843),
and fails Best-of-N identically. Picking a better head on frozen features is
worth roughly nine times more than unfreezing the encoder.

**The original architecture ranking was a learning-rate artefact.** A 6x3 sweep
found `lr=1e-3` -- the value the first protocol fixed -- to be the worst of three
for all six heads, costing 0.056 to 0.445 dev loss. That effect is 4-8x larger
than the architecture differences being measured, and it inverted the ranking:
`cnn` led at `1e-3`, `attention` leads once the rate is tuned. This is the
single most important methodological finding here, and it is the reason the
superseded result directories are kept rather than overwritten.

## Limitations

* **Training budget interacts with capacity.** Under matched `lr=1e-4`, best
  epochs run from 11 (`linear`) to 25 (`gru`); a fixed budget systematically
  favours smaller heads. Extending from 10 to 30 epochs moved every head by
  0.07-0.13 dev loss but did not change the ranking or the significance
  structure, so the conclusions hold -- but the comparison is still "best within
  30 epochs", not "best at convergence".
* **The learning-rate grid has no lower bound.** `1e-4` was best or near-best for
  every head and the trend is monotone, so the optimum lies below the range
  searched.
* **The LoRA arm is not compute-matched** (1 epoch against the frozen arms' 30).
  The handicap favours the frozen arms, which makes "LoRA shows no gain"
  conservative and a LoRA loss ambiguous.
* **Single seed for most runs, and the seed check predates the final
  protocol.** Three seeds were run for `attention`, `cnn` and `mlp`: the AUC
  ranges do not overlap and the architecture gaps are 3.5-5.5x the seed
  standard deviation. Those runs used `lr=1e-4` but the earlier 10-epoch cap,
  and five of the six picked epoch 10, so they had not converged. The check
  therefore supports the ranking at a 10-epoch budget, not the converged
  30-epoch numbers above, and the other three heads rest on one seed each.
* **The early-stopping check was run under the superseded protocol.** Raising
  patience from 2 to 4 did not change the ranking (`results_patience4/`), but
  that run used `lr=1e-3` and 10 epochs. It shows the *original* ranking was not
  a patience artefact; it says nothing direct about the final one, which uses
  patience 5 over 30 epochs.
* **`max_length=512` truncation.** About 9% of steps (concentrated in long
  trajectories, 80% of them error steps) are not encoded. Re-encoding the
  validation set at 2048 moved every head by +0.007 to +0.009 AUC and changed no
  ranking.
* **Single encoder, single training corpus.** Math-Shepherd's Monte-Carlo step
  labels are known to be noisy, and nothing here separates that from the
  architecture question.

## Repository layout

Core pipeline:

- `encoder.py` — frozen step encoder; one hidden state per `ки` marker
- `precompute_embeddings.py`, `precompute_eval_embeddings.py` — build caches;
  `--lora_path` exports from a LoRA-adapted encoder instead
- `dataset.py` — Math-Shepherd loader; `collate_fn` aligns labels to the step
  markers that survive tokenization
- `reward_heads.py` — the six heads behind one interface
- `pqm_loss.py` — PQM comparative ranking loss
- `train_from_cache.py` — trains a head from cached embeddings
- `train_lora.py` — the LoRA arm: adapters on the encoder plus a value head

Evaluation (`eval/`):

- `eval_step_metrics.py` — held-out step metrics
- `eval_bon.py` — Best-of-N against majority voting and an oracle
- `eval_single_from_cache.py` — single-solution scoring from a dedicated cache
- `eval_single_from_step_cache.py` — the in-distribution single-solution control
- `eval_coin_flip_baseline.py`, `summarize_results.py`

Analysis (`analysis/`):

- `analyze_data_bias.py` — label audit, majority and position-only baselines
- `analyze_head_behavior.py` — stratified, first-error-boundary and perturbation
  analyses
- `analyze_offline_pruning.py` — causal-prefix scoring and risk-budgeted pruning
- `bootstrap_step_metrics.py`, `bootstrap_causal_metrics.py` — paired
  trajectory-level confidence intervals
- `holm_correction.py` — Holm-Bonferroni correction over those pairwise
  comparisons, from the CSVs they already wrote
- `compare_lora_frozen.py` — paired comparison across two encoders' caches

Run evaluation and analysis as modules from the repository root, for example
`python -m eval.eval_step_metrics ...`.

## Getting started

```bash
pip install -r requirements.txt
python -m unittest discover -v
```

`TRAINING_WORKFLOW.md` is the step-by-step manual. `RESULTS_MAP.md` says which
result directory answers which question; read `results_conv/` for the final
numbers. Four of the eighteen tests are invariants rather than unit tests --
pointwise heads must score prefixes and full trajectories identically, plain
attention must be permutation equivariant — and two of them exist because the
corresponding bug had already corrupted a published conclusion.
