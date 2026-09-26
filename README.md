# Lightweight PRM Reward Heads on a Frozen Encoder

Trains six lightweight reward heads on top of a frozen Qwen2.5-0.5B step encoder
with the PQM comparative ranking loss, and asks which head architecture is worth
the parameters. The heads split into pointwise (`linear`, `mlp`) and contextual
(`cnn`, `gru`, `attention`), plus `attention_pe`, an order-aware variant added
after the perturbation analysis.

The anchor is *Process Reward Model with Q-Value Rankings* (PQM, Li & Li).
This repository reuses PQM's loss and its Math-Shepherd training corpus, but
deviates in two ways that matter: it freezes a 0.5B encoder where PQM fine-tunes
a 7B one, and it adds an architecture axis that PQM never ablates. A LoRA arm
tests what the first deviation costs.

All numbers below come from `results_final/` and `results_ablations/`, computed
on correctly parsed data (see [Data parsing](#data-parsing-and-what-it-invalidated)).
Held-out metrics use 2,223 validation trajectories that play no part in training,
early stopping or threshold selection; intervals are 95% trajectory-level (or,
for Best-of-N, question-level) paired bootstrap intervals over 2,000 resamples,
and "significant" means significant after Holm correction within the relevant
family of comparisons.

## What the experiments found

**Contextual heads win on held-out step scoring.** Held-out step ROC-AUC, each
head at `lr=1e-4` with up to 30 epochs and patience 5:

| head | params | ROC-AUC [95% CI] |
|---|---|---|
| `attention` | 6.43M | **0.8692** [0.8548, 0.8831] |
| `attention_pe` | 6.43M | 0.8651 [0.8511, 0.8787] |
| `gru` | 788k | 0.8527 [0.8382, 0.8675] |
| `cnn` | 394k | 0.8486 [0.8336, 0.8633] |
| `mlp` | 263k | 0.8374 [0.8224, 0.8521] |
| `linear` | 897 | 0.7672 [0.7501, 0.7839] |
| *position-only baseline* | 0 | *0.6313* |
| *majority baseline* | 0 | *0.5000* |

13 of the 15 pairwise differences are significant; `{attention, attention_pe}`
and `{gru, cnn}` are the two indistinguishable groups. Every head clears the
position-only baseline by a wide margin, so the frozen representation carries
step-correctness signal and not just step position. The ranking holds across
seeds (attention > cnn > mlp for seeds 42, 43 and 44; cnn and mlp are
indistinguishable at seed 43) and at `lr=3e-5` (below).

**Most of that advantage comes from reading future steps.** The encoder is
causal, but a bidirectional head re-introduces access to later steps. Scoring
step *t* from the first *t* embeddings only -- the setting any online
early-exit policy faces -- removes most of the gap:

| head | causal ROC-AUC [95% CI] | penalty vs full trajectory |
|---|---|---|
| `attention_pe` | 0.8431 [0.8288, 0.8575] | −0.0220 |
| `mlp` | 0.8374 [0.8224, 0.8521] | 0.0000 |
| `gru` | 0.8321 [0.8171, 0.8463] | −0.0206 |
| `attention` | 0.8251 [0.8087, 0.8405] | −0.0441 |
| `cnn` | 0.7866 [0.7716, 0.8017] | −0.0620 |
| `linear` | 0.7672 [0.7501, 0.7839] | 0.0000 |

Under causal scoring the 263k-parameter `mlp` is statistically
indistinguishable from the best contextual heads, `cnn` loses the most but stays
above `linear`, and 11 of 15 comparisons are significant. Which contextual head
comes first is *not* stable: it is `attention_pe` at `lr=1e-4` and `gru` at
`lr=3e-5`. The exactly-zero penalty for the pointwise heads is an invariant,
asserted in the test suite, as is the exactly-zero effect of reordering steps on
plain `attention`.

**Frozen heads can judge whole solutions in distribution, but not on GSM8K.**
Scoring a whole solution by its minimum step score, held-out Math-Shepherd
trajectories reach ROC-AUC 0.72 (`linear`) to 0.87 (`attention`). On Qwen's own
GSM8K candidates every head drops to 0.63-0.72, with `linear` the best at both
learning rates -- the representation's step-correctness signal does not
survive the change of generator and task.

**No frozen head beats majority voting at Best-of-N.** Reranking 16
Qwen2.5-0.5B candidates per GSM8K question (1,319 questions):

| selector | BoN@16 [95% CI] |
|---|---|
| oracle (any of 16 correct) | 0.7604 [0.7369, 0.7839] |
| **majority voting** (no reward model) | **0.4655** [0.4374, 0.4927] |
| best frozen head, PRM-weighted voting (`attention`, mean) | 0.4761, vs majority +0.011 [−0.003, +0.024], n.s. |
| best frozen head, highest score (`attention`, last) | 0.3699, vs majority −0.096 [−0.121, −0.070] |
| random selection | 0.3282 [0.3002, 0.3555] |

Of 36 frozen selectors (6 heads x 3 aggregations x {argmax, weighted vote}),
none beats majority voting after Holm correction and 23 are significantly worse;
the same holds at `lr=3e-5` (0 better, 21 worse) and for the BCE-trained heads
(0 better, 9 of 18 worse). Taking the single highest-scoring candidate is far
worse than letting the reward reweight a vote.

**Unfreezing the encoder is what changes that.** LoRA (r=16 on q/k/v/o, one
pass over the 440k training trajectories) with the same linear value head:

| comparison | ROC-AUC difference [95% CI] |
|---|---|
| LoRA linear vs frozen linear | **+0.0779** [+0.0636, +0.0927] |
| LoRA linear vs frozen attention | −0.0241 [−0.0356, −0.0130] |

On held-out steps LoRA recovers most, not all, of what a 6.4M-parameter head on
frozen features achieves -- but on GSM8K it is the only configuration that beats
majority voting. Its PRM-weighted vote wins with all three aggregations, each
significant after Holm correction across its six selectors:

| LoRA selector | BoN@16 | vs majority [95% CI] |
|---|---|---|
| weighted vote, last | 0.4920 | +0.027 [+0.010, +0.043] |
| weighted vote, min | 0.4897 | +0.024 [+0.008, +0.040] |
| weighted vote, mean | 0.4867 | +0.021 [+0.008, +0.034] |
| highest score, last | 0.4594 | −0.006 [−0.029, +0.017], n.s. |

The gain is small and the arm is not compute-matched, but the direction is
clear: better step metrics on frozen features did not buy Best-of-N, while
adapting the encoder did. What limits a frozen-encoder PRM here is the frozen
representation, not the head placed on it.

**The PQM ranking loss does not beat pointwise BCE here.** Training the same
heads with per-step binary cross-entropy instead:

| head | PQM − BCE, held-out ROC-AUC [95% CI] |
|---|---|
| `linear` | −0.0117 [−0.0161, −0.0072] |
| `mlp` | −0.0151 [−0.0220, −0.0084] |
| `attention` | +0.0008 [−0.0086, +0.0100], n.s. |

BCE is significantly better for the pointwise heads and equivalent for
`attention`. This is the opposite of the anchor paper's central claim, in a
setting that differs from it in the two ways named above, so it bounds rather
than refutes that claim.

**Learning rate matters as much as architecture.** On corrected data `lr=3e-4`
is worse than `1e-4` for five of six heads by 0.017-0.048 AUC -- as large as
most architecture gaps -- and `3e-5` reaches a lower development loss than
`1e-4` for five of six. Rerunning the entire evaluation chain on the `3e-5`
heads (`results_final_lr3e-5/`) reproduces the held-out ranking with the same
significance structure (only the indistinguishable top pair swaps), the causal
collapse, the in-distribution control and the Best-of-N failure. Two finer
claims do not reproduce, and are therefore not made above: which contextual head
leads causally, and that larger heads transfer worse to GSM8K. The margin `zeta`
matters little: against the default 4, only `linear` at `zeta=2` (−0.004) and
`attention` at `zeta=8` (−0.013) are significantly worse.

## Data parsing and what it invalidated

`dataset.py` originally split each Math-Shepherd solution on newlines and
labelled every line not ending in `+` or `-` as an incorrect step. A step can
span several lines -- most often the final `Step k: ...\n\n# Answer\n\n42` -- so
one labelled step became up to three, the extra ones marked wrong. On this
repository's data that made 13.5% of all error labels spurious and marked 17% of
fully correct trajectories as containing an error, with the fake errors placed
just before the final step. Steps are now the spans between `ки` markers in the
input and each label is the `+`/`-` that replaces its marker; a record whose
label text is not exactly its input with markers replaced is rejected (114 of
440,208 training and 2 of 4,447 validation records).

Every cache, checkpoint and result directory built before the fix is kept only
as a record (`RESULTS_MAP.md`). Two conclusions changed materially: the LoRA arm
had shown no gain from unfreezing (it now shows the largest effect in the
study), and `cnn` had fallen below `linear` under causal scoring (it no longer
does). The study's earlier history -- a protocol that fixed `lr=1e-3`, which the
first learning-rate grid showed to be the worst value for every head and which
had inverted the architecture ranking -- also predates the fix, and `1e-3` was
not re-tested on corrected data.

## Limitations

* **The learning-rate grid has no lower bound.** `3e-5` beats `1e-4` on
  development loss for five heads and the trend is monotone, so the optimum may
  lie lower. The main results keep the pre-specified `1e-4`; the full
  evaluation at `3e-5` shows which conclusions depend on that choice.
* **The LoRA arm is not compute-matched** (one pass against the frozen arms' up
  to 30 epochs) and uses a single seed. The handicap favours the frozen arms, so
  its advantage is if anything understated -- but its Best-of-N margin is small.
* **Seeds.** Three seeds were run for `attention`, `cnn` and `mlp` under the
  final protocol; the other three heads and all ablations rest on one seed.
* **`max_length=512` truncation.** 8.4% of validation steps are not encoded,
  75% of them error steps, in 507 of 4,445 trajectories. Re-encoding the
  validation set at 2048 tokens changes every head's AUC by −0.002 to +0.008,
  none significant after Holm, and changes no ranking. The in-distribution
  control reads its "fully correct" target from the full labels; the pruning
  analysis classifies trajectories from the encoded steps only, which can count
  a trajectory whose only error was truncated as clean.
* **Training budget.** Best epochs range from 6 (`linear`) to 30; the two heads
  that stopped at the 30-epoch cap (`mlp`, `attention_pe`) were rerun with a
  60-epoch cap, selected the same epoch-30 checkpoint and early-stopped at 35.
* **Single encoder, single training corpus.** Math-Shepherd's Monte-Carlo step
  labels are noisy, and nothing here separates that from the architecture
  question. Best-of-N uses one generator (Qwen2.5-0.5B) on one benchmark.

## Repository layout

Core pipeline:

- `encoder.py` — frozen step encoder; one hidden state per `ки` marker
- `precompute_embeddings.py`, `precompute_eval_embeddings.py` — build caches;
  `--lora_path` exports from a LoRA-adapted encoder instead
- `dataset.py` — Math-Shepherd parser and loader; `collate_fn` aligns labels to
  the step markers that survive tokenization
- `reward_heads.py` — the six heads behind one interface
- `pqm_loss.py` — PQM comparative ranking loss, plus the pointwise step-BCE
  baseline it is compared against
- `train_from_cache.py` — trains a head from cached embeddings (`--loss pqm|bce`)
- `train_lora.py` — the LoRA arm: adapters on the encoder plus a value head
- `run_experiments.py` — resumable driver for every cache-based experiment
  above, the LoRA arm, and their evaluation

Evaluation (`eval/`):

- `eval_step_metrics.py` — held-out step metrics
- `eval_bon.py` — Best-of-N (argmax and PRM-weighted voting) against majority
  voting and an oracle, with paired, Holm-corrected differences to majority voting
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
- `compare_lora_frozen.py` — paired comparison of arms that may use different
  caches or checkpoints (LoRA vs frozen, and the loss / zeta / lr / seed ablations)
- `holm_correction.py` — Holm-Bonferroni correction over those pairwise
  comparisons, within a chosen family

Run evaluation and analysis as modules from the repository root, for example
`python -m eval.eval_step_metrics ...`.

## Getting started

```bash
pip install -r requirements.txt
python -m unittest discover -v
python run_experiments.py --dry_run
```

`TRAINING_WORKFLOW.md` is the step-by-step manual and `RESULTS_MAP.md` says
which result directory answers which question. Of the 27 tests, several are
invariants rather than unit tests -- pointwise heads must score prefixes and full
trajectories identically, plain attention must be permutation equivariant, a
multi-line final step must stay one step -- and each of those exists because the
corresponding bug had already corrupted a conclusion.

The results were produced with two environments: caches and the LoRA arm with
torch 2.13 / transformers 5.13 / peft 0.21, heads and evaluation with torch 2.5.1.
On unaffected records the encoder output matches the pre-fix caches to cosine
similarity ≥ 0.99996 (fp16 rounding). Encoding the 440k training trajectories
takes about 2.5 hours and the LoRA pass about 9.5 hours on an RTX 4060 (8 GB).
