# Results directory map

Each directory is one experiment run. They are kept separately rather than
overwritten because several of them exist precisely to show that an earlier
conclusion did not hold.

**Read `results_final/` and `results_ablations/`.** They are the only results
computed on correctly parsed data. Everything else predates the parser fix in
`dataset.py`, which had split multi-line Math-Shepherd steps into extra steps
labelled as errors (13.5% of error labels spurious, 17% of fully correct
trajectories marked as containing an error). Those directories are kept as a
record of how the study evolved, not as evidence.

## Final results (corrected data)

| Directory | Contents |
|---|---|
| `results_final/` | **Primary results.** Six heads at `lr=1e-4`, up to 30 epochs, patience 5, trained on `cache/train_fixed` and evaluated on `cache/val_fixed`: training logs, data-bias audit and baselines, held-out step metrics, behaviour / first-error / perturbation analysis, causal-prefix pruning, paired bootstrap CIs with Holm correction, Best-of-N (argmax and PRM-weighted voting), out-of-distribution and in-distribution single-solution evaluation, summary table, and the 512- vs 2048-token truncation check (`truncation_512_vs_2048.*`). |
| `results_final_lr3e-5/` | The same evaluation chain for the six heads trained at `lr=3e-5`. On the corrected data 3e-5 reaches a lower development loss than the protocol's 1e-4 for five of six heads, so this checks whether the conclusions depend on the learning rate. |
| `results_ablations/` | Paired comparisons against the final heads: PQM vs BCE loss (`loss_pqm_vs_bce.*`, `bon_bce/`), `zeta` 2 / 8 (`zeta.*`), seeds 43 / 44 (`seeds.*`), learning rates 3e-5 / 3e-4 (`lr.*`), a 60-epoch cap for mlp and attention_pe (`long.*`), and LoRA against the frozen encoder (`lora_vs_frozen.*`, `bon_lora/`, training log in `lora/`). Holm corrections (`*_holm.csv`) cover only the comparisons each ablation is about. Training logs in `train/`. |

`run_experiments.py` regenerates both from the caches.

## Before the parser fix -- superseded, kept for the record

In order of the study's history. Each was a valid step at the time; none of
their numbers should be quoted.

| Directory | What it was |
|---|---|
| `results/` | The original protocol: `lr=1e-3`, 10 epochs. Its architecture ranking (cnn best) turned out to be an artefact of the untuned learning rate. Also holds the first in-distribution control (on `cache/val_2048`, checkpoints not recorded). |
| `results_attnpe/` | First `attention_pe` arm, `lr=1e-3` only. |
| `results_single_mean/`, `results_single_last/` | Single-solution evaluation under the `mean` and `last` aggregations. |
| `results_lrsweep/` | The 6 heads x {1e-3, 3e-4, 1e-4} grid. **This is where the lr=1e-3 conclusions broke**: 1e-3 was the worst value for every head. |
| `results_best/` | Six heads at their best lr from the grid, 10 epochs. |
| `results_long/` | `attention @ 1e-4` for 30 epochs: the 10-epoch budget had truncated it. |
| `results_long30/`, `results_conv/` | Six heads at `lr=1e-4`, 30 epochs, and their full evaluation -- the protocol `results_final/` repeats on corrected data. |
| `results_patience4/`, `results_p4_eval/` | Patience check, at the superseded `lr=1e-3`. |
| `results_seeds/`, `results_seedeval/` | Seed check with a 10-epoch cap. |
| `results_val2048/` | Truncation check at 2048 tokens. |
| `results_sub10/` | Frozen heads on a 10% subset, for a scaled-down LoRA comparison later replaced by the full one. |
| `results_lora_full/`, `results_loraeval/` | LoRA on the full 440k for one epoch, and its comparison with the frozen encoder. `train_lora.py` reads labels through `dataset.py`, so this arm used the faulty parser too. |

## Conventions shared by every directory

All runs use `seed=42` for training unless the directory is a seed check,
`split_seed=42` and `calibration_fraction=0.5` for the trajectory-level
calibration/test split, and PQM `zeta=4.0` unless the directory is a zeta
check. Thresholds are always selected on the calibration half and applied
unchanged to the held-out half. Bootstrap intervals resample trajectories (or,
for Best-of-N, questions), never individual steps.

`*_step_predictions.pt` and `*_causal_predictions.pt` are excluded by
`.gitignore`; regenerate them with `analysis.analyze_head_behavior` and
`analysis.analyze_offline_pruning`.
