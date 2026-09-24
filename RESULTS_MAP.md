# Results directory map

Each directory is one experiment run. They are kept separately rather than
overwritten because several of them exist precisely to show that an earlier
conclusion did not hold: the learning-rate sweep invalidated the architecture
ranking produced under the original fixed `lr=1e-3`, so both sets are preserved.

Read `results_conv/` for the final numbers. Everything else is either a
superseded run kept for the record, or a sensitivity/robustness check.

## Final results

| Directory | Contents |
|---|---|
| `results_conv/` | **Primary results.** Six heads at `lr=1e-4`, 30 epochs (converged), evaluated on `cache/val_clean`: step metrics, paired bootstrap CIs, causal-prefix CIs, offline pruning, Best-of-N. |
| `results_long30/` | Training artefacts (loss history, curves, efficiency) for the converged 30-epoch runs whose checkpoints `results_conv/` evaluates. |

## Superseded, kept for the record

| Directory | Contents |
|---|---|
| `results/` | The original protocol: `lr=1e-3`, 10 epochs. Its architecture ranking (cnn best) was later shown to be an artefact of the untuned learning rate. |
| `results_best/` | Six heads at their best lr from the grid, 10 epochs. Intermediate step between `results/` and `results_conv/`. |
| `results_attnpe/` | First `attention_pe` arm, `lr=1e-3` only. Superseded once the sweep covered it. |
| `results_single_mean/`, `results_single_last/` | Single-solution evaluation under the `mean` and `last` aggregations (the `min` default lands in `results/`). |
| `results/single_indist_metrics.*` | First in-distribution single-solution control, computed on `cache/val_2048` without recording which checkpoints it used. Superseded by the rerun on `cache/val_clean` with `checkpoints/long30` into `results_conv/` (`run_extra_experiments.py --only eval`). |
| `results_sub10/` | Frozen heads trained on the 10% subset. Built for a scaled-down LoRA comparison that was replaced by the full-data one. |

## Sensitivity and robustness checks

| Directory | Question it answers |
|---|---|
| `results_lrsweep/` | Does the learning rate matter more than the architecture? Full 6 heads x {1e-3, 3e-4, 1e-4} grid. **This is where the original conclusions broke.** |
| `results_patience4/`, `results_p4_eval/` | Was the ranking an artefact of `patience=2` early stopping? (No -- but tested at `lr=1e-3`, 10 epochs, i.e. on the superseded ranking.) |
| `results_seeds/`, `results_seedeval/` | Is the ranking stable across random seeds? Three seeds for attention/cnn/mlp at `lr=1e-4` with a 10-epoch cap. (Yes; ranges do not overlap. Not rerun under the converged 30-epoch protocol.) |
| `results_long/` | Does the 10-epoch budget truncate the largest head? `attention @ 1e-4` for 30 epochs. (Yes, by 0.094 dev loss; best epoch 20.) |
| `results_val2048/` | Does the ~9% of steps lost to `max_length=512` truncation change anything? Re-encoded validation set at 2048. (No.) |
| `results_lora_full/`, `results_loraeval/` | Is the frozen-encoder premise costing anything, and is the Best-of-N failure caused by freezing? LoRA on the full 440k for one epoch. (No and no.) |

## Conventions shared by every directory

All runs use `seed=42` for training, `split_seed=42` and
`calibration_fraction=0.5` for the trajectory-level calibration/test split, and
PQM `zeta=4.0`. Thresholds are always selected on the calibration half and
applied unchanged to the held-out half. Bootstrap intervals resample
trajectories, never individual steps.

`*_step_predictions.pt` and `*_causal_predictions.pt` are excluded by
`.gitignore`; regenerate them with `analysis.analyze_head_behavior` and
`analysis.analyze_offline_pruning`.
