# Lightweight PRM: Frozen Encoder + Swappable Reward Head

Implements the architecture from the proposal:

```
Question + Steps  →  Frozen Qwen2.5-0.5B  →  per-step hidden states
                                                    │
                          ┌─────────────────────────┴─────────────────────────┐
                          │  Stage 2: swappable reward head (only this trains) │
                          │  linear / mlp / cnn / gru / attention              │
                          └─────────────────────────┬─────────────────────────┘
                                                    │
                                    PQM Comparative Ranking Loss (Eq.10)
```

## Files

- `encoder.py` — `FrozenStepEncoder`: wraps Qwen2.5-0.5B, freezes it, and extracts
  one hidden vector per reasoning step using a special `ки` step-marker token
  (Math-Shepherd convention). This part is IDENTICAL across every experiment
  in the proposal's comparison table — never touch it between runs.
- `reward_heads.py` — the 5 candidate architectures (`linear`, `mlp`, `cnn`,
  `gru`, `attention`), all sharing the interface
  `forward(step_hidden, step_mask) -> q_values`.
- `pqm_loss.py` — exact re-implementation of the official PQM loss
  (github.com/WindyLee0822/Process_Q_Model), Eq.10 of the paper.
- `dataset.py` — loads Math-Shepherd-style JSONL and formats
  `question + step ки step ки ...` strings for the encoder.
- `train.py` — CLI that ties the three stages together.

## Quick start

```bash
pip install torch transformers

python train.py \
  --head mlp \
  --model_name Qwen/Qwen2.5-0.5B \
  --train_file data/math_shepherd_train.jsonl \
  --batch_size 8 --epochs 3 --lr 1e-3 \
  --save_path checkpoints/mlp_head.pt
```

Swap `--head` between `linear | mlp | cnn | gru | attention` to reproduce each
row of the comparison table (Baseline C through Ours-Attention). The encoder
is downloaded/frozen identically each time, so only the head architecture
changes — this is what makes the comparison controlled.

## Data format expected by `dataset.py`

One JSON object per line:
```json
{"question": "...", "steps": ["step 1 text", "step 2 text", ...], "labels": [1, 1, 0, 1]}
```
`labels[i] = 1` if step i is correct, `0` if incorrect.

## Notes / things you'll likely want to extend

1. **LoRA baselines (A/B in the table)**: not included here since those train
   the LLM backbone itself — use `peft` LoRA on `AutoModelForCausalLM` +
   a linear head on top, with BCE (Baseline A) or `pqm_loss` (Baseline B).
2. **`--max_length` and step truncation**: if a trajectory's steps get cut off
   by tokenizer truncation, `train.py`'s `align_labels_to_steps` trims labels
   to match — but for real runs, set `max_length` generously so you're not
   silently dropping steps.
3. **Evaluation script** (BON@8/16, step accuracy, Q-value ranking accuracy,
   t-SNE/PCA on `step_hidden`) isn't included yet — happy to build that next
   once you've got training running, since it reuses `FrozenStepEncoder`
   directly.
4. Multi-GPU / DeepSpeed: the official PQM repo uses
   `torch.distributed.run` + 8 GPUs; this skeleton is single-GPU/CPU for
   clarity — wrap `head` in `DistributedDataParallel` if you scale up
   (encoder stays frozen so it doesn't need DDP wrapping at all).
