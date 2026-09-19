# 轻量级 PRM 完整实验手册

本文档是本项目唯一的完整实验执行说明，覆盖原有实验与新增分析。实验从环境准备、数据检查和编码器缓存开始，依次完成硬件基准、数据偏差审计、五种 Reward Head 训练、标准指标评估、单条轨迹评估、确定性与随机基线、分层分析、首错边界分析、扰动实验和最终结果汇总。

本项目受算力限制，明确不包含：

- Best-of-N 候选生成与 BON@8/BON@16 评估；
- 多随机种子重复训练；
- LoRA 或全参数微调编码器。

所有模型训练使用一个固定种子 `42`。这能保证单次实验可复现，但不能替代多随机种子方差分析。除最初的 embedding 预计算外，后续实验全部复用冻结编码器缓存，不再运行 Qwen。

## 0. 完整实验顺序

必须按以下顺序执行：

```text
环境与数据准备
        ↓
可选硬件基准测试
        ↓
冻结 Qwen 编码器，生成 train/val cache
        ↓
数据与位置偏差审计
        ↓
训练 Linear / MLP / CNN / BiGRU / Attention
        ↓
检查 train/development loss 与最佳 epoch
        ↓
Held-out step-level 标准评估
        ↓
可选 single-solution 评估
        ↓
Majority / Position-only / Coin-flip baseline
        ↓
按轨迹长度、首错位置、错误数量进行分层分析
        ↓
首个错误边界与同位置正确边界对照
        ↓
可选确定性扰动实验
        ↓
汇总最终表格、曲线与报告结论
```

完整实验要回答以下问题：

1. 数据中的类别比例或步骤位置是否已经可以预测正确性？
2. 冻结编码器的单步表示是否包含正确性信号？
3. CNN、BiGRU、Attention 等上下文模型是否优于 Linear/MLP？
4. 不同网络的差异发生在短轨迹、长轨迹、局部错误还是多个连续错误中？
5. Reward 是否真的在首个错误步骤附近发生下降？
6. 控制步骤位置后，这种边界效应是否仍然存在？
7. 顺序和局部信息被扰动后，不同网络是否表现出与其结构一致的敏感性？

## 1. 环境准备

进入项目目录并安装依赖：

```bash
pip install -r requirements.txt
```

运行基础检查：

```bash
python -m unittest discover -v
python -m py_compile *.py tests/*.py
```

设备选择顺序为：

```text
CUDA → MPS → CPU
```

也可以通过 `--device cuda`、`--device mps` 或 `--device cpu` 强制指定。

## 2. 数据准备

### 2.1 训练集与验证集

训练集和验证集使用 Math-Shepherd 风格 JSONL，每行一条推理轨迹：

```json
{"question": "题目", "steps": ["步骤 1", "步骤 2"], "labels": [1, 0]}
```

其中：

- `question`：数学题；
- `steps`：按顺序排列的推理步骤；
- `labels[i] = 1`：第 `i` 步正确；
- `labels[i] = 0`：第 `i` 步错误。

建议目录：

```text
data/train.jsonl
data/val.jsonl
```

### 2.2 可选单条轨迹评估集

如果需要判断整条解答是否正确，准备：

```json
{"question": "题目", "steps": ["步骤 1", "步骤 2"], "final_correct": 1}
```

建议路径：

```text
data/single_eval.jsonl
```

### 2.3 数据隔离原则

本项目把验证 cache 按轨迹固定切成两半：

- calibration/development 半区：早停和分类阈值选择；
- held-out test 半区：只用于最终指标和行为分析。

所有脚本使用相同的 `calibration_fraction=0.5` 和 `split_seed=42`，保证切分完全一致。同一条轨迹的步骤不会跨越两个子集。

## 3. 可选：硬件吞吐基准实验

如果还没有 embedding cache，建议先估计预计算时间：

```bash
python benchmark.py \
  --model_name Qwen/Qwen2.5-0.5B \
  --head mlp \
  --batch_size 8 \
  --seq_len 512 \
  --n_batches 20 \
  --dataset_size 445000
```

把 `dataset_size` 替换为实际训练样本数。该实验记录：

- batch/second；
- example/second；
- token/second；
- CUDA 峰值显存或 MPS 当前内存；
- 估算的每个 epoch 时间。

如果 train/val cache 已经存在，可以跳过该实验。

## 4. 冻结编码器并生成缓存

### 4.1 训练集 cache

```bash
python precompute_embeddings.py \
  --train_file data/train.jsonl \
  --cache_dir cache/train \
  --batch_size 16 \
  --max_length 512 \
  --dtype float16
```

### 4.2 验证集 cache

```bash
python precompute_embeddings.py \
  --train_file data/val.jsonl \
  --cache_dir cache/val \
  --batch_size 16 \
  --max_length 512 \
  --dtype float16
```

### 4.3 可选 single-eval cache

```bash
python precompute_eval_embeddings.py \
  --eval_file data/single_eval.jsonl \
  --cache_dir cache/single_eval \
  --batch_size 16 \
  --max_length 512 \
  --dtype float16
```

预期目录结构：

```text
cache/train/hidden_size.txt
cache/train/shard_*.pt
cache/val/hidden_size.txt
cache/val/shard_*.pt
cache/single_eval/hidden_size.txt       # 可选
cache/single_eval/shard_*.pt            # 可选
```

三个 split 必须使用相同的：

- `model_name`；
- step token；
- `max_length`；
- tokenizer 配置。

如果缓存来自另一台机器，复制后不要重新运行编码器，直接进入实验一。

## 5. 实验一：数据质量与位置偏差审计

在比较网络之前，先确认数据中是否存在类别比例或步骤位置偏差：

```bash
python -m analysis.analyze_data_bias \
  --cache_dir cache/val \
  --results_dir results \
  --position_bins 5 \
  --calibration_fraction 0.5 \
  --split_seed 42
```

该实验统计：

- 正确/错误步骤数量与比例；
- 轨迹长度分布；
- 同时包含正确和错误步骤的轨迹比例；
- `正确→错误` 与 `错误→正确` 转移次数；
- 第一个错误步骤的相对位置；
- 不同相对位置区间的错误率。

同时计算两个确定性基线：

- `majority`：始终预测 calibration 半区的多数类；
- `position_only`：只使用步骤的相对位置区间预测，不读取 embedding。

输出：

```text
results/data_bias.json
results/data_bias.png
results/position_label_rates.csv
results/deterministic_baselines.json
```

这一实验首先回答：后续模型是否只是利用“越靠后越容易错误”的数据规律。

## 6. 实验二：训练五种轻量 Reward Head

训练模型：

- Linear；
- MLP；
- CNN；
- BiGRU；
- Attention。

五个模型必须使用完全相同的：

- train cache；
- development 切分；
- 最大 10 epochs；
- 学习率 `1e-3`；
- PQM margin `zeta=4.0`；
- 固定随机种子 `42`；
- patience 为 2 的早停规则。

运行：

```bash
for head in linear mlp cnn gru attention; do
  python train_from_cache.py \
    --cache_dir cache/train \
    --val_cache_dir cache/val \
    --head "$head" \
    --epochs 10 \
    --early_stopping_patience 2 \
    --seed 42 \
    --calibration_fraction 0.5 \
    --split_seed 42 \
    --lr 1e-3 \
    --zeta 4.0 \
    --save_path "checkpoints/${head}_head.pt" \
    --results_dir results
done
```

训练协议：

1. 每个模型最多训练 10 epochs；
2. 每个 epoch 在 calibration/development 半区计算 PQM loss；
3. 连续两个 epoch 没有改善则提前停止；
4. 始终保留 development loss 最低的 checkpoint；
5. held-out test 半区不参与 checkpoint 选择；
6. 每个 epoch 内记录约五个训练 loss 点，避免只有少量曲线点。

每个模型输出：

```text
checkpoints/{head}_head.pt
results/{head}_efficiency.json
results/{head}_loss_history.json
results/{head}_loss_curve.png
```

其中：

- `epochs`：实际运行的 epoch 数；
- `max_epochs`：最大值 10；
- `best_epoch`：最终保留 checkpoint 对应的 epoch；
- `stopped_early`：是否触发早停；
- `final_train_loss`：最佳 epoch 的训练 loss；
- `final_eval_loss`：最佳 epoch 的 development loss。

注意：CNN 的 padding mask 已改为在每层卷积后重新应用。旧实现训练得到的 CNN checkpoint 应重新训练。

## 7. 实验三：训练过程与效率比较

训练完成后检查：

```text
results/{head}_loss_curve.png
results/{head}_loss_history.json
results/{head}_efficiency.json
```

比较：

- train loss 是否持续下降；
- development loss 是否趋于稳定或开始上升；
- 不同 head 的最佳 epoch；
- 可训练参数量；
- 总训练时间；
- CUDA 峰值显存。

报告中不要仅比较最终 loss。建议画参数量—性能、训练时间—性能 Pareto 图或在最终表中同时给出效率指标。

## 8. 实验四：Held-out Step-level 标准评估

```bash
for head in linear mlp cnn gru attention; do
  python -m eval.eval_step_metrics \
    --cache_dir cache/val \
    --head "$head" \
    --checkpoint "checkpoints/${head}_head.pt" \
    --calibration_fraction 0.5 \
    --split_seed 42 \
    --results_dir results
done
```

评估流程：

1. 在 calibration 半区搜索分类阈值；
2. 在 held-out test 半区固定使用该阈值；
3. 不允许在 test 半区重新选择阈值。

报告指标：

- Step Accuracy；
- Balanced Accuracy；
- ROC-AUC；
- Average Precision；
- 同一轨迹内的 Q-value Ranking Accuracy。

主要指标建议使用：

- ROC-AUC；
- Average Precision；
- Q-value Ranking Accuracy。

这些指标不依赖在 test 上调出的阈值。普通 Accuracy 只作为辅助指标。

输出：

```text
results/{head}_step_metrics.json
```

## 9. 实验五：可选 Single-solution 评估

如果存在 `cache/single_eval`：

```bash
for head in linear mlp cnn gru attention; do
  python -m eval.eval_single_from_cache \
    --cache_dir cache/single_eval \
    --head "$head" \
    --checkpoint "checkpoints/${head}_head.pt" \
    --agg min \
    --calibration_fraction 0.5 \
    --split_seed 42 \
    --results_dir results
done
```

该实验先把每一步 Q-value 聚合成整条轨迹分数。当前支持：

- `min`：最低步骤分数；
- `mean`：平均步骤分数；
- `last`：最后一步分数。

默认使用 `min`。如需比较聚合策略，可以分别运行三次，但应保存到不同结果目录，避免后一次覆盖前一次。

报告：

- held-out Accuracy；
- Balanced Accuracy；
- ROC-AUC；
- Average Precision；
- Pairwise Separation。

输出：

```text
results/{head}_single_metrics.json
```

## 10. 实验六：随机掷硬币基线

随机 baseline 不训练模型，计算成本可以忽略：

```bash
python -m eval.eval_coin_flip_baseline \
  --val_cache_dir cache/val \
  --single_cache_dir cache/single_eval \
  --results_dir results \
  --trials 100 \
  --seed 42 \
  --calibration_fraction 0.5 \
  --split_seed 42
```

如果不存在 single-eval cache，删除：

```text
--single_cache_dir cache/single_eval
```

该 baseline 只在与其他模型相同的 held-out test 半区上运行。它不能替代实验一中的 majority 和 position-only baseline。

输出：

```text
results/coin_flip_efficiency.json
results/coin_flip_step_metrics.json
results/coin_flip_single_metrics.json       # 有 single cache 时
```

## 11. 实验七：架构分层与首错边界分析

运行核心行为分析：

```bash
python -m analysis.analyze_head_behavior \
  --cache_dir cache/val \
  --checkpoint_dir checkpoints \
  --results_dir results \
  --heads linear mlp cnn gru attention \
  --calibration_fraction 0.5 \
  --split_seed 42
```

脚本为每个 head 做一次轻量前向，并保存逐步 Q-value。不会重新运行编码器，也不会重新训练。

### 11.1 Pointwise 与 Contextual Head

模型分组：

- Pointwise：Linear、MLP；
- Contextual：CNN、BiGRU、Attention。

首先比较 pointwise 模型与 position-only baseline。如果 Linear/MLP 明显超过位置基线，说明冻结 encoder 表示本身包含步骤正确性信号。

再比较 contextual 与 pointwise 模型。如果 contextual head 进一步提高性能，说明步骤间关系可能提供额外信息。

### 11.2 分层实验

`behavior_by_group.csv` 和对应图片按以下维度拆分：

- 轨迹长度：short、medium、long；
- 首错位置：early、middle、late；
- 错误数量：one_error、multiple_errors。

这些结果用于检验：

- CNN 是否主要在局部错误附近受益；
- BiGRU 是否在长轨迹或多错误轨迹上更稳定；
- Attention 的优势是否随轨迹长度增加；
- 模型优势是否只出现在后段错误中。

### 11.3 首个错误边界

把所有含错误轨迹按照首个错误步骤对齐：

```text
offset=-2   offset=-1   offset=0   offset=+1   offset=+2
前两步        前一步      首个错误      后一步       后两步
```

不同 head 的 Q-value 尺度不同，因此先使用 calibration 半区的分数进行 head 内标准化。

报告三个量：

- `first_error_boundary_drop`：错误前一步的标准化 Q 减去首错 Q；
- `matched_correct_boundary_drop`：相同相对位置区间内，`正确→正确` 转移的平均下降；
- `position_controlled_boundary_effect`：上述两者之差。

如果最后一个指标为正，说明首错处的下降大于普通位置变化能够解释的下降。但这仍是关联性证据，不能直接宣称模型因果地理解了推理错误。

输出：

```text
results/{head}_step_predictions.pt
results/{head}_behavior_metrics.json
results/behavior_summary.csv
results/behavior_summary.md
results/behavior_by_group.csv
results/behavior_by_group.png
results/first_error_boundary_curves.csv
results/first_error_boundary.png
```

## 12. 实验八：可选确定性扰动实验

该实验不训练模型，只增加几次 reward head 前向：

```bash
python -m analysis.analyze_head_behavior \
  --cache_dir cache/val \
  --checkpoint_dir checkpoints \
  --results_dir results \
  --heads linear mlp cnn gru attention \
  --calibration_fraction 0.5 \
  --split_seed 42 \
  --run_perturbations
```

扰动包括：

- `reverse`：反转有效步骤顺序；
- `swap_adjacent`：交换相邻步骤；
- `mask_previous`：把首错前一步 embedding 置零；
- `mask_first_error`：把首错步骤 embedding 置零。

主要观察扰动前后的：

- ROC-AUC 变化；
- Q-value Ranking Accuracy 变化。

预期用途：

- Linear/MLP 对单纯重排应基本不敏感；
- CNN 对局部邻接破坏更敏感；
- BiGRU 对顺序反转或相邻交换更敏感；
- Attention 的变化反映全局交互对当前任务的贡献。

限制：缓存中的 encoder hidden state 已经编码原始上下文和位置信息，因此这些实验属于敏感性分析，而不是严格的因果干预。

输出：

```text
results/perturbation_results.csv
results/perturbation_sensitivity.png
```

## 13. 实验九：最终结果汇总

所有实验完成后运行：

```bash
python -m eval.summarize_results \
  --results_dir results \
  --out_csv results/summary.csv \
  --out_md results/summary.md
```

最终表格包含：

- 模型参数量；
- 实际 epoch 数和最佳 epoch；
- 训练时间和峰值显存；
- train/development loss；
- step-level Accuracy、Balanced Accuracy、ROC-AUC、AP；
- Q-value Ranking Accuracy；
- optional single-solution 指标；
- 首错边界下降；
- 位置匹配的正确边界下降；
- position-controlled boundary effect；
- majority、position-only、coin-flip baseline。

输出：

```text
results/summary.csv
results/summary.md
```

## 14. 使用 Notebook 一次执行完整流程

主入口：

```text
complete_training.ipynb
```

Notebook 已按本手册顺序组织：

1. 设置 cache、checkpoint、results 路径；
2. 检查 cache；
3. 数据与位置偏差审计；
4. 训练五种 head；
5. 展示 loss 曲线；
6. step-level held-out 评估；
7. 分层、首错边界和扰动分析；
8. 可选 single-solution 评估；
9. coin-flip baseline；
10. 汇总并展示最终表格。

默认配置：

```python
EPOCHS = 10
SEED = 42
EARLY_STOPPING_PATIENCE = 2
CALIBRATION_FRACTION = 0.5
RUN_PERTURBATIONS = True
HEADS = ["linear", "mlp", "cnn", "gru", "attention"]
```

Notebook 默认不会重新生成 embedding cache。

## 15. 最终报告的推荐顺序

报告应按照证据链组织，而不是逐个网络孤立汇报：

1. 项目目标：冻结编码器是否可以配合轻量 Reward Head 完成过程奖励建模；
2. 数据概况：标签比例、轨迹长度、首错位置和位置偏差；
3. 基线：majority、position-only、coin-flip；
4. 训练过程：统一 10 epochs 上限、早停和最佳 checkpoint；
5. 总体表现：五个 head 的 held-out ROC-AUC、AP 和 ranking accuracy；
6. 单步语义：Linear/MLP 是否超过位置基线；
7. 上下文价值：CNN/BiGRU/Attention 是否超过 pointwise head；
8. 能力来源：长度、首错位置和错误数量分层；
9. 可解释行为：首错边界 reward drop 与正确边界对照；
10. 结构敏感性：顺序与局部 mask 扰动；
11. 效率：参数量、时间、显存和性能之间的权衡；
12. 限制：单一训练种子、无 BON、无 LoRA、缓存扰动不等于因果解释。

不要仅根据总体 Accuracy 宣称某种网络“理解了推理”。架构结论至少应同时得到以下证据支持：

- 超过确定性偏差基线；
- held-out threshold-free 指标更好；
- 对应分层样本上的优势；
- 合理的首错边界行为；
- 与网络结构一致的扰动敏感性。

## 16. 实验结束后的完整审计清单

### 16.1 代码检查

```bash
python -m unittest discover -v
python -m py_compile *.py tests/*.py
```

### 16.2 配置一致性

确认所有 head 使用：

- 相同 train cache；
- 相同 validation cache；
- 相同 `seed=42`；
- 相同 `split_seed=42`；
- 相同 `calibration_fraction=0.5`；
- 相同学习率和 PQM `zeta`；
- 相同早停规则。

### 16.3 防止数据泄漏

确认：

- early stopping 只使用 calibration/development 半区；
- 阈值只在 calibration 半区选择；
- test 半区没有参与模型、epoch 或阈值选择；
- summary 中所有可比较模型使用相同 test 半区。

### 16.4 必需输出

至少保留：

```text
checkpoints/*.pt
results/*_loss_history.json
results/*_loss_curve.png
results/*_efficiency.json
results/*_step_metrics.json
results/*_behavior_metrics.json
results/data_bias.json
results/deterministic_baselines.json
results/behavior_by_group.csv
results/first_error_boundary_curves.csv
results/perturbation_results.csv             # 执行扰动时
results/summary.csv
results/summary.md
```

当前仓库不包含真实 cache、checkpoint 或实验结果。完成真实数据运行后，再根据上述清单进行最终结果审计。
