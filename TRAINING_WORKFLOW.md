# 轻量级 PRM 完整实验手册

本文档是本项目唯一的完整实验执行说明。实验从环境准备、数据检查和编码器缓存开始，依次完成数据偏差审计、Reward Head 训练、标准指标评估、置信区间、分层与首错边界分析、扰动实验、因果前缀检查、离线剪枝、Best-of-N 重排序、以及 LoRA 对照。

## 协议已修订，旧版结论作废

本手册的初版规定 `lr=1e-3`、最多 10 epochs、patience=2，并据此比较了五种架构。后续的 6x3 学习率网格证明 **`1e-3` 是三个候选值中对全部六个头都最差的一个**（代价 0.056–0.445 dev loss），而这个影响比被比较的架构差异大 4–8 倍。在调好学习率后架构排名发生反转。

**因此：任何在 `lr=1e-3` 下得到的架构结论都不成立。** 当前协议为：

| 项 | 旧版 | 现行 |
|---|---|---|
| 学习率 | `1e-3` | `1e-4`（网格内对 6 个头中的 5 个最优） |
| epoch 上限 | 10 | 30 |
| early stopping patience | 2 | 5 |
| 主指标 | step-level 指标 | step-level 指标 + **Best-of-N** |

原先因算力排除的三项现已全部完成，不再是 scope 之外：

- **Best-of-N**：编码器缓存是唯一昂贵的部分，而它已经存在，打分只需几秒。BoN 是锚点论文 PQM 自己的主指标，排除它是初版最严重的方法论失误。
- **多随机种子**：`attention` / `cnn` / `mlp` 各跑 3 个种子。
- **LoRA**：全量 440k 过一轮，作为"冻结编码器"这一前提的对照。

## 数据解析已修复，修复前的全部结果作废

`dataset.py` 原先按换行切分步骤，把不以 `+`/`-` 结尾的行都当成错误步骤。Math-Shepherd 的一个步骤可以跨多行（最常见的是结尾的 `Step k: …\n\n# Answer\n\n42`），于是一个步骤被切成最多三个，多出的都被标成错误：13.5% 的错误标签是假的，17% 的全对轨迹被标成含错误，且假错误集中在最后一步之前。现在改为按 `ки` 标记切分步骤、从 `label` 同位置的 `+`/`-` 读标签；不符合该格式的记录会被拒绝并计数（114 条，0.03%）。

`cache/train_clean`、`cache/val_clean`、`cache/val_lora` 以及由它们得到的所有 checkpoint 和结果目录（`results_conv/`、`results_long30/`、`results_lrsweep/`、`results_seeds/`、`results_patience4/`、`results_loraeval/` 等）都建立在错误标签上，仅作记录保留。

### 路径约定

本手册的命令使用修复后的路径：

| 用途 | 路径 |
|---|---|
| 训练 / 验证 cache | `cache/train_fixed`、`cache/val_fixed` |
| 最终 checkpoint | `checkpoints/final/{head}_head.pt` |
| 训练记录与最终评估结果 | `results_final/` |
| 消融（loss、zeta、lr、种子） | `checkpoints/ablations/`、`results_ablations/` |

`run_experiments.py` 按本手册的顺序执行全部基于缓存的实验（见 14G）。第 14A、14E、14F 节保留的是修复前实际运行过的命令，作为历史记录。

训练仍使用固定种子 `42`（多种子实验另用 43、44）。除 embedding 预计算与 LoRA 训练外，其余实验全部复用缓存，不再运行 Qwen。

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
训练 Linear / MLP / CNN / BiGRU / Attention / Attention-PE
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
完整轨迹分数与因果前缀分数比较
        ↓
校准剪枝阈值并进行 held-out 离线回放
        ↓
误剪—检出—安全步骤节省权衡
        ↓
学习率网格（决定上面所有架构结论是否成立）
        ↓
轨迹级配对 bootstrap 置信区间
        ↓
Best-of-N 重排序 vs majority voting
        ↓
LoRA 对照（冻结前提是否成立）
        ↓
早停与种子敏感性
        ↓
汇总最终表格、曲线与报告结论
```

学习率网格排在架构分析之后，是因为历史顺序如此；**若从头重做，它应当排在训练之前** —— 先确定每个头的学习率，再比较架构。

完整实验要回答以下问题：

1. 数据中的类别比例或步骤位置是否已经可以预测正确性？
2. 冻结编码器的单步表示是否包含正确性信号？
3. CNN、BiGRU、Attention 等上下文模型是否优于 Linear/MLP？
4. 不同网络的差异发生在短轨迹、长轨迹、局部错误还是多个连续错误中？
5. Reward 是否真的在首个错误步骤附近发生下降？
6. 控制步骤位置后，这种边界效应是否仍然存在？
7. 顺序和局部信息被扰动后，不同网络是否表现出与其结构一致的敏感性？
8. 去除未来步骤信息后，各奖励头的性能还能保留多少？
9. 在限制正确轨迹误剪率的条件下，奖励信号能否转化为安全的理论步骤节省？

## 1. 环境准备

进入项目目录并安装依赖：

```bash
pip install -r requirements.txt
```

运行基础检查：

```bash
python -m unittest discover -v
python -m py_compile *.py analysis/*.py eval/*.py tests/*.py
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
  --cache_dir cache/train_fixed \
  --batch_size 16 \
  --max_length 512 \
  --dtype float16
```

### 4.2 验证集 cache

```bash
python precompute_embeddings.py \
  --train_file data/val.jsonl \
  --cache_dir cache/val_fixed \
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
cache/train_fixed/hidden_size.txt
cache/train_fixed/shard_*.pt
cache/val_fixed/hidden_size.txt
cache/val_fixed/shard_*.pt
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
  --cache_dir cache/val_fixed \
  --results_dir results_final \
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
results_final/data_bias.json
results_final/data_bias.png
results_final/position_label_rates.csv
results_final/deterministic_baselines.json
```

这一实验首先回答：后续模型是否只是利用“越靠后越容易错误”的数据规律。

## 6. 实验二：训练六种轻量 Reward Head

训练模型：

- Linear；
- MLP；
- CNN；
- BiGRU；
- Attention；
- Attention + 正弦位置编码（`attention_pe`，在扰动实验发现无位置编码的 attention 置换等变后加入）。

六个模型必须使用完全相同的：

- train cache；
- development 切分；
- 最大 30 epochs；
- 学习率 `1e-4`；
- PQM margin `zeta=4.0`；
- 固定随机种子 `42`；
- patience 为 5 的早停规则。

运行：

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

训练协议：

1. 每个模型最多训练 30 epochs；
2. 每个 epoch 在 calibration/development 半区计算 PQM loss；
3. 连续五个 epoch 没有改善则提前停止；
4. 始终保留 development loss 最低的 checkpoint；
5. held-out test 半区不参与 checkpoint 选择；
6. 每个 epoch 内记录约五个训练 loss 点，避免只有少量曲线点。

每个模型输出：

```text
checkpoints/final/{head}_head.pt
results_final/{head}_efficiency.json
results_final/{head}_loss_history.json
results_final/{head}_loss_curve.png
```

其中：

- `epochs`：实际运行的 epoch 数；
- `max_epochs`：最大值 30；
- `best_epoch`：最终保留 checkpoint 对应的 epoch；
- `stopped_early`：是否触发早停；
- `final_train_loss`：最佳 epoch 的训练 loss；
- `final_eval_loss`：最佳 epoch 的 development loss。

注意：CNN 的 padding mask 已改为在每层卷积后重新应用。旧实现训练得到的 CNN checkpoint 应重新训练。

## 7. 实验三：训练过程与效率比较

训练完成后检查：

```text
results_final/{head}_loss_curve.png
results_final/{head}_loss_history.json
results_final/{head}_efficiency.json
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
results_final/{head}_step_metrics.json
```

## 9. 实验五：可选 Single-solution 评估

如果存在 `cache/single_eval`：

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
results_final/{head}_single_metrics.json
```

## 10. 实验六：随机掷硬币基线

随机 baseline 不训练模型，计算成本可以忽略：

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

如果不存在 single-eval cache，删除：

```text
--single_cache_dir cache/single_eval
```

该 baseline 只在与其他模型相同的 held-out test 半区上运行。它不能替代实验一中的 majority 和 position-only baseline。

输出：

```text
results_final/coin_flip_efficiency.json
results_final/coin_flip_step_metrics.json
results_final/coin_flip_single_metrics.json       # 有 single cache 时
```

## 11. 实验七：架构分层与首错边界分析

运行核心行为分析：

```bash
python -m analysis.analyze_head_behavior \
  --cache_dir cache/val_fixed \
  --checkpoint_dir checkpoints/final \
  --results_dir results_final \
  --heads linear mlp cnn gru attention attention_pe \
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
results_final/{head}_step_predictions.pt
results_final/{head}_behavior_metrics.json
results_final/behavior_summary.csv
results_final/behavior_summary.md
results_final/behavior_by_group.csv
results_final/behavior_by_group.png
results_final/first_error_boundary_curves.csv
results_final/first_error_boundary.png
```

## 12. 实验八：可选确定性扰动实验

该实验不训练模型，只增加几次 reward head 前向：

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
results_final/perturbation_results.csv
results_final/perturbation_sensitivity.png
```

## 13. 实验九：因果前缀与离线启发式剪枝

完整轨迹上的 CNN、BiGRU 和 Attention 分数可能利用后续 step embedding，不能直接模拟在线早停。本实验对第 `t` 步只输入前 `t` 个缓存 embedding，并取前缀最后一步的奖励。冻结 Qwen 不会重新运行。

轨迹分为：

- `clean`：所有步骤正确；
- `monotone_error`：首次错误后不再恢复；
- `recovery`：出现 `正确→错误→正确`。

主剪枝指标只使用前两类；恢复型轨迹单独报告，避免把可能恢复的路径错误地当成必然应被剪枝的路径。

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

脚本比较两种策略：

- `single_low`：当前奖励低于阈值时停止；
- `two_consecutive`：连续两个奖励都低于阈值时停止。

每个阈值只在 calibration 半区确定，并分别限制全正确轨迹误剪预算为 1%、5% 和 10%。阈值固定后才在 held-out test 半区评估。默认主比较点是 `single_low` 在 5% 误剪预算下的结果。

主要指标：

- `clean_false_prune_rate`：全正确轨迹误剪率；
- `pre_error_false_prune_rate`：首错前错误停止的比例；
- `error_coverage`：首错后成功停止的错误轨迹比例；
- `detection_at_0/1/2`：在首错当步、后一步、后两步以内检出的比例；
- `median_detection_delay`：首错到停止的中位延迟；
- `safe_step_saving_rate`：只把首错后正确触发带来的剩余步骤计为收益；
- `oracle_efficiency_ratio`：实现了多少比例的理想首错剪枝空间。

1000 次轨迹级 bootstrap 只估计评估不确定性，不会重新训练模型。输出：

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

`theoretical_step_saving_rate` 与 `safe_step_saving_rate` 都是基于缓存轨迹长度的 step-equivalent 指标，不能写成真实 wall-clock 或 FLOPs 加速。

## 14. 实验十：最终结果汇总

所有实验完成后运行：

```bash
python -m eval.summarize_results \
  --results_dir results_final \
  --out_csv results_final/summary.csv \
  --out_md results_final/summary.md
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
- 完整轨迹与因果前缀的分数差异；
- 5% 误剪预算下的检出率、延迟和安全步骤节省率；
- majority、position-only、coin-flip baseline。

输出：

```text
results_final/summary.csv
results_final/summary.md
```

## 14A. 实验十一：学习率网格（最重要的一步）

初版协议把 `lr` 固定在 `1e-3` 然后比较架构。这一步检验那个固定值是否站得住。

> 历史记录：下面的网格运行在修复前的 `_clean` cache 上。修复后，`run_experiments.py --only lr` 在 `_fixed` cache 上用 `{3e-5, 3e-4}` 重新检验选定的 `1e-4`（`1e-3` 在修复前对全部六个头都明显最差，不再重跑）。

```bash
for lr in 1e-3 3e-4 1e-4; do
  for head in linear mlp cnn gru attention attention_pe; do
    python train_from_cache.py       --cache_dir cache/train_clean --val_cache_dir cache/val_clean       --head "$head" --epochs 30 --early_stopping_patience 5 --seed 42       --calibration_fraction 0.5 --split_seed 42 --lr "$lr" --zeta 4.0       --save_path "checkpoints/lrsweep/${head}_lr${lr}_head.pt"       --results_dir "results_lrsweep/${head}_lr${lr}"
  done
done
```

实测结果：**`1e-3` 对六个头全部最差**，改善幅度 0.056（linear）到 0.445（attention），
而架构间的差异不超过 0.05。排名因此反转 —— `cnn` 在 `1e-3` 下领先，`attention` 在调好后领先。

两条必须写进 limitation 的边界：网格没有探到下界（`1e-4` 对 5/6 个头最优且趋势单调），
以及不同容量的头达到最优所需的 epoch 不同（`linear` 约 11，`gru` 约 25）。

## 14B. 实验十二：主指标的置信区间

架构排名依赖 0.005 量级的差值，没有区间就无法判断是否为噪声。

```bash
python -m analysis.bootstrap_step_metrics   --cache_dir cache/val_fixed --checkpoint_dir checkpoints/final   --results_dir results_final --heads linear mlp cnn gru attention attention_pe   --bootstrap_samples 2000 --calibration_fraction 0.5 --split_seed 42 --seed 42

python -m analysis.bootstrap_causal_metrics   --results_dir results_final --heads linear mlp cnn gru attention attention_pe   --bootstrap_samples 2000 --seed 42
```

重采样单位是**轨迹**而非步骤（同一条解答内的步骤高度相关），且所有头在**同一次重采样**上打分，
这样头与头之间的差值才有正确的配对区间。

`bootstrap_causal_metrics` 复用 `analyze_offline_pruning` 已存的 causal predictions，零前向开销。

六个头两两比较是每个指标 15 次检验，逐个看区间是否跨 0 会放大假阳性。用 Holm 校正控制族错误率（只读上面两个 pairwise CSV，不需要模型或缓存）：

```bash
python -m analysis.holm_correction   results_final/step_metrics_ci_pairwise.csv results_final/causal_metrics_ci_pairwise.csv   --bootstrap_samples 2000
```

输出同名的 `*_pairwise_holm.csv`，原文件不变。

输出：`step_metrics_ci.json` / `.csv` / `_pairwise.csv`、`causal_metrics_ci.json`。

## 14C. 实验十三：Best-of-N 重排序

这是锚点论文 PQM 的主指标，也是 PRM 最贴近实际的用途。

```bash
python -m eval.eval_bon   --cache_dir cache/single_eval --eval_file data/single_eval.jsonl   --source_file data/gsm8k_qwen0.5b_bon16.jsonl   --checkpoint_dir checkpoints/final --heads linear mlp cnn gru attention attention_pe   --aggs min mean last --ks 1 2 4 8 16 --subsets_per_question 20   --bootstrap_samples 2000 --results_dir results_final --seed 42
```

两条参照线缺一不可：

- `majority_vote`（self-consistency）：只数最终答案，不用奖励模型。**PRM 必须打败它才有部署价值**；
- `oracle`：16 个候选里只要有一个对就算对，给出重排序的上限。

`k < 16` 时每题抽多个随机子集，避免结果依赖候选的生成顺序。置信区间按**题目**重采样
（同一题的 16 个候选不独立）。

## 14D. 实验十四：同分布 single-solution 对照

若 BoN / OOD single-solution 表现差，需要区分"模型做不了轨迹级判断"与"迁移不过去"。

```bash
python -m eval.eval_single_from_step_cache   --cache_dir cache/val_fixed --checkpoint_dir checkpoints/final   --heads linear mlp cnn gru attention attention_pe --aggs min mean last   --results_dir results_final --bootstrap_samples 2000   --calibration_fraction 0.5 --split_seed 42 --seed 42
```

它复用验证集缓存，把"全部步骤正确"作为轨迹标签，因此与 OOD 版本只差分布。

## 14E. 实验十五：LoRA 对照

冻结编码器是**本项目自己引入的简化**，不是 PQM 的做法（后者在 8 卡上全量微调 7B）。
这一步检验该前提的代价。

> 历史记录：下面的 LoRA 实验在修复前的数据上运行（`train_lora.py` 同样经由 `dataset.py` 读取标签，`cache/val_lora` 也用旧解析器生成），其结论需要在修复后的数据上重跑才能成立。

```bash
python train_lora.py   --train_file data/train.jsonl --epochs 1 --batch_size 4 --max_length 512   --lr 1e-4 --zeta 4.0 --seed 42   --save_dir checkpoints/lora_full --results_dir results_lora_full

# LoRA 改变了编码器，评估缓存必须重新生成
python precompute_embeddings.py --train_file data/val.jsonl   --cache_dir cache/val_lora --lora_path checkpoints/lora_full/adapter   --batch_size 16 --max_length 512 --dtype float16

python -m analysis.compare_lora_frozen   --arm "lora_linear:cache/val_lora:linear:checkpoints/lora_eval/linear_head.pt"   --arm "frozen_linear:cache/val_clean:linear:checkpoints/long30/linear_head.pt"   --arm "frozen_attention:cache/val_clean:attention:checkpoints/long30/attention_head.pt"   --results_dir results_loraeval --bootstrap_samples 2000
```

在 8GB 显存上可用的操作点只有 `batch_size=4, max_length=512`；`batch_size=8` 会溢出并慢 6.8 倍。
全量一轮约 10 小时。

**解读时必须声明遍历次数不对称**（LoRA 1 轮 vs 冻结 30 轮）。该偏差对冻结侧有利，
因此"LoRA 无收益"是保守结论，而"LoRA 落败"无法区分于"轮数不够"。

## 14F. 实验十六：早停与种子敏感性

```bash
# patience 是否扭曲了排名
for head in linear mlp cnn gru attention; do
  python train_from_cache.py --cache_dir cache/train_clean --val_cache_dir cache/val_clean     --head "$head" --epochs 10 --early_stopping_patience 4 --seed 42     --calibration_fraction 0.5 --split_seed 42 --lr 1e-3 --zeta 4.0     --save_path "checkpoints/patience4/${head}_head.pt"     --results_dir "results_patience4"
done

# 种子方差是否大于架构差异
for seed in 43 44; do
  for head in attention cnn mlp; do
    python train_from_cache.py --cache_dir cache/train_clean --val_cache_dir cache/val_clean       --head "$head" --epochs 10 --early_stopping_patience 2 --seed "$seed"       --calibration_fraction 0.5 --split_seed 42 --lr 1e-4 --zeta 4.0       --save_path "checkpoints/seeds/${head}_s${seed}_head.pt"       --results_dir "results_seeds/${head}_s${seed}"
  done
done
```

判据是**区间是否重叠**，而不是点估计谁高。实测三个头的 AUC 区间两两不重叠，
架构差距是种子标准差的 3.5–5.5 倍。

注意这两项检查都不是在现行协议下做的：patience 检查用的是作废的 `lr=1e-3`，只能说明旧排名不是 patience 造成的；种子检查用的是 `lr=1e-4` 但只有 10 epochs，6 次中有 5 次在第 10 轮取到最佳，尚未收敛。因此它们支持的是 10 轮预算下的排名，而不是 `results_conv/` 中收敛后的数字。两者也都运行在修复前的数据上；修复后的种子检查见 14G 的 `seeds` 组。

## 14G. 一键运行：修复后的主实验与补充消融

`run_experiments.py` 在 `cache/train_fixed` / `cache/val_fixed` 上按顺序执行本手册中全部基于缓存的实验。训练步骤在 checkpoint 与 efficiency 文件都存在时自动跳过，中断后可直接重跑续上；评估步骤每次都重跑。

| 组 | 内容 | 训练量 |
|---|---|---|
| `main` | 六个头，现行协议（`lr=1e-4`、30 epochs、patience 5、`zeta=4`、种子 42） | 6 次 |
| `main_eval` | 第 5–14D 节的全部评估：偏差审计与基线、held-out 指标、行为 / 首错 / 扰动、因果前缀剪枝、bootstrap 置信区间与 Holm 校正、BoN（argmax 与 PRM 加权投票）、OOD 与同分布 single-solution、汇总表 | 无 |
| `bce` | PQM 排序 loss 是否优于逐步 BCE？这是锚点论文的核心主张 | linear / mlp / attention |
| `zeta` | 固定的 `zeta=4.0` 是否敏感？ | linear / attention × `zeta` ∈ {2, 8} |
| `seeds` | 现行协议下排名是否对种子稳健？ | attention / cnn / mlp × 种子 43、44 |
| `lr` | 修复后 `1e-4` 是否仍是合适的学习率？ | 六个头 × {`3e-5`, `3e-4`} |
| `ablation_eval` | 上述各组与最终头的配对 bootstrap + Holm 校正；BCE 头的 BoN | 无 |

```bash
python run_experiments.py --dry_run                  # 先看将执行的命令
python run_experiments.py                            # 全部执行
python run_experiments.py --only main main_eval      # 只跑主实验
```

输出：

```text
checkpoints/final/{head}_head.pt                     # 主实验
results_final/                                       # 训练记录与全部主评估
checkpoints/ablations/{bce,zeta,seeds,lr}/
results_ablations/train/                             # 消融训练记录
results_ablations/{seeds,loss_pqm_vs_bce,zeta,lr}.json 与 *_pairwise(_holm).csv
results_ablations/bon_bce/bon_results.csv
```

解读注意：

- 不同 `zeta` 或不同 loss 的 development loss 数值**不可互相比较**（loss 定义不同），比较只看 held-out ROC-AUC / AP 与 BoN。
- 学习率之间的比较同时看各自最佳 development loss 与 held-out AUC；若某个学习率在第 30 轮才取到最佳，说明预算不足，不能据此判定其优劣。
- 编码使用 `transformers`；本机的 `Training` 环境没有安装它，缓存由 `node2` 环境生成。两个环境在未受影响的记录上得到的 embedding 余弦相似度 ≥ 0.99996（fp16 舍入差异）。

## 15. 使用 Notebook 一次执行完整流程

主入口：

```text
complete_training.ipynb
```

Notebook 已按本手册顺序组织：

1. 设置 cache、checkpoint、results 路径；
2. 检查 cache；
3. 数据与位置偏差审计；
4. 训练六种 head；
5. 展示 loss 曲线；
6. step-level held-out 评估；
7. 分层、首错边界和扰动分析；
8. 因果前缀检查与离线启发式剪枝；
9. 可选 single-solution 评估；
10. coin-flip baseline；
11. 汇总并展示最终表格、剪枝权衡图和主工作点。

默认配置：

```python
LR = 1e-4
EPOCHS = 30
SEED = 42
EARLY_STOPPING_PATIENCE = 5
CALIBRATION_FRACTION = 0.5
RUN_PERTURBATIONS = True
HEADS = ["linear", "mlp", "cnn", "gru", "attention", "attention_pe"]
```

Notebook 默认不会重新生成 embedding cache。它读取 `cache/train_fixed` 与 `cache/val_fixed`，checkpoint 写入 `checkpoints/rerun/`、结果写入 `results_rerun/`，不会覆盖最终结果（`checkpoints/final/`、`results_final/`）或修复前的记录。

## 16. 最终报告的推荐顺序

报告应按照证据链组织，而不是逐个网络孤立汇报：

1. 项目目标：冻结编码器是否可以配合轻量 Reward Head 完成过程奖励建模；
2. 数据概况：标签比例、轨迹长度、首错位置和位置偏差；
3. 基线：majority、position-only、coin-flip；
4. 训练过程：统一 30 epochs 上限、patience=5 早停和最佳 checkpoint；
5. 总体表现：六个 head 的 held-out ROC-AUC、AP 和 ranking accuracy；
6. 单步语义：Linear/MLP 是否超过位置基线；
7. 上下文价值：CNN/BiGRU/Attention 是否超过 pointwise head；
8. 能力来源：长度、首错位置和错误数量分层；
9. 可解释行为：首错边界 reward drop 与正确边界对照；
10. 结构敏感性：顺序与局部 mask 扰动；
11. 因果前缀：去除未来信息后各 head 的性能变化；
12. 剪枝价值：固定误剪预算下的检出、延迟和安全步骤节省；
13. 效率：参数量、时间、显存和性能之间的权衡；
14. 学习率网格：说明为什么固定 `lr` 下的架构结论不成立；
15. Best-of-N：与 majority voting 和 oracle 对照；
16. LoRA 对照：冻结前提的代价；
17. 限制：每条都应附上量化它的那个实验，而不是笼统声明 ——
    epoch 预算与容量交互（已用 30 轮验证，排名不变）、
    lr 网格未探到下界、
    LoRA 非算力对齐（1 轮 vs 30 轮，偏差对冻结侧有利）、
    多数配置单种子（三头三种子验证过，区间不重叠）、
    `max_length=512` 丢失约 9% 步骤（2048 重编码验证过，±0.009 AUC）、
    离线步骤节省不等于真实加速。

不要仅根据总体 Accuracy 宣称某种网络“理解了推理”。架构结论至少应同时得到以下证据支持：

- 超过确定性偏差基线；
- held-out threshold-free 指标更好；
- 对应分层样本上的优势；
- 合理的首错边界行为；
- 与网络结构一致的扰动敏感性；
- 因果前缀下仍然有效的剪枝信号。

## 17. 实验结束后的完整审计清单

### 17.1 代码检查

```bash
python -m unittest discover -v
python -m py_compile *.py analysis/*.py eval/*.py tests/*.py
```

### 17.2 配置一致性

确认所有 head 使用：

- 相同 train cache；
- 相同 validation cache；
- 相同 `seed=42`；
- 相同 `split_seed=42`；
- 相同 `calibration_fraction=0.5`；
- 相同学习率和 PQM `zeta`；
- 相同早停规则。

### 17.3 防止数据泄漏

确认：

- early stopping 只使用 calibration/development 半区；
- 阈值只在 calibration 半区选择；
- 剪枝阈值只用 calibration 中的全正确轨迹校准；
- test 半区没有参与模型、epoch 或阈值选择；
- summary 中所有可比较模型使用相同 test 半区。

### 17.4 必需输出

至少保留：

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
results_final/perturbation_results.csv             # 执行扰动时
results_final/*_causal_predictions.pt
results_final/*_pruning_metrics.json
results_final/pruning_results.csv
results_final/pruning_by_group.csv
results_final/pruning_summary.md
results_final/summary.csv
results_final/summary.md
results_final/step_metrics_ci.json                 # 主指标置信区间
results_final/step_metrics_ci_pairwise.csv         # 配对差值与显著性
results_final/causal_metrics_ci.json               # 因果前缀置信区间
results_final/bon_results.csv                      # Best-of-N vs majority / oracle
results_final/single_indist_metrics.csv            # 同分布 single-solution 对照
results_final/lora_vs_frozen.json                  # LoRA 对照
```

### 17.5 结论审计

跑完不等于结论成立。交付前逐条核对：

1. **每个架构声明都有置信区间**，且区间不重叠或配对差值显著；
2. **pointwise 头的因果前缀分数必须精确等于完整轨迹分数**（差值应在 1e-6 量级）。
   若不为零，说明加载路径按 label 而非 `step_mask` 筛选了步骤 —— 这个 bug 曾使
   因果落差被夸大 40%；
3. **无位置编码的 attention 在 reverse / swap 扰动下 ΔAUC 必须精确为 0**
   （它是置换等变的）。若不为零说明实现有误；
4. **学习率不是固定的**，或若固定则已用网格证明该值合理；
5. **Best-of-N 与 majority voting 对照过** —— 只报 step-level 指标不足以支撑部署结论；
6. 上述 1–3 条已写成 `tests/test_core.py` 中的断言，`python -m unittest discover` 会检查。

结果目录的用途见 `RESULTS_MAP.md`；最终数字取自 `results_final/`。
