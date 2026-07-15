# 复现说明书：Frozen Encoder + Lightweight Reward Approximator PRM

本项目验证：能否用一个**冻结**的小语言模型（Qwen2.5-0.5B）做语义编码，
只训练一个**轻量级**的奖励打分头（Linear / MLP / CNN / GRU / Attention），
来平替传统"整个大 Transformer 都参与训练"的 Process Reward Model。

按下面的顺序执行，就能完整复现从训练到出最终对比表的全过程。

---

## 0. 环境准备

```bash
pip install torch transformers scikit-learn matplotlib
```

设备会自动检测（`cuda` → `mps` → `cpu`），Mac 上跑 MPS，有 N 卡就跑 CUDA，都不用手动指定 `--device`（除非你想强制指定）。

## 0.5 准备数据

需要 Math-Shepherd 格式（或同结构）的 JSONL，每行一条：
```json
{"question": "...", "steps": ["step 1", "step 2", ...], "labels": [1, 1, 0, 1]}
```
`labels[i]=1` 表示第 i 步是正确的推理步骤，`0` 表示错误。

准备**训练集**和**验证/测试集**两份文件，例如：
- `data/train.jsonl`
- `data/val.jsonl`

如果要做 BON@8/16 评估，还需要**另一份**格式的数据（每题多个候选完整解答 + 最终答案对错），见步骤 5。

---

## 1. 预计算编码器输出（只需跑一次，训练集 + 验证集各跑一次）

编码器是冻结的，输出不随训练变化，所以整个项目里**只需要让 0.5B 的 LLM 前向跑这一次**，之后所有训练、评估都复用缓存。

```bash
# 训练集
python precompute_embeddings.py \
  --train_file data/train.jsonl \
  --cache_dir cache/train \
  --batch_size 16

# 验证/测试集（用于后面的评估）
python precompute_embeddings.py \
  --train_file data/val.jsonl \
  --cache_dir cache/val \
  --batch_size 16
```

> 建议先用几千条数据的小样本跑通整条 pipeline，确认没问题后再上全量数据（全量可能要跑几十小时，具体时间用 `benchmark.py` 先在你的机器上实测，见下方"效率评测"）。

---

## 2. 训练全部 5 种轻量奖励头

依次训练 `linear`（Baseline C）/ `mlp` / `cnn` / `gru` / `attention`，全部读同一份训练集缓存，不再碰 LLM：

```bash
mkdir -p checkpoints results

for head in linear mlp cnn gru attention; do
  python train_from_cache.py \
    --cache_dir cache/train \
    --head $head \
    --epochs 10 \
    --lr 1e-3 \
    --save_path checkpoints/${head}_head.pt \
    --results_dir results
done
```

每个 head 训练完会自动在 `results/` 下写一份 `{head}_efficiency.json`（可训练参数量、训练耗时、峰值显存）。

---

## 3. 评估：Step-level 指标 + Q-value 排序准确率

```bash
for head in linear mlp cnn gru attention; do
  python eval_step_metrics.py \
    --cache_dir cache/val \
    --head $head \
    --checkpoint checkpoints/${head}_head.pt \
    --results_dir results
done
```

会在 `results/` 下写出 `{head}_step_metrics.json`，包含：
- Step-level Reward Accuracy（网格搜索最优阈值后的准确率）
- Q-value Ranking Accuracy（同一条轨迹内，正确步骤 Q 值高于错误步骤的比例——这是最能反映 PQM 训练目标本身效果的指标）

---

## 4. 评估：单条解答准确率（推荐先做这个，比 BON 快很多）

如果你不需要"从 N 个候选里选最好"的能力验证，只想快速看"PRM 打分和最终对错相不相关"，
可以用单条解答评估——同样是"大模型只推理一次，之后所有 head 复用缓存"的思路。

数据格式（每行一条解答，不是候选列表）：
```json
{"question": "...", "steps": ["step1", "step2", ...], "final_correct": 1}
```

```bash
# 4a. 预计算一次（只需跑一次，跟训练集的 precompute 是同一个思路）
python precompute_eval_embeddings.py \
  --eval_file data/single_eval.jsonl \
  --cache_dir cache/single_eval \
  --batch_size 16

# 4b. 对每个 head 评分（读缓存，几秒到几十秒每个，不再碰 LLM）
for head in linear mlp cnn gru attention; do
  python eval_single_from_cache.py \
    --cache_dir cache/single_eval \
    --head $head \
    --checkpoint checkpoints/${head}_head.pt \
    --results_dir results
done
```

会输出两个指标：
- **Best-threshold accuracy**：搜索最优阈值后，PRM 分数预测"最终对错"的准确率
- **Pairwise separation**：P(正确解答分数 > 错误解答分数)，AUC 等价指标，不依赖阈值

> 单条评估速度快、适合早期筛选架构，但它衡量的是"打分准不准"，不是"能不能挑出最好的候选"——
> 后者才是 PRM 实际要解决的问题。建议：先用单条评估快速筛掉明显不行的架构，
> 再对表现最好的 1-2 个架构额外跑一遍 BON@8/16（步骤 5）验证真正的重排序能力。

## 5. 评估：Best-of-N（BON@8 / BON@16，可选，更贴近实际场景但更耗时）

需要额外准备一份评估集，每道题带多个候选解答：
```json
{"question": "...", "candidates": [
  {"steps": ["...", "..."], "final_correct": 1},
  {"steps": ["...", "..."], "final_correct": 0},
  ...
]}
```
（这份数据通常需要用一个生成模型对同一道题采样多个解答，再判断每个解答最终答案对不对）

```bash
for head in linear mlp cnn gru attention; do
  python eval_bon.py \
    --eval_file data/bon_eval.jsonl \
    --head $head \
    --checkpoint checkpoints/${head}_head.pt \
    --ks 8 16 \
    --n_draws 20 \
    --results_dir results
done
```

---

## 6. 效率评测（可选，独立于上面几步，随时可以跑）

```bash
python benchmark.py \
  --model_name Qwen/Qwen2.5-0.5B \
  --head mlp \
  --batch_size 8 \
  --seq_len 512 \
  --n_batches 20 \
  --dataset_size 445000   # 换成你实际的训练集大小
```
会打印真实吞吐（tokens/sec）、峰值显存，并按你的数据集大小外推整个 epoch/多 epoch 需要多久。建议在你要用的每台机器上都跑一次。

---

## 7. 表征分析（解释"为什么"，不只是"多少"）

```bash
# 编码器输出的 t-SNE 可视化（只需跑一次，5 个 head 共用同一份编码器输出）
python analyze_representations.py --cache_dir cache/val --mode encoder_tsne

# 每个 head 各自的 Q-value 分布直方图（正确/错误步骤是否明显分开）
for head in linear mlp cnn gru attention; do
  python analyze_representations.py \
    --cache_dir cache/val \
    --mode qvalue_dist \
    --head $head \
    --checkpoint checkpoints/${head}_head.pt \
    --out results/qvalue_dist_${head}.png
done
```

---

## 8. 汇总成最终对比表

```bash
python summarize_results.py --results_dir results \
  --out_csv results/summary.csv \
  --out_md results/summary.md
```

生成的 `summary.md` 就是提案里那张"对比表"的雏形：每行一个架构，列包括可训练参数量、训练耗时、峰值显存、BON@8/16、Step Accuracy、Q-value Ranking Accuracy。可以随时重跑（比如只做完 3 个 head 就先看一眼），脚本会自动扫描已有的 JSON 文件，不要求全部跑完。

---

## 完整命令速查（假设已有 data/train.jsonl、data/val.jsonl、data/single_eval.jsonl、data/bon_eval.jsonl）

```bash
# 1. 预计算（训练集 + step-level 验证集）
python precompute_embeddings.py --train_file data/train.jsonl --cache_dir cache/train --batch_size 16
python precompute_embeddings.py --train_file data/val.jsonl   --cache_dir cache/val   --batch_size 16

# 1.5 预计算（单条解答评估集，独立于上面，格式不同）
python precompute_eval_embeddings.py --eval_file data/single_eval.jsonl --cache_dir cache/single_eval --batch_size 16

# 2-4. 训练 + 三类评估
mkdir -p checkpoints results
for head in linear mlp cnn gru attention; do
  python train_from_cache.py --cache_dir cache/train --head $head --epochs 10 \
    --save_path checkpoints/${head}_head.pt --results_dir results

  python eval_step_metrics.py --cache_dir cache/val --head $head \
    --checkpoint checkpoints/${head}_head.pt --results_dir results

  python eval_single_from_cache.py --cache_dir cache/single_eval --head $head \
    --checkpoint checkpoints/${head}_head.pt --results_dir results

  # 可选，更耗时：只对筛选出来的少数几个 head 跑
  # python eval_bon.py --eval_file data/bon_eval.jsonl --head $head \
  #   --checkpoint checkpoints/${head}_head.pt --ks 8 16 --results_dir results
done

# 7. 表征分析
python analyze_representations.py --cache_dir cache/val --mode encoder_tsne
for head in linear mlp cnn gru attention; do
  python analyze_representations.py --cache_dir cache/val --mode qvalue_dist \
    --head $head --checkpoint checkpoints/${head}_head.pt --out results/qvalue_dist_${head}.png
done

# 8. 汇总
python summarize_results.py --results_dir results
```

## 文件一览

| 文件 | 作用 |
|---|---|
| `encoder.py` | 冻结的 Qwen2.5-0.5B 语义编码器 |
| `reward_heads.py` | 5 种可训练的轻量奖励头 |
| `pqm_loss.py` | PQM 官方 Comparative Ranking Loss（Eq.10） |
| `dataset.py` | Math-Shepherd 格式数据加载 |
| `precompute_embeddings.py` | 预计算并缓存编码器输出（训练/验证集，只需跑一次） |
| `train_from_cache.py` | 用缓存快速训练任意 head |
| `train.py` | 不用缓存、每次都重新跑编码器的训练脚本（调试用，正式实验建议走缓存流程） |
| `benchmark.py` | 实测吞吐量/显存，估算训练时长 |
| `eval_step_metrics.py` | Step-level Accuracy + Q-value Ranking Accuracy（读缓存） |
| `precompute_eval_embeddings.py` | 预计算单条解答评估集的编码器输出，只需跑一次 |
| `eval_single_from_cache.py` | 单条解答评估：打分准确率 + 分离度（读缓存，推荐先做这个） |
| `eval_bon.py` | BON@8 / BON@16（自带候选批量编码，未走缓存，因为候选数据集通常和训练/验证集不同） |
| `analyze_representations.py` | t-SNE/PCA 可视化 + Q-value 分布图 |
| `summarize_results.py` | 汇总所有结果成一张对比表 |

## 常见问题

- **报错找不到 huggingface 模型**：检查网络能否访问 huggingface.co，或者设置 `HF_ENDPOINT` 镜像源。
- **MPS 上某个算子报错（常见于 GRU 头）**：加环境变量 `PYTORCH_ENABLE_MPS_FALLBACK=1`，会自动把不支持的算子退回 CPU 跑。
- **显存/内存不够**：调小 `precompute_embeddings.py` 和 `benchmark.py` 的 `--batch_size`，缓存机制下正式训练阶段几乎不占什么显存，通常不会是瓶颈。
- **想换更大的编码器**（比如验证"编码器越大越好"）：`--model_name` 换成别的 HuggingFace 模型 id 即可，`encoder.py` 里的 hidden_size 会自动适配，但要重新跑一遍 `precompute_embeddings.py`。
