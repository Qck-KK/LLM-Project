"""
eval_intrinsic.py
=================
Intrinsic Evaluation for Trajectory-level correctness.
This script evaluates the model as a binary classifier directly against the 
trajectory labels (e.g., from ProcessBench), skipping the Best-of-N generator loop.

It calculates:
1. ROC-AUC: The core ranking metric (how well it separates correct vs incorrect trajectories).
2. Accuracy: Binary classification accuracy based on a threshold (default 0.5).

Usage:
    python eval_intrinsic.py --eval_file data/processbench_bon_gsm8k.jsonl \
        --head gru --checkpoint checkpoints/gru_clean.pt
"""

import argparse
import json
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score

from encoder import FrozenStepEncoder
from reward_heads import build_reward_head


def load_trajectory_data(file_path):
    """
    解析 JSONL，提取所有候选轨迹及其对应的布尔标签
    将数据展平为一维列表，每个元素是一个 (question, steps, label) 样本
    """
    samples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            question = data["question"]
            
            for cand in data.get("candidates", []):
                # 兼容不同的步骤格式
                if "steps" in cand:
                    steps = cand["steps"]
                else:
                    raw_text = cand.get("text", cand.get("response", cand.get("content", "")))
                    steps = [s for s in raw_text.split('\n') if s.strip()]
                
                # 提取正确的二分类标签 (1 or 0)
                correct_val = cand.get("final_correct", cand.get("label", cand.get("is_correct", cand.get("correct", 0))))
                label = 1 if correct_val else 0
                
                samples.append((question, steps, label))
                
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_file", required=True)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--head", required=True,
                        choices=["linear", "mlp", "cnn", "gru", "attention"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    # 1. 设备配置
    device = args.device or ("cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"))

    print(f"🚀 初始化 Intrinsic Evaluation 脚本...")
    print(f"📦 设备: {device} | 模型: {args.model_name} | 头部: {args.head}")

    # 2. 加载 Encoder 和 Reward Head
    encoder = FrozenStepEncoder(model_name=args.model_name).to(device)
    encoder.eval()

    head = build_reward_head(args.head, hidden_size=encoder.hidden_size).to(device)
    head.load_state_dict(torch.load(args.checkpoint, map_location=device))
    head.eval()

    # 3. 加载并扁平化数据
    print(f"📂 正在加载并解析数据集: {args.eval_file} ...")
    samples = load_trajectory_data(args.eval_file)
    print(f"✅ 共解析出 {len(samples)} 条独立的解题轨迹样本。\n")

    y_true = []
    y_scores = []

    # 4. 推理打分循环
    with torch.no_grad():
        for question, steps, label in tqdm(samples, desc="Scoring Trajectories"):
            if not steps:
                continue
                
            text = question.strip() + "\n"
            for s in steps:
                text += s.strip() + f" {encoder.step_token}\n"
                
            # 编码文本
            step_hidden, step_mask = encoder.encode_texts([text], device=device, max_length=args.max_length)
            
            # 模型前向传播获取 q_values
            q_values = head(step_hidden, step_mask)[0]
            
            # 提取有效的分数，我们使用 "last" 策略作为轨迹级正误的判定分数
            valid_q = q_values[step_mask[0]]
            if valid_q.numel() == 0:
                continue
                
            score = valid_q[-1].item()  # 提取最后一个 step 的分数
            
            y_scores.append(score)
            y_true.append(label)

    # 5. 计算评估指标
    if len(y_true) == 0:
        print("❌ 没有提取到任何有效样本！")
        return

    # 计算 ROC-AUC
    try:
        auc = roc_auc_score(y_true, y_scores)
    except ValueError:
        auc = 0.0
        print("⚠️ 无法计算 AUC，可能是数据集中全为正样本或全为负样本。")

    # 计算硬标签准确率 (Accuracy)
    # 因为很多奖励模型在训练时是拟合 0/1 标签并带了 Sigmoid，这里默认以 0.5 作为二分类阈值
    threshold = 0.5
    y_pred = [1 if s >= threshold else 0 for s in y_scores]
    acc = accuracy_score(y_true, y_pred)

    # 打印最终报告
    print("\n" + "="*50)
    print("📊 最终内在评估结果 (Intrinsic Evaluation)")
    print("="*50)
    print(f"Total Valid Samples : {len(y_true)}")
    print(f"ROC-AUC             : {auc:.4f}  <-- 核心排序区分度指标")
    print(f"Accuracy            : {acc:.4f}  (Threshold={threshold})")
    print("="*50)


if __name__ == "__main__":
    main()