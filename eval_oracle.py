import json
import math
import argparse

def pass_at_k(n, c, k):
    """计算从n个样本中抽取k个，至少包含1个正确样本(共c个)的概率"""
    if n - c < k:
        return 1.0
    # 等价于 1 - 组合数之比
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_file", type=str, default="data/gsm8k_qwen0.5b_bon16.jsonl")
    args = parser.parse_args()

    questions = []
    with open(args.eval_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))

    ks = [1, 4, 8, 16]
    oracle_scores = {k: [] for k in ks}

    for q in questions:
        cands = q["candidates"]
        n = len(cands)  # 理论上是 16
        
        # 统计这 16 条轨迹中有几条是正确的 (利用之前修复过的 get 方法)
        c = sum(1 for cand in cands if cand.get("final_correct", False) in [True, 1])

        for k in ks:
            if k > n:
                continue
            oracle_scores[k].append(pass_at_k(n, c, k))

    print(f"成功加载 {len(questions)} 道题目的生成数据。")
    print("\n=== Oracle Pass@k (生成器能力上限) ===")
    for k in ks:
        if oracle_scores[k]:
            avg_score = sum(oracle_scores[k]) / len(oracle_scores[k])
            print(f"Oracle@{k}: {avg_score:.4f}")

if __name__ == "__main__":
    main()