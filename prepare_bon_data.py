import json
from datasets import load_dataset
from collections import defaultdict

def convert_to_bon_format():
    print("正在下载并加载 ProcessBench 数据集...")
    # 加载数据集，这里以 gsm8k 分支为例
    dataset = load_dataset('Qwen/ProcessBench', split='gsm8k')
    
    # 使用字典来按题目聚合候选解答
    bon_data = defaultdict(list)
    
    print("正在聚合数据...")
    for item in dataset:
        # 将原数据集的 'problem' 映射为 eval_bon.py 预期的 'question'
        question_text = item['problem']
        
        # 组装单个候选解答 (Candidate)
        # 注意：你需要根据 eval_bon.py 的判断逻辑，选择合适的正确性标签
        # 这里假设用 final_answer_correct 代表这条轨迹最终是否正确
        candidate = {
            "steps": item["steps"],
            "label": item["final_answer_correct"] 
        }
        
        bon_data[question_text].append(candidate)
    
    output_file = "processbench_bon_gsm8k.jsonl"
    print(f"聚合完成！共处理了 {len(bon_data)} 道独立的数学题。")
    print(f"正在保存为 BoN 格式的 JSONL 文件: {output_file} ...")
    
    # 写入本地文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for q, cands in bon_data.items():
            bon_item = {
                "question": q,
                "candidates": cands
            }
            f.write(json.dumps(bon_item, ensure_ascii=False) + '\n')
            
    print("转换成功！你现在可以用 eval_bon.py 跑这个文件了。")

if __name__ == "__main__":
    convert_to_bon_format()