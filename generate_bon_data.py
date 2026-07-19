import os
import json
import re
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# ================= 配置参数 =================
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct" 
# 明确指定保存到 data 文件夹下
OUTPUT_DIR = "f:/LLMproject/data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "gsm8k_qwen0.5b_bon16.jsonl")

N_SAMPLES = 16          # 每道题采样的轨迹数 
BATCH_SIZE = 16          # 显存如果够大，可以改成 8 或 16 提速
TEMPERATURE = 0.7       
TOP_P = 0.95
MAX_NEW_TOKENS = 512
# ============================================

def extract_ground_truth(answer_str):
    """从 GSM8K 的标准答案中提取最终数字"""
    match = re.search(r'####\s*(-?\d+)', answer_str)
    return match.group(1) if match else None

def extract_model_prediction(text):
    """提取模型输出的最终数字"""
    matches = re.findall(r'-?\d+', text.replace(',', ''))
    return matches[-1] if matches else None

def main():
    # 安全检查：如果 data 文件夹不存在，就自动创建它
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"已自动创建输出文件夹: {OUTPUT_DIR}")

    print(f"Loading model {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        
        dtype=torch.float16
    ).to("cuda")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading GSM8K test set...")
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    
    results = []
    
    # 遍历测试集
    for i, item in enumerate(tqdm(dataset)):
        question = item["question"]
        gold_answer_str = item["answer"]
        gold_num = extract_ground_truth(gold_answer_str)
        
        messages = [
            {"role": "system", "content": "You are a helpful mathematical reasoning assistant. Please solve the math problem step by step and end your response with the final answer."},
            {"role": "user", "content": question}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = tokenizer([prompt] * BATCH_SIZE, return_tensors="pt", padding=True).to(model.device)
        
        candidates = []
        for _ in range(N_SAMPLES // BATCH_SIZE):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            for out in outputs:
                generated_text = tokenizer.decode(out[inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                pred_num = extract_model_prediction(generated_text)
                is_correct = 1 if (pred_num == gold_num and gold_num is not None) else 0
                
                candidates.append({
                    "text": generated_text,
                    "final_correct": is_correct,
                    "pred_num": pred_num
                })
        
        # 组装 BoN 所需的层级格式
        # 映射键名为 eval_bon.py 预期的 "question"
        results.append({
            "question": question,
            "gold_answer": gold_num,
            "candidates": candidates
        })
        
        # 实时写入 data 文件夹
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            for res in results:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()