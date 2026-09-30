"""Builds every data file the experiments read.

    python prepare_data.py split      # Math-Shepherd -> data/train.jsonl, data/val.jsonl
    python prepare_data.py generate   # GSM8K test -> data/gsm8k_qwen0.5b_bon16.jsonl
    python prepare_data.py convert    # candidates -> data/single_eval.jsonl

split
    Downloads peiyi9979/Math-Shepherd (444,655 trajectories) from the Hugging
    Face Hub and holds out 1% (4,447) for validation, as in the reported runs.
    The original split used an unrecorded shuffle, so this seeded split has the
    same sizes and format but not the same rows; results reproduce up to that
    sampling difference.

generate
    Samples 16 solutions per GSM8K test question (1,319) from
    Qwen/Qwen2.5-0.5B-Instruct with temperature 0.7, top-p 0.95 and at most 512
    new tokens, and marks each correct when its last number equals the gold
    answer. The released data/gsm8k_qwen0.5b_bon16.jsonl was generated without
    a fixed seed, so a rerun gives a different sample; the file itself is
    committed so that the Best-of-N results can be reproduced exactly.

convert
    Turns each candidate into the step format the encoder reads: steps are the
    paragraphs of the solution (split on blank lines), and every candidate keeps
    its question_id and candidate_id. Deterministic; on Windows the output is
    byte-identical to the data/single_eval.jsonl used for the reported results.
"""

import argparse
import json
import os
import re

GSM8K_SYSTEM_PROMPT = ("You are a helpful mathematical reasoning assistant. Please solve "
                       "the math problem step by step and end your response with the "
                       "final answer.")


def split(args):
    from datasets import load_dataset

    dataset = load_dataset("peiyi9979/Math-Shepherd", split="train")
    parts = dataset.train_test_split(test_size=args.val_fraction, seed=args.seed)
    os.makedirs(os.path.dirname(args.train_out) or ".", exist_ok=True)
    for name, path in (("train", args.train_out), ("test", args.val_out)):
        with open(path, "w", encoding="utf-8") as f:
            for row in parts[name]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print("[split] {0:,} trajectories -> {1}".format(len(parts[name]), path))


def gsm8k_gold(answer):
    match = re.search(r"####\s*(-?\d+)", answer)
    return match.group(1) if match else None


def last_number(text):
    matches = re.findall(r"-?\d+", text.replace(",", ""))
    return matches[-1] if matches else None


def generate(args):
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    dataset = load_dataset("openai/gsm8k", "main", split="test")
    if args.limit:
        dataset = dataset.select(range(args.limit))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for index, item in enumerate(dataset):
            messages = [{"role": "system", "content": GSM8K_SYSTEM_PROMPT},
                        {"role": "user", "content": item["question"]}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                                   add_generation_prompt=True)
            inputs = tokenizer([prompt] * args.n_samples, return_tensors="pt",
                               padding=True).to(device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                         temperature=args.temperature, top_p=args.top_p,
                                         do_sample=True, pad_token_id=tokenizer.pad_token_id)
            gold = gsm8k_gold(item["answer"])
            candidates = []
            for output in outputs:
                text = tokenizer.decode(output[inputs["input_ids"].shape[1]:],
                                        skip_special_tokens=True)
                pred = last_number(text)
                candidates.append({"text": text,
                                   "final_correct": int(gold is not None and pred == gold),
                                   "pred_num": pred})
            f.write(json.dumps({"question": item["question"], "gold_answer": gold,
                                "candidates": candidates}, ensure_ascii=False) + "\n")
            f.flush()
            print("[generate] question {0}/{1}".format(index + 1, len(dataset)), flush=True)


def solution_steps(text):
    """Paragraphs of a free-form solution, one step each."""
    return [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]


def convert(args):
    with open(args.source, encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]
    if args.limit:
        questions = questions[:args.limit]
    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for question_id, item in enumerate(questions):
            for candidate_id, candidate in enumerate(item["candidates"]):
                f.write(json.dumps({"question": item["question"],
                                    "steps": solution_steps(candidate["text"]),
                                    "final_correct": candidate["final_correct"],
                                    "question_id": question_id,
                                    "candidate_id": candidate_id}, ensure_ascii=False) + "\n")
                n += 1
    print("[convert] {0:,} candidates from {1:,} questions -> {2}".format(
        n, len(questions), args.out))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("split", help="Math-Shepherd train/validation split")
    p.add_argument("--val_fraction", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_out", default="data/train.jsonl")
    p.add_argument("--val_out", default="data/val.jsonl")
    p.set_defaults(func=split)

    p = sub.add_parser("generate", help="sample Best-of-N candidates on GSM8K")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--n_samples", type=int, default=16)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0, help="first N questions only; 0 = all")
    p.add_argument("--out", default="data/gsm8k_qwen0.5b_bon16.jsonl")
    p.set_defaults(func=generate)

    p = sub.add_parser("convert", help="candidates -> step-format evaluation file")
    p.add_argument("--source", default="data/gsm8k_qwen0.5b_bon16.jsonl")
    p.add_argument("--limit", type=int, default=0, help="first N questions only; 0 = all")
    p.add_argument("--out", default="data/single_eval.jsonl")
    p.set_defaults(func=convert)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
