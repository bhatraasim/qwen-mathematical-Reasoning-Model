
"""
Unified GSM8K evaluation for every stage of the pipeline.

Usage examples (run each as its own process):

  # GRPO model
  python evaluate_all.py --adapter outputs/grpo_tagged

  # SFT model (for the baseline comparison)
  python evaluate_all.py --adapter outputs/sft_distilled_tagged

  # Base model, same tagged prompt as the trained models
  python evaluate_all.py --adapter none

  # Base model with the ORIGINAL harness prompt (reproduces your ~38% number)
  python evaluate_all.py --adapter none --prompt_style legacy

  # Quick smoke test on a subset
  python evaluate_all.py --adapter outputs/grpo_tagged --num_examples 100

Every run uses greedy decoding, the same test split, and the same
extraction logic, so the numbers are directly comparable. Each run also
writes a JSON file with every raw response for manual inspection.
"""

import argparse
import json
import math
import re
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Imported, not copied, so the eval prompt can never drift from training.
from data_generation.prepare_tagged_dataset import SYSTEM_PROMPT, build_user_prompt

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

NUM_RE = r"(-?\d[\d,]*(?:\.\d+)?)"

STRICT_FORMAT_RE = re.compile(
    r"^\s*<think>.+?</think>\s*<answer>.+?</answer>\s*$", re.DOTALL
)


def legacy_user_prompt(question):
    # The original pre-tags harness prompt (used for the ~38% base number).
    return (
        "Solve the following math problem step by step.\n\n"
        "At the end, give your final numerical answer "
        "in this exact format:\n"
        "Answer: <number>\n\n"
        f"Problem:\n{question}"
    )


def extract_answer(text):
    """Returns (answer_string_or_None, which_pattern_matched)."""
    patterns = [
        ("answer_tag", rf"<answer>\s*\$?\s*{NUM_RE}\s*</answer>"),
        ("boxed", rf"\\boxed\{{\s*\$?\s*{NUM_RE}\s*\}}"),
        ("answer_colon", rf"(?:Answer|Final Answer)\s*[:=]\s*\$?\s*{NUM_RE}"),
        ("answer_is", rf"(?:the\s+)?(?:final\s+)?answer\s+is\s+\$?\s*{NUM_RE}"),
    ]
    for name, pat in patterns:
        matches = re.findall(pat, text, re.IGNORECASE)
        if matches:
            return matches[-1].replace(",", ""), name
    return None, "no_match"


def numeric_equal(pred, actual):
    try:
        return abs(float(pred) - float(actual)) < 1e-6
    except (TypeError, ValueError):
        return False


def wilson_interval(correct, total, z=1.96):
    """95% confidence interval for a binomial proportion."""
    if total == 0:
        return 0.0, 0.0
    p = correct / total
    denom = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return centre - margin, centre + margin


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="outputs/grpo_tagged",
                        help="Path to a LoRA adapter dir, or 'none' for the base model")
    parser.add_argument("--prompt_style", choices=["tagged", "legacy"], default="tagged")
    parser.add_argument("--num_examples", type=int, default=None,
                        help="Subset size; default = full 1319-question test set")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=640)
    parser.add_argument("--out", default=None, help="Where to save raw responses (JSON)")
    args = parser.parse_args()

    use_adapter = args.adapter.lower() != "none"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Model:        {'base' if not use_adapter else args.adapter}")
    print(f"Prompt style: {args.prompt_style}")


    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")   

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    if use_adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.to(device)
    model.eval()

    test_dataset = load_dataset("openai/gsm8k", "main")["test"]
    if args.num_examples is not None:
        test_dataset = test_dataset.select(range(args.num_examples))
    total = len(test_dataset)
    print(f"Evaluating on {total} test questions\n")

    records = []
    correct = 0
    well_formed = 0
    truncated = 0

    for start in range(0, total, args.batch_size):
        end = min(start + args.batch_size, total)
        questions = [test_dataset[i]["question"] for i in range(start, end)]

        if args.prompt_style == "tagged":
            messages = [
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(q)},
                ]
                for q in questions
            ]
        else:
            messages = [
                [{"role": "user", "content": legacy_user_prompt(q)}]
                for q in questions
            ]

        prompts = [
            tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages
        ]
        inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        for j in range(len(questions)):
            generated = outputs[j][inputs["input_ids"].shape[1]:]
            n_tokens = int((generated != tokenizer.pad_token_id).sum().item())
            response = tokenizer.decode(generated, skip_special_tokens=True)

            predicted, pattern = extract_answer(response)
            actual = test_dataset[start + j]["answer"].split("####")[-1].strip()
            is_correct = numeric_equal(predicted, actual)

            is_well_formed = bool(STRICT_FORMAT_RE.match(response))
            hit_limit = n_tokens >= args.max_new_tokens

            correct += is_correct
            well_formed += is_well_formed
            truncated += hit_limit

            records.append({
                "index": start + j,
                "actual": actual,
                "predicted": predicted,
                "pattern": pattern,
                "correct": is_correct,
                "well_formed": is_well_formed,
                "hit_token_limit": hit_limit,
                "response_tokens": n_tokens,
                "response": response,
            })

        print(f"Progress: {end}/{total} | Accuracy: {correct / end * 100:.2f}%")

    accuracy = correct / total
    lo, hi = wilson_interval(correct, total)

    out_path = args.out or str(
        Path("results/evaluations")
        / (
            "eval_" + ("base" if not use_adapter else args.adapter.strip("/").split("/")[-1])
            + f"_{args.prompt_style}.json"
        )
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)

    patterns_used = {}
    for r in records:
        patterns_used[r["pattern"]] = patterns_used.get(r["pattern"], 0) + 1

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Correct:            {correct}/{total}")
    print(f"Accuracy:           {accuracy * 100:.2f}%   (95% CI: {lo * 100:.1f}% - {hi * 100:.1f}%)")
    print(f"Well-formed output: {well_formed}/{total} ({well_formed / total * 100:.1f}%)")
    print(f"Hit token limit:    {truncated}/{total} ({truncated / total * 100:.1f}%)")
    print(f"Answer patterns:    {patterns_used}")
    print(f"Raw responses saved to: {out_path}")


if __name__ == "__main__":
    main()
