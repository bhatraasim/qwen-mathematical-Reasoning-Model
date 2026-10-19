"""
Generate verified-correct reasoning traces on GSM8K using a strong
math-specialized teacher model (Qwen2.5-Math-7B-Instruct).

Tuned for: NVIDIA Quadro RTX 8000, 48GB VRAM, Turing architecture.
  - Turing has NO flash-attention-2 support and NO native bf16 tensor
    cores -- we load in fp16 with default/sdpa attention, not bf16,
    not flash_attention_2.
  - 48GB comfortably fits the 7B model in full fp16 (~14-15GB) with
    plenty of headroom for large batches and long sequences -- no
    need for 4-bit quantization here, full precision is both simpler
    and slightly higher quality.
  - With this much more throughput, you can realistically cover the
    FULL GSM8K train set (7473 examples) rather than a subset.

Still time-boxed and resumable for safety, but the budget knobs below
assume you have real headroom now rather than being tightly squeezed.
"""

import json
import os
import re
import time

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

TEACHER_MODEL = "Qwen/Qwen2.5-Math-7B-Instruct"

# --- Budget knobs ---
NUM_QUESTIONS = 10         # full GSM8K train set is realistic now
TIME_BUDGET_MINUTES = 5    # safety cap -- adjust to whatever you have
BATCH_SIZE = 32               # RTX 8000 has plenty of VRAM; try 48-64 if
                               # you want to push it, drop to 16 if you see OOM
MAX_NEW_TOKENS = 400

# Set >1 to sample multiple completions per question (helps recover
# questions the teacher gets wrong on the first try). With this much
# compute, 2-3 is very affordable. Set to 1 for max throughput / simplicity.
SAMPLES_PER_QUESTION = 1

OUTPUT_PATH = "data_generation/distill_correct.jsonl"      # verified-correct traces
REJECTED_PATH = "data_generation/distill_rejected.jsonl"   # wrong answers, for inspection


def build_prompt(question):
    return (
        "Solve the following math problem step by step.\n\n"
        "At the end, give your final numerical answer "
        "in this exact format:\n"
        "Answer: <number>\n\n"
        f"Problem:\n{question}"
    )


def extract_answer(text):
    patterns = [
        r"(?:Answer|Final Answer)\s*[:=]\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)",
        r"\\boxed\{\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)\s*\}",
        r"(?:the\s+)?(?:final\s+)?answer\s+is\s+\$?\s*(-?\d[\d,]*(?:\.\d+)?)",
    ]
    for pat in patterns:
        matches = re.findall(pat, text, re.IGNORECASE)
        if matches:
            return matches[-1].replace(",", "")
    return None


def load_already_done(path):
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done.add(rec["index"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading teacher model in fp16 (full precision, no quantization)...")
    model = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL,
        dtype=torch.float16,
        attn_implementation="sdpa",   # Turing has no flash-attention-2 support
    )
    model.to("cuda")
    model.eval()
    print("Teacher model loaded. VRAM used:",
          f"{torch.cuda.memory_allocated() / 1e9:.1f} GB")

    print("Loading GSM8K train set...")
    dataset = load_dataset("openai/gsm8k", "main")["train"].select(range(NUM_QUESTIONS))

    already_done = load_already_done(OUTPUT_PATH) | load_already_done(REJECTED_PATH)
    print(f"Already processed: {len(already_done)} (resuming, will skip these)")

    remaining_indices = [i for i in range(len(dataset)) if i not in already_done]
    print(f"Remaining to process: {len(remaining_indices)}")

    start_time = time.time()
    deadline = start_time + TIME_BUDGET_MINUTES * 60

    out_f = open(OUTPUT_PATH, "a")
    rej_f = open(REJECTED_PATH, "a")

    correct_count = 0
    processed_count = 0

    for batch_start in range(0, len(remaining_indices), BATCH_SIZE):
        if time.time() >= deadline:
            print(f"\nTime budget ({TIME_BUDGET_MINUTES} min) reached. Stopping.")
            break

        batch_indices = remaining_indices[batch_start:batch_start + BATCH_SIZE]
        questions = [dataset[i]["question"] for i in batch_indices]
        gt_answers = [
            dataset[i]["answer"].split("####")[-1].strip() for i in batch_indices
        ]

        messages_batch = [
            [{"role": "user", "content": build_prompt(q)}] for q in questions
        ]
        prompts = [
            tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages_batch
        ]

        inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(model.device)

        for sample_idx in range(SAMPLES_PER_QUESTION):
            do_sample = SAMPLES_PER_QUESTION > 1
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=do_sample,
                    temperature=0.7 if do_sample else None,
                    top_p=0.9 if do_sample else None,
                    pad_token_id=tokenizer.pad_token_id,
                )

            for j, idx in enumerate(batch_indices):
                # Only record each question once it's confirmed correct
                # (skip re-checking ones already written this run)
                generated_tokens = outputs[j][inputs["input_ids"].shape[1]:]
                response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

                predicted = extract_answer(response)
                is_correct = predicted is not None and predicted == gt_answers[j]

                record = {
                    "index": idx,
                    "question": questions[j],
                    "ground_truth": gt_answers[j],
                    "predicted": predicted,
                    "teacher_response": response,
                }

                if is_correct:
                    out_f.write(json.dumps(record) + "\n")
                    correct_count += 1
                    break  # got a correct one for this question, move on
                elif sample_idx == SAMPLES_PER_QUESTION - 1:
                    # exhausted all samples without a correct answer
                    rej_f.write(json.dumps(record) + "\n")

            processed_count = min(processed_count + len(batch_indices), len(remaining_indices))

        out_f.flush()
        rej_f.flush()

        elapsed_min = (time.time() - start_time) / 60
        remaining_min = TIME_BUDGET_MINUTES - elapsed_min
        rate = correct_count / processed_count * 100 if processed_count else 0
        print(
            f"Processed {processed_count}/{len(remaining_indices)} | "
            f"Correct: {correct_count} ({rate:.1f}%) | "
            f"Elapsed: {elapsed_min:.1f}min | Remaining budget: {remaining_min:.1f}min"
        )

    out_f.close()
    rej_f.close()

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"Verified-correct traces saved to: {OUTPUT_PATH} ({correct_count} examples)")
    print(f"Rejected (wrong-answer) traces saved to: {REJECTED_PATH}")
    print("Re-run this script anytime to resume and add more examples.")


if __name__ == "__main__":
    main()
