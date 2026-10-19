"""
Generate verified-correct reasoning traces on GSM8K using
Qwen2.5-Math-7B-Instruct as a teacher, for distillation into
Qwen2.5-0.5B-Instruct.

Key points:
  - SYSTEM_PROMPT is required for Qwen2.5-Math-Instruct to behave
    correctly (without it, correct-rate collapsed from ~95% to ~10%).
  - Greedy first attempt, temperature-sampled retries, and each retry
    round only regenerates questions still unsolved.
  - Records token length for every trace so you can filter out
    repetition loops and suspiciously short "lucky guess" traces
    before they reach SFT.
  - Rejects record how many attempts were burned, as a difficulty signal.
  - Resumable: skips indices already present in either output file.
"""

import json
import os
import re
import time

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

TEACHER_MODEL = "Qwen/Qwen2.5-Math-7B-Instruct"

SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

# --- Budget knobs ---
NUM_QUESTIONS = 7473
TIME_BUDGET_MINUTES = None     # None = no time limit. Set a number to cap it.
BATCH_SIZE = 32
MAX_NEW_TOKENS = 640           # teacher traces run longer than GSM8K originals

SAMPLES_PER_QUESTION = 4
RETRY_TEMPERATURE = 0.7

OUTPUT_PATH = "data_generation/distill_correct.jsonl"
REJECTED_PATH = "data_generation/distill_rejected.jsonl"


def build_user_prompt(question):
    return (
        "Solve the following math problem step by step.\n\n"
        "At the end, give your final numerical answer "
        "in this exact format:\n"
        "Answer: <number>\n\n"
        f"Problem:\n{question}"
    )


def extract_answer(text):
    patterns = [
        r"\\boxed\{\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)\s*\}",
        r"(?:Answer|Final Answer)\s*[:=]\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)",
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
                    done.add(json.loads(line)["index"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def generate_batch(model, tokenizer, questions, do_sample):
    messages_batch = [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(q)},
        ]
        for q in questions
    ]
    prompts = [
        tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages_batch
    ]
    inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(model.device)

    gen_kwargs = dict(
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = RETRY_TEMPERATURE
        gen_kwargs["top_p"] = 0.9

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    results = []
    for j in range(len(questions)):
        generated_tokens = outputs[j][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        # token length of the generated portion only (excludes padding/prompt)
        n_tokens = int((generated_tokens != tokenizer.pad_token_id).sum().item())
        results.append((text, n_tokens))
    return results


def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading teacher model in fp16...")
    model = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL,
        dtype=torch.float16,
        attn_implementation="sdpa",
    )
    model.to("cuda")
    model.eval()
    print(f"Teacher loaded. VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    dataset = load_dataset("openai/gsm8k", "main")["train"].select(range(NUM_QUESTIONS))

    already_done = load_already_done(OUTPUT_PATH) | load_already_done(REJECTED_PATH)
    print(f"Already processed: {len(already_done)} (will skip)")

    remaining_indices = [i for i in range(len(dataset)) if i not in already_done]
    print(f"Remaining to process: {len(remaining_indices)}")

    start_time = time.time()
    deadline = None if TIME_BUDGET_MINUTES is None else start_time + TIME_BUDGET_MINUTES * 60

    out_f = open(OUTPUT_PATH, "a")
    rej_f = open(REJECTED_PATH, "a")

    correct_count = 0
    processed_count = 0

    for batch_start in range(0, len(remaining_indices), BATCH_SIZE):
        if deadline is not None and time.time() >= deadline:
            print(f"\nTime budget reached. Stopping.")
            break

        batch_indices = remaining_indices[batch_start:batch_start + BATCH_SIZE]
        questions = {i: dataset[i]["question"] for i in batch_indices}
        gt_answers = {
            i: dataset[i]["answer"].split("####")[-1].strip() for i in batch_indices
        }

        pending = list(batch_indices)
        last_failure = {}   # idx -> (predicted, response, n_tokens, attempts_used)

        for attempt in range(SAMPLES_PER_QUESTION):
            if not pending:
                break
            if deadline is not None and time.time() >= deadline:
                break

            do_sample = attempt > 0
            results = generate_batch(
                model, tokenizer, [questions[i] for i in pending], do_sample
            )

            still_pending = []
            for idx, (response, n_tokens) in zip(pending, results):
                predicted = extract_answer(response)
                is_correct = predicted is not None and predicted == gt_answers[idx]

                if is_correct:
                    out_f.write(json.dumps({
                        "index": idx,
                        "question": questions[idx],
                        "ground_truth": gt_answers[idx],
                        "predicted": predicted,
                        "teacher_response": response,
                        "attempt": attempt + 1,
                        "response_tokens": n_tokens,
                    }) + "\n")
                    correct_count += 1
                else:
                    last_failure[idx] = (predicted, response, n_tokens, attempt + 1)
                    still_pending.append(idx)

            pending = still_pending

        for idx in pending:
            predicted, response, n_tokens, attempts_used = last_failure.get(
                idx, (None, "", 0, 0)
            )
            rej_f.write(json.dumps({
                "index": idx,
                "question": questions[idx],
                "ground_truth": gt_answers[idx],
                "predicted": predicted,
                "teacher_response": response,
                "attempts_used": attempts_used,   # difficulty signal
                "response_tokens": n_tokens,
            }) + "\n")

        processed_count += len(batch_indices)
        out_f.flush()
        rej_f.flush()

        elapsed_min = (time.time() - start_time) / 60
        rate = correct_count / processed_count * 100 if processed_count else 0
        print(
            f"Processed {processed_count}/{len(remaining_indices)} | "
            f"Correct: {correct_count} ({rate:.1f}%) | Elapsed: {elapsed_min:.1f}min"
        )

    out_f.close()
    rej_f.close()

    print("\n" + "=" * 60)
    print(f"DONE -- {correct_count} verified traces in {OUTPUT_PATH}")
    print(f"Rejects in {REJECTED_PATH}")


if __name__ == "__main__":
    main()
