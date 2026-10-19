import torch
import re

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from pathlib import Path

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
BASE_DIR = Path(__file__).resolve().parent.parent

ADAPTER = BASE_DIR / "outputs" / "grpo_tagged"


def extract_answer(text):
    patterns = [
        r"<answer>\s*(-?\d[\d,]*(?:\.\d+)?)\s*</answer>",
        r"\\boxed\{\s*(-?\d[\d,]*(?:\.\d+)?)\s*\}",
        r"(?:Answer|Final Answer)\s*[:=]\s*(-?\d[\d,]*(?:\.\d+)?)",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].replace(",", "")

    return None


# -----------------------------
# Load model
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    padding_side="left"
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float16 if device == "cuda" else torch.float32
)

print("Loading GRPO adapter...")
model = PeftModel.from_pretrained(model, ADAPTER)

model.to(device)
model.eval()

print("Model loaded.")
print("Device:", device)


# -----------------------------
# Load GSM8K test set
# -----------------------------
dataset = load_dataset("openai/gsm8k", "main")["test"]

correct = 0
total = len(dataset)

print("\nEvaluating", total, "questions...\n")


# -----------------------------
# Evaluation
# -----------------------------
for i, example in enumerate(dataset):

    question = example["question"]

    actual = example["answer"].split("####")[-1].strip()

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful mathematical reasoning assistant. "
                "Solve the problem step by step. "
                "Put all reasoning inside <think> and </think>. "
                "Put only the final numerical answer inside "
                "<answer> and </answer>."
            ),
        },
        {
            "role": "user",
            "content": (
                "Solve the following math problem.\n\n"
                f"Problem:\n{question}"
            ),
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=640,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )

    generated = output[0][inputs["input_ids"].shape[1]:]

    response = tokenizer.decode(
        generated,
        skip_special_tokens=True
    )

    predicted = extract_answer(response)

    try:
        is_correct = (
            predicted is not None
            and abs(float(predicted) - float(actual)) < 1e-6
        )
    except:
        is_correct = False

    if is_correct:
        correct += 1

    if (i + 1) % 25 == 0:
        print(
            f"{i + 1}/{total} | "
            f"Accuracy: {correct / (i + 1) * 100:.2f}%"
        )


# -----------------------------
# Final result
# -----------------------------
print("\n" + "=" * 50)
print("GRPO EVALUATION")
print("=" * 50)

print(f"Correct:  {correct}/{total}")
print(f"Accuracy: {correct / total * 100:.2f}%")