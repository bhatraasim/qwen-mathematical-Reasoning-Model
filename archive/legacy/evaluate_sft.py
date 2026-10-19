import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import re
from datasets import load_dataset

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

SFT_PATH = "outputs/sft_distilled_tagged"

# Must match prepare_tagged_dataset.py EXACTLY -- this is the recurring
# bug in this whole project, so keep these two files in lockstep.
SYSTEM_PROMPT = (
    "You are a helpful mathematical reasoning assistant. "
    "Solve the problem step by step. "
    "Put all reasoning inside <think> and </think>. "
    "Put only the final numerical answer inside <answer> and </answer>."
)


def build_user_prompt(question):
    return (
        "Solve the following math problem.\n\n"
        f"Problem:\n{question}"
    )


def main():

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        SFT_PATH,
        padding_side="left"
    )

    print("Loading base model...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16 if device == "cuda" else torch.float32
    )

    print("Loading SFT LoRA adapter...")

    model = PeftModel.from_pretrained(
        base_model,
        SFT_PATH
    )

    model.to(device)
    model.eval()

    print("SFT model loaded successfully!")
    print("Device:", device)

    print("Loading GSM8K test dataset...")

    dataset = load_dataset("openai/gsm8k", "main")
    test_dataset = dataset["test"]

    print("Test examples:", len(test_dataset))

    def extract_predicted_answer(text):
        # <answer>N</answer> checked FIRST -- this is the format the
        # model was actually trained to produce.
        matches = re.findall(
            r"<answer>\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)\s*</answer>",
            text,
            re.IGNORECASE
        )
        if matches:
            return matches[-1].replace(",", "")

        # Fallbacks below catch a model that reverts to an older
        # format under distribution shift -- keep them as a safety
        # net, but <answer> should dominate if training worked.
        matches = re.findall(
            r"\\boxed\{\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)\s*\}",
            text
        )
        if matches:
            return matches[-1].replace(",", "")

        matches = re.findall(
            r"(?:Answer|Final Answer)\s*[:=]\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)",
            text,
            re.IGNORECASE
        )
        if matches:
            return matches[-1].replace(",", "")

        matches = re.findall(
            r"(?:the\s+)?(?:final\s+)?answer\s+is\s+\$?\s*(-?\d[\d,]*(?:\.\d+)?)",
            text,
            re.IGNORECASE
        )
        if matches:
            return matches[-1].replace(",", "")

        return None

    BATCH_SIZE = 32

    correct = 0
    total = len(test_dataset)

    for start in range(0, total, BATCH_SIZE):

        end = min(start + BATCH_SIZE, total)

        questions = [
            test_dataset[i]["question"]
            for i in range(start, end)
        ]

        # Include the SYSTEM turn -- matches training exactly.
        messages = [
            [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": build_user_prompt(question)
                }
            ]
            for question in questions
        ]

        prompts = [
            tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True
            )
            for message in messages
        ]

        inputs = tokenizer(
            prompts,
            padding=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=640,   # up from 384 -- match training trace lengths
                do_sample=False
            )

        for j in range(len(questions)):

            generated_tokens = outputs[j][inputs["input_ids"].shape[1]:]

            response = tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True
            )

            predicted = extract_predicted_answer(response)

            actual = test_dataset[start + j]["answer"].split("####")[-1].strip()

            if predicted == actual:
                correct += 1

        processed = end

        print(
            f"Progress: {processed}/{total} | "
            f"Accuracy: {correct / processed * 100:.2f}%"
        )

    accuracy = correct / total

    print("\n==============================")
    print("SFT EVALUATION RESULTS")
    print("==============================")
    print("Correct:", correct)
    print("Total:", total)
    print(f"Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()