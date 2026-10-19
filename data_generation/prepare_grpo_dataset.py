"""
Build the GRPO prompt dataset: just prompts + ground-truth answers,
no reasoning traces needed (the policy generates those itself).

CRITICAL: system prompt and user prompt format must match SFT exactly,
or the policy starts GRPO already off-distribution from what it was
trained on. Imported directly from prepare_tagged_dataset.py rather
than redefined here, so the two can never drift apart.
"""

from datasets import Dataset, load_dataset

from data_generation.prepare_tagged_dataset import SYSTEM_PROMPT, build_user_prompt

GSM8K_SPLIT = "train"   # never "test" -- keep GRPO on the same split SFT used


def prepare_grpo_dataset(num_examples=None):
    dataset = load_dataset("openai/gsm8k", "main")[GSM8K_SPLIT]
    if num_examples is not None:
        dataset = dataset.select(range(num_examples))

    examples = []
    for ex in dataset:
        ground_truth = ex["answer"].split("####")[-1].strip()
        examples.append({
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(ex["question"])},
            ],
            "ground_truth": ground_truth,
        })

    print(f"GRPO prompt dataset: {len(examples)} examples")
    return Dataset.from_list(examples)


if __name__ == "__main__":
    ds = prepare_grpo_dataset(num_examples=20)
    print(ds[0])