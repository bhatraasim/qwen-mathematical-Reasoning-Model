"""
Build the SFT dataset from distill_correct.jsonl, wrapping each teacher
trace in <think>...</think><answer>...</answer> tags.

WHY TAGS IN SFT:
GRPO's format reward can only reinforce a behaviour the policy already
produces sometimes. If the SFT model never emits tags, the format reward
is ~0 for every sample in the group, contributes no gradient signal, and
the model has to stumble into the format by chance. Teaching the format
in SFT (cheap, supervised) means GRPO starts at ~100% format compliance
and can spend its whole budget on correctness instead of syntax.

The teacher trace becomes the <think> content; the verified ground-truth
number becomes the <answer> content.
"""

import json
import re
import statistics

from datasets import Dataset

DISTILL_PATH = "data_generation/distill_correct.jsonl"   # adjust to your actual project layout

MIN_RESPONSE_TOKENS = 60    # below this, the trace is too terse to teach reasoning
MAX_RESPONSE_TOKENS = 600   # near max_new_tokens -> likely a repetition loop
# (named _RESPONSE_ deliberately -- distinct from max_length in SFTConfig,
#  which caps the FULL tokenized sequence: system + user + this response)

SYSTEM_PROMPT = (
    "You are a helpful mathematical reasoning assistant. "
    "Solve the problem step by step. "
    "Put all reasoning inside <think> and </think>. "
    "Put only the final numerical answer inside <answer> and </answer>."
)


def build_user_prompt(question):
    return f"Solve the following math problem.\n\nProblem:\n{question}"


def strip_boxed_tail(text):
    """
    The teacher ends with something like '... is \\boxed{72}.' -- that
    final sentence is redundant once the answer lives in <answer> tags,
    and leaving it in teaches the model two competing answer formats.
    Trim the trailing boxed sentence if present.
    """
    text = text.rstrip()
    # cut from the last sentence that contains \boxed{...}
    match = None
    for m in re.finditer(r"\\boxed\{[^}]*\}", text):
        match = m
    if match is None:
        return text
    # walk back to the start of that sentence/line
    start = max(text.rfind("\n", 0, match.start()), text.rfind(". ", 0, match.start()))
    if start == -1:
        return text
    return text[:start].rstrip()


def prepare_tagged_dataset(verbose=True, strip_boxed=True):
    records = [json.loads(line) for line in open(DISTILL_PATH)]
    total = len(records)

    kept, too_short, too_long = [], 0, 0
    for r in records:
        n = r.get("response_tokens")
        if n is not None and n < MIN_RESPONSE_TOKENS:
            too_short += 1
            continue
        if n is not None and n > MAX_RESPONSE_TOKENS:
            too_long += 1
            continue
        kept.append(r)

    examples = []
    for r in kept:
        reasoning = r["teacher_response"]
        if strip_boxed:
            reasoning = strip_boxed_tail(reasoning)

        assistant_content = (
            f"<think>\n{reasoning.strip()}\n</think>\n"
            f"<answer>{r['ground_truth']}</answer>"
        )

        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(r["question"])},
                {"role": "assistant", "content": assistant_content},
            ]
        })

    if verbose:
        lengths = [r["response_tokens"] for r in kept if r.get("response_tokens")]
        print(f"Loaded {total} verified traces")
        print(f"  dropped too short (<{MIN_RESPONSE_TOKENS} tok): {too_short}")
        print(f"  dropped too long  (>{MAX_RESPONSE_TOKENS} tok): {too_long}")
        print(f"  kept: {len(examples)}")
        if lengths:
            print(f"  token length -- mean {statistics.mean(lengths):.0f}, "
                  f"median {statistics.median(lengths):.0f}, max {max(lengths)}")

    return Dataset.from_list(examples)


if __name__ == "__main__":
    ds = prepare_tagged_dataset()
    print("\n--- Example assistant turn ---")
    print(ds[0]["messages"][-1]["content"][:600])
    print("   ...")
    print(ds[0]["messages"][-1]["content"][-200:])