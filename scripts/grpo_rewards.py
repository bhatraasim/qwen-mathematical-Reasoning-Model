"""
Reward functions for GRPO on GSM8K with <think>/<answer> formatting.

Design notes:

FORMAT REWARD is structural, not substring-based. A substring check
(`"<think>" in completion`) grants full reward to malformed output:
wrong tag order, duplicated tags, empty tag bodies, trailing junk.
Empty bodies matter most -- `<think></think><answer></answer>` is the
cheapest possible way to farm format reward, and policies find it fast.

REWARD MAGNITUDES: correctness must dominate. If format reward is a
large fraction of the achievable total, the policy optimises syntax over
solving. Roughly 1.0 correctness : 0.2 format is a sane starting ratio --
format is a nudge toward parseable output, not a goal in itself.
"""

import re

# Anchored, single-occurrence, non-empty bodies, nothing trailing.
FORMAT_RE = re.compile(
    r"^\s*<think>(?P<think>.*?)</think>\s*<answer>(?P<answer>.*?)</answer>\s*$",
    re.DOTALL,
)

NUMBER_RE = re.compile(r"^-?\d[\d,]*(?:\.\d+)?$")

MIN_THINK_CHARS = 30   # a <think> body shorter than this isn't reasoning


def _parse(completion):
    """Return (think_body, answer_body) or None if structurally invalid."""
    m = FORMAT_RE.match(completion)
    if not m:
        return None
    # reject duplicated tags hiding inside the bodies
    think, answer = m.group("think"), m.group("answer")
    for tag in ("<think>", "</think>", "<answer>", "</answer>"):
        if tag in think or tag in answer:
            return None
    return think.strip(), answer.strip()


def format_reward(completions, **kwargs):
    """
    0.2 for well-formed output. Partial credit (0.1) for correct tag
    structure with a too-short think body, so the policy still gets a
    gradient toward the right shape without being paid full price for
    an empty reasoning block.
    """
    rewards = []
    for completion in completions:
        text = completion[0]["content"] if isinstance(completion, list) else completion
        parsed = _parse(text)
        if parsed is None:
            rewards.append(0.0)
            continue
        think, answer = parsed
        if not answer:
            rewards.append(0.0)
        elif len(think) < MIN_THINK_CHARS:
            rewards.append(0.1)
        else:
            rewards.append(0.2)
    return rewards


def answer_parseable_reward(completions, **kwargs):
    """
    Small extra nudge (0.1) when <answer> holds a bare number rather than
    a sentence. Keeps the output machine-checkable without needing the
    answer to be correct.
    """
    rewards = []
    for completion in completions:
        text = completion[0]["content"] if isinstance(completion, list) else completion
        parsed = _parse(text)
        if parsed is None:
            rewards.append(0.0)
            continue
        _, answer = parsed
        rewards.append(0.1 if NUMBER_RE.match(answer) else 0.0)
    return rewards


def correctness_reward(completions, ground_truth, **kwargs):
    """
    1.0 when the extracted answer matches ground truth numerically.
    This should dominate the total reward.

    `ground_truth` is passed through from your dataset column of the
    same name -- TRL forwards extra dataset columns to reward functions
    as kwargs.
    """
    rewards = []
    for completion, gt in zip(completions, ground_truth):
        text = completion[0]["content"] if isinstance(completion, list) else completion
        parsed = _parse(text)
        if parsed is None:
            # fall back to a loose extraction so a correct-but-malformed
            # answer still gets correctness credit (it just loses format reward)
            loose = re.findall(r"(-?\d[\d,]*(?:\.\d+)?)", text)
            pred = loose[-1].replace(",", "") if loose else None
        else:
            pred = parsed[1].replace(",", "")

        try:
            ok = pred is not None and abs(float(pred) - float(gt)) < 1e-6
        except ValueError:
            ok = False
        rewards.append(1.0 if ok else 0.0)
    return rewards


# Pass all three to GRPOTrainer(reward_funcs=[...]); TRL sums them.
REWARD_FUNCS = [correctness_reward, format_reward, answer_parseable_reward]