"""
Compare two evaluation runs (JSON files written by evaluate_all.py) on the
SAME questions, without re-running any model.

Why a paired test: both runs answer the same 1,319 questions, so the right
question isn't "do the two accuracy confidence intervals overlap?" but
"on the questions where the two models DISAGREE, does one win more often
than chance would explain?" That is McNemar's test, and it is much more
sensitive than eyeballing overlapping intervals.

Why --flexible: the strict extractor only accepts a bare number inside
\\boxed{...} or <answer>...</answer>. A correct answer written as
\\boxed{18 \\text{ dollars}} or \\boxed{\\$18} gets scored as a miss. That
penalises a model that never learned the tag format (the base model) more
than one that did. --flexible re-scores the SAVED raw responses with a
lenient extractor (the same idea as lm-eval-harness's "flexible-extract"),
so you can report strict and flexible numbers side by side.

Usage:
  python compare_runs.py eval_base_legacy.json eval_sft_distilled_tagged_tagged.json
  python compare_runs.py eval_base_legacy.json eval_sft_distilled_tagged_tagged.json --flexible
"""

import argparse
import json
import math
import re

NUM = r"-?\d[\d,]*(?:\.\d+)?"


def _first_number(s):
    m = re.search(NUM, s)
    return m.group(0).replace(",", "") if m else None


def _last_boxed_content(text):
    """Content of the last \\boxed{...}, with proper brace matching."""
    start = text.rfind("\\boxed{")
    if start == -1:
        return None
    i = start + len("\\boxed{")
    depth, out = 1, []
    while i < len(text) and depth > 0:
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(ch)
        i += 1
    return "".join(out)


def flexible_extract(text):
    tags = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if tags:
        n = _first_number(tags[-1])
        if n:
            return n

    boxed = _last_boxed_content(text)
    if boxed:
        n = _first_number(boxed)
        if n:
            return n

    m = re.findall(rf"(?:Answer|Final Answer)\s*[:=]\s*\$?\s*({NUM})", text, re.IGNORECASE)
    if m:
        return m[-1].replace(",", "")
    m = re.findall(rf"answer\s+is\s+\$?\s*({NUM})", text, re.IGNORECASE)
    if m:
        return m[-1].replace(",", "")

    # last resort: the last number anywhere in the response
    nums = re.findall(NUM, text)
    return nums[-1].replace(",", "") if nums else None


def numeric_equal(pred, actual):
    try:
        return abs(float(pred) - float(actual)) < 1e-6
    except (TypeError, ValueError):
        return False


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value from the discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def load(path, flexible):
    with open(path) as f:
        records = json.load(f)
    correct = {}
    for r in records:
        if flexible:
            correct[r["index"]] = numeric_equal(flexible_extract(r["response"]), r["actual"])
        else:
            correct[r["index"]] = bool(r["correct"])
    return correct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_a")
    parser.add_argument("run_b")
    parser.add_argument("--flexible", action="store_true",
                        help="re-score saved responses with a lenient extractor")
    args = parser.parse_args()

    a = load(args.run_a, args.flexible)
    b = load(args.run_b, args.flexible)
    shared = sorted(set(a) & set(b))
    n = len(shared)

    acc_a = sum(a[i] for i in shared) / n
    acc_b = sum(b[i] for i in shared) / n

    only_a = [i for i in shared if a[i] and not b[i]]
    only_b = [i for i in shared if b[i] and not a[i]]
    both = sum(1 for i in shared if a[i] and b[i])
    neither = sum(1 for i in shared if not a[i] and not b[i])

    p = mcnemar_exact(len(only_a), len(only_b))

    mode = "flexible" if args.flexible else "strict (as saved)"
    print(f"Scoring mode: {mode}")
    print(f"Questions compared: {n}\n")
    print(f"A = {args.run_a}:  {acc_a * 100:.2f}%")
    print(f"B = {args.run_b}:  {acc_b * 100:.2f}%")
    print(f"Difference (B - A): {(acc_b - acc_a) * 100:+.2f} points\n")
    print(f"Both correct:        {both}")
    print(f"Neither correct:     {neither}")
    print(f"Only A correct:      {len(only_a)}")
    print(f"Only B correct:      {len(only_b)}")
    print(f"\nMcNemar exact p-value: {p:.4f}")
    if p < 0.05:
        print("-> the difference is unlikely to be chance (p < 0.05)")
    else:
        print("-> cannot rule out chance at the 5% level")

    print("\nFirst few question indices where only A / only B was right "
          "(read these raw responses before drawing conclusions):")
    print("  only A:", only_a[:10])
    print("  only B:", only_b[:10])


if __name__ == "__main__":
    main()