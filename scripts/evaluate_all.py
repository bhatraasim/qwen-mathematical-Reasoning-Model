
"""
Unified GSM8K evaluation for every stage of the pipeline.

Usage examples (run each as its own process, from the repository root):

  # GRPO model
  python -m scripts.evaluate_all --adapter outputs/grpo_tagged

  # SFT model (for the baseline comparison)
  python -m scripts.evaluate_all --adapter outputs/sft_distilled_tagged

  # Base model, same tagged prompt as the trained models
  python -m scripts.evaluate_all --adapter none

  # Base model with the ORIGINAL harness prompt (reproduces your ~38% number)
  python -m scripts.evaluate_all --adapter none --prompt_style legacy

  # Quick smoke test on a subset
  python -m scripts.evaluate_all --adapter outputs/grpo_tagged --num_examples 100

Every run uses greedy decoding, the same test split, and the same
extraction logic, so the numbers are directly comparable. Each run also
writes a JSON file with every raw response for manual inspection, plus a
manifest_eval_*.json sidecar recording the adapter, prompt hash, split,
decoding settings, package versions, and git commit.
"""

import argparse
import hashlib
import json
import math
import platform
import re
import subprocess
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
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
    return max(0.0, centre - margin), min(1.0, centre + margin)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def git_dirty():
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
        )
        return bool(status.strip())
    except Exception:
        return None


def package_version(name):
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def generated_length(token_ids, eos_token_id):
    """Generated tokens up to and including the first EOS, plus whether EOS was emitted."""
    ids = token_ids.tolist()
    if eos_token_id is not None and eos_token_id in ids:
        return ids.index(eos_token_id) + 1, True
    return len(ids), False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="outputs/grpo_tagged",
                        help="Path to a LoRA adapter dir, or 'none' for the base model")
    parser.add_argument("--prompt_style", choices=["tagged", "legacy"], default="tagged")
    parser.add_argument("--num_examples", type=int, default=None,
                        help="Subset size; default = full 1319-question test set")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=640)
    parser.add_argument("--out", default=None, help="Where to save raw responses (JSON)")
    parser.add_argument("--run_name", default=None,
                        help="Label for this run (used in the default output filename)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow overwriting an existing output file")
    args = parser.parse_args()

    use_adapter = args.adapter.lower() != "none"

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    default_label = "base" if not use_adapter else args.adapter.strip("/").split("/")[-1]
    run_label = args.run_name or default_label
    out_path = args.out or str(
        Path("results/evaluations") / f"eval_{run_label}_{args.prompt_style}.json"
    )
    out_stem = Path(out_path).stem
    run_name = out_stem[len("eval_"):] if out_stem.startswith("eval_") else out_stem

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    if Path(out_path).exists() and not args.overwrite:
        raise SystemExit(
            f"Refusing to overwrite existing results: {out_path}\n"
            "Use --overwrite or pass a new --out path."
        )

    print(f"Model:        {'base' if not use_adapter else args.adapter}")
    print(f"Prompt style: {args.prompt_style}")
    print(f"Run name:     {run_name}")
    print(f"Output:       {out_path}")


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
            n_tokens, ended_with_eos = generated_length(generated, tokenizer.eos_token_id)
            response = tokenizer.decode(generated, skip_special_tokens=True)

            predicted, pattern = extract_answer(response)
            actual = test_dataset[start + j]["answer"].split("####")[-1].strip()
            is_correct = numeric_equal(predicted, actual)

            is_well_formed = bool(STRICT_FORMAT_RE.match(response))
            hit_limit = (not ended_with_eos) and n_tokens >= args.max_new_tokens

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

    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)

    adapter_path = Path(args.adapter) / "adapter_model.safetensors" if use_adapter else None
    prompt_template = (
        SYSTEM_PROMPT + "\n\n" + build_user_prompt("{question}")
        if args.prompt_style == "tagged"
        else legacy_user_prompt("{question}")
    )
    manifest = {
        "run_name": run_name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "git_dirty": git_dirty(),
        "eval_script_sha256": sha256_file(Path(__file__).resolve()),
        "model": MODEL_NAME,
        "adapter": args.adapter if use_adapter else None,
        "adapter_sha256": (
            sha256_file(adapter_path)
            if adapter_path is not None and adapter_path.exists()
            else None
        ),
        "prompt_style": args.prompt_style,
        "prompt_sha256": hashlib.sha256(prompt_template.encode()).hexdigest(),
        "dataset": "openai/gsm8k",
        "split": "test",
        "num_examples": total,
        "num_examples_requested": args.num_examples,
        "decoding": "greedy",
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "dtype": str(next(model.parameters()).dtype),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": package_version("transformers"),
            "peft": package_version("peft"),
            "datasets": package_version("datasets"),
            "trl": package_version("trl"),
        },
        "metrics": {
            "correct": correct,
            "accuracy_pct": round(accuracy * 100, 3),
            "accuracy_ci95_pct": [round(lo * 100, 3), round(hi * 100, 3)],
            "well_formed_pct": round(well_formed / total * 100, 3),
            "hit_token_limit_pct": round(truncated / total * 100, 3),
            "no_match": sum(1 for r in records if r["pattern"] == "no_match"),
            "mean_response_tokens": round(
                sum(r["response_tokens"] for r in records) / total, 1
            ),
        },
    }
    manifest_path = Path(out_path).with_name(f"manifest_{out_stem}.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

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
    if total and truncated / total > 0.05:
        print(
            f"\nWARNING: {truncated}/{total} responses hit the "
            f"{args.max_new_tokens}-token limit. Accuracy is not meaningful at this "
            "truncation rate; increase --max_new_tokens (640+ for the tagged format)."
        )
    print(f"Answer patterns:    {patterns_used}")
    print(f"Raw responses saved to: {out_path}")
    print(f"Manifest saved to:      {manifest_path}")


if __name__ == "__main__":
    main()
