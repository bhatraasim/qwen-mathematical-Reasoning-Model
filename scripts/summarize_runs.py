"""
Summarize every evaluation run in one go and save the result.

Reads all eval_*.json files (written by evaluate_all.py), and produces:
  1. A table of accuracy per run (strict and flexible scoring, with 95% CIs),
     plus format-compliance and truncation stats.
  2. A paired comparison of every run against a reference run (McNemar exact
     test on the questions where the two runs disagree).

Writes results/summary.md and results/summary.csv, and prints the markdown.
No model is loaded -- this only re-reads the saved raw responses.

Usage (from the repository root):
  python scripts/summarize_runs.py
  python scripts/summarize_runs.py --reference sft_distilled_tagged_tagged
  python scripts/summarize_runs.py --files results/evaluations/eval_base_legacy.json results/evaluations/eval_checkpoint-200_tagged.json
  python scripts/summarize_runs.py --out results/my_results
"""

import argparse
import csv
import glob
import json
import math
import os
import re

# Works whether run as `python scripts/summarize_runs.py`
# or as `python -m scripts.summarize_runs`.
try:
    from compare_runs import flexible_extract, mcnemar_exact, numeric_equal
except ImportError:
    from scripts.compare_runs import flexible_extract, mcnemar_exact, numeric_equal


def wilson_interval(correct, total, z=1.96):
    if total == 0:
        return 0.0, 0.0
    p = correct / total
    denom = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return centre - margin, centre + margin


def run_name(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem[len("eval_"):] if stem.startswith("eval_") else stem


def manifest_for(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    return os.path.join(os.path.dirname(path), f"manifest_{stem}.json")


def load_manifest(path):
    manifest_path = manifest_for(path)
    if not os.path.exists(manifest_path):
        return {}
    with open(manifest_path) as f:
        return json.load(f)


def sort_key(name):
    """Pipeline order: base, SFT, GRPO checkpoints by step, then anything else."""
    m = re.search(r"checkpoint-(\d+)", name)
    if name.startswith("base"):
        return (0, 0, name)
    if m:
        return (2, int(m.group(1)), name)
    if "sft" in name:
        return (1, 0, name)
    return (3, 0, name)


def load_run(path):
    with open(path) as f:
        records = json.load(f)
    manifest = load_manifest(path)

    strict, flex = {}, {}
    well_formed = token_limit = no_match = 0
    tokens = []
    for r in records:
        i = r["index"]
        strict[i] = bool(r["correct"])
        flex[i] = numeric_equal(flexible_extract(r["response"]), r["actual"])
        well_formed += bool(r["well_formed"])
        token_limit += bool(r["hit_token_limit"])
        no_match += r["pattern"] == "no_match"
        tokens.append(r["response_tokens"])

    return {
        "adapter": (manifest.get("adapter") or "base") if manifest else "?",
        "prompt": manifest.get("prompt_style", "?"),
        "split": manifest.get("split", "?"),
        "n": len(records),
        "strict": strict,
        "flex": flex,
        "well_formed": well_formed,
        "token_limit": token_limit,
        "no_match": no_match,
        "mean_tokens": sum(tokens) / len(tokens),
    }


def paired(ref_run, run, key):
    """Compare `run` against `ref_run` on shared questions for one scoring mode."""
    a, b = ref_run[key], run[key]
    shared = sorted(set(a) & set(b))
    only_ref = sum(1 for i in shared if a[i] and not b[i])
    only_run = sum(1 for i in shared if b[i] and not a[i])
    delta = (sum(b[i] for i in shared) - sum(a[i] for i in shared)) / len(shared) * 100
    return delta, mcnemar_exact(only_ref, only_run), only_ref, only_run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="*", default=None,
                        help="eval JSON files (default: every results/evaluations/eval_*.json file)")
    parser.add_argument("--reference", default=None,
                        help="run name to compare the others against (default: the SFT run)")
    parser.add_argument("--out", default="results/summary",
                        help="output file prefix; writes <prefix>.md and <prefix>.csv")
    args = parser.parse_args()

    files = args.files or sorted(glob.glob("results/evaluations/eval_*.json"))
    if not files:
        raise SystemExit("No results/evaluations/eval_*.json files found. Run evaluation first.")

    runs = {run_name(f): load_run(f) for f in files}
    names = sorted(runs, key=sort_key)

    ref = args.reference
    if ref is None:
        ref = next((n for n in names if "sft" in n), names[0])
    elif ref not in runs:
        raise SystemExit(f"--reference must be one of: {names}")

    # ---- per-run table ----
    rows = []
    for name in names:
        r = runs[name]
        n = r["n"]
        sc, fc = sum(r["strict"].values()), sum(r["flex"].values())
        s_lo, s_hi = wilson_interval(sc, n)
        f_lo, f_hi = wilson_interval(fc, n)
        rows.append({
            "run": name,
            "adapter": r["adapter"],
            "prompt": r["prompt"],
            "split": r["split"],
            "n": n,
            "strict_acc": sc / n * 100,
            "strict_ci_lo": s_lo * 100,
            "strict_ci_hi": s_hi * 100,
            "flex_acc": fc / n * 100,
            "flex_ci_lo": f_lo * 100,
            "flex_ci_hi": f_hi * 100,
            "well_formed_pct": r["well_formed"] / n * 100,
            "token_limit_pct": r["token_limit"] / n * 100,
            "no_match": r["no_match"],
            "mean_tokens": r["mean_tokens"],
        })

    # ---- provenance checks ----
    unknown = [r["run"] for r in rows if r["adapter"] == "?"]
    if unknown:
        print(
            "WARNING: no manifest found for "
            f"{', '.join(unknown)}; Adapter/Prompt/Split show '?'."
        )
    prompts = sorted({r["prompt"] for r in rows if r["prompt"] != "?"})
    splits = sorted({r["split"] for r in rows if r["split"] != "?"})
    sizes = sorted({r["n"] for r in rows})
    if len(prompts) > 1:
        print(f"WARNING: runs use different prompt styles: {prompts}; accuracy is not directly comparable.")
    if len(splits) > 1:
        print(f"WARNING: runs use different splits: {splits}; accuracy is not directly comparable.")
    if len(sizes) > 1:
        print(f"WARNING: runs use different example counts: {sizes}; compare with care.")

    # ---- paired comparison vs reference ----
    ref_run = runs[ref]
    paired_rows = []
    for name in names:
        if name == ref:
            continue
        ds, ps, _, _ = paired(ref_run, runs[name], "strict")
        df, pf, only_ref, only_run = paired(ref_run, runs[name], "flex")
        paired_rows.append((name, ds, ps, df, pf, only_ref, only_run))

    # ---- markdown ----
    md = []
    md.append("# Evaluation summary\n")
    md.append("## Accuracy by run\n")
    md.append("| Run | Adapter | Prompt | Split | N | Strict acc | Strict 95% CI "
              "| Flexible acc | Flexible 95% CI | Well-formed | Hit token limit | no_match | Mean tokens |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        md.append(
            f"| {r['run']} | {r['adapter']} | {r['prompt']} | {r['split']} | {r['n']} "
            f"| {r['strict_acc']:.2f}% "
            f"| {r['strict_ci_lo']:.1f}-{r['strict_ci_hi']:.1f}% "
            f"| {r['flex_acc']:.2f}% "
            f"| {r['flex_ci_lo']:.1f}-{r['flex_ci_hi']:.1f}% "
            f"| {r['well_formed_pct']:.1f}% | {r['token_limit_pct']:.1f}% "
            f"| {r['no_match']} | {r['mean_tokens']:.0f} |"
        )

    md.append(f"\n## Paired comparison vs `{ref}`\n")
    md.append("| Run | Change (strict) | p (strict) | Change (flexible) | p (flexible) "
              "| Only reference right | Only this run right |")
    md.append("|---|---|---|---|---|---|---|")
    for name, ds, ps, df, pf, only_ref, only_run in paired_rows:
        md.append(
            f"| {name} | {ds:+.2f} pts | {ps:.4f} | {df:+.2f} pts | {pf:.4f} "
            f"| {only_ref} | {only_run} |"
        )

    md.append(
        "\nNotes: *strict* uses the extractor stored in each JSON; *flexible* re-scores the "
        "saved responses with a lenient extractor (units, dollar signs, last-number fallback). "
        "The last two columns use flexible scoring. p-values are exact McNemar tests on the "
        "questions where the two runs disagree; they are not corrected for multiple "
        "comparisons, so treat p just under 0.05 with caution. "
        "Adapter/Prompt/Split come from the `manifest_<run>.json` sidecar written by "
        "evaluate_all.py; '?' means the run predates the manifest."
    )

    md_text = "\n".join(md) + "\n"

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(f"{args.out}.md", "w") as f:
        f.write(md_text)

    with open(f"{args.out}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})

    print(md_text)
    print(f"Saved: {args.out}.md and {args.out}.csv")


if __name__ == "__main__":
    main()
