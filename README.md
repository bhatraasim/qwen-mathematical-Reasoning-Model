# Qwen Math Reasoning: SFT and GRPO on GSM8K

This repository contains the code and evaluation notes for supervised fine-tuning (SFT) and GRPO experiments on GSM8K. The base model is `Qwen/Qwen2.5-0.5B-Instruct`. The SFT stage teaches `<think>...</think><answer>...</answer>` output; GRPO then optimizes answer correctness while rewarding valid formatting.

## Repository layout

```text
archive/           Retained legacy scripts, not part of the supported pipeline
data_generation/   Dataset generation and preparation code
results/           Experiment notes and compact result summaries
scripts/           Training, evaluation, reward, and analysis scripts
outputs/           Local adapters and checkpoints (ignored by Git)
```

Generated training data, raw evaluation response dumps, model adapters, and checkpoints are intentionally kept out of this Git repository. They are reproducible or large artifacts; publish model files through a model host such as Hugging Face, or use Git LFS if you specifically need them in GitHub. Check the licenses and terms for the base model, datasets, and generated traces before redistributing them.

## Setup

Use Python 3.10 or newer and install PyTorch for your hardware, then install the project libraries:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch
python -m pip install -r requirements.txt
```

Install a PyTorch build that matches your accelerator. The remaining dependency versions are pinned in `requirements.txt` to match the versions recorded with the saved model artifacts. Record your PyTorch/CUDA versions when reproducing a run, since those depend on your hardware.

The scripts load GSM8K (`openai/gsm8k`) and the base model from Hugging Face, so the first run needs network access and accepts their respective terms.

## Run the pipeline

1. Generate teacher traces (this writes local JSONL files under `data_generation/`):

   ```bash
   python -m data_generation.generate_data
   ```

2. Train the tagged SFT adapter:

   ```bash
   python -m scripts.train_sft
   ```

3. Train GRPO from the SFT adapter:

   ```bash
   python -m scripts.train_grpo
   ```

4. Evaluate an adapter or the base model. Evaluation JSON files are written to `results/evaluations/`:

   ```bash
   python -m scripts.evaluate_all --adapter outputs/grpo_tagged
   python -m scripts.evaluate_all --adapter outputs/sft_distilled_tagged
   python -m scripts.evaluate_all --adapter none
   ```

5. Summarize saved evaluation files:

   ```bash
   python -m scripts.summarize_runs
   ```

The training scripts save local checkpoints under `outputs/`. Keep only checkpoints you need; intermediate checkpoints include optimizer and scheduler state and take considerably more space than a LoRA adapter.

## Current reported evaluation

The checked-in summary in [results/summary.md](results/summary.md) compares four saved runs on the 1,319-example GSM8K test split. The `checkpoint-200_tagged` run reached 44.96% strict accuracy and 45.11% flexible accuracy, compared with 42.15% and 42.23% for the SFT run. These figures are the experiment's recorded results, not a claim of independent reproduction.

## Notes

- `data_generation/prepare_tagged_dataset.py` defines the shared tagged prompt used by SFT, GRPO, and evaluation. Keep prompt changes consistent with that source.
- `data_generation/generate_data.py` is the current resumable trace generator. Earlier scripts are kept in `archive/legacy/` for reference.
- The files in `results/` describe individual runs and the aggregate evaluation summary.
- This repository does not yet include a license. Add one that reflects the intended reuse before publishing it publicly; also check the terms for Qwen and GSM8K separately.
# qwen-mathematical-Reasoning-Model
