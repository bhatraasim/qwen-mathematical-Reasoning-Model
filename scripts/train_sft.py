import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from peft import LoraConfig

from trl import (
    SFTTrainer,
    SFTConfig
)

from data_generation.prepare_tagged_dataset import prepare_tagged_dataset


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def main():

    print("=" * 70)
    print("DISTILLED SFT TRAINING (tagged <think>/<answer> format)")
    print("=" * 70)


    # ------------------------------------------------------------
    # 1. Check GPU
    # ------------------------------------------------------------

    print("\n[1] Checking GPU...")

    if torch.cuda.is_available():
        device = "cuda"
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        device = "cpu"
        print("WARNING: CUDA is not available.")


    # ------------------------------------------------------------
    # 2. Load tokenizer
    # ------------------------------------------------------------

    print("\n[2] Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        padding_side="right"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Tokenizer loaded.")


    # ------------------------------------------------------------
    # 3. Load model
    # ------------------------------------------------------------

    print("\n[3] Loading model...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16 if device == "cuda" else torch.float32
    )

    model.to(device)

    print("Model loaded.")
    print(
        "Model device:",
        next(model.parameters()).device
    )


    # ------------------------------------------------------------
    # 4. Configure LoRA
    # ------------------------------------------------------------

    print("\n[4] Configuring LoRA...")

    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM"
    )

    print("LoRA configuration created.")
    print("Rank (r):", lora_config.r)
    print("Alpha:", lora_config.lora_alpha)
    print("Dropout:", lora_config.lora_dropout)
    print("Target modules:", lora_config.target_modules)


    # ------------------------------------------------------------
    # 5. Load tagged distillation dataset
    # ------------------------------------------------------------

    print("\n[5] Loading tagged distillation dataset...")

    train_dataset = prepare_tagged_dataset()

    print("Training examples:", len(train_dataset))

    print("\nFirst example (assistant turn):")
    print(train_dataset[0]["messages"][-1]["content"][:1000])


    # ------------------------------------------------------------
    # 6. Training configuration
    # ------------------------------------------------------------

    print("\n[6] Creating training configuration...")

    training_args = SFTConfig(
        output_dir="outputs/sft_distilled_tagged",

        num_train_epochs=1,

        per_device_train_batch_size=4,

        gradient_accumulation_steps=4,

        learning_rate=1e-4,

        logging_steps=10,

        save_steps=200,

        save_total_limit=2,

        fp16=True,

        report_to="none",

        max_length=1024,   # up from 512 -- tagged teacher traces run longer

        packing=False,
    )


    # ------------------------------------------------------------
    # 7. Create SFT trainer
    # ------------------------------------------------------------

    print("\n[7] Creating SFT trainer...")

    trainer = SFTTrainer(
        model=model,
        args=training_args,

        train_dataset=train_dataset,

        processing_class=tokenizer,

        peft_config=lora_config
    )

    print("SFT trainer created.")


    # ------------------------------------------------------------
    # 8. Start training
    # ------------------------------------------------------------

    print("\n[8] Starting distilled SFT training...")
    print("=" * 70)

    trainer.train()

    print("\nDistilled SFT training complete!")


    # ------------------------------------------------------------
    # 9. Save model
    # ------------------------------------------------------------

    print("\n[9] Saving distilled SFT model...")

    trainer.save_model("outputs/sft_distilled_tagged")

    tokenizer.save_pretrained("outputs/sft_distilled_tagged")

    print("Model saved to: outputs/sft_distilled_tagged")


if __name__ == "__main__":
    main()