"""
train.py — Fine-tune Microsoft Phi-2 on Medical CoT dataset using Unsloth + LoRA.

Usage:
    python train.py --hf_token <YOUR_HF_TOKEN> [--max_steps 60] [--push_to_hub]

Requirements: see requirements.txt
"""

import argparse
import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported
from huggingface_hub import login, whoami

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
MODEL_NAME       = "unsloth/phi-2"
MAX_SEQ_LENGTH   = 2048
LOAD_IN_4BIT     = True
LORA_RANK        = 16
OUTPUT_DIR       = "outputs"
DATASET_ID       = "FreedomIntelligence/medical-o1-reasoning-SFT"
DATASET_SPLIT    = "train[:500]"
HUB_REPO         = "priyanshuxsinha/phi2-medical-expert"

TRAIN_PROMPT = (
    "Below is an instruction that describes a task, paired with an input that provides further context.\n"
    "Write a response that appropriately completes the request.\n"
    "Before answering, think carefully about the question and create a step-by-step chain of thoughts "
    "to ensure a logical and accurate response.\n\n"
    "### Instruction:\n"
    "You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning.\n"
    "Please answer the following medical question.\n\n"
    "### Question:\n{question}\n\n"
    "### Response:\n\n{cot}\n\n{answer}"
)

INFERENCE_PROMPT = (
    "### Question:\n{question}\n\n"
    "### Answer:\n<think>"
)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def load_model(hf_token: str):
    """Load Phi-2 with Unsloth 4-bit optimisation."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,          # auto-detect bf16 / fp16
        load_in_4bit=LOAD_IN_4BIT,
        token=hf_token,
    )
    return model, tokenizer


def apply_lora(model):
    """Wrap the base model with LoRA adapters."""
    return FastLanguageModel.get_peft_model(
        model=model,
        r=LORA_RANK,
        target_modules=["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"],
        lora_alpha=LORA_RANK,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3047,
    )


def prepare_dataset(tokenizer):
    """Download and format the medical reasoning dataset."""
    raw = load_dataset(DATASET_ID, "en", split=DATASET_SPLIT, trust_remote_code=True)
    eos = tokenizer.eos_token

    def _format(examples):
        texts = []
        for q, cot, ans in zip(examples["Question"], examples["Complex_CoT"], examples["Response"]):
            text = TRAIN_PROMPT.format(question=q, cot=cot, answer=ans) + eos
            texts.append(text)
        return {"texts": texts}

    return raw.map(_format, batched=True)


def make_formatting_func(tokenizer):
    """Return a formatting function compatible with SFTTrainer."""
    eos = tokenizer.eos_token

    def _fmt(examples):
        texts = []
        for q, cot, ans in zip(examples["Question"], examples["Complex_CoT"], examples["Response"]):
            text = (
                f"### Question:\n{q}\n\n"
                f"### Reasoning:\n{cot}\n\n"
                f"### Final Answer:\n{ans}"
                f"{eos}"
            )
            texts.append(text)
        return texts

    return _fmt


def build_trainer(model_lora, tokenizer, dataset, max_steps: int):
    return SFTTrainer(
        model=model_lora,
        tokenizer=tokenizer,
        train_dataset=dataset,
        formatting_func=make_formatting_func(tokenizer),
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=1,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            max_steps=max_steps,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            output_dir=OUTPUT_DIR,
            report_to="none",
        ),
    )


def run_inference(model, tokenizer, question: str) -> str:
    """Run a single inference pass and return the answer text."""
    FastLanguageModel.for_inference(model)
    prompt = INFERENCE_PROMPT.format(question=question)
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=1200,
        use_cache=True,
    )
    decoded = tokenizer.batch_decode(outputs)[0]
    parts = decoded.split("### Answer:")
    return parts[1] if len(parts) > 1 else decoded


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Phi-2 for medical Q&A")
    parser.add_argument("--hf_token",   required=True,  help="Hugging Face write token")
    parser.add_argument("--max_steps",  type=int, default=60, help="Training steps (default: 60)")
    parser.add_argument("--push_to_hub", action="store_true", help="Push trained model to HF Hub")
    parser.add_argument("--save_local",  default="phi2_medical_lora", help="Local save directory")
    args = parser.parse_args()

    # ── Auth ──────────────────────────────────
    login(token=args.hf_token)
    print("✅ Logged in as:", whoami(token=args.hf_token)["name"])

    # ── Load base model ───────────────────────
    print("⏳ Loading Phi-2 ...")
    model, tokenizer = load_model(args.hf_token)

    # ── Pre-training inference test ───────────
    sample_q = (
        "A 61-year-old woman with a long history of involuntary urine loss during activities like "
        "coughing or sneezing but no leakage at night undergoes a gynecological exam and Q-tip test. "
        "What would cystometry most likely reveal?"
    )
    print("\n── Pre-fine-tune answer ──")
    print(run_inference(model, tokenizer, sample_q))

    # ── LoRA ──────────────────────────────────
    print("\n⚙️  Applying LoRA adapters ...")
    model_lora = apply_lora(model)

    # Clear stale state if present
    if hasattr(model, "_unwrapped_old_generate"):
        del model._unwrapped_old_generate

    # ── Dataset ───────────────────────────────
    print("📦 Loading dataset ...")
    dataset = prepare_dataset(tokenizer)

    # ── Train ─────────────────────────────────
    print(f"🚀 Training for {args.max_steps} steps ...")
    trainer = build_trainer(model_lora, tokenizer, dataset, args.max_steps)
    trainer.train()
    print("✅ Training complete.")

    # ── Post-training inference test ──────────
    print("\n── Post-fine-tune answer ──")
    print(run_inference(model_lora, tokenizer, sample_q))

    # ── Save ──────────────────────────────────
    model_lora.save_pretrained(args.save_local)
    tokenizer.save_pretrained(args.save_local)
    print(f"💾 Model saved to ./{args.save_local}")

    if args.push_to_hub:
        model_lora.push_to_hub(HUB_REPO, token=args.hf_token)
        tokenizer.push_to_hub(HUB_REPO, token=args.hf_token)
        print(f"☁️  Pushed to https://huggingface.co/{HUB_REPO}")


if __name__ == "__main__":
    main()
