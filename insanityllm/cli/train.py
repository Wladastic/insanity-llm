"""
insanityllm.cli.train
Training CLI for Insanity LLM (QLoRA + DPO).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig
from transformers import AutoTokenizer

from insanityllm.config import setup_environment

# Optional wandb import
try:
    import wandb
except ImportError:
    wandb = None

def parse_args():
    parser = argparse.ArgumentParser(description="QLoRA + DPO Finetuning for Qwen3 models")
    parser.add_argument("--model_id", required=True, help="HF model ID or path, e.g. Qwen/Qwen3-14B-Base")
    parser.add_argument("--dataset", required=True,
                        help="HF dataset name or path to JSONL file with prompt/chosen/rejected columns")
    parser.add_argument("--output_dir", required=True, help="Output directory for the trained model")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=1200)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--beta", type=float, default=0.1, help="DPO temperature (reward sharpening)")
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--use_4bit", action="store_true", help="Load base model in 4-bit NF4 (recommended for 14B)")
    parser.add_argument("--checkpointing", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report_to", default="wandb", choices=["wandb", "none"], help="Tracking backend")
    return parser.parse_args()

def prepare_dataset(ds_path: str, datasets_dir: str):
    """Load HF dataset or JSONL file with prompt/chosen/rejected columns."""
    if ds_path.endswith(".json") or ds_path.endswith(".jsonl"):
        if not os.path.isabs(ds_path):
            ds_path = os.path.join(datasets_dir, ds_path)
        return load_dataset("json", data_files=ds_path, split="train")
    else:
        return load_dataset(ds_path, split="train")

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # Setup environment and get configuration
    config = setup_environment()
    datasets_dir = config['DATASETS_DIR']
    
    # Dataset
    train_ds = prepare_dataset(args.dataset, datasets_dir)
    print(f"Dataset loaded: {len(train_ds):,} pairs")

    # Model & Tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        args.model_id,
        load_in_4bit=args.use_4bit,
        max_seq_length=args.max_seq_length,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    # DPO Trainer
    dpo_config = DPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        beta=args.beta,
        seed=args.seed,
        logging_steps=10,
        save_steps=100,
        save_total_limit=5,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to=args.report_to,
        bf16=True,
        gradient_checkpointing=args.checkpointing,
    )

    trainer = DPOTrainer(
        model=model,
        tokenizer=tokenizer,
        args=dpo_config,
        train_dataset=train_ds,
        beta=args.beta,
        max_length=args.max_seq_length,
        max_prompt_length=args.max_seq_length // 2,
    )

    print("🚀 Starting DPO training...")
    trainer.train()

    print("💾 Saving model...")
    trainer.save_model()

    print("🎯 Unloading and merging model...")
    model = FastLanguageModel.for_inference(model)
    model.save_pretrained_merged(args.output_dir, tokenizer)

    print(f"✅ Training complete! Model saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
