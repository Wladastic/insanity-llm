#!/usr/bin/env python3
"""Minimal CLI‑Script für QLoRA‑DPO‑Finetuning mit Unsloth.

Beispielaufruf:
    python scripts/train_dpo.py \
        --model_id Qwen/Qwen3-14B-Base \
        --dataset sam-paech/gutenbergs_1_2_3_4-antislop-dpo \
        --output_dir models/DeliriumQwen3-14B \
        --batch_size 1 --grad_accum 8 --max_steps 1200
        
Notwendige Pakete (siehe README):
    pip install "unsloth[torch]" datasets trl bitsandbytes einops peft accelerate
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

# Optional wandb import
try:
    import wandb
except ImportError:
    wandb = None

# ---------------------------------------------------------------------------
# Argument‑Parser
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="QLoRA + DPO Finetuning für Qwen3‑Modelle")
    parser.add_argument("--model_id", required=True, help="HF‑ID oder Pfad des Basismodells, z. B. Qwen/Qwen3-14B-Chat")
    parser.add_argument("--dataset", required=True,
                        help="HF‑Dataset‑Name oder Pfad zur JSONL‑Datei mit Spalten prompt/ chosen/ rejected")
    parser.add_argument("--output_dir", required=True, help="Zielverzeichnis für das gemergte Modell")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=1200)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--beta", type=float, default=0.1, help="DPO‑Temperatur (Reward‑Scharfstellung)")
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--use_4bit", action="store_true", help="Basismodell in 4‑Bit NF4 laden (empfohlen bei 14B)")
    parser.add_argument("--checkpointing", action="store_true", help="Gradient Checkpointing aktivieren")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report_to", default="wandb", choices=["wandb", "none"], help="Tracking‑Backend")
    return parser.parse_args()

# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def prepare_dataset(ds_path: str):
    """Lädt HF‑Dataset oder JSONL‑Datei mit den Spalten prompt/chosen/rejected."""
    if ds_path.endswith(".json") or ds_path.endswith(".jsonl"):
        return load_dataset("json", data_files=ds_path, split="train")
    else:
        return load_dataset(ds_path, split="train")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # Dataset ---------------------------------------------------------------
    train_ds = prepare_dataset(args.dataset)
    print(f"Dataset loaded: {len(train_ds):,} pairs")

    # Modell & Tokenizer ----------------------------------------------------
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

    # DPO‑Trainer -----------------------------------------------------------
    training_args = DPOConfig(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=50,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=args.seed,
        output_dir=args.output_dir,
        report_to=args.report_to if args.report_to != "none" else [],
        gradient_checkpointing=args.checkpointing,
        remove_unused_columns=True,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # Let Unsloth handle reference model
        args=training_args,
        beta=args.beta,
        train_dataset=train_ds,
        tokenizer=tokenizer,
        max_length=args.max_seq_length,
        max_prompt_length=args.max_seq_length // 2,
    )

    trainer.train()

    # Modell mergen (16‑Bit), damit vLLM es direkt laden kann --------------
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save using Unsloth's save method
    model.save_pretrained_merged(
        args.output_dir,
        tokenizer,
        save_method="merged_16bit"
    )
    
    print(f"\n✅ Training abgeschlossen – gemergtes Modell in: {args.output_dir}\n")


if __name__ == "__main__":
    main()
