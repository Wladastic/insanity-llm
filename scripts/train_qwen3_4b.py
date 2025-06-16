#!/usr/bin/env python3
"""Training script specifically configured for Qwen3-4B model."""

import subprocess
import sys
from pathlib import Path

def main():
    """Run DPO training with Qwen3-4B optimized settings."""
    
    # Qwen3-4B specific configuration
    model_id = "Qwen/Qwen3-4B-Base"  # Using base version for better DPO training
    dataset = "data/sample_dpo_dataset.jsonl"  # Start with sample data
    output_dir = "models/DeliriumQwen3-4B"
    
    # Optimized settings for 4B model
    batch_size = 2
    grad_accum = 4
    max_steps = 100  # Start small for testing
    learning_rate = 5e-5
    lora_r = 16
    lora_alpha = 32
    max_seq_length = 1024
    
    print("🚀 Starting Qwen3-4B DPO Training")
    print(f"Model: {model_id}")
    print(f"Dataset: {dataset}")
    print(f"Output: {output_dir}")
    print(f"Settings: batch_size={batch_size}, grad_accum={grad_accum}, max_steps={max_steps}")
    print()
    
    # Check if dataset exists
    if not Path(dataset).exists():
        print(f"❌ Dataset not found: {dataset}")
        print("Creating sample dataset...")
        # The sample dataset should already exist from setup
        
    # Build the training command
    cmd = [
        "python3", "scripts/train_dpo.py",
        "--model_id", model_id,
        "--dataset", dataset,
        "--output_dir", output_dir,
        "--batch_size", str(batch_size),
        "--grad_accum", str(grad_accum),
        "--max_steps", str(max_steps),
        "--learning_rate", str(learning_rate),
        "--lora_r", str(lora_r),
        "--lora_alpha", str(lora_alpha),
        "--max_seq_length", str(max_seq_length),
        "--use_4bit",
        "--checkpointing",
        "--report_to", "none"  # Disable wandb for testing
    ]
    
    print("Running command:")
    print(" ".join(cmd))
    print()
    
    # Run the training
    try:
        subprocess.run(cmd, check=True)
        print("✅ Training completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with error: {e}")
        return 1
    except KeyboardInterrupt:
        print("⚠️ Training interrupted by user")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
