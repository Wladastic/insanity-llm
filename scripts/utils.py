#!/usr/bin/env python3
"""Utility functions for the Delirium QLoRA DPO project."""

from typing import Dict, Any, Optional
import json
import torch
from pathlib import Path

def check_gpu_memory() -> Dict[str, Any]:
    """Check available GPU memory and provide recommendations."""
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    
    gpu_info = {}
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / 1024**3
        gpu_info[f"GPU_{i}"] = {
            "name": props.name,
            "memory_gb": round(memory_gb, 1),
            "compute_capability": f"{props.major}.{props.minor}"
        }
    
    return gpu_info

def estimate_batch_size(model_size: str, available_memory_gb: float, use_4bit: bool = True) -> Dict[str, int]:
    """Estimate optimal batch size based on model size and available memory."""
    
    # Rough memory estimates (in GB) for different model sizes
    memory_estimates = {
        "7B": {"4bit": 8, "16bit": 14},
        "14B": {"4bit": 12, "16bit": 28},
        "32B": {"4bit": 24, "16bit": 64}
    }
    
    precision = "4bit" if use_4bit else "16bit"
    base_memory = memory_estimates.get(model_size, {}).get(precision, 16)
    
    # Leave some headroom for gradients and activations
    available_for_batch = available_memory_gb - base_memory - 2
    
    if available_for_batch <= 0:
        return {"recommended_batch_size": 1, "gradient_accumulation": 32, "warning": "Very tight memory"}
    
    # Rough estimate: each batch uses ~1-2GB additional memory
    max_batch_size = max(1, int(available_for_batch / 2))
    
    # Recommend smaller batch sizes with gradient accumulation
    if max_batch_size > 4:
        return {"recommended_batch_size": 2, "gradient_accumulation": 8}
    else:
        return {"recommended_batch_size": 1, "gradient_accumulation": max(8, 16 // max_batch_size)}

def validate_dataset_format(dataset_path: str) -> Dict[str, Any]:
    """Validate that a JSONL dataset has the required format for DPO."""
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            sample = json.loads(first_line)
        
        required_keys = ["prompt", "chosen", "rejected"]
        missing_keys = [key for key in required_keys if key not in sample]
        
        if missing_keys:
            return {"valid": False, "error": f"Missing keys: {missing_keys}"}
        
        return {"valid": True, "sample_keys": list(sample.keys())}
        
    except Exception as e:
        return {"valid": False, "error": str(e)}

def create_training_command(
    model_id: str,
    dataset: str,
    output_dir: str,
    **kwargs
) -> str:
    """Generate a training command with optimal parameters."""
    
    cmd = "python scripts/train_dpo.py \\\n"
    cmd += f"    --model_id {model_id} \\\n"
    cmd += f"    --dataset {dataset} \\\n"
    cmd += f"    --output_dir {output_dir}"
    
    # Add optional parameters
    for key, value in kwargs.items():
        if value is not None:
            if isinstance(value, bool):
                if value:  # Only add flag if True
                    cmd += f" \\\n    --{key}"
            else:
                cmd += f" \\\n    --{key} {value}"
    
    return cmd

if __name__ == "__main__":
    # Quick system check
    print("🔍 GPU Information:")
    gpu_info = check_gpu_memory()
    for gpu, info in gpu_info.items():
        if "error" not in info:
            print(f"  {gpu}: {info['name']} ({info['memory_gb']}GB)")
            
            # Provide recommendations
            recommendations = estimate_batch_size("14B", info['memory_gb'])
            print(f"    Recommended for 14B model: batch_size={recommendations['recommended_batch_size']}, "
                  f"grad_accum={recommendations['gradient_accumulation']}")
        else:
            print(f"  Error: {info['error']}")
