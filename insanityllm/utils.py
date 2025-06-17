"""
insanityllm.utils
Utility functions for Insanity LLM.
"""
import torch
from typing import Dict, Any

def get_gpu_memory_info() -> Dict[str, Any]:
    """Get GPU memory information."""
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device).total_memory
        allocated_memory = torch.cuda.memory_allocated(device)
        free_memory = total_memory - allocated_memory
        
        return {
            "device": device,
            "device_name": torch.cuda.get_device_name(device),
            "total_memory_gb": total_memory / (1024**3),
            "allocated_memory_gb": allocated_memory / (1024**3),
            "free_memory_gb": free_memory / (1024**3),
            "memory_usage_percent": (allocated_memory / total_memory) * 100
        }
    else:
        return {"error": "CUDA not available"}

def validate_dataset_format(dataset_path: str) -> Dict[str, Any]:
    """Validate DPO dataset format."""
    try:
        import json
        from pathlib import Path
        
        if not Path(dataset_path).exists():
            return {"valid": False, "error": f"File not found: {dataset_path}"}
        
        required_fields = {"prompt", "chosen", "rejected"}
        sample_count = 0
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        data = json.loads(line)
                        sample_count += 1
                        
                        if not isinstance(data, dict):
                            return {"valid": False, "error": f"Line {line_num}: Not a JSON object"}
                        
                        missing_fields = required_fields - set(data.keys())
                        if missing_fields:
                            return {"valid": False, "error": f"Line {line_num}: Missing fields: {missing_fields}"}
                            
                        if sample_count >= 10:  # Check first 10 samples
                            break
                            
                    except json.JSONDecodeError as e:
                        return {"valid": False, "error": f"Line {line_num}: Invalid JSON: {e}"}
        
        return {"valid": True, "samples_checked": sample_count}
        
    except Exception as e:
        return {"valid": False, "error": f"Validation error: {e}"}

def print_system_info():
    """Print system information for debugging."""
    print("🔧 System Information:")
    print(f"  Python: {torch.__version__ if hasattr(torch, '__version__') else 'Unknown'}")
    print(f"  PyTorch: {torch.__version__ if torch else 'Not installed'}")
    print(f"  CUDA Available: {torch.cuda.is_available() if torch else False}")
    
    if torch and torch.cuda.is_available():
        gpu_info = get_gpu_memory_info()
        print(f"  GPU: {gpu_info.get('device_name', 'Unknown')}")
        print(f"  GPU Memory: {gpu_info.get('free_memory_gb', 0):.1f}GB free / {gpu_info.get('total_memory_gb', 0):.1f}GB total")
