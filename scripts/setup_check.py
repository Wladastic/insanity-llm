#!/usr/bin/env python3
"""Quick setup verification script."""

import sys
import importlib
from pathlib import Path

def check_import(module_name: str) -> bool:
    """Check if a module can be imported."""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False

def main():
    print("🔍 Delirium QLoRA DPO Setup Verification")
    print("=" * 50)
    
    # Check Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"Python version: {python_version}")
    
    if sys.version_info < (3, 8):
        print("⚠️  Warning: Python 3.8+ recommended")
    else:
        print("✅ Python version OK")
    
    print("\n📦 Checking required packages:")
    
    required_packages = [
        "torch",
        "transformers", 
        "datasets",
        "accelerate",
        "peft",
        "trl",
        "bitsandbytes",
        "einops"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        if check_import(package):
            print(f"✅ {package}")
        else:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    # Check optional packages
    print("\n📦 Checking optional packages:")
    optional_packages = ["wandb", "jupyter"]
    
    for package in optional_packages:
        if check_import(package):
            print(f"✅ {package}")
        else:
            print(f"⚠️  {package} (optional)")
    
    # Check CUDA availability
    print("\n🔧 Checking CUDA:")
    if check_import("torch"):
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.device_count()} GPU(s)")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / 1024**3
                print(f"   GPU {i}: {props.name} ({memory_gb:.1f}GB)")
        else:
            print("⚠️  CUDA not available (CPU-only mode)")
    else:
        print("❌ Cannot check CUDA (torch not installed)")
    
    # Check file structure
    print("\n📁 Checking project structure:")
    required_dirs = ["scripts", "models", "data"]
    required_files = ["requirements.txt", "README.md", "scripts/train_dpo.py"]
    
    for dir_name in required_dirs:
        if Path(dir_name).exists():
            print(f"✅ {dir_name}/")
        else:
            print(f"❌ {dir_name}/")
    
    for file_name in required_files:
        if Path(file_name).exists():
            print(f"✅ {file_name}")
        else:
            print(f"❌ {file_name}")
    
    # Summary
    print("\n📋 Summary:")
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("   Run: pip install -r requirements.txt")
    else:
        print("✅ All required packages are installed!")
    
    if Path("data/sample_dpo_dataset.jsonl").exists():
        print("✅ Sample dataset is ready for testing")
    
    print("\n🚀 Next steps:")
    print("1. Install missing packages if any")
    print("2. Run: python scripts/utils.py (to check GPU memory)")
    print("3. Test with sample dataset: Ctrl+Shift+P > Tasks: Run Task > Train DPO (Sample Dataset)")

if __name__ == "__main__":
    main()
