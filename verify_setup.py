#!/usr/bin/env python3
"""Setup verification script for insanityllm package."""

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
    print("🔍 Insanity LLM Setup Verification")
    print("=" * 50)
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 10):
        print("❌ Python 3.10+ required")
        return False
    else:
        print("✅ Python version OK")
    
    # Check core dependencies
    dependencies = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("datasets", "Datasets"),
        ("unsloth", "Unsloth"),
        ("trl", "TRL"),
        ("peft", "PEFT"),
        ("bitsandbytes", "BitsAndBytes"),
        ("insanityllm", "Insanity LLM Package"),
    ]
    
    all_good = True
    for module, name in dependencies:
        if check_import(module):
            print(f"✅ {name}")
        else:
            print(f"❌ {name} - not installed")
            all_good = False
    
    # Check CUDA
    if check_import("torch"):
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available - {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  CUDA not available - CPU training only")
    
    # Test insanityllm package
    try:
        import insanityllm
        config = insanityllm.setup_environment()
        print("✅ Insanityllm package working correctly")
        print(f"   Datasets dir: {config['DATASETS_DIR']}")
        print(f"   Models dir: {config['MODELS_DIR']}")
    except Exception as e:
        print(f"❌ Insanityllm package error: {e}")
        all_good = False
    
    print("\n" + "=" * 50)
    if all_good:
        print("🎉 Setup verification completed successfully!")
        print("You can now use the CLI commands:")
        print("   insanity-download <dataset>")
        print("   insanity-train --model_id <model> --dataset <dataset> --output_dir <dir>")
    else:
        print("❌ Setup verification failed. Please install missing dependencies.")
        print("Run: pip install -e .")
    
    return all_good

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
