"""
insanityllm: QLoRA DPO fine-tuning toolkit for large language models

A modern toolkit for fine-tuning large language models using QLoRA and DPO techniques.
Designed for efficiency and ease of use with the Unsloth library.
"""

__version__ = "0.1.0"
__author__ = "Wladastic"

from .config import setup_environment
from .utils import get_gpu_memory_info, validate_dataset_format, print_system_info

__all__ = [
    "setup_environment",
    "get_gpu_memory_info", 
    "validate_dataset_format",
    "print_system_info",
]
