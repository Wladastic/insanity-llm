"""
insanityllm.cli: Command-line interface for Insanity LLM

Provides command-line tools for:
- downloading datasets (insanity-download)
- training models (insanity-train)
"""

from . import download, train

__all__ = ["download", "train"]
