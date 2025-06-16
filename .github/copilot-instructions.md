# Copilot Instructions for Delirium QLoRA DPO Project

<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

## Project Overview
This is a machine learning project focused on fine-tuning large language models using QLoRA (Quantized Low-Rank Adaptation) and DPO (Direct Preference Optimization) techniques with the Unsloth library.

## Key Technologies
- **Unsloth**: Fast and memory-efficient fine-tuning library
- **QLoRA**: Quantized Low-Rank Adaptation for efficient training
- **DPO**: Direct Preference Optimization for alignment training
- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face transformers library
- **PEFT**: Parameter-Efficient Fine-Tuning methods

## Code Style Guidelines
- Use type hints for all function parameters and return values
- Follow PEP 8 style guidelines
- Use descriptive variable names, especially for ML hyperparameters
- Add docstrings to all functions with clear parameter descriptions
- Use argparse for command-line interfaces
- Include proper error handling for file operations and model loading

## ML-Specific Guidelines
- Always set random seeds for reproducibility
- Use gradient checkpointing for memory efficiency when possible
- Include proper logging for training metrics
- Save models in a format compatible with inference frameworks (like vLLM)
- Use 4-bit quantization for large models (14B+ parameters) to save memory
- Implement proper dataset validation before training

## File Organization
- Training scripts go in `scripts/` directory
- Trained models are saved in `models/` directory
- Datasets and data files go in `data/` directory
- Use clear, descriptive naming for output directories and model checkpoints

## Performance Considerations
- Optimize batch sizes and gradient accumulation based on available GPU memory
- Use mixed precision training when supported
- Consider using DeepSpeed for multi-GPU setups
- Monitor GPU memory usage and adjust parameters accordingly
