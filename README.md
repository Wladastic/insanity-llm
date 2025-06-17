# Insanity LLM: QLoRA DPO Fine-tuning Toolkit

![Insanity LLM Banner](assets/insanity-llm-banner.png)

A minimal and efficient setup for fine-tuning large language models using QLoRA (Quantized Low-Rank Adaptation) and DPO (Direct Preference Optimization) with Unsloth.

I am using sam-paech's [Gutenberg dataset](https://huggingface.co/datasets/sam-paech/gutenbergs_1_2_3_4-antislop-dpo) for training, but you can use any Hugging Face dataset with the provided scripts. Their Delirium v1 Model was pretty fun to play with, so I wanted to try it out with QLoRA and DPO.

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended: 16GB+ VRAM for "bigger" models like 8B or higher)
- Git and Git LFS

### Installation

1. **Install the package:**
```bash
# Install in development mode
pip install -e .

# Or install from requirements
pip install -r requirements.txt
```

2. **Verify the setup:**
```bash
python3 verify_setup.py
```

2. **Configure directories (sort of optional. better to do it):**
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env to customize paths (optional - defaults work fine)
# nano .env
```

3. **Login to Hugging Face and Weights & Biases (optional, but some models require verification):**
```bash
huggingface-cli login
wandb login
```

### Basic Usage

1. **Download a dataset:**
```bash
# Download any Hugging Face dataset
insanity-download sam-paech/gutenbergs_1_2_3_4-antislop-dpo

# Or use a specific subset/split
insanity-download argilla/ultrafeedback-binarized-preferences-cleaned --subset default --split train
```

2. **Train a model with DPO:**
```bash
insanity-train \
    --model_id Qwen/Qwen3-4B-Base \
    --dataset data/sam-paech_gutenbergs_1_2_3_4-antislop-dpo_train.jsonl \
    --output_dir models/DeliriumQwen3-4B \
    --batch_size 2 \
    --grad_accum 4 \
    --max_steps 500 \
    --use_4bit \
    --checkpointing
```

### CLI Commands

The package provides two main CLI commands:

- `insanity-download` - Download and convert HuggingFace datasets to JSONL
- `insanity-train` - Train models using QLoRA + DPO

## 🎯 Configuration

### Environment Variables

The project uses a `.env` file to configure directories and cache locations. Copy `.env.example` to `.env` and customize as needed:

```bash
# Dataset Configuration
DATASETS_DIR=data                           # Where to save downloaded datasets
CACHE_DIR=cache                            # Base cache directory
HF_DATASETS_CACHE=cache/huggingface/datasets  # HuggingFace datasets cache
HF_MODELS_CACHE=cache/huggingface/models     # HuggingFace models cache

# Training Configuration
MODELS_DIR=models                          # Where to save trained models
LOGS_DIR=logs                             # Training logs directory
```

**Benefits:**
- All datasets download to your specified directory (not HuggingFace's default cache)
- Consistent cache management across all scripts
- Easy to change storage locations without modifying code
- Prevents accidental downloads to home directory

## 📁 Project Structure

```
insanity-llm/
├── insanityllm/           # Main Python package
│   ├── __init__.py        # Package initialization and exports  
│   ├── config.py          # Environment configuration
│   ├── utils.py           # Utility functions
│   └── cli/               # Command-line interface
│       ├── __init__.py
│       ├── download.py    # Dataset downloader CLI
│       └── train.py       # Model training CLI
├── bin/                   # Convenience scripts
│   ├── insanity-download
│   └── insanity-train  
├── examples/              # Configuration examples
│   └── train_config.sh    # Training configuration examples
├── models/                # Saved fine-tuned models
├── data/                  # Datasets and data files
├── pyproject.toml         # Package configuration
├── verify_setup.py        # Setup verification script
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🔧 Training Parameters

### Key Arguments

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_id` | Required | Hugging Face model ID (e.g., `Qwen/Qwen3-14B-Base`) |
| `--dataset` | Required | Dataset name or path to JSONL file |
| `--output_dir` | Required | Directory to save the fine-tuned model |
| `--batch_size` | 1 | Per-device training batch size |
| `--grad_accum` | 8 | Gradient accumulation steps |
| `--max_steps` | 1200 | Maximum training steps |
| `--learning_rate` | 2e-4 | Learning rate |
| `--beta` | 0.1 | DPO temperature parameter |
| `--use_4bit` | False | Use 4-bit quantization (recommended for 14B+ models) |
| `--checkpointing` | False | Enable gradient checkpointing for memory efficiency |

### LoRA Configuration

- **Rank (r)**: 32 (adjustable via `--lora_r`)
- **Alpha**: 32 (adjustable via `--lora_alpha`)
- **Target modules**: All linear layers (automatically selected by Unsloth)

## 📊 Datasets

### Downloading Datasets

Use the dataset downloader to get any Hugging Face dataset:

```bash
# Download a specific dataset
python scripts/download_datasets.py sam-paech/gutenbergs_1_2_3_4-antislop-dpo

# Download with specific subset and split
python scripts/download_datasets.py argilla/ultrafeedback-binarized-preferences-cleaned --subset default --split train

# Download with sample limit for testing
python scripts/download_datasets.py sam-paech/gutenbergs_1_2_3_4-antislop-dpo --max-samples 1000

# Save to custom directory
python scripts/download_datasets.py sam-paech/gutenbergs_1_2_3_4-antislop-dpo --output-dir my_datasets
```

The script will automatically:
- Download the specified dataset from Hugging Face
- Convert it to JSONL format
- Generate a descriptive filename
- Show you the first sample to verify the data structure

### Dataset Format

Your dataset should have three columns for DPO training:
- `prompt`: The input prompt
- `chosen`: The preferred response
- `rejected`: The less preferred response

Example JSONL format:
```json
{"prompt": "Explain quantum computing", "chosen": "Good explanation...", "rejected": "Bad explanation..."}
```

## 🎯 Supported Models

This script is optimized for Qwen3 models but should work with most instruction-tuned models:

- `Qwen/Qwen3-4B-Base` (recommended for most users)
- `Qwen/Qwen3-7B-Base`
- `Qwen/Qwen3-14B-Base` (requires 24GB+ VRAM)
- Other compatible models from Hugging Face

## 💡 Memory Optimization Tips

### For Large Models (14B+):
- Use `--use_4bit` for 4-bit quantization
- Enable `--checkpointing` for gradient checkpointing
- Reduce `--batch_size` and increase `--grad_accum`
- Consider using DeepSpeed for multi-GPU setups

### Example for 24GB GPU:
```bash
insanity-train \
    --model_id Qwen/Qwen3-4B-Base \
    --dataset your_dataset \
    --output_dir models/output \
    --batch_size 2 \
    --grad_accum 4 \
    --use_4bit \
    --checkpointing \
    --max_seq_length 1024
```

## 🔍 Monitoring Training

The script supports Weights & Biases logging by default. To disable:
```bash
python scripts/train_dpo.py ... --report_to none
```

## 🚀 Inference

After training, your model will be saved in 16-bit format for efficient inference:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("models/DeliriumQwen3-14B")
tokenizer = AutoTokenizer.from_pretrained("models/DeliriumQwen3-14B")
```

Or use with vLLM for fast inference:
```bash
python -m vllm.entrypoints.openai.api_server \
    --model models/DeliriumQwen3-14B \
    --port 8000
```

## 🎯 Quick Start with Qwen3-4B

For testing and smaller setups, Qwen3-4B is an excellent choice. Here's how to get started quickly:

### Training Qwen3-4B with Sample Data
```bash
# Activate virtual environment
source venv/bin/activate

# Quick test run (100 steps)
python3 scripts/train_qwen3_4b.py
```

### Manual Qwen3-4B Training
```bash
python3 scripts/train_dpo.py \
    --model_id Qwen/Qwen3-4B-Base \
    --dataset data/sample_dpo_dataset.jsonl \
    --output_dir models/DeliriumQwen3-4B \
    --batch_size 2 \
    --grad_accum 4 \
    --max_steps 500 \
    --learning_rate 5e-5 \
    --lora_r 16 \
    --lora_alpha 32 \
    --use_4bit \
    --checkpointing \
    --report_to none
```

### Memory Requirements
- **Qwen3-4B**: ~8-12GB VRAM (with 4-bit quantization)
- **Recommended GPU**: RTX 3080/4070 or better
- **Minimum**: RTX 3060 12GB

## 🛠️ Development

### Adding New Features
1. Add new functionality to the `insanityllm/` package
2. Update requirements.txt if new dependencies are needed
3. Update this README with usage instructions

### Common Issues
- **CUDA out of memory**: Reduce batch size, enable checkpointing, or use 4-bit quantization
- **Dataset format errors**: Ensure your dataset has `prompt`, `chosen`, and `rejected` columns
- **Model loading errors**: Check model ID and ensure you have access to private models

## 📝 License

This project is open source. Please check individual model licenses before commercial use.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

---

For more information about the underlying technologies:
- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [DPO Paper](https://arxiv.org/abs/2305.18290)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
