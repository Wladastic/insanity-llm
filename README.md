# Delirium QLoRA DPO Fine-tuning Project

A minimal and efficient setup for fine-tuning large language models using QLoRA (Quantized Low-Rank Adaptation) and DPO (Direct Preference Optimization) with Unsloth.

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- CUDA-compatible GPU (recommended: 16GB+ VRAM for 14B models)
- Git and Git LFS

### Installation

1. **Clone and setup the environment:**
```bash
# Install dependencies
pip install -r requirements.txt

# Or install manually:
pip install "unsloth[torch]" datasets trl bitsandbytes einops peft accelerate wandb
```

2. **Login to Hugging Face and Weights & Biases (optional):**
```bash
huggingface-cli login
wandb login
```

### Basic Usage

Train a model with DPO:

```bash
python scripts/train_dpo.py \
    --model_id Qwen/Qwen3-14B-Base \
    --dataset sam-paech/gutenbergs_1_2_3_4-antislop-dpo \
    --output_dir models/DeliriumQwen3-14B \
    --batch_size 1 \
    --grad_accum 8 \
    --max_steps 1200 \
    --use_4bit \
    --checkpointing
```

## 📁 Project Structure

```
delirium/
├── scripts/           # Training and utility scripts
│   └── train_dpo.py   # Main DPO training script
├── models/            # Saved fine-tuned models
├── data/              # Datasets and data files
├── requirements.txt   # Python dependencies
└── README.md         # This file
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

## 📊 Dataset Format

Your dataset should have three columns:
- `prompt`: The input prompt
- `chosen`: The preferred response
- `rejected`: The less preferred response

Example JSONL format:
```json
{"prompt": "Explain quantum computing", "chosen": "Good explanation...", "rejected": "Bad explanation..."}
```

## 🎯 Supported Models

This script is optimized for Qwen3 models but should work with most instruction-tuned models:

- `Qwen/Qwen3-7B-Base`
- `Qwen/Qwen3-14B-Base`
- Other compatible models from Hugging Face

## 💡 Memory Optimization Tips

### For Large Models (14B+):
- Use `--use_4bit` for 4-bit quantization
- Enable `--checkpointing` for gradient checkpointing
- Reduce `--batch_size` and increase `--grad_accum`
- Consider using DeepSpeed for multi-GPU setups

### Example for 24GB GPU:
```bash
python scripts/train_dpo.py \
    --model_id Qwen/Qwen3-14B-Base \
    --dataset your_dataset \
    --output_dir models/output \
    --batch_size 1 \
    --grad_accum 16 \
    --use_4bit \
    --checkpointing \
    --max_seq_length 2048
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
1. Create new scripts in the `scripts/` directory
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
