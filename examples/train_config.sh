# Training Configuration Examples
# Copy and modify these examples for your specific use case

# Qwen3-4B configuration (optimal for testing, works on 8GB+ GPU)
export MODEL_ID="Qwen/Qwen3-4B-Base"
export DATASET="data/sample_dpo_dataset.jsonl"
export OUTPUT_DIR="models/DeliriumQwen3-4B"
export BATCH_SIZE=2
export GRAD_ACCUM=4
export MAX_STEPS=100
export USE_4BIT=""  # 4B model doesn't need 4-bit quantization
export CHECKPOINTING="--checkpointing"

# Small model configuration (7B on 16GB GPU)
# export MODEL_ID="Qwen/Qwen3-7B-Chat"
# export DATASET="sam-paech/gutenbergs_1_2_3_4-antislop-dpo"
# export OUTPUT_DIR="models/DeliriumQwen3-7B"
# export BATCH_SIZE=2
# export GRAD_ACCUM=8
# export MAX_STEPS=1000
# export USE_4BIT="--use_4bit"
# export CHECKPOINTING="--checkpointing"

# Medium model configuration (14B on 24GB GPU)
# export MODEL_ID="Qwen/Qwen3-14B-Chat"
# export DATASET="sam-paech/gutenbergs_1_2_3_4-antislop-dpo"
# export OUTPUT_DIR="models/DeliriumQwen3-14B"
# export BATCH_SIZE=1
# export GRAD_ACCUM=16
# export MAX_STEPS=1200
# export USE_4BIT="--use_4bit"
# export CHECKPOINTING="--checkpointing"

# Large model configuration (32B on 48GB+ GPU)
# export MODEL_ID="Qwen/Qwen3-32B-Chat"
# export DATASET="sam-paech/gutenbergs_1_2_3_4-antislop-dpo"
# export OUTPUT_DIR="models/DeliriumQwen3-32B"
# export BATCH_SIZE=1
# export GRAD_ACCUM=32
# export MAX_STEPS=800
# export USE_4BIT="--use_4bit"
# export CHECKPOINTING="--checkpointing"

# Run training with current configuration
echo "🚀 Starting training with configuration:"
echo "Model: $MODEL_ID"
echo "Dataset: $DATASET"
echo "Output: $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE, Grad accum: $GRAD_ACCUM"
echo ""

python scripts/train_dpo.py \
    --model_id "$MODEL_ID" \
    --dataset "$DATASET" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size $BATCH_SIZE \
    --grad_accum $GRAD_ACCUM \
    --max_steps $MAX_STEPS \
    $USE_4BIT \
    $CHECKPOINTING
