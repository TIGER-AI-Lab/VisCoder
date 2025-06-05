#!/usr/bin/env bash
set -x

# ----------------------------
# WandB Logging Configuration
# ----------------------------

# Default run name if not provided
if [ -z "$RUN_NAME" ]; then
    RUN_NAME="VisCoder_7B"
fi

export WANDB_API_KEY="YOUR_WANDB_API_KEY"      # Get this from wandb.ai/settings
export WANDB_PROJECT="YOUR_WANDB_PROJECT"      # e.g., "VisCoder"
export WANDB_NAME=$RUN_NAME                    # Run name appears in the UI


# ----------------------------
# Model and Data Setup
# ----------------------------

MODEL_PATH="Qwen/Qwen2.5-Coder-7B-Instruct"
DATA_PATH="data/VisCode-200K.jsonl"
OUTPUT_DIR="output/VisCoder_7B"

# Create output dir if not exist
if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# ----------------------------
# Distributed Training Setup
# ----------------------------

DISTRIBUTED_ARGS="\
    --nproc_per_node 8 \
    --nnodes $WORLD_SIZE \
    --node_rank $RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

# ----------------------------
# Training Launch
# ----------------------------

torchrun ${DISTRIBUTED_ARGS} ms-swift/swift/cli/sft.py\
    --use_hf True \
    \
    --model $MODEL_PATH \
    --train_type full \
    --torch_dtype bfloat16 \
    --attn_impl flash_attn \
    \
    --dataset $DATA_PATH \
    --split_dataset_ratio 0 \
    --dataset_num_proc 8 \
    --streaming False \
    --strict False \
    --deepspeed zero3 \
    --remove_unused_columns False \
    --dataloader_num_workers 8 \
    \
    --truncation_strategy delete \
    \
    --output_dir $OUTPUT_DIR \
    --gradient_checkpointing True \
    --per_device_train_batch_size 4 \
    --weight_decay 0.05 \
    --learning_rate 5e-6 \
    --lr_scheduler_type "cosine" \
    --report_to "wandb" \
    --logging_first_step True \
    --logging_steps 1 \
    \
    --num_train_epochs 3 \
    --gradient_accumulation_steps 5 \
    --save_strategy "epoch" \
    --warmup_ratio 0.05 \
    --ddp_backend "nccl"