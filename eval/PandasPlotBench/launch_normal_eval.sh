#!/bin/bash

CONFIG="configs/config_normal_viscoder.yaml" # e.g., config_normal_baseline.yaml, config_normal_viscoder.yaml
LIMIT=None
GPUS=(0 1 2 3 4 5 6 7)

# Auto parse config and set log path
CONFIG_BASENAME=$(basename "$CONFIG" .yaml)
IFS='_' read -ra PARTS <<< "$CONFIG_BASENAME"
MODE=${PARTS[1]}          # "normal"
SUBMODE=${PARTS[2]}       # "baseline" or "viscoder"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/${MODE}_mode/${SUBMODE}"
mkdir -p "$LOG_DIR"

# CHECKPOINTS=(
#   "openai/gpt-4o"
#   "openai/gpt-4o-mini"
#   "meta-llama/Llama-3.2-3B-Instruct"
#   "Qwen/Qwen2.5-3B-Instruct"
#   "Qwen/Qwen2.5-Coder-3B-Instruct"
#   "meta-llama/Llama-3.1-8B-Instruct"
#   "Qwen/Qwen2.5-7B-Instruct"
#   "Qwen/Qwen2.5-Coder-7B-Instruct"
# )

CHECKPOINTS=(
  "TIGER-Lab/VisCoder-7B"
  "TIGER-Lab/VisCoder-3B"
)

GPU_COUNT=${#GPUS[@]}
TOTAL=${#CHECKPOINTS[@]}

for ((i=0; i<TOTAL; i+=GPU_COUNT)); do
  echo "[INFO] Starting batch from checkpoint $i..."
  for ((j=0; j<GPU_COUNT && i+j<TOTAL; j++)); do
    GPU_ID=${GPUS[$j]}
    CKPT=${CHECKPOINTS[$((i+j))]}
    CKPT_NAME=$(basename "$CKPT")
    LOG_FILE="$LOG_DIR/eval_${CKPT_NAME}_${TIMESTAMP}.log"

    echo "[INFO] Launching on GPU $GPU_ID -> $CKPT"
    CUDA_VISIBLE_DEVICES=$GPU_ID python batch_eval_run.py \
      --cuda=$GPU_ID \
      --model="$CKPT" \
      --limit="$LIMIT" \
      --config="$CONFIG" \
      > "$LOG_FILE" 2>&1 &

    sleep 0.5
  done

  wait
done

echo "[INFO] All evaluation jobs completed."
