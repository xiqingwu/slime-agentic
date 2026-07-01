#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MS_SWIFT_ROOT=${MS_SWIFT_ROOT:-/home/xiqingwu/Documents/workspace/ms-swift}
MODEL=${MODEL:-Qwen/Qwen3-VL-4B-Instruct}
DATASET=${DATASET:-/data/MAT/mat_swift_sft.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-/data/AgentFlow_Qwen3VL_MAT_SWIFT_SFT}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
IMAGE_MAX_TOKEN_NUM=${IMAGE_MAX_TOKEN_NUM:-256}

export PYTHONPATH="${ROOT_DIR}:${MS_SWIFT_ROOT}:${PYTHONPATH:-}"
export NPROC_PER_NODE CUDA_VISIBLE_DEVICES IMAGE_MAX_TOKEN_NUM
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

swift sft \
  --model "${MODEL}" \
  --dataset "${DATASET}" \
  --tuner_type lora \
  --lora_rank 16 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --freeze_vit true \
  --freeze_aligner true \
  --torch_dtype bfloat16 \
  --attn_impl flash_attn \
  --padding_free true \
  --packing true \
  --gradient_checkpointing true \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 \
  --num_train_epochs 3 \
  --max_length 4096 \
  --deepspeed zero2 \
  --split_dataset_ratio 0.02 \
  --eval_steps 50 \
  --save_steps 50 \
  --logging_steps 5 \
  --save_total_limit 3 \
  --create_checkpoint_symlink true \
  --dataset_num_proc 4 \
  --dataloader_num_workers 4 \
  --output_dir "${OUTPUT_DIR}" \
  --report_to tensorboard
