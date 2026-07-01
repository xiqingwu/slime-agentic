#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MS_SWIFT_ROOT=${MS_SWIFT_ROOT:-/home/xiqingwu/Documents/workspace/ms-swift}
MODEL=${MODEL:-/data/AgentFlow_Qwen3VL_MAT_SWIFT_SFT/last}
DATASET=${DATASET:-/data/MAT/mat_swift_rl.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-/data/AgentFlow_Qwen3VL_MAT_SWIFT_GRPO}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1,2,3}
NPROC_PER_NODE=${NPROC_PER_NODE:-3}
IMAGE_MAX_TOKEN_NUM=${IMAGE_MAX_TOKEN_NUM:-256}
MAX_STEPS=${MAX_STEPS:--1}

export PYTHONPATH="${ROOT_DIR}:${MS_SWIFT_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES NPROC_PER_NODE IMAGE_MAX_TOKEN_NUM
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

EXTRA_ARGS=()
if [ "${MAX_STEPS}" -ge 0 ]; then EXTRA_ARGS+=(--max_steps "${MAX_STEPS}"); fi

swift rlhf \
  --rlhf_type grpo \
  --model "${MODEL}" \
  --dataset "${DATASET}" \
  --tuner_type lora \
  --lora_rank 16 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --freeze_vit true \
  --freeze_aligner true \
  --external_plugins "${ROOT_DIR}/swift_plugin.py" \
  --reward_funcs mat_agentflow_reward \
  --use_vllm true \
  --vllm_mode server \
  --vllm_server_host 127.0.0.1 \
  --vllm_server_port 8000 \
  --vllm_server_pass_dataset true \
  --vllm_enable_lora true \
  --multi_turn_scheduler mat_agentflow \
  --max_turns 5 \
  --torch_dtype bfloat16 \
  --attn_impl flash_attn \
  --padding_free true \
  --gradient_checkpointing true \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --steps_per_generation 4 \
  --num_generations 6 \
  --max_completion_length 4096 \
  --learning_rate 1e-5 \
  --num_train_epochs 1 \
  --beta 0.001 \
  --temperature 1.0 \
  --deepspeed zero2 \
  --offload_optimizer true \
  --offload_model true \
  --save_steps 20 \
  --logging_steps 1 \
  --save_total_limit 3 \
  --create_checkpoint_symlink true \
  --log_completions true \
  --report_to tensorboard \
  --output_dir "${OUTPUT_DIR}" \
  "${EXTRA_ARGS[@]}"
