#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MS_SWIFT_ROOT=${MS_SWIFT_ROOT:-/home/xiqingwu/Documents/workspace/ms-swift}
MODEL=${MODEL:-Qwen/Qwen3-VL-4B-Instruct}
ADAPTER=${ADAPTER:-/data/AgentFlow_Qwen3VL_MAT_SWIFT_SFT/last}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
IMAGE_MAX_TOKEN_NUM=${IMAGE_MAX_TOKEN_NUM:-256}

export PYTHONPATH="${ROOT_DIR}:${MS_SWIFT_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES IMAGE_MAX_TOKEN_NUM

swift rollout \
  --model "${MODEL}" \
  --adapters "${ADAPTER}" \
  --vllm_use_async_engine true \
  --vllm_enable_lora true \
  --vllm_max_lora_rank 32 \
  --vllm_max_model_len 16384 \
  --vllm_gpu_memory_utilization 0.75 \
  --vllm_limit_mm_per_prompt '{"image": 6}' \
  --external_plugins "${ROOT_DIR}/swift_plugin.py" \
  --multi_turn_scheduler mat_agentflow \
  --max_turns 5
