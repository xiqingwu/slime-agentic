#!/bin/bash
# MAT-Coding SFT for Qwen3-VL-4B — 1x RTX 4090 (24GB).
#
# Usage: MAT_MAX_PIXELS=200704 bash train_mat_sft_1gpu.sh

set -euo pipefail
set -x

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${ROOT_DIR}/scripts/models/qwen3-vl-4B.sh"

N_GPUS=1
TP_SIZE=1
MODEL_PATH=${MODEL_PATH:-/data/models/qwen3_vl_4b}
DATA_PATH=${DATA_PATH:-/root/autodl-tmp/MAT/mat_coding_sft.jsonl}
SAVE_PATH=${SAVE_PATH:-/data/AgentFlow_Qwen3VL_MAT_SFT}
MAT_MAX_PIXELS=${MAT_MAX_PIXELS:-50176}
WANDB_PROJECT=${WANDB_PROJECT:-MAT_SFT}
WANDB_GROUP=${WANDB_GROUP:-qwen3vl_4b}

if [ "${SKIP_PROCESS_KILL:-0}" != "1" ]; then
    ray stop --force 2>/dev/null || true
    pkill -9 ray 2>/dev/null || true
fi

export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export PYTHONUNBUFFERED=1
export MAT_MAX_PIXELS

ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${N_GPUS}" --disable-usage-stats

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_PATH:-/root/autodl-tmp/Megatron-LM}:${SCRIPT_DIR}:${ROOT_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"0\",
    \"MAT_MAX_PIXELS\": \"${MAT_MAX_PIXELS}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"expandable_segments:True\",
    \"WANDB_API_KEY\": \"${WANDB_API_KEY:-}\",
    \"WANDB_PROJECT\": \"${WANDB_PROJECT}\",
    \"WANDB_MODE\": \"${WANDB_MODE:-online}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 "${ROOT_DIR}/train_async.py" \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node "${N_GPUS}" \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "${MODEL_PATH}" \
  --save "${SAVE_PATH}" \
  --save-interval 50 \
  --rollout-function-path sft_mat.generate_rollout \
  --prompt-data "${DATA_PATH}" \
  --input-key messages \
  --multimodal-keys '{"image": "image_path"}' \
  --metadata-key metadata \
  --rollout-shuffle \
  --num-epoch 3 \
  --rollout-batch-size 1 \
  --global-batch-size 1 \
  --loss-mask-type qwen3 \
  --loss-type sft_loss \
  --calculate-per-token-loss \
  --disable-compute-advantages-and-returns \
  --debug-train-only \
  --tensor-model-parallel-size "${TP_SIZE}" \
  --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 \
  --expert-model-parallel-size 1 \
  --expert-tensor-parallel-size 1 \
  --use-dynamic-batch-size \
  --max-tokens-per-gpu 256 \
  --recompute-granularity full \
  --recompute-method uniform \
  --recompute-num-layers 1 \
  --init-model-with-meta-device \
  --optimizer adam \
  --lr 2e-6 \
  --lr-decay-style cosine \
  --min-lr 2e-7 \
  --lr-warmup-fraction 0.05 \
  --weight-decay 0.1 \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --optimizer-cpu-offload \
  --main-params-dtype fp16 \
  --use-precision-aware-optimizer \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --main-grads-dtype bf16 \
  --attention-softmax-in-fp32 \
  --attention-backend flash \
  --megatron-to-hf-mode bridge \
  --use-wandb \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-group "${WANDB_GROUP}"
