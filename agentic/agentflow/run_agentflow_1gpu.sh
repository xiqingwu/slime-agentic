#!/bin/bash
# AgentFlow minimal demo: 1 GPU Qwen2.5-0.5B GRPO on dapo-math-17k
#
# Prerequisites:
#   1. slime env built (build_conda.sh complete) with slime/ moved to slime_bak/
#   2. Qwen2.5-0.5B-Instruct at /root/models/Qwen2.5-0.5B-Instruct
#   3. dapo-math-17k at /root/datasets/dapo-math-17k
#
# Usage: bash run_agentflow_1gpu.sh

set -ex

# Cleanup
if [ "${SKIP_PROCESS_KILL}" != "1" ]; then
    pkill -9 sglang 2>/dev/null || true
    sleep 2
    ray stop --force 2>/dev/null || true
    pkill -9 ray 2>/dev/null || true
    sleep 2
    pkill -9 ray 2>/dev/null || true
fi

export PYTHONUNBUFFERED=1
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ── Model config (Qwen2.5-0.5B) ───────────────────────────────────────────────
source "${ROOT_DIR}/scripts/models/qwen2.5-0.5B.sh"

MODEL_PATH="/root/models/Qwen2.5-0.5B-Instruct"

# ── Reduce TMS margin for tight single-GPU memory ─────────────────────────────
TMS_ARGS=(
   --train-memory-margin-bytes 0
)

# Conservative params for 0.5B model on 24GB VRAM
CKPT_ARGS=(
   --hf-checkpoint "${MODEL_PATH}"
   --ref-load "${MODEL_PATH}"
)

ROLLOUT_ARGS=(
   --prompt-data /root/datasets/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --rollout-shuffle
   --reward-key score
   --num-epoch 1
   --num-rollout 2
   --rollout-batch-size 2
   --n-samples-per-prompt 2
   --rollout-max-response-len 4096
   --rollout-temperature 0.7
   --global-batch-size 2
   --balance-data
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 2048
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.0
   --eps-clip 0.2
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

# Single GPU: minimal SGLang memory usage
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.3
   --sglang-disable-cuda-graph
   --sglang-context-length 8192
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --actor-num-nodes 1
   --actor-num-gpus-per-node 1
   --colocate
   --megatron-to-hf-mode bridge
   --ci-test
)

CUSTOM_ARGS=(
   --custom-generate-function-path rollout_1gpu.generate
   --custom-rm-path rollout_1gpu.reward_func
   --custom-eval-rollout-log-function-path rollout_1gpu.eval_log
   --custom-convert-samples-to-train-data-path custom_convert.custom_convert
)

# ── Launch ────────────────────────────────────────────────────────────────────

# Unset proxies to avoid Ray connection issues
for v in http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; do unset $v 2>/dev/null; done

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 1 --disable-usage-stats

RUNTIME_ENV_JSON='{
  "env_vars": {
    "PYTHONPATH": "/root/autodl-tmp/Megatron-LM/:'"${SCRIPT_DIR}"'",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_NVLS_ENABLE": "0"
  }
}'

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 /root/autodl-tmp/slime/train.py \
   ${MODEL_ARGS[@]} \
   ${TMS_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${CUSTOM_ARGS[@]}
