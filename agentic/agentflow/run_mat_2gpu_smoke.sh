#!/bin/bash
# MAT visual AgentFlow smoke test — 2×4090 (GPU0=SGLang, GPU1=Megatron)
#
# Prerequisites:
#   1. Qwen3-VL-4B-Instruct at /data/models/qwen3_vl_4b
#   2. MAT dataset at /data/MAT/mat_coding_agentflow.jsonl
#   3. 2 GPUs available
#
# Usage: MAT_MAX_PIXELS=200704 bash run_mat_2gpu_smoke.sh

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
export MAT_MAX_PIXELS="${MAT_MAX_PIXELS:-200704}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ── Model config (Qwen3-VL-4B) ─────────────────────────────────────────────
source "${ROOT_DIR}/scripts/models/qwen3-vl-4B.sh"

MODEL_PATH="/data/models/qwen3_vl_4b"

TMS_ARGS=(--train-memory-margin-bytes 0)

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_PATH}"
   --ref-load "${MODEL_PATH}"
)

ROLLOUT_ARGS=(
   --prompt-data /data/MAT/mat_smoke_subset.jsonl
   --input-key problem
   --label-key gt
   --apply-chat-template
   --rollout-shuffle
   --reward-key score
   --num-epoch 1
   --num-rollout 2
   --rollout-batch-size 2
   --n-samples-per-prompt 2
   --rollout-max-response-len 2048
   --rollout-temperature 0.7
   --global-batch-size 2
   --balance-data
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
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

# GPU0 dedicated to SGLang — can use most of the 24GB
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --rollout-num-gpus 1
   --colocate
   --sglang-mem-fraction-static 0.7
   --sglang-context-length 16384
)

# NOT colocated — SGLang on one GPU, Megatron on the other
MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --actor-num-nodes 1
   --actor-num-gpus-per-node 2
   --optimizer-cpu-offload
   --use-precision-aware-optimizer
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --megatron-to-hf-mode bridge
   --ci-test
)

# MAT custom hooks
CUSTOM_ARGS=(
   --custom-generate-function-path rollout_mat.generate
   --custom-rm-path core.mat_rewards.mat_reward_func
   --custom-rollout-log-function-path rollout_mat.log_rollout
   --custom-eval-rollout-log-function-path rollout_mat.eval_log
   --custom-convert-samples-to-train-data-path custom_convert.custom_convert
)

# ── Launch ─────────────────────────────────────────────────────────────────

for v in http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; do unset $v 2>/dev/null; done
export MASTER_ADDR="127.0.0.1"
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 2 --disable-usage-stats

RUNTIME_ENV_JSON='{
  "env_vars": {
    "PYTHONPATH": "/root/autodl-tmp/Megatron-LM/:'"${SCRIPT_DIR}"'",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_NVLS_ENABLE": "0",
    "MAT_MAX_PIXELS": "'"${MAT_MAX_PIXELS}"'"
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
