#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# AgentFlow × MAT-Coding (multimodal visual agentic RL) — training launcher.
# Plan: docs/algorithms/plan_agentflow_visual.md §5.7. Target: 8×4090 (24GB).
#
# This drives the online MAT loop (problem→code→answer with real cv2 execution)
# via the multimodal wiring built in M1–M3:
#   data : prepare_mat_data.py        → JSONL of 1200 trajectory start points
#   gen  : rollout_mat.generate       → MATSolver + OpenCV_Editor_Tool
#   rm   : core.mat_rewards.mat_reward_func  (format+diagnosis+code_exec+outcome)
#   conv : custom_convert.custom_convert     (turn-split + multimodal alignment)
#
# ── BEFORE RUNNING (one-time, on the GPU box) ────────────────────────────────
# 1. Get the data + images:
#      huggingface-cli download laolao77/MAT --repo-type dataset --local-dir /data/MAT
#    then build the rollout JSONL (embeds absolute image paths via --image-root):
#      python prepare_mat_data.py \
#        --input  /data/MAT/MAT-Training/rft_agent_code_1_2k.json \
#        --output /data/MAT/mat_coding_agentflow.jsonl \
#        --image-root /data/MAT/MAT-Training/images   # <-- adjust to the real image dir
# 2. Convert the Qwen3-VL checkpoint to Megatron-dist (slime's qwen3_vl converter).
#    The LLM-backbone MODEL_ARGS are provided in scripts/models/qwen3-vl-4B.sh
#    (verified vs HF config). For another size, mirror it as qwen3-vl-<size>.sh.
# 3. Start SGLang servers for the tool engines if your rollout uses them. The MAT
#    loop only needs the policy engine (router) + the cv2 tool (no extra LLM port),
#    so unlike the math AgentFlow you do NOT need ports 30000/30001 here.
# ─────────────────────────────────────────────────────────────────────────────

# Cleanup previous runs
if [ "${SKIP_PROCESS_KILL}" != "1" ]; then
    pkill -9 sglang
    sleep 3
    ray stop --force
    pkill -9 ray
    pkill -9 python
    sleep 3
    pkill -9 ray
    pkill -9 python
fi

set -ex

# Save training trajectories to trajectories/ (default off)
SAVE_TRAJECTORY=${SAVE_TRAJECTORY:-"0"}

# Max online MAT steps per trajectory (problem→code→answer); read by rollout_mat.
MAT_MAX_STEPS=${MAT_MAX_STEPS:-"5"}

# Per-turn generation cap. MAT steps are short (a diagnosis set / a cv2 snippet / a
# brief answer); 1024 avoids wasting budget and speeds rollout. Raise if steps truncate.
MAT_MAX_NEW_TOKENS=${MAT_MAX_NEW_TOKENS:-"1024"}

# A4 correction-targeted reward: generate a no-tool baseline answer per rollout so the
# reward can score Call Gain/Harm. Costs one extra (greedy) generation per rollout; set
# to 0 to disable and fall back to outcome+shaping only.
MAT_CORRECTION_REWARD=${MAT_CORRECTION_REWARD:-"1"}

# B3 cv2 exec backend: 1 = warm forkserver (faster, avoids re-importing cv2 per code
# step); 0 = per-call resource-limited subprocess. Keep 0 until validated on the GPU box.
MAT_CV2_FORKSERVER=${MAT_CV2_FORKSERVER:-"0"}

export PYTHONBUFFERED=16
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
# cv2 in the tool subprocess must stay headless / single-threaded.
export OPENCV_IO_ENABLE_OPENEXR=0
export TOKENIZERS_PARALLELISM=false

# Detect NVLink availability
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then HAS_NVLINK=1; else HAS_NVLINK=0; fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# ── Model configuration ───────────────────────────────────────────────────────
# qwen3-vl-4B.sh holds the Qwen3-VL-4B LLM-backbone MODEL_ARGS (verified against the
# HF config; vision tower handled by slime's qwen3_vl support). For another size,
# create scripts/models/qwen3-vl-<size>.sh the same way and override MODEL_CONFIG.
MODEL_CONFIG="${MODEL_CONFIG:-${SCRIPT_DIR}/../../scripts/models/qwen3-vl-4B.sh}"
source "${MODEL_CONFIG}"

# Number of GPUs to use (8×4090).
N_GPUS=${N_GPUS:-8}

# Checkpoint arguments  (TODO: point at your converted Qwen3-VL paths)
CKPT_ARGS=(
   --hf-checkpoint   /data/models/qwen3_vl_4b
   --ref-load        /data/models/qwen3_vl_4b_dist/
   --save            /data/AgentFlow_Qwen3VL_MAT/
   --save-interval   100
)

# Rollout arguments
ROLLOUT_ARGS=(
   --prompt-data /data/MAT/mat_coding_agentflow.jsonl
   --input-key   problem
   --label-key   gt
   --multimodal-keys '{"image": "image_path"}'   # image_path is a LIST (see prepare_mat_data)
   --metadata-key metadata                        # carries corruption_gt / input_image_path
   --rollout-shuffle
   --reward-key score
   --num-epoch 3
   --rollout-batch-size 16
   --n-samples-per-prompt 8                        # = Visual-ARFT num_generations
   --rollout-max-response-len 4096                 # MAT steps are short; keep small for 4090
   --rollout-temperature 1.0
   --global-batch-size 64
   --balance-data
)

# Evaluation arguments (MAT-Bench: 200 = 70 simple + 130 hard).
# Build an eval JSONL the same way (prepare_mat_data on the benchmark split) and enable:
# EVAL_ARGS=(
#    --eval-interval 20
#    --eval-prompt-data mat /data/MAT/mat_bench_agentflow.jsonl
#    --n-samples-per-eval-prompt 1
#    --eval-max-response-len 4096
#    --eval-top-p 0.95
# )
EVAL_ARGS=()

# Performance arguments — TP=2 for 8×4090 (plan §5.7)
PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 8192        # smaller than math run: 24GB cards + vision tokens
)

# GRPO arguments (low_var_kl, same as math AgentFlow)
GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.001
   --kl-loss-type low_var_kl
   --entropy-coef 0.0
   --eps-clip 0.2
   --eps-clip-high 0.3
)

# Optimizer arguments
OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

# WandB (off by default)
WANDB_ARGS=()
# WANDB_ARGS=(
#    --use-wandb
#    --wandb-project AgentFlow_Visual
#    --wandb-group   AgentFlow-Qwen3VL-MAT
#    --wandb-key ${WANDB_KEY:-"your_wandb_key_here"}
# )

# SGLang arguments — context-length cut down hard for 4090 (plan §7.5)
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.7
   --sglang-context-length 16384
)

# Misc arguments
MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

# Custom MAT generation / reward / convert / logging hooks (the M1–M3 deliverables)
CUSTOM_ARGS=(
   --custom-generate-function-path rollout_mat.generate
   --custom-rm-path core.mat_rewards.mat_reward_func
   --custom-rollout-log-function-path rollout_mat.log_rollout   # logs mat/* reward+mechanism stats
   --custom-eval-rollout-log-function-path rollout_mat.eval_log
   --custom-convert-samples-to-train-data-path custom_convert.custom_convert
)

# Optional reward-weight overrides (defaults in core/mat_rewards.DEFAULT_WEIGHTS).
# Uncomment to tune without editing code; they're read by mat_reward_func.
# export MAT_W_OUTCOME=1.0 MAT_W_FORMAT=0.05 MAT_W_DIAGNOSIS=0.1 MAT_W_CODE_EXEC=0.05 MAT_W_CORRECTION=0.5

# Launch Ray head node
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${N_GPUS} \
   --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# Runtime environment (PYTHONPATH must include the agentflow dir so `core` / `tools`
# / rollout_mat / custom_convert import the same way the offline tests do).
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${SCRIPT_DIR}:/root/slime\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"SAVE_TRAJECTORY\": \"${SAVE_TRAJECTORY}\",
    \"MAT_MAX_STEPS\": \"${MAT_MAX_STEPS}\",
    \"MAT_MAX_NEW_TOKENS\": \"${MAT_MAX_NEW_TOKENS}\",
    \"MAT_CORRECTION_REWARD\": \"${MAT_CORRECTION_REWARD}\",
    \"MAT_CV2_FORKSERVER\": \"${MAT_CV2_FORKSERVER}\",
    \"TOKENIZERS_PARALLELISM\": \"false\",
    \"SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN\": \"1\"
  }
}"

# Submit training job
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node ${N_GPUS} \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${CUSTOM_ARGS[@]}
