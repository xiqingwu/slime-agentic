# AgentFlow-ARFT on ms-swift

Independent migration of the MAT visual AgentFlow pipeline to ms-swift. The old
slime implementation is not imported at runtime.

## Architecture

```text
Qwen3-VL LoRA SFT
  -> vLLM rollout server
  -> MATMultiTurnScheduler
       <problem> -> <code> -> OpenCV -> repaired image -> <answer>
  -> rule reward (F1 + format + diagnosis + code execution)
  -> LoRA GRPO
```

The implementation targets the local ms-swift checkout at commit `798af9d`:

```text
/home/xiqingwu/Documents/workspace/ms-swift
```

## Layout

- `mat_agentflow/plugin.py`: ms-swift scheduler and reward registrations.
- `mat_agentflow/image_executor.py`: resource-limited OpenCV execution.
- `scripts/prepare_data.py`: raw MAT -> ms-swift SFT and GRPO JSONL.
- `scripts/train_sft_lora.sh`: 4-GPU Qwen3-VL LoRA bootstrap.
- `scripts/start_rollout_server.sh`: 1-GPU asynchronous vLLM rollout server.
- `scripts/train_grpo_lora.sh`: 3-GPU LoRA GRPO training.
- `evaluate.py`: standalone multi-turn VllmEngine evaluation.

## Environment

```bash
cd /home/xiqingwu/Documents/workspace/ms-swift
pip install -e .
pip install -r /path/to/slime-agentic/ms-swift-agentflow/requirements.txt
```

All launchers accept `MS_SWIFT_ROOT`; its default is the local path above.
After installation, validate the exact local API contract with:

```bash
PYTHONPATH=/path/to/ms-swift:/path/to/ms-swift-agentflow \
  python scripts/check_local_swift.py
```

## 1. Prepare data

```bash
cd /path/to/slime-agentic/ms-swift-agentflow
python scripts/prepare_data.py raw-mat \
  --input /data/MAT/MAT-Training/rft_agent_code_1_2k.json \
  --image-root /data/MAT/MAT-Training/images \
  --sft-output /data/MAT/mat_swift_sft.jsonl \
  --rl-output /data/MAT/mat_swift_rl.jsonl
```

Existing AgentFlow or synthesized ChartQA JSONL can be converted directly:

```bash
python scripts/prepare_data.py agentflow-jsonl \
  --input /data/MAT/chartqa_mat.jsonl \
  --output /data/MAT/chartqa_swift_rl.jsonl
```

## 2. LoRA SFT

```bash
MODEL=/data/models/Qwen3-VL-4B-Instruct \
DATASET=/data/MAT/mat_swift_sft.jsonl \
bash scripts/train_sft_lora.sh
```

The ViT and aligner are frozen. With 4 GPUs, per-device batch 1 and gradient
accumulation 8 produce an effective batch of 32.

GRPO computes a greedy no-tool baseline for correction-targeted reward by
default. Set `MAT_CORRECTION_REWARD=0` to disable the extra generation.

## 3. Online LoRA GRPO

Terminal A (GPU 0):

```bash
MODEL=/data/models/Qwen3-VL-4B-Instruct \
ADAPTER=/data/AgentFlow_Qwen3VL_MAT_SWIFT_SFT/last \
bash scripts/start_rollout_server.sh
```

Terminal B (GPUs 1-3):

```bash
MODEL=/data/AgentFlow_Qwen3VL_MAT_SWIFT_SFT/last \
DATASET=/data/MAT/mat_swift_rl.jsonl \
bash scripts/train_grpo_lora.sh
```

For a one-step infrastructure smoke test, use `scripts/run_grpo_smoke.sh`.

## 4. Evaluate

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
  --model /data/models/Qwen3-VL-4B-Instruct \
  --adapter /data/AgentFlow_Qwen3VL_MAT_SWIFT_SFT/last \
  --data /data/MAT/mat_swift_eval.jsonl \
  --baseline \
  --output mat_swift_eval.json
```

The summary reports F1, EM, format accuracy, diagnosis accuracy, tool-call rate,
code-execution success, and average turns.
With `--baseline`, it also reports baseline EM and Call Gain/Harm/net.

## Validation gates

1. SFT loss decreases and held-out generations contain `<problem>` then `<code>`.
2. Smoke logs show at least one `code executed; image updated` path.
3. GRPO reward is non-constant within generation groups.
4. Evaluation reports non-zero tool-call and code-execution rates.
