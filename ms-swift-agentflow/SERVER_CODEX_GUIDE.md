# ms-swift AgentFlow Server Handoff

This document is the primary handoff for a Codex agent working on the server.
Read it before changing code or launching training.

## 1. Project goal

This directory is a standalone migration of the MAT visual AgentFlow ARFT
pipeline from slime to ms-swift. It does not import the old
`agentic/agentflow` implementation at runtime.

The intended model behavior is a strict multi-turn loop:

```text
corrupted image + visual question
  -> <problem> diagnose corruption
  -> <code> generate OpenCV repair code
  -> execute code and append repaired image
  -> <answer> answer from the repaired image
```

Training has two stages:

1. LoRA SFT teaches the protocol and provides an adapter that can reliably
   produce `<problem>` and `<code>` actions.
2. LoRA GRPO uses online multi-turn rollouts and rule-based reward to improve
   answer quality, diagnosis, executable repairs, and correction benefit.

## 2. Repository and compatibility assumptions

- Project directory: `ms-swift-agentflow/`
- Git branch: `ms-swift-migration`
- Reference ms-swift checkout: commit `798af9d`
- Local development ms-swift path used during migration:
  `/home/xiqingwu/Documents/workspace/ms-swift`
- Target model: `Qwen3-VL-4B-Instruct`
- Intended 4-GPU allocation: GPU 0 for vLLM rollouts; GPUs 1-3 for GRPO
  training. SFT uses all four GPUs by default.

Do not assume newer ms-swift releases preserve every plugin API. Run
`scripts/check_local_swift.py` against the exact server checkout before GPU
jobs. The important APIs are `MultiTurnScheduler`, `RolloutInferRequest`,
`rollout_infos`, dynamic `images`, external ORM registration, and the vLLM
server mode arguments.

## 3. Code map

| Path | Responsibility |
| --- | --- |
| `swift_plugin.py` | External plugin entrypoint loaded by ms-swift CLI |
| `mat_agentflow/plugin.py` | Multi-turn scheduler, online baseline, reward ORM registration |
| `mat_agentflow/protocol.py` | Strict action grammar, parsers, system prompt, repair tips |
| `mat_agentflow/image_executor.py` | Resource-limited subprocess for generated OpenCV code |
| `mat_agentflow/rewards.py` | Answer metrics and aggregate MAT reward |
| `scripts/prepare_data.py` | Raw MAT or AgentFlow JSONL conversion to ms-swift schemas |
| `scripts/train_sft_lora.sh` | Qwen3-VL LoRA SFT launcher |
| `scripts/start_rollout_server.sh` | Async vLLM rollout server on one GPU |
| `scripts/train_grpo_lora.sh` | Three-GPU LoRA GRPO trainer |
| `scripts/run_grpo_smoke.sh` | One-step GRPO infrastructure smoke test |
| `evaluate.py` | Standalone multi-turn evaluation and no-tool baseline comparison |
| `tests/` | CPU tests for protocol, conversion, reward, executor, scheduler |

## 4. Runtime call chain

ms-swift loads `swift_plugin.py`, which imports `mat_agentflow.plugin`. Importing
that module registers:

```python
multi_turns["mat_agentflow"] = MATMultiTurnScheduler
orms["mat_agentflow_reward"] = MATReward
```

For every GRPO trajectory:

1. `on_trajectory_start()` optionally performs a greedy, no-tool VQA
   generation. This becomes the baseline for correction reward. Set
   `MAT_CORRECTION_REWARD=0` to disable it and reduce rollout cost.
2. `on_turn_end()` parses exactly one of `<problem>`, `<code>`, or `<answer>`.
   An answer or malformed action ends the trajectory.
3. `step()` responds to a diagnosis with a repair tip. For a code action it
   executes OpenCV against the most recent image, appends the repaired PIL
   image, and returns a tool response containing another `<image>` marker.
4. Scheduler metadata is returned in `rollout_infos`: planner steps, code
   execution outcomes, final answer, tool messages, baseline output, and the
   current image list.
5. `MATReward` combines dataset labels with this metadata and returns one
   scalar per completion to ms-swift GRPO.

The image executor replaces the literal paths
`path_to_input_image.jpg`/`path_to_output_image.jpg` with temporary files. It
blocks obvious process, network, dynamic-import, and shell operations, then
runs code in a subprocess with timeout, CPU, address-space, and file-size
limits. This is a basic containment layer, not a hardened multi-tenant sandbox.

## 5. Data contracts

### GRPO JSONL

Each line must resemble:

```json
{
  "messages": [
    {"role": "system", "content": "...strict MAT protocol..."},
    {"role": "user", "content": "<image>\nWhat is the number?"}
  ],
  "images": ["/absolute/path/to/corrupted.png"],
  "solution": "7500",
  "answers": ["7500", "7,500"],
  "corruption_gt": ["blur"],
  "trajectory_id": "unique-id",
  "input_image": "/absolute/path/to/corrupted.png"
}
```

Image paths must be absolute and valid on every worker. Before training, check
row counts, unique trajectory IDs, missing files, empty answers, and the
distribution of `corruption_gt`.

### SFT JSONL

Each line contains system/user/assistant messages, one image, `trajectory_id`,
and `step_type`. Raw MAT conversion emits separate examples for problem, code,
and answer steps. This stage is essential: if the SFT adapter never emits
`<code>`, GRPO cannot learn from a useful tool-use exploration distribution.

Prepare both datasets:

```bash
cd /path/to/slime-agentic/ms-swift-agentflow
python scripts/prepare_data.py raw-mat \
  --input /data/MAT/MAT-Training/rft_agent_code_1_2k.json \
  --image-root /data/MAT/MAT-Training/images \
  --sft-output /data/MAT/mat_swift_sft.jsonl \
  --rl-output /data/MAT/mat_swift_rl.jsonl
```

Existing synthesized AgentFlow JSONL can be converted with the
`agentflow-jsonl` subcommand. The converter does not synthesize new QA pairs;
it only changes schema and preserves each row's image, question, and answer.

## 6. Reward definition

The scalar reward is:

```text
R = answer_F1
  + 0.05 * format
  + 0.10 * diagnosis
  + 0.05 * code_exec
  + 0.50 * correction
```

- `answer_F1`: maximum normalized token F1 over accepted answers.
- `format`: fraction of parsed planner steps containing exactly one valid
  action tag.
- `diagnosis`: 1 only when the first problem set exactly matches
  `corruption_gt`.
- `code_exec`: mean of successful code executions that also changed the
  image. With no code, it is 1 only for the `none` corruption class.
- `correction`: +1 when the tool trajectory fixes a baseline error, -1 when it
  changes a correct baseline into a wrong answer, otherwise 0.

Exact match is reported as a metric but is not an additional outcome term.
With correction enabled, every trajectory incurs an extra greedy baseline
generation. Inspect reward variance within each six-generation GRPO group;
constant group rewards provide no useful policy gradient.

## 7. Server setup and fastest validation path

Install the server's ms-swift checkout and project dependencies:

```bash
cd /path/to/ms-swift
pip install -e .
pip install -r /path/to/slime-agentic/ms-swift-agentflow/requirements.txt

cd /path/to/slime-agentic/ms-swift-agentflow
PYTHONPATH=/path/to/ms-swift:$PWD python scripts/check_local_swift.py
pytest -q tests
bash -n scripts/*.sh
```

Then validate in this order. Do not start a long GRPO run before every earlier
gate passes.

### Gate A: data

- All image paths exist from Ray/trainer/vLLM processes.
- At least one SFT code example reads and writes the required literal paths.
- RL rows retain `answers` and `corruption_gt` after ms-swift dataset loading.

### Gate B: SFT behavior

```bash
MODEL=/data/models/Qwen3-VL-4B-Instruct \
DATASET=/data/MAT/mat_swift_sft.jsonl \
OUTPUT_DIR=/data/AgentFlow_Qwen3VL_MAT_SWIFT_SFT \
bash scripts/train_sft_lora.sh
```

Sample held-out generations. Require valid `<problem>` output and a non-zero
`<code>` rate before GRPO. A decreasing loss alone is not sufficient.

### Gate C: rollout server

```bash
MODEL=/data/models/Qwen3-VL-4B-Instruct \
ADAPTER=/data/AgentFlow_Qwen3VL_MAT_SWIFT_SFT/last \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/start_rollout_server.sh
```

Verify the adapter is loaded, multimodal requests succeed, and the server can
accept the configured number of images across turns.

### Gate D: one-step GRPO smoke

In a second terminal:

```bash
MODEL=/data/AgentFlow_Qwen3VL_MAT_SWIFT_SFT/last \
DATASET=/data/MAT/mat_swift_rl.jsonl \
CUDA_VISIBLE_DEVICES=1,2,3 \
bash scripts/run_grpo_smoke.sh
```

Require all of the following in logs or saved completions:

- at least one `<code>` action;
- `code executed; image updated` for at least one trajectory;
- a repaired image reaches the next model turn;
- non-constant rewards within a generation group;
- one optimizer step completes without OOM or tensor-shape failure.

### Gate E: evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
  --model /data/models/Qwen3-VL-4B-Instruct \
  --adapter /data/AgentFlow_Qwen3VL_MAT_SWIFT_SFT/last \
  --data /data/MAT/mat_swift_eval.jsonl \
  --baseline \
  --output mat_swift_eval.json
```

Track answer F1/EM, format, diagnosis, tool-call rate, code execution success,
average turns, baseline EM, and Call Gain/Harm/net. Use a held-out split; do
not report training-set metrics as final evaluation.

## 8. Resource defaults

The launch scripts use conservative defaults aimed at four GPUs:

| Job | Default allocation | Important defaults |
| --- | --- | --- |
| SFT | GPUs 0-3 | LoRA r=16, per-device batch 1, grad accumulation 8, ZeRO-2 |
| Rollout | GPU 0 | async vLLM, LoRA enabled, model length 16384, max 6 images |
| GRPO | GPUs 1-3 | LoRA r=16, batch 1, grad accumulation 4, 6 generations, ZeRO-2/offload |

Override paths and resources through environment variables rather than editing
launchers first: `MS_SWIFT_ROOT`, `MODEL`, `ADAPTER`, `DATASET`, `OUTPUT_DIR`,
`CUDA_VISIBLE_DEVICES`, `NPROC_PER_NODE`, `IMAGE_MAX_TOKEN_NUM`, and
`MAT_TOOL_TIMEOUT`.

If memory is tight, first reduce image tokens, completion length, generations,
or model length. Preserve multiple generations per prompt because GRPO needs
within-group comparisons. LoRA reduces trainable parameters but does not remove
activation, vision-token, rollout KV-cache, or reference-policy memory costs.

## 9. Known limitations and review targets

The migration has CPU-level test coverage, but has not been proven end to end
in this local environment because the ms-swift/vLLM/DeepSpeed GPU stack is not
installed here. Server validation is mandatory.

Review these points when the first server logs are available:

1. ms-swift API drift from reference commit `798af9d`.
2. Whether dynamic repaired images survive rollout server serialization and
   return through `rollout_infos` with correct token/loss-mask alignment.
3. Whether generated OpenCV code can import required libraries under the 8 GiB
   virtual-address limit.
4. Reward format accounting currently considers parsed/typed planner steps;
   malformed terminal actions should be inspected separately in evaluation.
5. Answer normalization is generic token normalization. Numeric variants such
   as `7,500` versus `7500` may need task-specific canonicalization.
6. The correction baseline doubles some rollout work and is not cached across
   identical prompts in a GRPO group.
7. The executor is suitable for controlled research workloads, not hostile
   untrusted code in a shared production environment.

When fixing server-only issues, add a focused regression test here whenever the
failure can be reproduced without GPUs. Keep changes independent from the old
slime implementation.

## 10. Current validation record

Completed locally during migration:

- `pytest -q ms-swift-agentflow/tests`: 11 tests passed.
- Python compilation passed for the full directory.
- Shell syntax validation passed for all launch scripts.
- Real OpenCV subprocess execution and forbidden-operation rejection passed.
- A 2,000-row synthesized ChartQA AgentFlow JSONL conversion produced 2,000
  unique QA rows with no missing image paths in the local dataset.

Not yet established by this record:

- server dependency installation and exact ms-swift API check;
- Qwen3-VL LoRA SFT completion;
- live multi-turn vLLM repaired-image round trip;
- GRPO optimizer smoke step;
- final held-out evaluation results.

Treat those as pending gates, not as completed claims.
