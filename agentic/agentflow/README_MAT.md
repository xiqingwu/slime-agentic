# AgentFlow × MAT-Coding — Multimodal Visual Agentic RL

Extends AgentFlow into a **multimodal visual agent** that reproduces and upgrades
Visual-ARFT's **MAT-Coding** task: diagnose an image's corruption → write OpenCV
code to repair it → answer a question about the repaired image. Built on
[slime](https://github.com/THUDM/slime) with Qwen3-VL.

Full design: [`docs/algorithms/plan_agentflow_visual.md`](../../docs/algorithms/plan_agentflow_visual.md).

## What's different from Visual-ARFT

| | Visual-ARFT (baseline) | This work |
|---|---|---|
| Multi-turn | offline pre-split steps | **online** problem→code→answer loop |
| Code execution in training | ❌ (code step hard-coded 0.9) | ✅ **real cv2 execution**, processed image fed forward |
| Step context | teacher-written | model-generated (+ tips) |
| Reward | step-GT `accuracy_reward` (needs per-step GT) | **path-independent** format + diagnosis + code-exec + outcome + **correction-targeted** (§4.0) |
| Mechanism metric | — | **Call Gain / Call Harm** (does the tool selectively help), wired into both training reward and eval |

## Architecture

```
corrupted image + question
  │
  ▼  MATSolver (core/mat_solver.py), online loop, max_steps
  ├─ <problem> diagnose corruption        (training turn, loss_mask=1)
  │     └─ system injects <tips> (crop/none/other)
  ├─ <code>    OpenCV fix                  (training turn, loss_mask=1)
  │     └─ OpenCV_Editor_Tool executes it → processed image fed to next turn
  └─ <answer>  final answer               (training turn, loss_mask=1)
  │
  ▼  mat_reward_func (core/mat_rewards.py), rule-based, NO LLM judge
     format + diagnosis(corruption_gt) + code_exec(changed) + outcome(max F1/EM)
     + correction(vs no-tool baseline: +1 rescue / -1 break)
  │
  ▼  custom_convert: per-turn split, reward amortized, multimodal aligned
```

## Files

| File | Role |
|---|---|
| `prepare_mat_data.py` | MAT train/benchmark JSON → AgentFlow start-point JSONL (`--benchmark` for MAT-Bench) |
| `core/mat_rewards.py` | path-independent rewards (format/diagnosis/code-exec/outcome/correction) + `mat_reward_func` |
| `core/image_tool.py` + `tools/opencv_editor/` | OpenCV code execution (extract→path-rewrite→**resource-limited subprocess**→verify); not a true sandbox — see module docstring |
| `core/image_worker.py` | optional warm forkserver executor (`MAT_CV2_FORKSERVER=1`) — avoids re-importing cv2 per code step |
| `core/llm_engine.py` | SGLang engine with multimodal path (sends `image_data`, returns `multimodal_train_inputs`) |
| `core/mat_solver.py` | online MAT loop (`MATSolver`) + protocol prompt/tips |
| `custom_convert.py` | turn-split + multimodal_train_inputs alignment |
| `rollout_mat.py` | slime hooks: `generate` / `reward_func` |
| `core/mat_eval_metrics.py` + `eval_mat.py` | F1/EM by split + Call Gain/Harm; benchmark runner |
| `agentflow_qwen3vl_mat.sh` | training launcher (8×4090, TP=2) |

## Data prep

```bash
huggingface-cli download laolao77/MAT --repo-type dataset --local-dir /data/MAT

# training data: 3500 pre-split rows -> 1200 trajectory start points
python prepare_mat_data.py \
  --input  /data/MAT/MAT-Training/rft_agent_code_1_2k.json \
  --output /data/MAT/mat_coding_agentflow.jsonl \
  --image-root /data/MAT/MAT-Training/images

# benchmark: 200 items (70 simple + 130 hard)
python prepare_mat_data.py --benchmark \
  --input  /data/MAT/MAT-Benchmark/MAT-Coding.json \
  --output /data/MAT/mat_bench_agentflow.jsonl \
  --image-root /data/MAT/MAT-Benchmark/MAT-Coding-image
```

> Adjust `--image-root` to wherever the images actually live in your download.

## Train

```bash
# fill in checkpoint/data paths inside the script first
bash agentflow_qwen3vl_mat.sh
```

Uses `scripts/models/qwen3-vl-4B.sh` for the LLM backbone. Convert the Qwen3-VL
checkpoint with slime's `qwen3_vl` converter beforehand.

## Evaluate

```bash
python eval_mat.py --model /data/AgentFlow_Qwen3VL_MAT/ \
  --eval-data /data/MAT/mat_bench_agentflow.jsonl \
  --baseline --output mat_eval_results.json
```

`--baseline` adds a no-tool single-turn pass and reports Call Gain / Call Harm. The
baseline is the *same* definition used by the training correction reward
(`core.mat_solver.baseline_answer`), so the training signal and the eval metric
measure the same counterfactual.

## Env knobs (set in the launcher, exported into the Ray runtime)

| Env | Default | Effect |
|---|---|---|
| `MAT_MAX_STEPS` | 5 | max online steps per trajectory (problem→code→answer) |
| `MAT_MAX_NEW_TOKENS` | 1024 | per-turn generation cap (MAT steps are short) |
| `MAT_CORRECTION_REWARD` | 1 | generate the no-tool baseline + add the correction term |
| `MAT_CV2_FORKSERVER` | 0 | cv2 exec backend: warm forkserver (1) vs per-call subprocess (0) |

Rollout efficiency: the OpenCV tool is instantiated once and shared; the no-tool
baseline runs concurrently with the tool loop; and the N identical greedy baselines
of one GRPO group are collapsed to a single computation via in-flight dedup
(`core.mat_solver.single_flight`), cleared per call so a later step recomputes fresh.

## Tests (offline, no GPU)

```bash
cd agentic/agentflow
for t in tests/test_*.py; do python "$t"; done
```

93 unit tests cover rewards (incl. correction-targeted), data conversion, the cv2
tool (real OpenCV) and the forkserver executor, the multimodal engine,
custom_convert alignment, the online solver loop (incl. real-cv2 integration,
temp-file cleanup, and baseline single-flight dedup), the shared no-tool baseline,
and eval metrics.

## Status

Offline implementation complete and tested. Remaining steps require the GPU box:
Qwen3-VL checkpoint conversion, end-to-end training, baseline reproduction, and
the M4 experiments (F1/EM + Call Gain/Harm + ablations). See the plan's milestones.
