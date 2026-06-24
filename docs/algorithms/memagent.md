# MemAgent 算法介绍文档

> 代码位置：[`agentic/memagent/`](../../agentic/memagent/)
> 复现论文：**MemAgent: Reshaping Long-Context LLM with Multi-Conv RL-based Memory Agent**（[arXiv:2507.02259](https://arxiv.org/abs/2507.02259)）
> 训练框架：基于 [slime](https://github.com/THUDM/slime) 的自定义 rollout / reward / convert 钩子

---

## 1. 一句话概述

MemAgent 把**任意长的文档**逐块（chunk-by-chunk）压缩进一个**固定大小的循环记忆（recurrent memory）**，
然后只凭这段记忆回答问题——模型**从头到尾都不会一次性看到全文**。
RL（GRPO）施加在**所有"记忆更新"轮次**上，采用 **Multi-Conversation** 训练目标，
让模型学会跨 chunk 保留关键信息。

核心价值：用一个 **7B** 模型 + 固定大小记忆，处理 7K → 448K token 的超长上下文，且效果稳定超过更大的基线模型。

---

## 2. 复现的方法（论文核心思想）

| 概念 | 说明 |
|---|---|
| 循环记忆 | 一个固定 token 上限的文本 memory，逐块被 LLM 覆盖式更新 |
| 分块阅读 | 把超长文档切成固定大小 chunk，模型每次只看「问题 + 当前记忆 + 一个 chunk」 |
| Multi-Conv RL | 每个记忆更新轮次是一段**独立的对话/训练序列**，互不依赖（context-independent） |
| 奖励平摊 | 最终答案的奖励被**均匀分摊到所有记忆更新轮次**，对齐论文的 Multi-Conv 目标 |
| O(n) 复杂度 | 因为每次只处理固定大小输入，长文处理从平方复杂度降到线性 |

### 处理流程

```
输入：question + 超长 document
  │
  ▼
memory = "No previous memory"
  │
  └─► for chunk in split(document, CHUNK_TOKENS):
        └─ LLM(problem, memory, chunk) → 覆盖更新 memory        (loss_mask=1，每个 chunk 一条训练序列)
  │
  ▼
LLM(problem, memory) → 最终答案放进 \boxed{}                    (loss_mask=1，但奖励同样平摊)
  │
  ▼
Reward：与 ground truth 做 EM / 等价判定（取多答案的 max）
        奖励 / 轮次数，均匀分摊到所有 turn
```

代码主线：[`rollout.generate`](../../agentic/memagent/rollout.py#L93)（直接在一个函数里完成整个循环，无独立 solver）。

---

## 3. 数据怎么做

MemAgent 是三个方法里**唯一有完整数据转换脚本**的（[`prepare_data.py`](../../agentic/memagent/prepare_data.py)），
因为源数据是字段复杂的 parquet。

### 源数据

- 训练：**HotpotQA**（[BytedTsinghua-SIA/hotpotqa](https://huggingface.co/datasets/BytedTsinghua-SIA/hotpotqa)，parquet）
- 评测：**RULER-HQA**（`eval_<length>.json`，7K → 448K 多种上下文长度）

### 转换规则（parquet → slime JSONL）

| 源字段（MemAgent parquet） | → slime JSONL |
|---|---|
| `prompt[0].content` | `prompt`（问题） |
| `reward_model.ground_truth[0]` | `label`（首个答案） |
| `context`（超长文档） | `metadata.context` |
| 全部可接受答案 | `metadata.ground_truth`（reward 做多答案匹配用） |
| `extra_info.num_docs` / `data_source` | `metadata.num_docs` / `metadata.data_source` |

输出格式：

```jsonc
{
  "prompt":  "Which film was released first?",
  "label":   "Titanic",
  "metadata": {"context": "<长文档>", "ground_truth": ["Titanic", ...], "num_docs": 200}
}
```

### 脚本能力（[`prepare_data.py`](../../agentic/memagent/prepare_data.py)）

- 三种输入：本地 parquet（`--input`）、HF 直拉（`--hf-dataset/--hf-split`）、HF 单文件（`--hf-file`，处理非标准的 `eval_*.json`）。
- 自动识别两种格式：训练集（`prompt`/`reward_model`）与评测集 RULER-HQA（`input`/`answers`）。
- 默认走 `hf-mirror.com` 镜像；用 `list_repo_files` 精确列文件，规避多 split 格式不一致的推断错误。

> **关键设计**：长文档进 `metadata.context`，**不进 prompt**——这正是 MemAgent「永不一次性看到全文」的根基。

用法示例：

```bash
# 本地 parquet
python agentic/memagent/prepare_data.py \
    --input /data/hotpotqa_hf/hotpotqa_train_process.parquet \
    --output /data/hotpotqa_slime/train.jsonl

# 直接从 HF
python agentic/memagent/prepare_data.py \
    --hf-dataset BytedTsinghua-SIA/hotpotqa --hf-split train \
    --output /data/hotpotqa_slime/train.jsonl
```

---

## 4. rollout 怎么做（核心循环）

见 [`generate`](../../agentic/memagent/rollout.py#L93)：

1. 从 `metadata.context` 取长文 → tokenizer 编码 → 按 `CHUNK_TOKENS` 切块，最多 `MAX_CHUNKS` 块。
2. `memory = "No previous memory"`，逐块调用 LLM，用 `_MEMORY_TEMPLATE` 拼「problem + memory + chunk」，
   输出**覆盖**成新 memory（每块上限 `MEM_MAX_MEMORY` token）。
3. 跑完所有块后，用 `_FINAL_TEMPLATE` 拼「problem + memory」，让模型把答案放进 `\boxed{}`（上限 `MEM_MAX_FINAL`）。
4. 每个 chunk 更新 + 最终回答都各记一个 turn（`tokens / response_length / loss_mask=[1]* / rollout_log_probs`），
   塞进 `sample.train_metadata = {"turns": ...}`。

Prompt 模板与原始 MemAgent 的 `TEMPLATE / TEMPLATE_FINAL_BOXED` 严格对齐
（[`rollout.py:46-74`](../../agentic/memagent/rollout.py#L46)）。

### 可调超参（环境变量）

| 环境变量 | 默认 | 含义 |
|---|---|---|
| `MEM_CHUNK_TOKENS` | 训练 5000 / 代码默认 2048 | 每块 token 数 |
| `MEM_MAX_MEMORY` | 1024 | 记忆最大 token |
| `MEM_MAX_FINAL` | 256 | 最终答案最大 token |
| `MEM_MAX_CHUNKS` | 512 | 最多切多少块 |

---

## 5. Reward 怎么设计

奖励是 **0/1 精确匹配**，并尽量对齐原始 MemAgent 的实现（见 [`reward_func`](../../agentic/memagent/rollout.py#L272)）：

```
1) 取 final_output 末尾 300 字符并小写化（与原实现一致，省 token）
2) 从中抽最后一个 \boxed{} 内容作为 pred
3) 对 metadata.ground_truth 里的每个候选答案做等价判定 _is_equiv，命中任一即 score = 1.0（取 max）
```

判等函数 `_is_equiv` / `_strip_string` / `_last_boxed_only_string` 都**逐行移植自原始 MemAgent 的 `hotpotqa.py`**，
做了大量 LaTeX/格式归一化（去 `\left\right`、`tfrac→frac`、去空格等），保证与论文判分一致。

### 从单标量到逐 turn 优势（Multi-Conv ↔ Flow-GRPO）

[`custom_convert`](../../agentic/memagent/custom_convert.py) **直接复用 AgentFlow 的实现**（`from agentflow.custom_convert import custom_convert`）：

1. 同题 `n_samples_per_prompt` 条 rollout 一组做 GRPO 归一化：`(r - mean) / (std + 1e-6)`。
2. 把每条轨迹的奖励**按 turn 数平摊**（`reward / T`），使各 turn 求和 = 求平均 —— 这正对应论文的 Multi-Conv RL 目标
   （奖励均匀分摊到所有记忆更新轮次）。
3. 每个 turn 展开成独立训练序列，trim 到 `global_batch_size` 整数倍。

---

## 6. 训练配置（关键超参）

来自 [`run_memagent_7b.sh`](../../agentic/memagent/run_memagent_7b.sh)：

| 类别 | 参数 | 值 |
|---|---|---|
| 模型 | base | Qwen2.5-7B（可替换） |
| Rollout | `rollout-batch-size` | 16 |
| Rollout | `n-samples-per-prompt` | 16（GRPO 组大小） |
| Rollout | `global-batch-size` | 256 |
| Rollout | `rollout-max-response-len` | 8192 |
| 算法 | `advantage-estimator` | grpo |
| 算法 | `use-kl-loss` / `kl-loss-coef` / `kl-loss-type` | ✅ / 0.001 / `low_var_kl` |
| 优化器 | `lr` / decay | 1e-6 / constant |
| 并行 | TP | 4 |
| 性能 | `use-dynamic-batch-size` / `max-tokens-per-gpu` | ✅ / 16384 |
| SGLang | `sglang-context-length` | 131072（配 YaRN 处理超长上下文） |
| 记忆 | `MEM_CHUNK_TOKENS` / `MEM_MAX_MEMORY` / `MEM_MAX_FINAL` / `MEM_MAX_CHUNKS` | 5000 / 1024 / 256 / 512 |

> 注意：`rollout-max-response-len` 只是框架层单次调用上限；**实际每轮输出 token 由 `MEM_MAX_MEMORY` / `MEM_MAX_FINAL` 控制**。

### slime 自定义钩子

```bash
--custom-generate-function-path              rollout.generate
--custom-rm-path                             rollout.reward_func
--custom-convert-samples-to-train-data-path  custom_convert.custom_convert
```

（训练脚本里 eval 默认注释关闭，需要时再开 `--eval-prompt-data` 并准备 `dev.jsonl`。）

---

## 7. 评测

- 评测集：**RULER-HQA**，覆盖 7K → 448K 多种上下文长度。
- 离线脚本：[`eval_ruler_hqa.py`](../../agentic/memagent/eval_ruler_hqa.py)（HQA）、
  [`eval_ruler_general.py`](../../agentic/memagent/eval_ruler_general.py)（通用 RULER）、
  [`eval_all_checkpoints.sh`](../../agentic/memagent/eval_all_checkpoints.sh)（批量）。
- 指标：**F1 / EM（exact match） / sub_EM（子串匹配）**，评测复用与训练相同的分块记忆流程。
- 评测同样受 `MEM_CHUNK_TOKENS`（默认 5000）/ `MEM_MAX_CHUNKS`（默认 512）控制。

### 结果

- 在 RULER-HQA 上（5 次取最优），MemAgent（7B）在所有上下文长度下**稳定超过所有基线，包括远大于它的模型**。
- 权重已发布：[LMIS-ORG/MemAgent_Slime_Agentic_Qwen2.5_7B](https://huggingface.co/LMIS-ORG/MemAgent_Slime_Agentic_Qwen2.5_7B)。

---

## 8. 训练细节与注意事项

- **记忆是覆盖式而非追加式**：每个 chunk 让模型重写整段 memory（固定上限），这是"固定大小记忆"的关键；
  prompt 模板明确要求"保留旧记忆中的相关信息 + 加入新信息"。
- **stop-token 文本级清洗**：`no_stop_trim=True` 时生成文本可能残留 `<|im_end|>` / `<|endoftext|>`，
  [`_strip_stop_tokens`](../../agentic/memagent/rollout.py#L84) 在把 memory 当作下一轮 prompt 之前先清掉
  （等价于原始 MemAgent 在 token 级 `remove_eos`）。
- **空 context 直接放弃**：没有 `metadata.context` 的样本 → `Sample.Status.ABORTED`。
- **多答案匹配**：reward 用 `metadata.ground_truth`（全部候选答案）取 max，比只用单个 `label` 更鲁棒。
- **YaRN 长上下文**：SGLang 用 `--sglang-context-length 131072` 配合 YaRN，承载长记忆 + 长 chunk。
- **奖励对齐细节**：只看 final_output **末尾 300 字符**且小写，是刻意与原论文 `hotpotqa.py` 保持一致（也顺带省 token）。
- **turn 数可变**：长文档 turn 多、短文档 turn 少，奖励按各自 `T` 平摊，保证不同长度样本的梯度尺度可比。
- **custom_convert 与 AgentFlow 共用**：两者的训练数据组织方式（多 turn 展开 + 奖励平摊）本质相同，所以直接 import 复用。
