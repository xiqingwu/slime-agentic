# ToolOrchestra 算法介绍文档

> 代码位置：[`agentic/ToolOrchestra/`](../../agentic/ToolOrchestra/)
> 复现论文：**ToolOrchestra**（[arXiv:2511.21689](https://arxiv.org/abs/2511.21689)）
> 训练框架：基于 [slime](https://github.com/THUDM/slime) 的自定义 rollout / reward / convert 钩子

---

## 1. 一句话概述

ToolOrchestra 是一个 **Orchestrator–Expert（编排器–专家）多智能体框架**：
一个中心 **Orchestrator LLM** 通过多轮工具调用，学会把任务**路由**给最合适的专家模型和工具。
RL（GRPO）施加在 Orchestrator 的决策轨迹上，让它在**没有人工标注中间步骤**的情况下，
学会"在答对的前提下，尽量便宜、快、用对专家"。

与前两个方法最大的不同：**奖励是多目标的**——不只看对错，还把**调用成本、延迟、路由偏好**一起加权进来。

---

## 2. 复现的方法（论文核心思想）

| 概念 | 说明 |
|---|---|
| Orchestrator | 被训练的小模型（Qwen3-8B），负责决策"下一步调哪个工具/专家" |
| Expert | 一组固定的、不同规模/特长的专家模型（Qwen3-32B / 30B-A3B / 14B / Math / Coder / DeepSeek-R1 …） |
| 抽象路由 | Orchestrator 只看到抽象名 `expert-1/2/3`，真实模型由数据的 `model_mapping` 决定 → 策略不绑死具体模型 |
| 多目标奖励 | 在"答对"前提下，综合 accuracy / cost / latency / 路由偏好做偏好加权 |
| 工具调用 | retrieval（FAISS 检索）、call_expert（路由专家）、answer（出最终答案） |

### 决策流程

```
输入问题
  │
  ▼
Orchestrator LLM                       ← 决定下一步调哪个工具          (loss_mask=1)
  │
  └─► for turn in range(max_turns=12):
        ├─ parse_tool_call()           ← 解析模型输出里的 <tool_call>
        ├─ tool call                   ← FAISS 检索服务 (port 8000)    (结果注入 prompt，loss_mask=0)
        ├─ call_expert ──────────────► 路由到专家模型 (各自独立端口)    (loss_mask=0)
        └─ answer ──────────────────► 出最终答案 → 结束循环
  │
  ▼
GenerationOutput：所有 turn 的 token_ids + log_probs 拼接
                  loss_mask：Orchestrator 输出=1 / 工具结果=0
```

代码主线：[`rollout.generate`](../../agentic/ToolOrchestra/rollout.py#L29) →
[`OrchestraSolver.solve`](../../agentic/ToolOrchestra/orchestra_solver.py) →
按 `category` 分流到 `_solve_qa` 或 `_solve_func_call`。

---

## 3. 双任务架构（关键）

ToolOrchestra 的数据/流程实际上是**两类任务混编**，由 `metadata.category` 分流
（[`orchestra_solver.py:104`](../../agentic/ToolOrchestra/orchestra_solver.py#L104)）：

| category | 数量 | 任务来源 | 判定方式 |
|---|---|---|---|
| `qa` | 930（`stem____*`） | `problem` 是真实问题，`answer` 是标准答案 | 规则匹配 + DashScope LLM 判等 |
| `func_call` | 6297（medicine/movie/bank/weather… 等 domain） | `problem` **为空**，任务藏在 `metadata.example.user_scenario` | **tau2 子进程仿真**判定成败 |

### (a) QA 路径 [`_solve_qa`](../../agentic/ToolOrchestra/orchestra_solver.py#L113)
- 工具：`search`（走 FAISS 检索服务）、`enhance_reasoning`、`answer`。
- `PromptBuilder.build_qa` 拼 prompt，Orchestrator 多轮调工具/专家直到 `answer`，最后比对 `answer` 字段。

### (b) func_call 路径 [`_solve_func_call`](../../agentic/ToolOrchestra/orchestra_solver.py#L173)
- 起一个 **tau2 子进程**（[`tau2_adapter.py`](../../agentic/ToolOrchestra/tau2_adapter.py)），通过文件协议
  `input_N.json` / `output_N.json` 交互：tau2 扮演用户 + 环境，Orchestrator 做 function-calling 应答。
- 把 `metadata.example` 写成临时任务文件喂给 tau2，`model_mapping` / `tool_pricing` 一并传入，让子进程动态构建 `call_expert` 工具。
- 任务成败完全由 tau2 仿真给出（不靠 `answer` 字段）。

---

## 4. 多服务（多端口）部署架构

来自 [`launch.sh`](../../agentic/ToolOrchestra/launch.sh)。Orchestrator 训练在 GPU 4-7，其余 GPU 跑固定专家与检索：

| GPU | 服务 | 端口 |
|---|---|---|
| 0 | FAISS 检索服务（Qwen3-Embedding-8B） | 8000 |
| 0 | SGLang Qwen3-32B-FP8 | 30001 |
| 0 | SGLang Qwen2.5-Math-7B | 30003 |
| 1 | SGLang DeepSeek-R1-Distill-Qwen-32B / Qwen3-30B-A3B | 30005 / 30006 |
| 2 | SGLang Qwen2.5-Coder-32B / Qwen3-14B | 30002 / 30007 |
| 3 | SGLang Qwen2.5-Math-72B | 30004 |
| 4,5,6,7 | **训练 Orchestrator Qwen3-8B**（TP=2, DP=2） | — |

- 检索服务 [`retrieval_general_thought.py`](../../agentic/ToolOrchestra/retrieval_general_thought.py)：
  用 Qwen3-Embedding-8B 编码 query，在 FAISS 索引上检索（语料来自 HF 数据集 `multi-train/index` 的 `train.index` + `train.jsonl`）。
- 专家调用见 [`tools/expert_caller/`](../../agentic/ToolOrchestra/tools/)，`EXPERT_ENGINE_MAP` 把抽象专家名映射到上述端口。

---

## 5. 数据怎么做

仓库里 [`data/`](../../agentic/ToolOrchestra/data/) 有两份等价数据（各 7227 行）：

| 文件 | 角色 | top-level 字段 |
|---|---|---|
| [`data.jsonl`](../../agentic/ToolOrchestra/data/data.jsonl) | ToolOrchestra **原始扁平格式** | `id, index, category, model_mapping, tool_pricing, problem, tools, answer, example, pref_vec` |
| [`data_slime_full.jsonl`](../../agentic/ToolOrchestra/data/data_slime_full.jsonl) | **slime 适配版（训练实际用）** | `problem, answer, tools, metadata{…}` |

适配 = **字段下沉**：slime 只认 `--input-key problem` / `--label-key answer`，
所以 `problem/answer/tools` 留顶层，其余全塞进 `metadata`。
（⚠️ 训练数据是**预构建好直接入库**的，仓库没有 committed 这个转换脚本；`eid` 如 `weather____14577` 来自 ToolOrchestra 官方任务 id。）

### 一条样本的关键字段

```jsonc
{
  "problem": "",                       // func_call 为空,任务在 example 里;qa 才有真实题面
  "answer":  "",                       // func_call 不靠它,靠 tau2 判定
  "tools":   [{call_expert 等的 function-calling schema}],
  "metadata": {
    "eid": "weather____14577",
    "category": "func_call" | "qa",                          // 决定 rollout 分支
    "model_mapping": {"expert-1": "Qwen/Qwen3-32B", ...},    // 抽象名 → 真实模型
    "tool_pricing": {"Qwen/Qwen3-32B": {input/output 每百万 token 单价}},  // 算成本
    "pref_vec":  {"accuracy":3, "cost":0.1, "latency":0.1, "expert-1":3, ...}, // 多目标权重
    "example":   {tau2 任务定义: user_scenario / instructions / ...}          // func_call 的用户剧本
  }
}
```

设计要点：
- **`model_mapping` 是灵魂**：Orchestrator 永远只看抽象名 `expert-1/2/3`，映射由数据决定，学到的路由策略不绑死具体模型。
- **`tool_pricing` + `pref_vec`** 让 reward 能同时算 准确率 / 成本 / 延迟 / 路由偏好 的多目标加权。

### 评测数据（有脚本）

[`prepare_eval_data.py`](../../agentic/ToolOrchestra/prepare_eval_data.py)：
把 `frames.jsonl`（QA benchmark）+ tau2 `original_tasks.json`（func_call benchmark）转成
`eval_tau2.jsonl`（269）/ `eval_frames.jsonl` / `eval_combined.jsonl`（469），
并保证 `model_mapping` / `tool_pricing` **与训练数据严格一致**。

---

## 6. Reward 怎么设计（多目标，核心）

reward 故意**拆成两层**，因为多目标 min-max 归一化必须看到同一题的所有 rollout：

| 阶段 | 文件 | 做什么 |
|---|---|---|
| ① 抽特征 | [`reward.py`](../../agentic/ToolOrchestra/reward.py) `reward_func` | 单条 rollout → 4 个原始特征 |
| ② 算奖励 | [`custom_convert.py`](../../agentic/ToolOrchestra/custom_convert.py) `custom_convert` | 同题 N 条 → 偏好加权 + GRPO |

### 阶段 ①：抽 4 个原始特征 [`extract_features`](../../agentic/ToolOrchestra/reward.py#L276)

```
correctness   : 0/1     # qa: 归一化精确匹配 + DashScope LLM 判等兜底;func_call: tau2 reward>0
total_cost    : float   # 真金白银的 token 成本(用 tool_pricing 折算)
total_latency : float   # 各 turn latency_ms 之和
tool_counts   : {role_name: 次数}  # 每个专家被调几次
```

`total_cost` 逐 turn 累加两部分（[`reward.py:309-328`](../../agentic/ToolOrchestra/reward.py#L309)）：
```
total_cost += orch_in × price_in(orch) + orch_out × price_out(orch)         # Orchestrator 自身
model_name  = model_mapping[role_name]    # expert-1 → "Qwen/Qwen3-32B"
total_cost += expert_in × price_in(model) + expert_out × price_out(model)   # 调用的专家
```
- Orchestrator 单价 = `tool_pricing[trained_model_type]`，查不到就用 [`_infer_orch_pricing`](../../agentic/ToolOrchestra/reward.py#L259) 取**最便宜模型**当代理。
- **QA LLM 判等兜底**：qa 且规则判错时，调 DashScope（`qwen-turbo-latest`）问"学生答案是否等价"，
  判对则 correctness 改回 1（环境变量 `QA_REWARD_JUDGE_ENABLED` 控制，默认开）。

这阶段返回的 `score` 只是 `correctness`，完整奖励留到阶段 ②。各 turn 的埋点字段
（`orch_input_tokens` / `latency_ms` / `role_name` / `tau2_reward_info`）由
[`orchestra_solver.py`](../../agentic/ToolOrchestra/orchestra_solver.py#L135) 记录。

### 阶段 ②：偏好加权 + GRPO [`custom_convert.py`](../../agentic/ToolOrchestra/custom_convert.py)

**2a. 同题 N 条做组内 min-max 归一化 + 偏好加权**（[`_compute_preference_rewards`](../../agentic/ToolOrchestra/custom_convert.py#L53)）：
```
fv[role]       = tool_counts[role]
fv["accuracy"] = correctness
fv["cost"]     = -total_cost       # 取负!成本越低越好
fv["latency"]  = -total_latency    # 取负!越快越好

# 答错(correctness<0.5)→ reward = 0(没资格比成本/延迟)
# 答对 → reward = Σ_key  pref_vec[key] × minmax_组内(fv[key])
```

**2b. 组内 GRPO 标准化 + 过滤**（[`_grpo_normalize_and_filter`](../../agentic/ToolOrchestra/custom_convert.py#L114)）：
```
advantage = (reward - mean) / (std + 1e-6),  clip 到 [-3, 3]
std < 0.1 的组 → 整组丢弃(没学习信号,keep_mask=False,loss_mask 清零)
```

**2c. 多轮轨迹拆成独立训练序列**（[`custom_convert.py:267`](../../agentic/ToolOrchestra/custom_convert.py#L267)）：
一条 rollout 的每个 Orchestrator turn 拆成一条序列，**共享同一 advantage**；
无效格式的 turn（[`_turn_has_valid_format`](../../agentic/ToolOrchestra/custom_convert.py#L181)）或无效答案的样本
（[`_has_valid_answer`](../../agentic/ToolOrchestra/custom_convert.py#L158)）loss_mask 清零；最后 trim 到 `global_batch_size` 整数倍。

> 一句话：`tool_pricing` 把"调了哪些专家、用了多少 token"折算成成本特征，
> `pref_vec` 把 准确率/成本/延迟/路由偏好 加权成单一标量，
> 再经组内 min-max + GRPO 变成 advantage —— Orchestrator 学到"答对前提下尽量便宜、快、用对专家"。

---

## 7. 训练配置（关键超参）

来自 [`train_orchestra.sh`](../../agentic/ToolOrchestra/train_orchestra.sh)：

| 类别 | 参数 | 值 |
|---|---|---|
| 模型 | Orchestrator | Qwen3-8B（可替换） |
| 数据 | `prompt-data` / `input-key` / `label-key` | `data/data_slime_full.jsonl` / problem / answer |
| Rollout | `num-epoch` | 2 |
| Rollout | `rollout-batch-size` | 32 |
| Rollout | `n-samples-per-prompt` | 8（GRPO 组大小） |
| Rollout | `global-batch-size` | 128 |
| Rollout | `rollout-temperature` | 0.7 |
| Rollout | `rollout-max-response-len` | 16384 |
| 算法 | `advantage-estimator` | grpo |
| 算法 | `use-kl-loss` / `kl-loss-coef` / `kl-loss-type` | ✅ / 0.001 / `low_var_kl` |
| 算法 | `eps-clip` / `eps-clip-high` | 0.2 / 0.3 |
| 优化器 | `lr` / decay | 1e-6 / constant |
| 并行 | TP | 2（DP=2） |
| 性能 | `max-tokens-per-gpu` / `log-probs-max-tokens-per-gpu` | 8192 / 131072 |
| SGLang | `sglang-context-length` | 131072 |
| rollout | `enable_thinking` | True（Orchestrator 开 thinking） |

### slime 自定义钩子

```bash
--custom-generate-function-path              rollout.generate
--custom-rm-path                             rollout.reward_func
--custom-convert-samples-to-train-data-path  custom_convert.custom_convert
--reward-key                                 score
```

---

## 8. 评测

- benchmark：**τ²-Bench**（默认）、**FRAMES**、**HLE**，由 `BENCHMARK` 环境变量切换。
- 流程：先 [`convert_to_hf.sh`](../../agentic/ToolOrchestra/convert_to_hf.sh) 把 torch_dist checkpoint 转回 HF，
  再 [`eval_orchestra.sh`](../../agentic/ToolOrchestra/eval_orchestra.sh) 起全部服务跑评测
  （[`eval_orchestra.py`](../../agentic/ToolOrchestra/eval_orchestra.py)）。
- 结果落在 `/data/eval_results/{benchmark}_{timestamp}/`。

### 结果

| 模型 | 数据集 | Baseline (Qwen3-8B) | ToolOrchestra | 提升 |
|---|---|---|---|---|
| Qwen3-8B | τ²-Bench | 0.278 | 0.388 | **+0.110** |

权重已发布：[LMIS-ORG/ToolOrchestra_Slime_Agentic_Qwen3_8B](https://huggingface.co/LMIS-ORG/ToolOrchestra_Slime_Agentic_Qwen3_8B)。

---

## 9. 训练细节与注意事项

- **强依赖外部服务**：训练前必须先 `launch.sh` 起好检索服务（8000）+ 全部专家 SGLang（30001-30007）。
  检索服务加载慢（8B emb + 大 FAISS 索引），脚本里有 `wait_port` 等待逻辑，且默认**不重启**已运行的检索服务。
- **LLM API Key**：τ² 用户模拟器 + QA 判分都调 DashScope（阿里云百炼），需 `export DASHSCOPE_API_KEY=...`。
- **per-sample 超时**：`ROLLOUT_SAMPLE_TIMEOUT = 360s`，func_call 的 tau2 仿真可能很慢，超时样本标 `FAILED`，
  避免长尾拖垮整个 batch（[`rollout.py:26`](../../agentic/ToolOrchestra/rollout.py#L26)）。
- **`max_turns = 12`**：Orchestrator 最多 12 轮决策。
- **func_call 的 reward 来自子进程**：tau2 把 `reward_info` 写回 turn（`tau2_reward_info`），
  reward / convert 都从这里读，而不是 `answer` 字段。
- **`_is_eval` 标记**：`generate` 把 `evaluation` 写进 `metadata["_is_eval"]`，
  [`_save_rollout`](../../agentic/ToolOrchestra/reward.py#L338) 据此把 rollout 日志分别落到 `train/` 或 `eval/`（`ROLLOUT_LOG_DIR`）。
- **std<0.1 整组丢弃**：多目标奖励可能让同题各 rollout 差异很小，过滤掉无信号的组，避免无效梯度。
- **答错即 0**：偏好奖励的硬约束——成本/延迟/路由偏好只在"答对"后才比较，防止模型为省钱而摆烂。
- **显存紧张**：多个 32B/72B 专家 + KV cache 共享 GPU（脚本注释里有逐卡显存预算），
  部分服务用 `mem-fraction≈0.45` 双开，部署时需按实际显卡调整。
- **抽象路由的迁移性**：因为训练只见 `expert-1/2/3`，理论上换一套 `model_mapping` 即可把学到的路由策略迁移到新专家集合。
