# AgentFlow 算法介绍文档

> 代码位置：[`agentic/agentflow/`](../../agentic/agentflow/)
> 复现论文：**AgentFlow: In-the-Flow Optimization of Agentic Systems**（[arXiv:2510.05592](https://arxiv.org/abs/2510.05592)）
> 训练框架：基于 [slime](https://github.com/THUDM/slime) 的自定义 rollout / reward / convert 钩子

---

## 1. 一句话概述

AgentFlow 把"单步 LLM 推理"扩展成一个 **Planner → Executor → Verifier** 的多轮智能体闭环，
并且**只对 Planner 的生成轨迹**施加 RL 信号（GRPO）。这样模型无需任何人工标注的中间步骤，
就能在"流程内（in-the-flow）"学会更好的工具使用与推理规划能力。

核心训练对象只有一个：**Planner**。Executor / Verifier / 各工具都用固定的基座模型，不参与梯度。

---

## 2. 复现的方法（论文核心思想）

| 概念 | 说明 |
|---|---|
| In-the-Flow 优化 | 不把 agent 当黑盒，而是在多轮智能体流程**内部**直接优化决策模型（Planner） |
| Flow-GRPO 目标 | 把一条多轮轨迹的奖励，按 turn 数平均分摊到每个 Planner 决策步上 |
| 模块解耦 | Planner（决策，可训练）/ Executor（生成工具命令，固定）/ Verifier（判断是否停止，固定） |
| 无中间标注 | 只需要最终问题的 ground truth，中间每一步用 RL 的相对优势自动学习 |

### 智能体流程

```
输入问题
  │
  ▼
Planner.plan()                       ← 分析问题、制定整体策略           (loss_mask=1，训练序列 #0)
  │
  └─► for step in range(max_steps=5):
        ├─ Planner.generate_next_step()    ← 选下一个工具 + 子目标       (loss_mask=1，训练序列 #1,#2,…)
        ├─ Executor.generate_tool_command()← 生成具体工具调用命令          (固定模型，不训练)
        │  + Executor.execute_command()    ← 真正执行工具                  (排除出序列)
        ├─ Verifier.verificate_context()   ← 判断 CONTINUE / STOP         (固定模型，不训练)
        └─ Memory.add_action()             ← 记录本步执行结果
  │
  ▼
Planner.generate_final_output()      ← 汇总记忆、产出最终答案            (固定 final_output 引擎，loss_mask=0)
  │
  ▼
Rewarder.compute_reward()            ← LLM-as-Judge 比对答案与 ground truth
```

代码主线：[`rollout.generate`](../../agentic/agentflow/rollout.py) → [`Solver.solve`](../../agentic/agentflow/core/solver.py) →
（[`Planner`](../../agentic/agentflow/core/planner.py) / [`Executor`](../../agentic/agentflow/core/executor.py) /
[`Verifier`](../../agentic/agentflow/core/verifier.py) / [`Memory`](../../agentic/agentflow/core/memory.py)）。

---

## 3. 多引擎（多端口）架构

AgentFlow 的不同角色被映射到不同的 SGLang 服务端口，关键设计是**把"被训练的 Planner"与"固定的评判/执行模型"物理隔离**，
保证奖励信号在整个 RL 过程中稳定（见 [`rollout.py:122-144`](../../agentic/agentflow/rollout.py#L122)）：

| 角色 (engine_map key) | 端口 | 模型 | 是否训练 |
|---|---|---|---|
| `planner` / `default` | sglang router（训练侧权重，随训练更新） | Qwen2.5-7B | ✅ 训练 |
| `executor` / `verifier` / `base_generator` / `final_output` | `127.0.0.1:30000` | 固定 Qwen2.5-7B-Instruct | ❌ 固定 |
| `python_coder` | `127.0.0.1:30001` | 固定 Qwen2.5-Coder-7B | ❌ 固定 |

> `final_output` 故意走固定基座模型：最终答案的生成不参与 loss（loss_mask=0），
> 这样"答得好不好"的奖励完全归因于 Planner 的规划质量，而不是最终复述能力。

### 工具系统（自动发现）

工具放在 [`tools/`](../../agentic/agentflow/tools/) 下，每个子目录一个 `tool.py`，导出 `TOOL_NAME` / `TOOL_DESCRIPTION`。
[`Planner._discover_tools`](../../agentic/agentflow/core/planner.py) 启动时扫描目录自动注册：

| 工具 | 说明 |
|---|---|
| `base_generator` | 通用文本生成工具，直接用 LLM 回答子任务 |
| `python_coder` | 生成并执行 Python 代码，用于数学计算 / 算法题 |

新增工具只需放一个新的 `tools/<name>/tool.py` 即可，无需改主流程。

---

## 4. 数据怎么做

AgentFlow **没有数据转换脚本**——它直接消费已是 slime JSONL 约定的数据文件。

### 数据格式

```jsonc
{"prompt": "数学题题面", "label": "标准答案"}
```

训练脚本通过 `--input-key prompt --label-key label` 接入（见
[`agentflow_qwen25_7b_rl_v2.sh`](../../agentic/agentflow/agentflow_qwen25_7b_rl_v2.sh)）。

| 用途 | 数据集 | 路径 |
|---|---|---|
| 训练 | **DAPO-Math-17K** | `/data/dapo-math-17k/dapo-math-17k.jsonl` |
| 评测 | **AIME 2024** | `/data/aime-2024/aime-2024.jsonl` |

### rollout 阶段对 sample 的处理

- 原始问题在 [`generate`](../../agentic/agentflow/rollout.py#L104) 里被存进 `sample.metadata["original_question"]`，
  因为 `sample.prompt` 之后会被 solver 拼好的多轮文本覆盖。reward / eval 都从 metadata 取原题，避免污染。
- solver 把每个 Planner turn 单独记录在 `out.turns` 里（含 `tokens / response_length / loss_mask / rollout_log_probs`），
  最后塞进 `sample.train_metadata = {"turns": ...}`，供 `custom_convert` 展开。

---

## 5. Reward 怎么设计

AgentFlow 的奖励**只看最终答案对不对**，是一个 0/1 的标量，由两级判定（见
[`reward_func`](../../agentic/agentflow/rollout.py#L209) + [`Rewarder`](../../agentic/agentflow/core/rewarder.py)）：

```
1) 规则快路：从 final_output 抽 \boxed{} → normalize → 若与 label 完全相等 → score = 1.0
2) LLM-as-Judge：否则调用固定的 Rewarder（端口 30000，严格数学判等 prompt）
                 → 模型输出 "VERDICT: True/False" → 1.0 / 0.0
```

设计要点：

- **判分模型用固定基座（30000），不是被训练的 Planner**，保证奖励信号在整个 RL 过程中不漂移
  （[`reward_func` 注释](../../agentic/agentflow/rollout.py#L213)）。
- LLM judge 的 prompt 明确要求"宁严勿松"（`Do NOT be lenient. When in doubt, output False`），
  数学等价（如 `1/2 == 0.5`、`1,000 == 1000`）算对。
- 返回 `{"score", "acc", "pred", "gt"}`，`--reward-key score` 指定用 `score` 做 RL 信号。

### 从单标量到逐 turn 优势（Flow-GRPO）

奖励是"整条轨迹一个分数"，但训练样本是"每个 Planner turn 一条序列"。
[`custom_convert`](../../agentic/agentflow/custom_convert.py) 做两件事：

1. **轨迹级 GRPO 归一化**：同一题的 `n_samples_per_prompt` 条 rollout 一组，
   `advantage = (r - mean) / (std + 1e-6)`（组内相对优势，无需 critic）。
2. **按 turn 平摊**：一条轨迹有 `T` 个 turn，每个 turn 的奖励设为 `normalized_reward / T`。
   这样"对各 turn 求和"恰好等于"对各 turn 求平均"，与论文的 **J_Flow-GRPO** 目标（跨 turn 平均）对齐
   （[`custom_convert.py` 注释](../../agentic/agentflow/custom_convert.py)）。
3. 把每个 turn 展开成独立训练序列，并 trim 到 `global_batch_size` 的整数倍（turn 展开后样本数不再天然可整除）。

---

## 6. 训练配置（关键超参）

来自 [`agentflow_qwen25_7b_rl_v2.sh`](../../agentic/agentflow/agentflow_qwen25_7b_rl_v2.sh)：

| 类别 | 参数 | 值 |
|---|---|---|
| 模型 | base / coder | Qwen2.5-7B / Qwen2.5-Coder-7B |
| Rollout | `rollout-batch-size` | 8 |
| Rollout | `n-samples-per-prompt` | 8（GRPO 组大小） |
| Rollout | `global-batch-size` | 64 |
| Rollout | `rollout-temperature` | 0.7 |
| Rollout | `rollout-max-response-len` | 32768 |
| 算法 | `advantage-estimator` | grpo |
| 算法 | `use-kl-loss` / `kl-loss-coef` / `kl-loss-type` | ✅ / 0.001 / `low_var_kl` |
| 算法 | `eps-clip` / `eps-clip-high` | 0.2 / 0.3（非对称裁剪） |
| 算法 | `entropy-coef` | 0.0 |
| 优化器 | `lr` / decay / `weight-decay` | 1e-6 / constant / 0.1 |
| 优化器 | adam-beta | (0.9, 0.98) |
| 并行 | TP / PP / CP | 4 / 1 / 1 |
| 性能 | `use-dynamic-batch-size` / `max-tokens-per-gpu` | ✅ / 16384 |
| 性能 | recompute | full / uniform / 1 layer |
| SGLang | `rollout-num-gpus-per-engine` | 4 |
| SGLang | `sglang-mem-fraction-static` / `context-length` | 0.75 / 131072 |
| 评测 | `eval-interval` | 20 |

### slime 自定义钩子

```bash
--custom-generate-function-path              rollout.generate
--custom-rm-path                             rollout.reward_func
--custom-eval-rollout-log-function-path      rollout.eval_log
--custom-convert-samples-to-train-data-path  custom_convert.custom_convert
```

---

## 7. 评测

- 评测集：**AIME 2024**，训练中每 `eval-interval=20` 跑一次。
- [`eval_log`](../../agentic/agentflow/rollout.py#L44) 自定义日志：把每步 eval 的 question / pred / label / score / final_output
  写入 `eval_scores.json`（返回 `False` 让框架继续跑默认日志）。
- 离线脚本：[`eval_agentflow.sh`](../../agentic/agentflow/eval_agentflow.sh)（训练后模型）、
  [`eval_baseline.sh`](../../agentic/agentflow/eval_baseline.sh)（基线）、
  [`eval_all_checkpoints.sh`](../../agentic/agentflow/eval_all_checkpoints.sh)（批量扫所有 checkpoint）。

### 结果

| 模型 | 数据集 | Baseline | AgentFlow | 提升 |
|---|---|---|---|---|
| Qwen2.5-7B-Instruct | AIME 2024 | 10.0% | 30.0% | **+20.0%** |

权重已发布：[LMIS-ORG/AgentFlow_Slime_Agentic_Qwen2.5_7B](https://huggingface.co/LMIS-ORG/AgentFlow_Slime_Agentic_Qwen2.5_7B)。

---

## 8. 训练细节与注意事项

- **不支持 partial rollout**：[`generate`](../../agentic/agentflow/rollout.py#L109) 开头有
  `assert not args.partial_rollout`，多轮轨迹无法做部分续跑。
- **token 预算保护**：solver 内有 `MAX_TOTAL_TOKENS = 131072`，累计超限会提前 STOP，避免单条轨迹爆显存
  （[`solver.py:161`](../../agentic/agentflow/core/solver.py#L161)）。
- **最大步数** `max_steps=5`：Verifier 输出 STOP 或步数耗尽即结束循环。
- **loss_mask 语义**：只有 Planner 的 `plan` / `generate_next_step` 输出 mask=1；
  拼接序列时工具结果、verifier、final_output 的 prompt 部分 mask=0、log_probs 填 0
  （[`solver.py:196-200`](../../agentic/agentflow/core/solver.py#L196)）。
- **enable_thinking=False**：Planner 引擎关闭 thinking 模式（[`rollout.py:113`](../../agentic/agentflow/rollout.py#L113)）。
- **必须先起两个固定 SGLang 服务**（30000 base、30001 coder），否则 Executor / Verifier / Rewarder / coder 工具会连不上。
- **轨迹落盘**：设 `SAVE_TRAJECTORY=1` 会把每条完整轨迹存到 `trajectories/`，便于调试（默认关闭）。
- **异常即丢弃**：solver 返回 None → `Sample.Status.ABORTED`；generate 抛异常 → `FAILED`，
  这些样本在 convert 阶段会被相应处理，不污染训练。
- **稳定性取舍**：训练侧保留 `TP=4`（7B 模型稳定性优先），KL 用 `low_var_kl` 配小系数 0.001 抑制策略漂移。
