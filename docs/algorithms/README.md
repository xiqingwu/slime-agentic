# Agentic RL 算法介绍文档

本目录收录 [slime-agentic](../../README.md) 中三个 Agentic RL 复现方法的**深度介绍文档**，
覆盖：复现的论文、核心方法、智能体流程、数据怎么做、reward 怎么设计、训练配置、评测、以及训练细节与注意事项。

| 文档 | 方法 | 复现论文 | 核心思想 | 训练对象 |
|---|---|---|---|---|
| [AgentFlow](./agentflow.md) | Planner→Executor→Verifier 多轮闭环 | [arXiv:2510.05592](https://arxiv.org/abs/2510.05592) | In-the-flow 优化决策模型 | Planner |
| [MemAgent](./memagent.md) | 固定大小循环记忆 + 分块阅读 | [arXiv:2507.02259](https://arxiv.org/abs/2507.02259) | Multi-Conv RL 压缩超长上下文 | 记忆更新模型 |
| [ToolOrchestra](./toolorchestra.md) | Orchestrator–Expert 多智能体路由 | [arXiv:2511.21689](https://arxiv.org/abs/2511.21689) | 多目标(对错/成本/延迟)路由 | Orchestrator |

## 进行中的方案

- [Plan：AgentFlow × 多模态视觉 Agentic RL](./plan_agentflow_visual.md) —— 基于 Visual-ARFT / MAT-Coding,在 AgentFlow 在线多轮 RL 脚手架上做视觉工具调用(Qwen3-VL,8×4090)。

## 三者的共性（与 slime 的集成方式）

每个方法都通过 slime 的四个自定义钩子接入训练循环：

```bash
--custom-generate-function-path              rollout.generate    # 多步 agent rollout,返回带 turns 的轨迹
--custom-rm-path                             rollout.reward_func # 自定义奖励
--custom-convert-samples-to-train-data-path  custom_convert.custom_convert  # 把多 turn 展开成训练序列
--custom-eval-rollout-log-function-path      rollout.eval_log    # 自定义 eval 日志(部分方法)
```

共同模式：

1. **多 turn 轨迹**：`generate` 把 agent 的每一步决策记成一个 turn，塞进 `sample.train_metadata["turns"]`。
2. **GRPO 组内相对优势**：同题 `n_samples_per_prompt` 条 rollout 一组，`(r - mean) / (std + eps)`，无需 critic。
3. **奖励平摊 / 加权**：AgentFlow 与 MemAgent 把单条轨迹奖励**按 turn 平摊**（J_Flow-GRPO / Multi-Conv）；
   ToolOrchestra 则做**多目标偏好加权**后再 GRPO。
4. **loss_mask 精确控制**：只有被训练角色的输出 mask=1，工具结果 / 固定模型输出 mask=0。
5. **KL 约束**：三者都用 `--kl-loss-type low_var_kl --kl-loss-coef 0.001` 抑制策略漂移。

## 关键区别速查

| 维度 | AgentFlow | MemAgent | ToolOrchestra |
|---|---|---|---|
| 基座模型 | Qwen2.5-7B | Qwen2.5-7B | Qwen3-8B |
| 数据集 | DAPO-Math-17K / AIME | HotpotQA / RULER-HQA | 自有 7227 条 / τ²-Bench·FRAMES·HLE |
| 数据转换脚本 | 无（现成 jsonl） | ✅ `prepare_data.py` | 训练数据预构建；评测有 `prepare_eval_data.py` |
| 奖励 | 0/1（规则 + LLM judge） | 0/1（EM 等价判定） | 多目标（对错 + 成本 + 延迟 + 路由偏好） |
| 奖励→turn | 平摊 `r/T` | 平摊 `r/T`（复用 AgentFlow） | 偏好加权 + 组内 min-max + GRPO |
| 外部服务 | 2 个固定 SGLang（base/coder） | 单引擎 + YaRN 长上下文 | 检索服务 + 7 个专家 SGLang + tau2 子进程 |
