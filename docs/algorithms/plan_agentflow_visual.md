# Plan：把 AgentFlow 扩展为多模态视觉 Agentic RL(基于 Visual-ARFT / MAT-Coding）

> 状态：规划中（草案 v2 — 已采用方案 C：在线多轮 + 路径无关奖励）
> 作者上下文：8×4090(24GB) · slime 已确认可跑 Qwen3-VL · 目标是一篇有现成 baseline 的论文
> 关联文档：[AgentFlow 介绍](./agentflow.md) · [ToolOrchestra 介绍](./toolorchestra.md)

---

## 0. TL;DR

在 **AgentFlow 的在线多轮 RL 脚手架**上，复现并改造 **Visual-ARFT 的 MAT-Coding 任务**(图像诊断→写 OpenCV 代码修图→回答)。
- **Baseline**：Visual-ARFT（Qwen2.5-VL-3B，8 GPU，数据全开，规则可验证奖励，已超 GPT-4o）。
- **我们的 delta**：把 Visual-ARFT 的「离线预拆步级 RFT、训练不执行代码」升级为「**在线多轮 rollout + 真工具执行 + Planner/Verifier 解耦 + verifier 门控**」，并用 Call Gain / Call Harm 指标证明结构带来「真驾驭工具」。
- **基座**：Qwen3-VL（slime 已支持转换器 `qwen3_vl`）。

---

## 1. 背景与动机

### 1.1 为什么是 Visual-ARFT（而非 DeepEyes 系）
- DeepEyes/DeepEyesV2：算力重、数据未完全开源、直接竞争 SOTA → 排除。
- Visual-ARFT 满足全部约束：
  - **算力**：Qwen2.5-VL-**3B** 选项 + 8 GPU + DeepSpeed ZeRO-3 **offload** → 8×4090 可行。
  - **数据全开且小**：MAT-Coding **1,200** 条、MAT-Search **20** 条（HuggingFace）。
  - **奖励廉价稳定**：纯规则可验证（F1 + format + 步类型校验），**无 LLM judge、训练不执行代码**。
  - **强 baseline**：MAT-Coding 上 +18.56% F1 / +13.00% EM（7B），超 GPT-4o。

### 1.2 关键洞察（决定改造设计）
Visual-ARFT 有「两个循环」：

| | 训练 `grpo_agent_code.py` | 评测 `evaluation_mat_coding_visual_arft.py` |
|---|---|---|
| 多轮 | ❌ 轨迹**离线预拆**成单步样本（`type`=pre_problem/pre_code/pre_answer） | ✅ 真 agent loop |
| 执行代码 | ❌ 训练中从不 `exec`，code 步固定给 0.9 | ✅ `extract_and_run_code` 真 `exec` OpenCV 改图 |
| context | 数据里老师写好的 `context` 字段 | 上一轮真实输出 + `<tips>` |

→ **AgentFlow 的本质（在线多轮 + 真执行 + turns 拆分训练）正好补上训练侧缺的东西。这就是论文的差异化。**

#### 1.2.1 更正：Visual-ARFT 训练是「单轮 + 无 SFT + 离线预拆步级」（已核实源码）
- **无 SFT / 无 cold-start**：repo 只有 `run_grpo_*` 脚本，纯 RFT 从 instruct 模型直接 GRPO。
- **训练单轮**：`make_conversation_image` 每条只有**一个 `{"role":"user"}`** 消息，模型对该 prompt 生成**一个 completion（一步）**；`num_generations=8` 是对同一 prompt 的 GRPO 组。无 assistant 轮、无多轮。
- **"多步"是离线伪造**：轨迹拆成 pre_problem/pre_code/pre_answer 三类独立样本，前序步骤被老师写进 `context`（teacher-forced），每步有自己的 GT `solution`。
- **直接后果**：其 `accuracy_reward` **依赖每步 GT `solution`**，只在离线预拆下成立。**AgentFlow 在线多轮里模型走自己的路径、没有逐步 GT，故该 reward 不能直接套** → 这正是采用方案 C 的原因（见 §4.0）。

#### 1.2.2 监督密度权衡（为什么是 C 不是 B）
| | Visual-ARFT（单轮步级） | AgentFlow 在线多轮 |
|---|---|---|
| 监督密度 | 稠密（每步对 GT） | 稀疏（通常只有 final outcome） |
| context | 老师给 | 模型自生成 |
| 训练中工具执行 | ❌ | ✅ 真执行 |
| 1200 条下 RL 难度 | 低 | 高（稀疏 + 多步信用分配） |

纯 outcome（方案 B）在 1200 条小数据 + 多步上风险高；方案 C 用**路径无关的廉价可验证信号**补回稠密监督。

### 1.3 风险认知（来自 disentangling 论文 arXiv:2602.01334）
- crop-zoom 类视觉工具 RL，>70% 增益来自「内在能力变强」，工具本身只占 ~20-30%，模型多半学会「与工具安全共存」而非「驾驭」。
- 对策：奖励要面向 **selective successful correction**（原本失败→工具救回给正分；原本成功→工具搞砸给惩罚）；用 **Call Gain / Call Harm** 评测。AgentFlow 的 **Verifier 门控**天然适配这个故事。

---

## 2. 目标与贡献

### 2.1 研究论点
> Visual-ARFT 用**单轮、离线预拆步级、teacher-forced context、训练不执行工具、无 SFT** 的 RFT 做视觉 agentic 训练。我们引入 **AgentFlow 式在线多轮 rollout（自生成 context）+ Planner–Verifier 解耦 + 真工具执行 + 路径无关的可验证奖励（format + code-exec + diagnosis + outcome，方案 C）+ correction-targeted 整形**，在**相同 MAT 基准、相同工具、相同基座**下，以更少无效工具调用换取更高 F1，并用 Call Gain / Call Harm 证明结构有效性。

### 2.2 可交付
- [ ] 在 slime+AgentFlow 上跑通 MAT-Coding 的在线多轮 RL（Qwen3-VL，8×4090）。
- [ ] 复现 Visual-ARFT baseline 数字（原 repo，作对照）。
- [ ] 报告 F1/EM + Call Gain/Call Harm 对比。

### 2.3 不做（避免范围蔓延）
- 不追 DeepEyesV2 SOTA。
- MAT-Search（仅 20 条，信号太稀）先放一边，主攻 MAT-Coding（1,200）。

---

## 3. Baseline 拆解（Visual-ARFT MAT-Coding，已扒源码）

### 3.1 任务
诊断图像问题 `{rotation90, rotation180, dark, overexposure, blur, noise, crop, none}` → 写 OpenCV 代码修复 → 基于修复后的图回答问题。

### 3.2 数据格式 `rft_agent_code_1_2k.json`
```jsonc
{
  "type": "pre_problem | pre_code | pre_answer",     // 轨迹的哪一步（离线预拆用）
  "image_path": "xxx.jpg",
  "problem": "任务/问题文本",
  "context": "前序步骤文本（pre_code/pre_answer 才有）",  // = AgentFlow 的 memory
  "solution": "<think>…</think><code>```python…```</code>",  // GT 下一步
  "gt": "最终标准答案"
}
```
- prompt = `SYSTEM_PROMPT_AGENT_CODE + '\n' + problem (+ context)`，图像作 `{"type":"image"}`。
- 输出协议（每步只出一个，前置 `<think>`）：`<problem>{...}</problem>` / `<code>```python ...```</code>` / `<answer>...</answer>`。
- 代码约定：读 `'path_to_input_image.jpg'`、写 `'path_to_output_image.jpg'`。

### 3.3 奖励（训练侧，两项相加，纯规则）
`accuracy_reward(completions, solution)` —— 按 GT `solution` 标签分支：
- GT `<answer>`：`compute_f1(学生答案, gt答案)`；学生错出 code/problem → 0
- GT `<code>`：学生也出 code 步 → **0.9**（代码正确性不验证，codebleu/exact 被注释）；错出 answer/problem → 0
- GT `<problem>`：比对诊断问题集合，全对 1.0 / rotation 半对 0.5 / 否则 0

`format_reward(completions, solution)` —— 严格正则匹配该步类型；重复标签(≥2) → 0。

辅助：`compute_f1` / `normalize` / `extract_problems` / `compute_code_similarity`(codebleu，未启用)。

### 3.4 评测侧真 agent loop（要对标的结构）
- 模型出一步 → 解析：
  - `<problem>`：注入 `<tips>`（crop 给 bbox 提示 / none 直接答 / 其它给修复提示），继续。
  - `<code>`：`exec` OpenCV，`path_to_input`→`path_to_output` 路径替换，crop 用归一化 bbox；成功则把输出图当作下一轮输入图。
  - `<answer>`：算 F1/EM，停止。
- MAT-Coding eval：200 条（70 simple + 130 hard）。

### 3.5 训练超参（3B，8 GPU 原配置）
DeepSpeed ZeRO-3 offload · `per_device_bs=1` · `grad_accum=2` · `num_generations=8` · `max_pixels=401408` · 图 resize 720×720 · 10 epoch · `save_steps=100` · flash_attention_2 · bf16。

### 3.6 数据规模与程序化扩充

#### 3.6.1 ✅ 已在真数据核实：是 1,200 个 distinct 起点（不是 ~400，担忧解除）
**核实结论**(`prepare_mat_data.py` 跑真数据)：`rft_agent_code_1_2k.json` 实为 **3,500 行 = 1,200 pre_problem + 1,200 pre_answer + 1,100 pre_code**。即文件里那「1,200」**本就是轨迹数**(1,200 个 original data points)，被离线预拆成 3,500 步行；其中 1,100 条三步(含 code)、100 条两步(`none` 无 code)。

| | Visual-ARFT（步级） | AgentFlow 在线（轨迹级） |
|---|---|---|
| distinct 起点 / GRPO 组 | 3,500 步行 | **1,200 轨迹**（实测，非 ~400） |
| 每 epoch 训练**序列**数 | 3,500 × 8 = 28,000 | 1,200 × (2~3 turn) × 8 ≈ 26k |

- **之前 "~400" 的估计是错的**：误以为 1,200 是步数。实际 distinct 起点 = 1,200，与 Visual-ARFT 的原始问题数一致 → **GRPO 组数不缩水，过拟合担忧大幅缓解**。
- 真实损坏分布(实测)：8 个单损坏类各 100；8 个双损坏(均含 rotation90/180)各 50 → 共 800 单 + 400 双 = 1,200。**双损坏必含 rotation 这点正好对上 `diagnosis_reward` 的 rotation 半对逻辑**。

#### 3.6.2 损坏是程序化的，且方案 C 不需要老师 CoT → 可无限合成
- 损坏类型 `{rotation90/180, dark, overexposure, blur, noise, crop, none}` 全是**确定性 cv2 操作**。
- 原 pipeline 里 GPT-4o 唯一的用处是生成逐步 `solution`/CoT；而**方案 C 用可验证奖励（format / exec / diagnosis 对已知损坏标签 / outcome F1），根本不需要老师 CoT** → 数据生成可完全自动、零成本。

合成 recipe（每条 = 一个轨迹起点）：
```
取干净图 + 已知答案的 (image, question, answer)   # 来自任意 VQA 数据集
  → 程序化施加一种已知损坏 (cv2)
  → 轨迹起点：{
        image: 损坏图,
        problem: question,
        gt: answer,                          # outcome reward 的 GT（继承自干净图 VQA）
        metadata.corruption_gt: 已施加的损坏类型  # diagnosis reward 的 GT（自己施加，已知）
     }
```
- diagnosis 与 outcome 两个 GT 都**由构造过程已知**，无需任何标注模型。
- 想要多少轨迹造多少 → 起点多样性可补回甚至超过原版。**不被 ~400 条卡住。**

#### 3.6.3 数据策略
- **阶段性**：先用原 1,200 拆出的 ~400 轨迹跑通 pipeline（M0–M3）；正式实验再程序化扩到 **2k–5k 轨迹**。
- **多样性来源**：干净图来自现成 VQA（保证 question/answer 真实）+ 多种损坏 × 多参数（角度/模糊核/噪声强度/crop 框）组合。
- **可控难度**：单损坏（simple）vs 多损坏叠加（hard），对齐 MAT-Bench 的 simple/hard 划分。

---

## 4. 改造映射：MAT-Coding → AgentFlow

### 4.0 方案 C：路径无关的奖励设计（核心，替代「原样搬 reward」）

在线多轮没有逐步 GT，故把奖励重构为**与模型实际路径无关、无需 teacher context**的廉价可验证信号，逐 turn 计算后由 AgentFlow 的 reward 平摊机制汇总：

| 奖励项 | 作用对象 | 是否需要 GT | 实现 |
|---|---|---|---|
| **format reward** | 每个 Planner 步 | 否 | 复用 Visual-ARFT 的 `format_reward` 正则（think+problem / think+code / think+answer），重复标签惩罚 |
| **code-execution reward** | code 步 | 否 | cv2 代码**真执行**成功 + 产出 output 图 → 正分；报错/无产出 → 0。**比 Visual-ARFT 强**（它训练不执行） |
| **diagnosis reward** | problem 步 | 用数据内已知损坏标签 | 损坏类型在数据里已知（原 `<problem>` GT 集合），存进 metadata，作路径无关信号；全对/半对/错（沿用其 1.0/0.5/0） |
| **outcome reward** | final answer | 用 `gt` | `compute_f1(final_answer, gt)`；经 custom_convert 按 turn 平摊到整条轨迹 |
| **(可选) correction-targeted 整形** | 轨迹级 | — | 原本失败→工具救回给正分、原本成功→工具搞砸给惩罚（对接 Call Gain/Harm 故事） |

> 关键：**diagnosis 的损坏标签 + final 的 gt** 这两个 GT 是「轨迹级、路径无关」的，不依赖模型走哪条路；format/exec 完全无需 GT。这样既保住稠密监督，又适配在线 rollout。

### 4.1 组件映射

| Visual-ARFT 组件 | AgentFlow 对应 | 改法 |
|---|---|---|
| `SYSTEM_PROMPT_AGENT_CODE`（think/problem/code/answer 协议） | `core/planner.py` 的 prompt | 协议搬进 Planner：`<problem>`=诊断、`<code>`=工具步、`<answer>`=final |
| `type` 预拆步 + `context` 字段 | **在线生成** turns + `core/memory.py` | 不预拆、不要 teacher context！`solver.solve` 在线跑 problem→code→answer，context 由 Memory 累积 |
| `extract_and_run_code`（`exec` cv2，换图） | `tools/python_coder/tool.py` + Executor | 执行 OpenCV、读 input→写 output、新图喂回下一轮；产出 exec 成功标志供 code-execution reward 用 |
| `<tips>` 提示注入 | Verifier / Executor 回灌 | crop 给 bbox、none 直接答 → 放进 Verifier 的 continue/stop + 提示逻辑 |
| `format_reward` + `compute_f1` + `extract_problems` | `rollout.reward_func`（按 §4.0 重构） | format/compute_f1/extract_problems 复制；`accuracy_reward` **不直接用**，改为 §4.0 的 exec+diagnosis+outcome |
| 损坏类型 GT（原 `<problem>` solution） | `sample.metadata["corruption_gt"]` | 数据转换时抽出，供 diagnosis reward |
| 数据 `image_path` | `sample.multimodal_inputs` | slime `--multimodal-keys '{"image":"image_path"}'`，图走 processor |
| `gt` | `sample.label` | `--label-key gt` |
| `num_generations=8` | `--n-samples-per-prompt 8` | 一致 |

---

## 5. AgentFlow 侧改动清单（文件级）

1. ✅ **数据转换脚本** `agentic/agentflow/prepare_mat_data.py`：MAT 轨迹数据 → AgentFlow 起点格式 `{problem(含<image>占位), gt, image_path(list), metadata.corruption_gt}`（在线 rollout 只需起点，丢弃 pre_* 预拆与 teacher `context`；从 pre_problem 的 `<problem>` solution 抽损坏类型 GT、从 pre_answer 的 `<answer>` 抽 final answer 作 gt）。**坑已踩**：① 一条轨迹的步行 `image_path` 在 `_proc`(损坏,起点)/`_ori`(干净,答案步)间变化，按去掉后缀的 base id 分组；② slime 要求 `image_path` 是 **list** 且 prompt 内含 `<image>` 占位(见 slime/utils/data.py)，否则多模态计数 assert 失败。
2. ✅ **引擎 `core/llm_engine.py`**：`SGLangEngine` 加 `processor` 参数，有图时用 processor 出 `input_ids`+`multimodal_train_inputs`、payload 发 `input_ids`+base64 `image_data`，`GenerationOutput` 返回 `multimodal_train_inputs`。参照 slime `rollout/sglang_rollout.py:120-151`。已做：惰性 slime 导入 + 纯函数抽离 + 10/10 离线测；**待 GPU 真验**(processor input_ids 与 SGLang 返回视觉 token 对齐)。**待接线**：`rollout.py` 把 `state.processor` 传进引擎。
3. **工具 `tools/python_coder/tool.py`**：换成 OpenCV 执行器（`exec` 代码、路径替换、返回新图路径）；加沙箱 + 超时。
4. ✅ **solver 协议**：用独立 `core/mat_solver.py`（`MATSolver`）实现 think/problem/code/answer 在线循环（不改 `core/solver.py`，math 路径不受影响）；图随轮更新（处理后新图回灌）、每 turn 存 `multimodal_train_inputs`；rollout 接线在 `rollout_mat.py`。
5. **reward `rollout.py:reward_func`**（按 §4.0 重构）：复制 `format_reward`+`compute_f1`+`extract_problems`；新增 code-execution reward（读工具 exec 成功标志）、diagnosis reward（对 `metadata.corruption_gt`）、outcome reward（`compute_f1(final, gt)`）。`accuracy_reward` 不直接用。custom_convert 仍按 turn 平摊 outcome。
6. **custom_convert `custom_convert.py`**：补 `multimodal_train_inputs` 的 emit（与每条训练序列一一对齐 append，无图填 None，trim 时同步切）。
7. ✅ **训练脚本** `agentflow_qwen3vl_mat.sh`：TP=2、`--multimodal-keys '{"image":"image_path"}'`、`--input-key problem --label-key gt`、`--metadata-key metadata`、context-length 16384、max-tokens-per-gpu 8192、custom hooks 指向 `rollout_mat.generate` / `core.mat_rewards.mat_reward_func` / `custom_convert.custom_convert`。`bash -n` 通过、四个 hook 符号已验证可解析。**待 GPU 补两件**：① `scripts/models/qwen3-vl-<size>.sh`(VLM backbone MODEL_ARGS,本仓无,需用 slime qwen3_vl 转换器配套);② checkpoint/数据/图像路径。

---

## 6. 分阶段实施（里程碑）

### M0 — 准备与对照（低风险，先做）
- [x] 数据转换脚本 `agentic/agentflow/prepare_mat_data.py`：MAT 3,500 步行 → **1,200** AgentFlow 起点 JSONL（`problem`+`<image>` / `image_path`(list) / `gt` / `metadata.corruption_gt`），已在真数据跑通(1200 条、0 空 gt/corruption)。测试 `tests/test_prepare_mat_data.py`(8/8)。
- [x] 数据下载 + 转换链路就绪：训练 `prepare_mat_data.py`（1200 起点）、评测 `prepare_mat_data.py --benchmark`（200=70+130，真数据跑通）。
- [x] Qwen3-VL backbone 配置 `scripts/models/qwen3-vl-4B.sh`（**核对 HF Qwen3-VL-4B config**，与 qwen3-4B 仅 rope_theta 5e6 不同；vision tower 由 slime qwen3_vl 处理）。
- [ ] **（需 GPU）** 在原 Visual-ARFT repo 跑通 3B baseline，记录 F1/EM（对照基准）。
- [ ] **（需 GPU）** 跑通 slime + Qwen3-VL 的最朴素单轮多模态 GRPO（验证转换器/架构这关）。

### M1 — Reward 先行（方案 C，部分可独立验证）✅ 已完成
- [x] 复制 `format_reward` / `compute_f1` / `extract_problems` 进 AgentFlow `reward_func`，离线单测对齐。
      → 落地于 `agentic/agentflow/core/mat_rewards.py`（primitives 逐字搬自 Visual-ARFT `grpo_agent_code.py`；`format_reward_step` 改成路径无关版：按 content 自身判定步类型而非 GT solution）。
- [x] 实现 diagnosis reward（对 `metadata.corruption_gt`）与 outcome reward（`compute_f1(final, gt)`），离线样本验证。
      → `diagnosis_reward` / `outcome_reward` + 轨迹级 `aggregate_mat_reward` + slime 入口 `mat_reward_func`（纯规则、无 LLM judge）。
- [x] code-execution reward 留到 M3（依赖真工具执行），M1 先打桩返回占位。
      → `code_exec_reward(exec_ok)`：`exec_ok=None` 时返回 0.0（M3 由 cv2 工具回填）。
- 测试：`agentic/agentflow/tests/test_mat_rewards.py`（17/17 通过，`python` 直接可跑，无 GPU/数据依赖）。
- 待 M3 由 solver/工具回填的 metadata 契约：`planner_steps`（每 Planner 步的 `{type, content}`）、`corruption_gt`、`code_exec_oks`；缺失时 `aggregate_mat_reward` 自动退化为 outcome-only。

### M2 — 引擎 + custom_convert 多模态打通（单轮）
- [x] **引擎发图 + 返回 `multimodal_train_inputs`**（`core/llm_engine.py` 重写，offline 可验部分完成）：
      - 新增 `processor` 参数；`messages_have_images` 检测图 → 走多模态路径(processor 出 `input_ids`+`multimodal_train_inputs`，payload 发 `input_ids`+base64 `image_data`)，否则原 text 路径(**向后兼容**，默认 `processor=None`)。镜像 `slime/rollout/sglang_rollout.py:120-151`。
      - slime 导入改**惰性**(`_post`/`_process_vision_info`/`_build_processor_kwargs`/`_encode_image` 模块级 seam)→ 模块无 slime 也能 import、且可 monkeypatch 离线测。
      - 抽出纯函数 `build_generate_payload` / `parse_generate_output` / `messages_have_images`。`GenerationOutput` 加 `multimodal_train_inputs` 字段。
      - 测试 `tests/test_llm_engine.py`(10/10：text/多模态两条 payload、processor 有图/无图分流、解析、真 slime `encode_image` roundtrip)。
- [x] **custom_convert emit 多模态**：`custom_convert.py` 给每条训练序列 append `multimodal_train_inputs`(无图 None)；turn 展开按 `turn["multimodal_train_inputs"]`、非 turn 按 `sample.multimodal_train_inputs`；trim 时与 `tokens_list` 同步切；仅当存在图时才 emit key(镜像 slime/ray/rollout.py:794)。测试 `tests/test_custom_convert.py`(5/5：非turn/turn展开/trim同步/混合/text-only 不 emit)。
- [x] **rollout 接线**：`rollout.py` 把 `state.processor` 传给三个 `SGLangEngine`(processor=None 时向后兼容,纯文本 AgentFlow 不受影响)。
- [ ] **（待做，需 GPU/真 processor）** 端到端验证：用真 Qwen3-VL processor + SGLang 跑 `plan()` 一步，验证 SGLang 返回 token 数与 processor `input_ids` 在视觉 token 上一致、loss 不报错。

### M3 — 在线多轮 + 真工具执行
- [x] **OpenCV 工具执行器**（offline 可验部分已完成）：
      - `core/image_tool.py`：`extract_code`（复用 Visual-ARFT 正则）→ `replace_paths`（占位→真路径，兼容单/双引号）→ killable 子进程 exec + timeout（plan §7.1 沙箱）→ 校验产出有效图；返回 `ImageExecResult.success` 即 M1 的 `code_exec_ok`。
      - `tools/opencv_editor/tool.py`：`OpenCV_Editor_Tool(BaseTool)`，`execute(code, input_image_path, output_image_path)` 直接跑模型自己的 `<code>`（区别于 python_coder 让 LLM 写码）。已在 `core/executor.py` 的 name mapping 注册。
      - 测试 `tests/test_image_tool.py`（11/11，真 cv2 合成损坏图：旋转修复成功 / 运行时错 / 不写输出 / 缺输入 / 无代码 / 超时 / wrapper 返回 dict）。
      - 简化：crop 的归一化 bbox 替换是 Visual-ARFT eval 的 teacher-forced hack，在线模型自己写真坐标，**不需要**；故未移植。
- [x] **solver 协议改造**（`core/mat_solver.py`，独立 `MATSolver`，不动 math solver）：
      - 在线循环 problem→(tips)→code→(exec+图回灌)→answer；植入 verbatim `SYSTEM_PROMPT_AGENT_CODE`。
      - 纯函数 `parse_step`（按 content 自身判 type + 抽 payload）、`build_tip`（crop/none/其它，文案搬自 eval）、`extract_answer`。
      - 每次模型生成=一个训练 turn（loss_mask=1 + 该轮 `multimodal_train_inputs`）；装配 `planner_steps`/`code_exec_oks`（对接 M1 契约）；`<code>` 步路由到注入的 `OpenCV_Editor_Tool`。
      - I/O（engine+tool）**注入式**→ 用 fake 脚本化整条轨迹离线测；测试 `tests/test_mat_solver.py`（9/9，含**真 cv2 工具+真旋转图**的集成测：只 fake LLM，验证图被修复并回灌）。
- [x] **处理后图重新编码进下一轮，turn 与「当轮用的图」对齐绑定**：`MATSolver` code 步成功后 `current_image=output`，下一轮 message 用新图；每 turn 存自己的 `multimodal_train_inputs`。失败则保留原图（已测）。
- [x] **rollout 接线** `rollout_mat.py`：`generate` 构 `SGLangEngine`(带 processor)+`OpenCV_Editor_Tool`+`MATSolver`，把 `final_output`/`planner_steps`/`code_exec_oks` 写进 `sample.metadata`、turns 进 `train_metadata`；`reward_func`=`mat_reward_func`。converter 增 `metadata.input_image_path`（cv2 读的损坏图路径）。**待 GPU 真验**。
- [x] **tips 门控**：以 `build_tip` 注入实现（crop 给 bbox / none 直接答 / 其它给修复提示）。注：当前是「系统按诊断注入 tips」而非独立 Verifier 打分门控；correction-targeted 的 Verifier 门控留待 M4 消融。

### M4 — 实验与对比
- [ ] F1/EM vs Visual-ARFT baseline。
- [ ] 实现 Call Gain / Call Harm 评测，证明结构有效。
- [ ] 消融：有/无 Verifier 门控、在线执行 vs 离线步、correction-targeted reward vs 原 reward。

---

## 7. 关键坑（务必盯）

1. **训练是否真执行代码**：Visual-ARFT 训练不执行（code 步固定 0.9）。走在线真执行能让 reward 升级为「基于处理后图能否答对」——更强，但要管 `exec` 安全（沙箱/超时）+ 处理后图的 token 成本。
2. **多轮中图会变**：原图→处理后图，每轮 `multimodal_train_inputs` 是不同的图；turn 拆分时必须绑定「那一轮当时的图」，不能只绑原图。
3. **视觉 token 的 loss_mask/prompt_len**：图像占位符展开成大量视觉 token，必须 mask=0、log_prob=0、计入 prompt_len；要验证 SGLang 返回 token 数与 processor input_ids 在视觉 token 上一致。
4. **custom_convert 对齐**：任何对 `tokens_list` 的操作（trim/过滤/turn 展开）都要对 `multimodal_train_inputs_list` 同步，否则图文错位训练崩。
5. **4090 显存**：沿用 ZeRO-3 offload 思路；context-length 砍小、图 resize、`max_pixels` 控制；TP=2。
6. **数据小（1200）**：提升空间有限，讲「低资源下结构更有效 + 工具选择性」的故事，别赌大涨点。

---

## 8. 评测方案 ✅ 代码就绪（`eval_mat.py` + `core/mat_eval_metrics.py`，9/9 离线测；待 GPU 真跑）

- **主指标**：MAT-Coding F1 / EM（200 条：70 simple + 130 hard）。→ `score_one`（max over answers）+ `aggregate`（overall/simple/hard）。
- **机制指标**：Call Gain（原本失败被工具救回）/ Call Harm（原本成功被工具搞砸）—— 来自 disentangling 论文。→ `call_gain_harm`（`eval_mat.py --baseline` 跑「无工具单轮」对照,EM 判对错,出 gain/harm/net/rate）。
- **对照**：Visual-ARFT 原版（同基座同数据） vs 本方案（AgentFlow 结构版）。
- **消融**：见 M4。
- **已知坑（实测）**：Visual-ARFT 的 `normalize` 会删冠词,单字母选择题答案 `"A"→"a"`(冠词)被删空 → **F1=0 但 EM=1**;选择题以 **EM** 为准(忠实于 baseline 的 normalize,已在 test 注释)。

---

## 9. 参考

- Visual-ARFT：arXiv:2505.14246 · GitHub `Liuziyu77/Visual-RFT`
- DeepEyes / DeepEyesV2：arXiv:2505.14362 / 2511.05271（仅作背景）
- Pixel Reasoner：arXiv:2505.15966（如何让模型肯用工具，避 learning trap）
- 工具效应拆解 + 评测指标：arXiv:2602.01334
- 本仓库 AgentFlow 实现：`agentic/agentflow/`
