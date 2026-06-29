# GPU 上机清单 — AgentFlow × MAT-Coding (Qwen3-VL-4B, 8×4090)

> 假设：服务器已有 CUDA、conda/pip 环境（Python 3.10+），可访问 HuggingFace 或镜像。
> 所有路径默认 `/data/`，可在脚本顶部统一改。

---

## 0. 代码 & 环境

```bash
# 把本仓库同步到服务器（或 git clone）
git clone <repo-url> /root/slime-agentic
cd /root/slime-agentic

# 切到开发分支
git checkout mat-agentflow-improvements

# 安装 slime 和依赖
pip install -e .
pip install sglang[all]                  # SGLang 推理引擎
pip install opencv-python-headless       # cv2（headless，无 GUI 依赖）
pip install qwen-vl-utils                # Qwen3-VL 图像预处理工具

# 验证 cv2 + MAT 离线测试能跑（无 GPU 也能跑）
cd /root/slime-agentic/agentic/agentflow
for t in tests/test_*.py; do python "$t" 2>&1 | tail -1; done
# 预期：全部 X/X passed（test_mat_solver.py 有 "Future exception" 警告属正常）
```

---

## 1. 数据准备

### 1-A. 下载 MAT 数据集

```bash
huggingface-cli download laolao77/MAT --repo-type dataset \
    --local-dir /data/MAT
```

下载后目录结构（关键路径）：
```
/data/MAT/
├── MAT-Training/
│   ├── rft_agent_code_1_2k.json   # 3500 行，1200 个轨迹起点
│   └── images/                    # 对应的图像文件
└── MAT-Benchmark/
    ├── MAT-Coding.json            # 200 条 benchmark
    └── MAT-Coding-image/          # benchmark 图像
```

### 1-B. 生成训练 JSONL

```bash
cd /root/slime-agentic/agentic/agentflow

python prepare_mat_data.py \
    --input      /data/MAT/MAT-Training/rft_agent_code_1_2k.json \
    --output     /data/MAT/mat_coding_agentflow.jsonl \
    --image-root /data/MAT/MAT-Training/images

# 验证：应输出 1200 行
wc -l /data/MAT/mat_coding_agentflow.jsonl
```

### 1-C. （可选但推荐）合成数据扩充

```bash
# 用 MAT 自己的干净图（_ori）重新施加程序化损坏，生成 4000 条额外起点
python synthesize_mat_data.py \
    --from-mat   /data/MAT/MAT-Training/rft_agent_code_1_2k.json \
    --image-root /data/MAT/MAT-Training/images \
    --num        4000 \
    --out-image-dir /data/MAT/synth_images \
    --output     /data/MAT/mat_synth.jsonl

# 合并真实集 + 合成集（真实集在前，保留 crop 类覆盖）
cat /data/MAT/mat_coding_agentflow.jsonl /data/MAT/mat_synth.jsonl \
    > /data/MAT/mat_train_all.jsonl

wc -l /data/MAT/mat_train_all.jsonl  # 应约 5200 行
```

如果用合成数据，把训练脚本里的 `--prompt-data` 改为 `mat_train_all.jsonl`。

### 1-D. 生成 Benchmark JSONL（用于 eval）

```bash
python prepare_mat_data.py --benchmark \
    --input      /data/MAT/MAT-Benchmark/MAT-Coding.json \
    --output     /data/MAT/mat_bench_agentflow.jsonl \
    --image-root /data/MAT/MAT-Benchmark/MAT-Coding-image

wc -l /data/MAT/mat_bench_agentflow.jsonl  # 应为 200 行
```

---

## 2. 模型准备

### 2-A. 下载 Qwen3-VL-4B-Instruct

```bash
huggingface-cli download Qwen/Qwen3-VL-4B-Instruct \
    --local-dir /data/models/qwen3_vl_4b
```

### 2-B. 转换为 Megatron-dist 格式

slime 训练需要 Megatron 分布式格式。

```bash
cd /root/slime-agentic
source scripts/models/qwen3-vl-4B.sh   # 加载 MODEL_ARGS

# 单卡转换即可（转换不需要多卡）
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /data/models/qwen3_vl_4b \
    --save          /data/models/qwen3_vl_4b_dist/
```

> **注**：`mbridge.AutoBridge` 需要 Megatron-LM 的 `megatron.bridge.models.qwen_vl` 支持。
> 如果报 `ImportError`，确认 `/root/Megatron-LM` 里有 Qwen3-VL bridge（`megatron/bridge/models/qwen_vl/`）。
> 部分 Megatron-LM 版本需要手动 checkout 含 VL 支持的分支。

转换成功后：
```bash
ls /data/models/qwen3_vl_4b_dist/      # 应有 model_optim_rng.pt 等分布式 ckpt 文件
```

---

## 3. 配置训练脚本

打开 `agentic/agentflow/agentflow_qwen3vl_mat.sh`，检查并修改以下段落：

### 3-A. CKPT_ARGS（路径）

```bash
CKPT_ARGS=(
   --hf-checkpoint   /data/models/qwen3_vl_4b          # HF 原始模型（给 processor / config）
   --ref-load        /data/models/qwen3_vl_4b_dist/    # 转换好的 Megatron-dist（ref policy）
   --save            /data/AgentFlow_Qwen3VL_MAT/       # 训练 ckpt 保存路径
   --save-interval   100
)
```

### 3-B. ROLLOUT_ARGS（数据路径）

```bash
--prompt-data /data/MAT/mat_coding_agentflow.jsonl   # 或 mat_train_all.jsonl（含合成）
```

### 3-C. （可选）开启评测

反注释 EVAL_ARGS 段落：

```bash
EVAL_ARGS=(
   --eval-interval      20
   --eval-prompt-data   mat /data/MAT/mat_bench_agentflow.jsonl
   --n-samples-per-eval-prompt 1
   --eval-max-response-len 4096
   --eval-top-p 0.95
)
```

### 3-D. （可选）开启 WandB

```bash
WANDB_ARGS=(
   --use-wandb
   --wandb-project AgentFlow_Visual
   --wandb-group   AgentFlow-Qwen3VL-MAT
   --wandb-key     <your_wandb_key>
)
```

### 3-E. RUNTIME_ENV_JSON 里的 PYTHONPATH

```json
"PYTHONPATH": "/root/Megatron-LM/:SCRIPT_DIR_PLACEHOLDER:/root/slime-agentic"
```

脚本里已用 `${SCRIPT_DIR}` 自动填，确认指向正确的 `agentic/agentflow/` 目录。

---

## 4. 小批量 e2e 验证（先别跑全量！）

正式训练前，先用极小配置验证整个 pipeline 不崩。

**临时改脚本里的这些参数**（或通过环境变量覆盖）：

```bash
# 只跑 2 步 rollout + 1 步训练，快速验证 pipeline
export ROLLOUT_BATCH_SIZE=2
export N_SAMPLES=2
export GLOBAL_BATCH_SIZE=4
export MAX_STEPS=2          # 在 MAT_MAX_STEPS 那行改
```

或者直接在脚本的 `ROLLOUT_ARGS` 段临时修改：

```bash
--rollout-batch-size 2
--n-samples-per-prompt 2
--global-batch-size 4
--num-epoch 1
```

启动：

```bash
bash agentic/agentflow/agentflow_qwen3vl_mat.sh
```

### 验证检查点（按顺序）

**① SGLang 服务是否成功启动**
```bash
# Ray job 提交后，等 SGLang 出现以下日志再往后看
grep "The server is fired up" /tmp/agentflow_logs/*.log 2>/dev/null
```

**② Processor 是否正常工作**

在 rollout 日志里找：
```
# 如果 multimodal 路径工作，会看到 input_ids 包含视觉 token（id 通常在 151xxx 段）
# 如果只有文字 token，说明 processor 没生效
```

**③ OpenCV 工具是否执行**

找 `[mat step X]` 日志中 `code executed` 或 `code failed`：
```
[mat step 1] unparseable step   # 说明模型输出格式不对（初期正常）
code executed; image updated    # 理想情况
code failed: ...                # 查具体报错
```

**④ reward 是否有值**

找 `[mat rollout 0]` 日志：
```
[mat rollout 0] score=0.xxx outcome=0.xxx format=0.xxx diagnosis=0.xxx code_exec=0.xxx correction=0.xxx
```
score 不应全为 0（否则 reward 函数没工作）。

**⑤ custom_convert 是否把 turn 拆开**

找 slime 训练端日志里类似：
```
custom_convert: trimming expanded samples from X to Y
```
说明 turn 展开 + trim 在工作。

---

## 5. 正式训练

验证通过后，恢复原始配置启动全量训练：

```bash
bash agentic/agentflow/agentflow_qwen3vl_mat.sh
```

关键环境变量（在脚本里已设置，可命令行覆盖）：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `MAT_MAX_PIXELS` | 401408 | = 512×28×28，每张图约 512 视觉 token。VRAM 富余可调大至 802816（1024×28×28）|
| `MAT_MAX_STEPS` | 5 | 每条轨迹最多几步（problem→code→answer）|
| `MAT_MAX_NEW_TOKENS` | 1024 | 每步最大生成 token 数，MAT 步骤短，不需要太大|
| `MAT_CORRECTION_REWARD` | 1 | A4 修正奖励（对比无工具 baseline），核心机制，保持开 |
| `MAT_CV2_FORKSERVER` | 0 | cv2 执行后端，先保持 0，验证稳定后可改 1 |
| `MAT_TOOL_TIMEOUT` | 30 | cv2 执行超时（秒）|

**VRAM 监控**（另开一个 terminal）：
```bash
watch -n 2 nvidia-smi
```

如果 OOM，优先调小 `MAT_MAX_PIXELS`（减半）或减少 `--max-tokens-per-gpu`。

**训练进度看这几个 log 位置**：
- `eval_scores.json`：每次 eval 的得分（若开启了 EVAL_ARGS）
- WandB `mat/outcome`、`mat/score`、`mat/tool_call_rate`：核心指标曲线

---

## 6. 评测

### 6-A. 转换 checkpoint 为 HF 格式

```bash
cd /root/slime-agentic
source scripts/models/qwen3-vl-4B.sh

# 单个 checkpoint
PYTHONPATH=/root/Megatron-LM python tools/convert_torch_dist_to_hf.py \
    ${MODEL_ARGS[@]} \
    --input-dir  /data/AgentFlow_Qwen3VL_MAT/iter_0000100/ \
    --output-dir /data/AgentFlow_Qwen3VL_MAT_HF/iter_0000100/ \
    --origin-hf-dir /data/models/qwen3_vl_4b
```

> 转换时需指定 `--model-name qwen3vlconfig`（如脚本不能自动识别时）。
> `convert_to_hf` 里的 `"qwen3vl" in model_name` 匹配 `Qwen3VLConfig` 的小写类名，通常能自动识别。

### 6-B. 运行 MAT-Coding 评测

```bash
cd /root/slime-agentic/agentic/agentflow

# 启动 SGLang 服务（评测只需一个 endpoint）
CUDA_VISIBLE_DEVICES=0,1 python -m sglang.launch_server \
    --model /data/AgentFlow_Qwen3VL_MAT_HF/iter_0000100/ \
    --port 30000 \
    --context-length 16384 \
    --tp 2 &

# 等服务就绪后运行评测
python eval_mat.py \
    --model      /data/AgentFlow_Qwen3VL_MAT_HF/iter_0000100/ \
    --eval-data  /data/MAT/mat_bench_agentflow.jsonl \
    --baseline \
    --output     mat_eval_results.json

# baseline 对比（无工具单轮，用 base 模型）
python eval_mat.py \
    --model      /data/models/qwen3_vl_4b \
    --eval-data  /data/MAT/mat_bench_agentflow.jsonl \
    --baseline \
    --output     mat_baseline_results.json
```

**评测报告包含**：
- F1/EM：overall + simple/hard 分组 + 每种 corruption 类型（`tool_by_type`）
- 机制统计：`diagnosis_accuracy`、`tool_call_rate`、`code_exec_success_rate`、`avg_steps`
- Call Gain / Call Harm（`--baseline` 时）：工具救回/搞砸的比例

---

## 7. 常见坑 & 排查

| 现象 | 可能原因 | 解决 |
|---|---|---|
| `ImportError: megatron.bridge.models.qwen_vl` | Megatron-LM 版本不含 VL bridge | 确认 Megatron-LM 有 `qwen_vl` 模块，或切对应分支 |
| SGLang OOM | 视觉 token 太多 | 减小 `MAT_MAX_PIXELS`，如 200704（= 256×28×28）|
| `code failed: timeout` | cv2 代码超时 | 增大 `MAT_TOOL_TIMEOUT`，检查代码是否死循环 |
| reward 全为 0 | processor 未生效，图没传到模型 | 检查 `processor` 是否 None，看 `messages_have_images` 输出 |
| `prompt must be a list` assert | slime processor assert | 确认 `--multimodal-keys` 已设置且 `image_path` 字段是 list |
| 视觉 token 对不上 | input_ids 与 SGLang logprob 长度不一致 | 这是 §7.3 验证点，检查 processor 输出 token 数 == SGLang logprob 数 |
| `no corrupted-image path` | 图路径不在 sample.prompt 里 | 检查 prepare_mat_data 输出的 `image_path` 字段 |
| 训练 loss 不降 | 学习率/奖励信号问题 | 看 mat/outcome 曲线，确认 reward 有方差 |

---

## 8. 实验消融（M4）

训练稳定后，依次做：

1. **消融 correction_reward**：`MAT_CORRECTION_REWARD=0 bash agentflow_qwen3vl_mat.sh`，对比 Call Gain/Harm 变化。
2. **消融 code_exec_reward**：`MAT_W_CODE_EXEC=0.0` 环境变量。
3. **消融合成数据**：只用真实 1200 条（vs 含合成 ~5200 条）对比 F1。
4. **MAT_MAX_PIXELS 消融**：512×28×28 vs 1024×28×28，看 F1 vs 训练速度的 tradeoff。

每组实验保存不同 `--save` 路径，用 `eval_mat.py` 统一评测后对比报告。
