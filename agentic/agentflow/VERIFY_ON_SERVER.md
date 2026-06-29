# 服务器验证清单 — slime v0.3.0 + AgentFlow 集成

> 目标:在服务器上**逐步确认** slime v0.3.0 升级 + 本分支改动
> (`rollout_ids`、去掉 custom_convert 的 trim、`max_pixels`)能端到端跑通。
>
> 策略:**先用单卡 0.5B 数学版冒烟测**(最便宜,直接验证 v0.3.0 训练链路不崩),
> 再上 MAT 视觉训练。每步都有「✅ 通过判据」,过了再走下一步。
>
> 路径按需替换为你服务器上的真实路径。

---

## 阶段 A — 代码与环境(5 分钟)

### A1. 同步代码到最新

```bash
cd /path/to/slime-agentic
git fetch origin
git checkout mat-agentflow-improvements
git pull origin mat-agentflow-improvements
git log --oneline -3
```

✅ 通过:最新 commit 含 `custom_convert: drop stale sample-count trim for slime v0.3.0`
(以及它之前的 `sync: replace vendor slime v0.2.2 with v0.3.0`)。

> ⚠️ 如果你在服务器上有本地未提交改动,先 `git stash` 再 pull,避免冲突。

---

### A2. 确认 slime 是 v0.3.0 且关键 API 在位

```bash
cd /path/to/slime-agentic
python -c "
from slime.utils.http_utils import post
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.processing_utils import process_vision_info, build_processor_kwargs, encode_image_for_rollout_engine, load_processor
from slime.utils.types import Sample
from slime.utils.metric_utils import compute_rollout_step
from slime.rollout.rm_hub.math_dapo_utils import last_boxed_only_string, remove_boxed, normalize_final_answer
import inspect
print('post sig     :', inspect.signature(post))
print('Sample fields:', 'rollout_id' in Sample.__dataclass_fields__, 'rollout_mask_sums OK')
print('ALL slime imports OK')
"
```

✅ 通过:打印 `post sig : (url, payload, max_retries=60, headers=None)` 且 `ALL slime imports OK`,无 ImportError。

❌ 若某个 import 报错:说明 v0.3.0 API 又漂了,**停下来**把报错贴回来再排查,不要继续。

---

### A3. 跑离线单测(无需 GPU)

```bash
cd /path/to/slime-agentic/agentic/agentflow
for t in tests/test_*.py; do printf "%-40s " "$(basename $t)"; python "$t" 2>/dev/null | tail -1; done
```

✅ 通过:每行 `X/X passed`,合计 120 个测试全过。

❌ 若有 FAIL:贴回失败的测试名 + 输出。

---

## 阶段 B — 单卡 0.5B 数学版冒烟测(核心验证,20–40 分钟)

> 这一步最关键:它真训练一次,直接验证 **v0.3.0 + 我们改的 custom_convert
> (rollout_ids 分组 / 去 trim)** 在端到端训练里不崩。用的是 `run_agentflow_1gpu.sh`。

### B1. 准备 0.5B 冒烟测所需文件

```bash
# 模型(若没有)
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir /root/models/Qwen2.5-0.5B-Instruct

# 数据(若没有)
huggingface-cli download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/datasets/dapo-math-17k
ls /root/datasets/dapo-math-17k/*.jsonl
```

✅ 通过:模型目录有 `config.json` + 权重;数据目录有 `dapo-math-17k.jsonl`。

---

### B2. 核对 `run_agentflow_1gpu.sh` 里的路径

打开 [run_agentflow_1gpu.sh](run_agentflow_1gpu.sh),确认这几处和你服务器实际路径一致:

| 行 | 变量 | 默认值 | 检查 |
|---|---|---|---|
| `MODEL_PATH` | 模型 | `/root/models/Qwen2.5-0.5B-Instruct` | ☐ |
| `--prompt-data` | 数据 | `/root/datasets/dapo-math-17k/dapo-math-17k.jsonl` | ☐ |
| `RUNTIME_ENV_JSON` PYTHONPATH | Megatron | `/root/autodl-tmp/Megatron-LM/` | ☐ |
| `ray job submit ... python3` | train.py | `/root/autodl-tmp/slime/train.py` | ☐ |

> 后两个路径是 autodl 模板留下的,**很可能要改成你的真实路径**
> (例如 `/path/to/slime-agentic/train.py` 和你的 Megatron-LM 目录)。

✅ 通过:四处路径都指向真实存在的文件/目录(`ls` 一下确认)。

---

### B3. 启动单卡冒烟测

```bash
cd /path/to/slime-agentic
bash agentic/agentflow/run_agentflow_1gpu.sh 2>&1 | tee /tmp/agentflow_1gpu_smoke.log
```

脚本会:杀旧进程 → 起 Ray(单卡)→ 提交训练 job(2 个 rollout step)。

---

### B4. 验证冒烟测的关键检查点(看 `/tmp/agentflow_1gpu_smoke.log`)

按顺序确认下面 5 个 gate,**这是验证我们改动的核心**:

**Gate 1 — SGLang 起来了**
```bash
grep -i "server is fired up\|Capture cuda graph\|Uvicorn running" /tmp/agentflow_1gpu_smoke.log | head
```
✅ SGLang server 启动成功(单卡 colocate 模式)。

**Gate 2 — rollout 真的生成了(AgentFlow solver 在跑)**
```bash
grep -i "rollout\|generate\|solver" /tmp/agentflow_1gpu_smoke.log | head
```
✅ 看到 rollout 生成进度,没有大片 traceback。

**Gate 3 — custom_convert 没炸(rollout_ids / 去 trim 生效)** ⭐最关键
```bash
grep -i "rollout_ids\|build_dp_schedule\|num_steps\|KeyError\|custom_convert\|Traceback" /tmp/agentflow_1gpu_smoke.log | head -30
```
✅ 通过:**没有** `KeyError: 'rollout_ids'`、**没有** `num_steps >= 1` 断言失败、**没有** custom_convert 相关 traceback。
❌ 若见 `KeyError: 'rollout_ids'` → custom_convert 没产出该字段(理论上不该,贴回来);
   若见 `num_rollouts (...) < global_batch_size` 断言 → 配置问题,调小 `--global-batch-size`。

**Gate 4 — 训练 step 真的跑了(loss 算出来了)**
```bash
grep -i "step\|loss\|grad_norm\|train" /tmp/agentflow_1gpu_smoke.log | grep -iv "rollout" | head
```
✅ 通过:看到至少 1 个训练 step 的 loss / grad_norm 数值(说明 v0.3.0 的 per-rollout loss 路径在我们缺 `rollout_mask_sums` 时正确退化成 per-sample,没崩)。

**Gate 5 — 整个 job 正常结束**
```bash
tail -30 /tmp/agentflow_1gpu_smoke.log
echo "EXIT CHECK:"; grep -i "Training finished\|Job .* succeeded\|SUCCEEDED" /tmp/agentflow_1gpu_smoke.log | tail
```
✅ 通过:job 跑完 2 个 step 正常退出,无未捕获异常。

> 🎯 **5 个 gate 全过 = slime v0.3.0 + 我们的 custom_convert 改动在真训练里验证通过。**
> 这是上 MAT 视觉训练前最重要的一关。

---

### B5. 清理

```bash
ray stop --force 2>/dev/null; pkill -9 sglang 2>/dev/null; pkill -9 ray 2>/dev/null
```

---

## 阶段 C — MAT 视觉训练(单卡/多卡,只有验证完 B 才做)

> ⚠️ MAT 视觉版(`agentflow_qwen3vl_mat.sh`)默认按 **8×4090** 配置(TP=2)。
> 如果你只有 1 张 4090,先把它当作「冒烟」用极小配置跑,别期望出效果。

### C1. 数据与模型(详见 [GPU_ONBOARDING.md](GPU_ONBOARDING.md) §1–§2)

```bash
# 数据
huggingface-cli download laolao77/MAT --repo-type dataset --local-dir /data/MAT
cd /path/to/slime-agentic/agentic/agentflow
python prepare_mat_data.py --input /data/MAT/MAT-Training/rft_agent_code_1_2k.json \
  --output /data/MAT/mat_coding_agentflow.jsonl --image-root /data/MAT/MAT-Training/images
wc -l /data/MAT/mat_coding_agentflow.jsonl   # ✅ 1200

# 模型 + 转 Megatron-dist
huggingface-cli download Qwen/Qwen3-VL-4B-Instruct --local-dir /data/models/qwen3_vl_4b
cd /path/to/slime-agentic && source scripts/models/qwen3-vl-4B.sh
PYTHONPATH=/path/to/Megatron-LM python tools/convert_hf_to_torch_dist.py ${MODEL_ARGS[@]} \
  --hf-checkpoint /data/models/qwen3_vl_4b --save /data/models/qwen3_vl_4b_dist/
```

✅ 通过:`mat_coding_agentflow.jsonl` 有 1200 行;`qwen3_vl_4b_dist/` 生成成功。
❌ 转换报 `ImportError: megatron.bridge.models.qwen_vl` → 你的 Megatron-LM 不含 Qwen3-VL bridge,需换分支。

---

### C2. MAT 极小 e2e 冒烟(改脚本临时参数)

在 [agentflow_qwen3vl_mat.sh](agentflow_qwen3vl_mat.sh) 的 `ROLLOUT_ARGS` 里临时改小:
```bash
--rollout-batch-size 2
--n-samples-per-prompt 2
--global-batch-size 4
--num-epoch 1
```
单卡时还要把 `N_GPUS=1`、`PERF_ARGS` 的 `--tensor-model-parallel-size 1`、`SGLANG_ARGS` 的 `--rollout-num-gpus-per-engine 1`。

```bash
MAT_MAX_PIXELS=200704 bash agentic/agentflow/agentflow_qwen3vl_mat.sh 2>&1 | tee /tmp/mat_smoke.log
```

验证(和 B4 类似,外加多模态专属 gate):

**Gate M1 — processor 多模态路径生效**
```bash
grep -i "image_data\|pixel_values\|multimodal\|processor" /tmp/mat_smoke.log | head
```
✅ 看到多模态输入在传(不是纯文本)。

**Gate M2 — cv2 工具执行**
```bash
grep -i "code executed\|code failed\|mat step\|opencv" /tmp/mat_smoke.log | head
```
✅ 看到 `code executed` 或至少 `mat step` 日志(初期模型格式不对、`unparseable` 属正常)。

**Gate M3 — MAT reward 有值**
```bash
grep -i "mat rollout\|outcome=\|score=" /tmp/mat_smoke.log | head
```
✅ 看到 `[mat rollout N] score=... outcome=...`,不全是 0。

**Gate M4 — 多模态 token 对齐 + 训练 step 不崩** ⭐
```bash
grep -i "Traceback\|KeyError\|assert\|size mismatch\|loss\|step" /tmp/mat_smoke.log | head -30
```
✅ 通过:无 traceback / 无 size mismatch,训练 step 算出 loss。
❌ 若见视觉 token 数量不匹配 / size mismatch → processor 的 input_ids 与 SGLang logprob 没对齐(plan §7.3 的已知风险点),贴回来排查。

---

### C3. 验证通过后,恢复正式配置跑全量

把 C2 改小的参数还原(或用 git checkout 还原脚本),按 [GPU_ONBOARDING.md](GPU_ONBOARDING.md) §5 起全量训练,`watch -n 2 nvidia-smi` 盯显存。

---

## 速查:出问题就贴这些

每个 gate 失败时,回贴对应的 grep 输出 + 最后 30 行日志:
```bash
tail -50 /tmp/agentflow_1gpu_smoke.log   # 或 /tmp/mat_smoke.log
```

---

## 验证顺序总览(打勾)

- [ ] A1 代码同步到含「drop stale trim」commit
- [ ] A2 slime v0.3.0 API 全部 import OK
- [ ] A3 120 离线测试全过
- [ ] B1–B2 0.5B 冒烟测文件 + 路径就绪
- [ ] B3–B4 **单卡冒烟测 5 个 gate 全过**(核心:验证 v0.3.0 + custom_convert 改动)
- [ ] C1 MAT 数据/模型/checkpoint 就绪
- [ ] C2 MAT 极小 e2e 4 个 gate 全过(核心:多模态 token 对齐)
- [ ] C3 全量训练启动
