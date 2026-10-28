# NS-2026-08 星云值班台：黑箱遥测世界模型闭环处置

难度：Hard（探索型）｜总分：0 ~ 1500｜题型：在线闭环控制 / AIOps agent（提交单文件 `agent.py`）

## 题目介绍

扮演值班 SRE：隐藏评测在每个事故任务中逐分钟调用 agent（horizon = 48 步），agent 只能看到脱敏、可能缺失/滞后/丢包的 telemetry，看不到真实根因，每步返回一个运维动作。

- **接口**：`class Agent` 需实现 `reset(task_info)`（每任务一次，给服务别名、拓扑、副本/容量/成本、流量预测、动作空间）和 `act(observation)`（每分钟一次，返回一个动作 dict）。
- **观测**：每服务 latency/error/saturation/queue 的 bucket 指标（含 `stale_steps`、`missing_fields`）、脱敏事件 `(channel, service, token, age)`、延迟带噪的 `probe_results`（`source_hint`、置信度 bucket、粗粒度 `action_hint`）。
- **动作**：`SCALE` / `ROLLBACK_RELEASE` / `ROLLBACK_CONFIG` / `RESTART` / `CIRCUIT_BREAK` / `SET_TIMEOUT`(100~1500) / `WARM_CACHE` / `REBUILD_INDEX` / `DRAIN_QUEUE` / `THROTTLE`(0~60) / `DIAGNOSTIC_PROBE` / `NOOP`，有冷却与副作用（见 `action_schema.json`）。
- **数据**：`train_traces.jsonl` 480 条公开交互轨迹（无根因标签）；`valid_tasks.jsonl` 64 个、`public_tasks.jsonl` 16 个 proxy 任务（带 proxy world，可本地评测）；`test_tasks.jsonl` 32 个正式任务 public view。
- **评分**：隐藏私有世界闭环评分，综合 SLO 损失、恢复效率、资源成本、动作成本、冷却违规、副作用、错误干预与稳健性，平台只返回总分。
- **约束**：CPU-only、禁网、Python 3.12.12；`reset` 5 s、`act` 2 s 超时，agent 子进程 ~120 s CPU / ~768 MiB；zip 根目录只能有 `agent.py`，不得附带权重/查表/rollout 缓存。

## 我们的做法

单文件、仅用 Python 标准库的**在线规则 agent**（未训练任何模型），核心是"轻量 belief state + 风险门控的贪心处置"：

1. **状态/belief**：每服务保留最近 8 步 bucket 历史，计算 pressure、delta、相对早期基线的 anomaly；自维护副本数估计、动作冷却与"已下发"状态。
2. **Channel→mode 约束求解（CSP）**：内置从 public/valid proxy 归纳的合法 `(mode, service)` 先验（8 种故障模式：release_instability、config_shift、heap_growth、traffic_surge、storage_wait、consumer_lag、cache_collapse、dependency_timeout）。把每个非背景 `(channel, service)` 当作 slot 枚举可行赋值，所有可行解一致则锁定该 channel 的 mode，否则按比例给弱证据。
3. **诊断探针**：`action_hint` 映射回 mode，权重 ≈ `(confidence+1)*(pressure+1)`，noisy 再 ×0.4；新 `(channel, service)` 未识别时优先 probe；同服务多 channel 时提前 probe 依赖侧（`svc_06 → svc_05 → svc_07 → svc_08`），发现被主故障掩盖的第二故障；最后 ~7 分钟停止 probe。
4. **动作选择**：每个 mode 有候选动作表（`HELPFUL`），过合法性、冷却、去重、CSP 排除、证据阈值、目标对齐、剩余窗口等门控后下发。关键规则：高风险动作需 probe/CSP 证据；`CIRCUIT_BREAK/SET_TIMEOUT/WARM_CACHE/REBUILD_INDEX/THROTTLE` 同目标只发一次；两种回滚全局各一次；`DRAIN_QUEUE` 14 分钟内不重发、`RESTART` 至少间隔 13 分钟（避免刷新生效窗口重新进入 delay gap）；`WARM_CACHE` 纯症状触发阈值收紧；预防性 `SCALE` 需 CSP 确认。
5. **确定性**：关键 set/dict 遍历排序，`PYTHONHASHSEED=1..5` 下 valid 分数一致。

**迭代过程**：以公开 `checker/aiops_world.py` 作为 proxy 世界模型，阅读动作成本/冷却/生效延迟/持续时间，每次只改一个门控或阈值，在 public 与 valid 上做 A/B，保留不退化的改动。版本线大致为：早期轻量事件/probe 规则（`agent_v6`~`v14`，约 500 行）→ 加入更多门控（`v15`~`v28`，约 650~675 行）→ 引入 CSP（`agent_v29_csp.py`）→ 症状回退（`agent_v30_sw_fallback.py`）→ `sub_v22`（CSP + probe + 风险门控，1127 行）→ 长期动作去重、活动窗、warm/drain/late-probe 门控、多故障依赖探测 → 最终版（1177 行）。

## 结果

| 阶段 | public uncapped | valid uncapped | 说明 |
|---|---:|---:|---|
| 早期轻量 submission | 1253.6699 | 未记录 | 简单事件/probe/回退规则 |
| CSP + probe 基线（sub_v22） | 1367.9459 | 1224.5817 | channel-mode CSP + 风险门控 |
| 长期动作去重 | 1371.3764 | 1231.0768 | |
| drain/restart 活动窗 | 1377.5668 | 1239.8470 | |
| warm/drain/late probe 门控 | 1378.3877 | 1242.2904 | |
| 多故障依赖探测（最终版） | 1378.3877 | 1243.7804 | valid calibrated 1403.2828 |

- **平台正式分数：`1336.427022981 / 1500`**（2026-05-27 14:42:57 提交；此前一次平台提交为 `1321.730329383`）。来源：writeup。
- `agent_vXX_NNNN.py` 文件名后缀数字（1197~1326）为当时的分数标记，但文档中未说明其对应哪个评测集。

## 目录与文件说明

| 路径 | 说明 |
|---|---|
| [`neuronspark2026-answer/NS-2026-08-answer.zip`](../../neuronspark/neuronspark2026-answer/NS-2026-08-answer.zip) | **最终提交包**（仅含 `agent.py`，SHA256 `f75968f0…a160`） |
| [`08/extracted/NS-2026-08/submission/agent.py`](../../neuronspark/08/extracted/NS-2026-08/submission/agent.py) | **最终提交脚本**（1177 行，与 zip 内 `agent.py` 完全一致，SHA256 `efafaf42…4a18`） |
| [`08/extracted/NS-2026-08/NS-2026-08-answer.zip`](../../neuronspark/08/extracted/NS-2026-08/NS-2026-08-answer.zip) | 与最终提交包相同的副本 |
| [`08/extracted/NS-2026-08/problem.md`](../../neuronspark/08/extracted/NS-2026-08/problem.md) | 题面；另有 [`agent_api.md`](../../neuronspark/08/extracted/NS-2026-08/agent_api.md)、[`action_schema.json`](../../neuronspark/08/extracted/NS-2026-08/action_schema.json)、[`DATA_NOTICE.md`](../../neuronspark/08/extracted/NS-2026-08/DATA_NOTICE.md) |
| [`08/extracted/NS-2026-08/checker/`](../../neuronspark/08/extracted/NS-2026-08/checker/) | 官方公开 proxy 评测：`aiops_world.py`（世界模拟器）、`checker.py`、`visualize_trace.py` |
| [`08/extracted/NS-2026-08/tools/`](../../neuronspark/08/extracted/NS-2026-08/tools/) | `check_format.py`（格式检查）、`run_public_eval.py`（公开评测） |
| [`08/extracted/NS-2026-08/inspect_ns08.py`](../../neuronspark/08/extracted/NS-2026-08/inspect_ns08.py) | 自写调试脚本：逐任务打印动作序列与 proxy 隐藏 mode |
| [`08/extracted/NS-2026-08/example_submission/`](../../neuronspark/08/extracted/NS-2026-08/example_submission/) | 官方示例 agent |
| `08/extracted/NS-2026-08/*.jsonl` | 数据：train_traces / valid_tasks / public_tasks / test_tasks |
| `08/NS-2026-08.zip`<sup>†</sup>、`neuronspark2026/2026-08.zip`<sup>†</sup> | 原始题目包 |
| [`neuronspark2026/NS-2026-08/`](../../neuronspark/neuronspark2026/NS-2026-08/) | 主要迭代工作目录：`agent_v6_backup.py`~`agent_v30_sw_fallback.py` 各版本、`agent_robust.py`（= v28）、`sub_*/agent.py` 历次候选提交（`sub_v22` 为最终版的直接前身）、`_eval/`、`_zip_test/` 临时目录，以及一份会话记录 txt |
| [`08/extracted/NS-2026-08/WRITEUP_NS_2026_08.md`](../../neuronspark/08/extracted/NS-2026-08/WRITEUP_NS_2026_08.md) | 早期较短版 writeup |

## 如何运行

依赖：Python 3（平台 3.12.12，本地验证 3.13.5），agent 仅用标准库；无需 GPU/联网。以下命令在题目目录下执行（`checker.py` 的 `--tasks` 按当前目录解析）：

```bash
cd neuronspark/08/extracted/NS-2026-08
python3 tools/check_format.py submission                                  # 格式检查
python3 tools/run_public_eval.py submission --tasks public_tasks.jsonl    # 公开评测
python3 checker/checker.py --tasks public_tasks.jsonl --submission submission
python3 checker/checker.py --tasks valid_tasks.jsonl --submission submission [--summary out.json]
python3 checker/visualize_trace.py train_traces.jsonl --limit 2           # 查看轨迹
python3 inspect_ns08.py [agent.py路径] [tasks.jsonl] [task_id ...]       # 逐任务动作分析（默认 submission/agent.py + valid_tasks.jsonl）
```

checker 输出 JSON，关注 `uncapped_score`、`score`（calibrated）与 `details`。

打包（zip 根目录只放 `agent.py`）：

```bash
cd submission && zip -j ../NS-2026-08-answer.zip agent.py && cd ..
python3 -m zipfile -t NS-2026-08-answer.zip
```

注意：writeup 中的复现命令使用了旧绝对路径 `/Users/tristonchen/Desktop/08/...` 与 `/Users/tristonchen/Desktop/neuronspark2026/...`，在本仓库中需替换为 `neuronspark/08/...` 与 `neuronspark/neuronspark2026/...`。脚本本身无硬编码绝对路径。

## 参考

- 最终 writeup：[`writeup_submit/08/NS-2026-08-writeup.md`](../../neuronspark/neuronspark2026/writeup_submit/08/NS-2026-08-writeup.md)（与 [`NS-2026-08-writeup-final.md`](../../neuronspark/08/extracted/NS-2026-08/NS-2026-08-writeup-final.md) 内容相同）
- 早期 writeup：[`WRITEUP_NS_2026_08.md`](../../neuronspark/08/extracted/NS-2026-08/WRITEUP_NS_2026_08.md)
- 题面：[`problem.md`](../../neuronspark/08/extracted/NS-2026-08/problem.md)；公开 checker 说明：[`checker/README.md`](../../neuronspark/08/extracted/NS-2026-08/checker/README.md)

---

† 标注的文件为比赛原始数据或过程记录，按数据协议未包含在公开仓库中，请从官方渠道获取题目数据包。
