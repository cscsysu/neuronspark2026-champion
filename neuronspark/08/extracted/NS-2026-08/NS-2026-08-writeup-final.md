# NS-2026-08 星云值班台：黑箱遥测世界模型闭环处置 Writeup

## 1. 解题概述

本题要求提交一个只能在 CPU、禁网环境中独立运行的 `agent.py`。隐藏评测逐分钟给出脱敏 observation，agent 需要在不可见真实根因的情况下选择一个运维动作，并同时控制 SLO 损失、恢复速度、资源成本、动作成本、冷却和误干预风险。

最终方案是一个单文件、无外部依赖的在线规则 agent。核心思想是把 telemetry 转成一个轻量 belief state：用服务指标历史估计异常压力，用事件 channel 与 service 的组合做故障模式约束求解，用诊断探针结果补充 mode 和 dependency 证据，再根据 belief 生成并筛选动作。策略整体偏风险敏感：证据不足先 probe；高风险动作需要 CSP/探针证据；同一目标的长期生效动作尽量不重复；靠近 horizon 末尾时停止无收益诊断和延迟动作。

最终提交记录：

- 平台正式提交时间：2026-05-27 14:42:57
- 平台正式分数：`1336.427022981 / 1500`
- 最终提交压缩包：`/Users/tristonchen/Desktop/neuronspark2026/neuronspark2026-answer/NS-2026-08-answer.zip`
- zip 内容：根目录仅包含 `agent.py`
- `agent.py` 行数：1177 行
- zip 大小：约 16 KB

本地 proxy 验证结果：

- `public_tasks.jsonl` uncapped：`1378.387728674`
- `valid_tasks.jsonl` uncapped：`1243.780415706`
- `valid_tasks.jsonl` calibrated：`1403.282848962`
- `tools/check_format.py submission`：OK

## 2. 关键改进

### 2.1 状态表示和 belief 更新

Agent 在 `reset(task_info)` 中读取服务别名、拓扑、基础副本数、最大副本数、服务角色和流量预测，并初始化以下状态：

- `history[svc]`：每个服务最近 8 步健康指标，字段包括 latency/error/saturation/queue bucket、`stale_steps` 和缺失字段。
- `events_seen[(channel, svc)]`：记录事件 channel 在哪个 service 上出现过及出现时间。
- `channel_mode`：由 CSP 和 probe 共同推断的 channel -> mode 映射。
- `svc_mode_score[svc][mode]`：某服务属于某故障模式的置信分。
- `dep_hint[svc][mode][source]`：probe 给出的 dependency/source hint 证据。
- `last_used_step`、`dispatched`、`global_dispatched`：动作冷却、动作去重和长期动作状态。
- `replicas`：agent 自己维护的副本数估计，避免重复扩容。

使用的 observation 字段：

- `time`
- `traffic_bucket`
- `global_health_bucket`
- `health[].service`
- `health[].latency_bucket`
- `health[].error_bucket`
- `health[].saturation_bucket`
- `health[].queue_bucket`
- `health[].stale_steps`
- `health[].missing_fields`
- `events[].channel`
- `events[].service`
- `events[].token`
- `events[].age`
- `events[].stale_steps`
- `probe_results[].target`
- `probe_results[].source_hint`
- `probe_results[].pressure_bucket`
- `probe_results[].coupling_bucket`
- `probe_results[].confidence_bucket`
- `probe_results[].action_hint`
- `probe_results[].noisy`
- `telemetry_quality`
- `recent_actions`

服务异常分数主要由最近几步 bucket 指标计算：

```text
pressure = mean(latency + error + saturation + queue over last 3 observed points)
delta = current combined bucket - previous max combined bucket
anomaly = pressure + 0.5 * max(0, pressure - early_baseline)
```

### 2.2 Channel-mode 约束求解

公开 checker 显示事件 channel 与真实 mode 存在稳定映射，但 hidden 只暴露脱敏 channel，不给 mode 文本。Agent 内置了从 public/valid proxy 和题目结构总结出的合法 `(mode, service)` 先验：

- `release_instability`: `svc_01/02/03/04/09`
- `config_shift`: `svc_00/01/02/03/04`
- `heap_growth`: `svc_00/02/03/04/09`
- `traffic_surge`: `svc_00/02/03`
- `storage_wait`: `svc_02/03/04`
- `consumer_lag`: `svc_09`
- `cache_collapse`: `svc_00/01/02`
- `dependency_timeout`: `svc_00/02/03/04`

每个非背景 `(channel, service)` 是一个待识别 slot。Agent 枚举所有满足以下条件的 assignment：

- 每个 channel 选择一个 mode。
- mode.service 必须能合法落在该 service。
- 同一任务中不同 channel 尽量对应不同 mode。
- 已由 probe 高置信确认的 channel 会被锁定。

若所有可行解对某个 channel 的 mode 一致，则写入 `channel_mode`，并给对应 `svc_mode_score` 一个强 boost。若有多个可行解，则按出现比例给弱证据，保留不确定性。

为了避免 Python hash seed 导致同分支下动作顺序波动，最终版对关键 set/dict 遍历做了排序或反序排序，本地 `PYTHONHASHSEED=1..5` 下分数一致。

### 2.3 诊断探针证据

Probe 结果的 `action_hint` 被映射回 mode：

```text
rollback_release -> release_instability
rollback_config  -> config_shift
restart_scale    -> heap_growth
scale_throttle   -> traffic_surge
storage_index    -> storage_wait
queue_drain      -> consumer_lag
cache_warm       -> cache_collapse
timeout_circuit  -> dependency_timeout
```

权重近似为：

```text
w = (confidence_bucket + 1) * (pressure_bucket + 1)
if noisy: w *= 0.4
```

Probe target 和 source_hint 都会被用于更新 belief。若 target 是 dependency 服务，例如 `svc_05/06/07/08`，但 mode 合法 service 不包含该 dependency，则把证据分发给与事件和拓扑更匹配的合法 service。

## 3. 轨迹建模、世界模型和调参方法

没有训练神经网络，也没有提交模型权重、checkpoint、查表文件或 rollout 缓存。策略来自公开 checker 和 public/valid proxy 任务上的离线调参。

使用的 surrogate/world model：

- 公开 `checker/aiops_world.py` 作为 proxy 世界模型，阅读其动作成本、冷却、flag 生效延迟、持续时间和 mode loss factor。
- `public_tasks.jsonl` 用于小规模快速回归。
- `valid_tasks.jsonl` 用于 proxy 稳健性验证。
- `train_traces.jsonl` 用于理解轨迹格式和 observation/action 交互方式，但未从中训练模型。

调参方式：

1. 先建立基线 agent，记录每个任务的动作序列、wrong action、action cost、resource cost 和 total loss。
2. 针对弱任务手动分析 hidden proxy mode、动作时序和 loss 峰值。
3. 每次只改一个门控或阈值，跑 public 和 valid，保留 public 不退化且 valid 提升的改动。
4. 对可能过拟合的改动，用 action wrong count、hard slice fraction 和 `PYTHONHASHSEED` 稳定性做筛查。

主要调参发现：

- `DRAIN_QUEUE` 在 checker 中 delay=2、duration=14，活动窗内重复执行会刷新 start，导致重新进入 delay gap，因此 14 分钟内不重发更好。
- `RESTART` 对 heap 有 duration=12，短期重复 restart 会带来动作成本和 transient loss，因此至少间隔 13 分钟。
- `WARM_CACHE` symptom-only 非常容易把 cache 依赖压力误认为 cache_collapse，需要更高阈值。
- 后 6-7 分钟继续 probe 通常性价比低，因为 probe 结果延迟返回后已没有足够动作收益窗口。
- 多 channel 同服务常出现“主故障盖住第二故障”，提前 probe dependency 侧能更早发现 `dependency_timeout` 或 `storage_wait`。

## 4. 诊断探针策略、价值评估和失败样例

Probe 使用策略：

- 新 `(channel, service)` pair 出现且 channel mode 未知时，优先 probe 该 service。
- 当 `global_health_bucket` 上升、已有 hypothesis 分数不足时，probe 信息增益最高的异常服务。
- 同一 service 出现多个 channel 且已识别出一个主 mode 时，优先 probe dependency 侧，最终顺序为 `svc_06 -> svc_05 -> svc_07 -> svc_08`，其中前两个用于更快发现 storage/dependency 相关故障。
- 对 event-bearing service，如果 channel 未解或 top score 偏低，继续做低频信息增益 probe。
- t > horizon - 7 时停止 probe，避免晚期纯成本。

价值评估：

- 在公开 checker 中 `DIAGNOSTIC_PROBE` 没有 wrong action penalty，但有动作成本，所以优化重点不是“探得越多越好”，而是早期探、关键依赖探、晚期少探。
- 典型收益来自 `valid_telemetry_058`：原先主故障 heap_growth 覆盖了后续 dependency_timeout/storage_wait，加入多 channel 依赖探测后 main fraction 从约 `0.61` 提升到约 `0.70`。

失败样例：

- `cache_collapse` 和 `dependency_timeout(dep=svc_07)` 都会提高 cache 侧 pressure，仍有少量 symptom-only `WARM_CACHE` 误判。
- 某些任务中 release/heap/consumer_lag 连续作用于 `svc_09`，容易把 release 事件误解释为 heap 或 queue 后续症状。
- late-start 多故障任务即便识别正确，动作延迟也限制了可恢复空间。
- 正式 hidden world 与 public/valid proxy 的 event channel 分布可能不同，因此保留了较多 CSP 不确定性分支，而不是硬编码每个 channel 的 mode。

## 5. 动作搜索、规则控制和风险权衡

动作不是一次性规划完整 48 步，而是每分钟基于当前 belief 做 greedy 风险控制。

每个 mode 的候选动作：

- `release_instability`: `ROLLBACK_RELEASE(service)`, `CIRCUIT_BREAK(svc_00)`
- `config_shift`: `ROLLBACK_CONFIG(service)`, `SET_TIMEOUT(service, 1000)`
- `heap_growth`: `RESTART(service)`, `SCALE(service, base+5)`
- `traffic_surge`: `SCALE(svc_00)`, `SCALE(service)`, `THROTTLE(svc_00, 12)`
- `storage_wait`: `REBUILD_INDEX(dependency)`, `SET_TIMEOUT(service, 1000)`
- `consumer_lag`: `DRAIN_QUEUE(svc_08)`, `SCALE(svc_09)`
- `cache_collapse`: `WARM_CACHE(svc_07)`, `THROTTLE(svc_00, 12)`
- `dependency_timeout`: `CIRCUIT_BREAK(service)`, `SET_TIMEOUT(dependency, 1000)`

动作筛选规则：

- 动作必须在 `HELPFUL[mode]` 中，并且 target 与 service/dependency/special service 对齐。
- 高风险动作需要 probe/CSP 证据，或极强症状证据。
- CSP 明确排除某 mode 时，不触发该 mode 的高风险动作。
- `SCALE` 需要副本数确实提升，且不能重复扩同一目标。
- `CIRCUIT_BREAK`、`SET_TIMEOUT`、`WARM_CACHE`、`REBUILD_INDEX`、`THROTTLE` 同一 target 只发一次。
- `ROLLBACK_CONFIG` 和 `ROLLBACK_RELEASE` 每任务全局只发一次，降低错误服务回滚风险。
- `DRAIN_QUEUE` 需要 queue 双侧强信号，或已有 consumer_lag 证据；14 分钟内不重复。
- `RESTART` 至少间隔 13 分钟。
- 最后一两分钟不发需要 delay 才生效的动作；`REBUILD_INDEX` 需要更长剩余窗口。

风险权衡：

- 更看重避免 wrong high-risk action，而不是追求所有 proxy task 的 oracle。
- 对 `SCALE` 保持保守，因为额外副本持续产生 resource cost。
- 对 `THROTTLE` 保守但不完全禁用，因为 traffic_surge/cache_collapse 中它能作为低风险补救。
- 对 queue/heap 的持续窗口动作，重点避免刷新窗口造成短期失效。

## 6. 验证与复现

最终提交压缩包：

```text
/Users/tristonchen/Desktop/neuronspark2026/neuronspark2026-answer/NS-2026-08-answer.zip
```

SHA256：

```text
NS-2026-08-answer.zip:
f75968f0e3cd17ad965ae4b4c73e0d0cb5bd2127eaef1e8b75c40891d119a160

zip 内 agent.py:
efafaf42fec7f715b3b09939b24812d6447d1d7141dc045a3723646206f24a18
```

复现命令：

```bash
cd /Users/tristonchen/Desktop/08/extracted/NS-2026-08
python3 tools/check_format.py submission
python3 checker/checker.py --tasks public_tasks.jsonl --submission submission
python3 checker/checker.py --tasks valid_tasks.jsonl --submission submission
python3 -m zipfile -t /Users/tristonchen/Desktop/neuronspark2026/neuronspark2026-answer/NS-2026-08-answer.zip
```

计算 SHA256：

```bash
openssl dgst -sha256 /Users/tristonchen/Desktop/neuronspark2026/neuronspark2026-answer/NS-2026-08-answer.zip
python3 - <<'PY'
import hashlib, zipfile
p = "/Users/tristonchen/Desktop/neuronspark2026/neuronspark2026-answer/NS-2026-08-answer.zip"
with zipfile.ZipFile(p) as z:
    print(hashlib.sha256(z.read("agent.py")).hexdigest())
PY
```

本地环境：

- 本地验证 Python：`Python 3.13.5`
- 平台承诺环境：Linux x86_64, Python `3.12.12`
- Agent 依赖：Python 标准库；不依赖 numpy/pandas/sklearn 等第三方包。
- 硬件：本地 CPU-only macOS；未使用 GPU。
- 网络：未联网、未下载模型或依赖。
- 调参时间：约 1 个工作会话，几十组 public/valid proxy rollout。

随机 seed：

- Agent 本身无随机数。
- 本地测试 `PYTHONHASHSEED=1..5`，`valid_tasks.jsonl` uncapped 均为 `1243.780415706`。

预算-分数曲线：

| 阶段 | public uncapped | valid uncapped | 说明 |
|---|---:|---:|---|
| 早期轻量 submission | 1253.6699 | 未完整记录 | 简单事件/probe/回退规则 |
| CSP + probe 基线 | 1367.9459 | 1224.5817 | 引入 channel-mode CSP 和风险门控 |
| 长期动作去重 | 1371.3764 | 1231.0768 | 减少重复 timeout/warm/throttle/rebuild |
| drain/restart 活动窗 | 1377.5668 | 1239.8470 | 避免刷新 active flag delay gap |
| warm/drain/late probe 门控 | 1378.3877 | 1242.2904 | 降低误判与晚期纯成本 |
| 多故障依赖探测最终版 | 1378.3877 | 1243.7804 | 提前发现被主故障掩盖的第二故障 |
| 平台 hidden 最终提交 | - | - | `1336.427022981 / 1500` |

## 7. 证据截图

建议随最终材料附以下截图：

1. 平台提交记录页：显示 `2026/5/27 14:42:57` 提交得分 `1336.427022981 / 1500`。
2. 本地 `python3 tools/check_format.py submission` 输出 OK。
3. 本地 public checker 输出 `uncapped_score = 1378.387728674`。
4. 本地 valid checker 输出 `uncapped_score = 1243.780415706`。
5. `unzip -l NS-2026-08-answer.zip` 显示 zip 根目录仅包含 `agent.py`。
6. SHA256 命令输出，包含 zip 与 zip 内 `agent.py` 的 SHA256。

## 8. 代码包说明

提交包只包含：

```text
agent.py
```

不包含：

- 模型权重
- checkpoint
- 策略权重
- 查表文件
- 预计算动作表
- rollout 缓存
- 第三方依赖包
- wheel/site-packages
- 配置文件或额外源码文件

`agent.py` 可在 CPU-only、禁网、Python 标准库环境中运行。每个任务只依赖 `reset(task_info)` 初始化的内存状态和之后逐分钟传入的 observation，不依赖跨任务或跨提交持久状态。

## 9. AI 使用声明与边界

本题按 A1 处理。AI 协助参与了：

- 阅读题面、checker 和公开 proxy 数据结构。
- 分析已有 agent 的动作序列和弱任务。
- 设计并执行本地 A/B 规则实验。
- 修改 `agent.py`。
- 生成提交 zip。
- 撰写和整理本 writeup。

AI 使用边界：

- AI 使用了题目公开文件、`public_tasks.jsonl`、`valid_tasks.jsonl`、`train_traces.jsonl` 的格式信息、公开 checker、本地实验结果和用户提供的提交反馈。
- AI 没有读取正式 `test_tasks.jsonl` 的条目内容；只接触了题面中关于 test public view 的文字描述和文件名。
- AI 接触了用户在对话中提供的提交反馈和最终平台总分，例如 `1321.730329383` 与 `1336.427022981`。
- AI 没有接触隐藏 replay、隐藏私有状态、逐任务 hidden 评测细节或平台内部日志。
- 没有使用外部数据集、预训练模型、开源项目代码或第三方模型服务。
- 没有联网下载依赖、调用外部 API、访问托管推理服务或探测平台网络。
- 最终提交生成过程由 AI 辅助完成，但最终文件为单个自包含 `agent.py`。

若按“训练模型或拟合数字孪生”口径说明：本方案没有训练可提交模型，也没有生成策略权重。公开 checker 被用作手工规则调参的 proxy world；所有参数和规则均硬编码在 `agent.py` 中，未提交 checkpoint/权重/缓存 rollout。

