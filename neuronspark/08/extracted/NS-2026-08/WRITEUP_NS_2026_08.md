# NS-2026-08 星云值班台 Writeup

## 1. 解题概述

本方案提交单文件 `agent.py`，不依赖网络、GPU、外部模型或额外数据文件。核心思路是把每分钟 observation 转换成一个轻量 belief state：维护服务级异常历史、事件 channel 到故障 mode 的约束求解结果、诊断 probe 的 mode/dep 证据，以及动作冷却和已生效动作状态。在线控制时先用 CSP/诊断证据识别故障，再按 mode 输出风险受控的处置动作；证据不足时优先 probe，避免盲目回滚或重启。

本地最终结果：

- `public_tasks.jsonl`: uncapped `1378.387728674`
- `valid_tasks.jsonl`: uncapped `1243.780415706`, calibrated `1403.282848962`
- `tools/check_format.py submission`: OK

最终提交文件：

- `submission/agent.py`
- `NS-2026-08-answer.zip`，zip 根目录直接包含 `agent.py`

## 2. 关键改进

状态表示和 belief 更新：

- 每个服务维护最近 8 步 bucket 指标，计算 pressure、delta、baseline anomaly。
- 将事件 `(channel, service)` 作为 mode slot，用公开 prior `LEGAL_MODE_SERVICE` 做 CSP 枚举；对所有可行解取交集，得到高置信 channel->mode。
- probe 结果按 `confidence_bucket * pressure_bucket` 加权写入 `svc_mode_score`，并记录 `dep_hint` 辅助确定 storage/cache/queue/dependency 目标。
- 对未完全识别的多 channel 服务保留不确定性，并用诊断探针补充依赖侧信息。

轨迹建模和规则调参：

- 直接阅读 checker 中的闭环机制：动作成本、冷却、flag 延迟、持续时间和 mode loss factor。
- 通过 public/valid proxy rollout 做 A/B：每次只改一个动作门控或阈值，比较 public 与 valid uncapped 分数、wrong action 数和弱任务动作序列。
- 没有训练模型或提交 checkpoint；所有调参结果收敛为 `agent.py` 内的规则和小型 CSP。

主要收益点：

- 对长期生效动作去重：`CIRCUIT_BREAK`、`SET_TIMEOUT`、`WARM_CACHE`、`REBUILD_INDEX`、`THROTTLE` 同一 target 只发一次；`ROLLBACK_CONFIG`/`ROLLBACK_RELEASE` 全局只发一次，避免同一任务内对错误服务重复回滚。
- 避免刷新活动窗口：`DRAIN_QUEUE` 14 分钟活动窗内不重发，`RESTART` 至少间隔 13 分钟，避免重置 start 后重新进入 delay gap。
- 收紧高误判动作：`WARM_CACHE` symptom-only 阈值提高到 `p07 >= 9.0` 且 caller pressure >= `8.5`；`DRAIN_QUEUE` 需要队列双侧强信号或 consumer_lag 证据。
- 晚期动作控制：t=42 后不再 probe；最后一分钟不发需要延迟才生效的动作，避免纯成本。
- 多故障依赖探测：同一服务出现多个 channel 时，优先 probe 依赖侧，尤其先试 `svc_06`/`svc_05` 的 storage 依赖，减少主故障掩盖第二故障。
- 排序所有关键 set/dict 迭代，消除 `PYTHONHASHSEED` 导致的分数抖动。

## 3. 验证与复现

复现命令：

```bash
cd /Users/tristonchen/Desktop/08/extracted/NS-2026-08
python3 tools/check_format.py submission
python3 checker/checker.py --tasks public_tasks.jsonl --submission submission
python3 checker/checker.py --tasks valid_tasks.jsonl --submission submission
python3 -m zipfile -t NS-2026-08-answer.zip
```

最终 SHA256：

```text
agent.py  efafaf42fec7f715b3b09939b24812d6447d1d7141dc045a3723646206f24a18
zip       f75968f0e3cd17ad965ae4b4c73e0d0cb5bd2127eaef1e8b75c40891d119a160
```

随机性：

- agent 无随机数。
- `PYTHONHASHSEED=1..5` 下 valid uncapped 均为 `1243.780415706`。

依赖和硬件：

- Python 3 标准库，无第三方依赖。
- CPU-only，本地 macOS 环境。
- 本轮调参约数十组 proxy rollout，未使用 GPU、外部服务或下载资源。

预算-分数曲线：

| 阶段 | public uncapped | valid uncapped | 说明 |
|---|---:|---:|---|
| 当前目录旧 submission | 1253.6699 | - | 361 行早期策略 |
| 续接上一轮 sub_v22 | 1367.9459 | 1224.5817 | CSP + probe + 风险门控 |
| 长期动作去重 | 1371.3764 | 1231.0768 | 去重 idempotent actions |
| drain/restart 活动窗 | 1377.5668 | 1239.8470 | 避免 active flag 被提前刷新 |
| warm/drain/late probe 门控 | 1378.3877 | 1242.2904 | 降低误判和晚期成本 |
| 多故障依赖探测最终版 | 1378.3877 | 1243.7804 | 提前识别被主故障掩盖的第二故障 |

## 4. 诊断探针策略

使用策略：

- 新 `(channel, service)` 出现且 mode 未知时强制 probe。
- 事故刚出现、强 hypothesis 不足时 probe pressure 高且信息增益高的服务。
- 对多 channel 服务，如果重复 probe 服务本身可能只返回主故障，则提前 probe storage/cache/queue 依赖侧。
- t=42 后停止 probe，因为结果即使返回，也很难转化为有延迟收益的动作。

价值评估：

- probe 在 checker 中不产生 wrong action penalty，但有动作成本，因此主要优化点是减少晚期无收益 probe。
- 多故障场景中，依赖侧 probe 能更早发现 `dependency_timeout`/`storage_wait`，例如 `valid_telemetry_058` 由约 0.61 main frac 提升到约 0.70 main frac。

失败样例：

- `cache_collapse` 与 `dependency_timeout(dep=svc_07)` 均会压高 cache 侧指标，symptom-only `WARM_CACHE` 仍可能误发。
- 同一服务连续出现 release/heap/consumer 信号时，`DRAIN_QUEUE` 和 `RESTART` 仍有少量错判风险。
- 部分 late-start 多故障任务受动作延迟限制，即使识别正确也难接近 oracle。

## 5. 动作控制和风险权衡

动作搜索不是全局 MCTS，而是按 belief 排序的规则控制：

- 对每个 `(mode, service, dependency)` 生成候选动作序列。
- 先过合法性、冷却、重复动作、CSP 排除、证据阈值、目标一致性、晚期收益门控。
- 高风险动作如 rollback/rebuild/restart/circuit 需要 probe/CSP 证据或极强症状，不直接由单个 bucket 触发。
- `SCALE` 持久增加资源成本，因此只在 traffic/heap/consumer/storage 相关证据下触发，且不重复降噪式扩容。
- `DRAIN_QUEUE` 和 `RESTART` 是有持续窗口的动作，重点避免在窗口内重复发导致 delay gap。

## 6. AI 使用声明与边界

本题 AI 协助等级按 A1。AI 参与了本地代码阅读、规则实验、阈值调参、提交包生成和本 writeup 草稿撰写。

数据使用边界：

- 使用了题目公开文件、`public_tasks.jsonl`、`valid_tasks.jsonl`、checker 和上一轮聊天记录。
- 本轮没有读取或利用 `test_tasks.jsonl` 的正式 public view。
- 没有使用排行榜增量反馈做自动搜索；仅使用用户提供的历史平台分数 `1321.73` 作为背景信息。
- 没有联网、没有调用外部 API、没有使用外部数据、预训练模型、闭源模型服务或托管推理服务。
- 最终提交不携带 checkpoint、模型权重、大型缓存或 rollout 表。

证据截图建议：

- 本地 `check_format.py` OK 输出。
- public/valid checker 输出中 uncapped score。
- 平台提交成功页和最终分数页。

