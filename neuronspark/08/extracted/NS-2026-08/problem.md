# NS-2026-08 星云值班台：黑箱遥测世界模型闭环处置

## 赛题信息

- **题号**：NS-2026-08
- **难度**：Hard（探索型）
- **评分范围**：0 ~ 1500 分

---

## 任务介绍

凌晨的星云业务集群开始抖动：一部分服务延迟上升，队列积压，缓存命中率异常，发布和配置变更可能同时影响多个依赖。你是值班 SRE，不能看到真实根因标签，只能持续观察脱敏 telemetry，并在事故尚未完全暴露时决定下一步处置。

本题要求提交一个 `agent.py`。隐藏评测会在每个事故任务中逐分钟调用你的 agent：

1. 先给出服务别名、拓扑、容量、成本、流量预测和观测空间定义
2. 在私有黑箱世界中不断产生 observation
3. 你的 agent 每分钟返回一个运维动作（扩缩容、回滚、重启、熔断、限流、索引维护、缓存预热、排队疏导、诊断探针或空动作）

公开数据不会给正式根因标签。你可以使用历史交互轨迹、public/valid proxy 任务和公开 checker 训练或调试；正式 test 只给 public view。最终成绩来自隐藏 judge 对闭环控制过程的评分，评分同时考虑服务质量、恢复效率、资源成本、动作成本、冷却限制、副作用、错误干预和整体稳健性。

---

## 需要解决的问题

1. 从缺失、延迟和丢包的 telemetry 中估计当前事故状态
2. 从公开轨迹中学习动作后果、延迟反馈、诊断探针价值和资源副作用
3. 在动作有冷却和副作用的情况下做风险敏感控制，避免盲目扩容或对无关服务做高风险操作
4. 将离线训练、规则调参、数字孪生或搜索结果收敛为一个可在平台 CPU 评测环境中独立运行的 `agent.py`

---

## 赛题数据

### 获取方式

在本页下方"相关资源"处下载数据包，或在 [ModelScope](https://modelscope.cn/datasets/SteamedFresh/NS-2026-AAA-data) 获取。

### 文件结构

```
NS-2026-08/
├── train_traces.jsonl          # 480 条公开交互轨迹
├── valid_tasks.jsonl           # 64 个 proxy 验证任务
├── public_tasks.jsonl          # 16 个小规模 proxy 调试任务
├── test_tasks.jsonl            # 32 个正式任务 public view
├── action_schema.json          # 动作参数定义
├── agent_api.md                # Agent API 文档
├── DATA_NOTICE.md              # 数据说明
├── example_submission/
│   └── agent.py                # 示例提交
├── tools/
│   ├── check_format.py         # 格式检查
│   └── run_public_eval.py      # 公开评测脚本
├── checker/
│   ├── aiops_world.py          # 世界模拟器
│   ├── checker.py              # 评测器
│   └── visualize_trace.py      # 轨迹可视化
└── README.md
```

### 数据规模

| 文件 | 数量/说明 |
|---|---|
| `train_traces.jsonl` | 480 条公开交互轨迹，包含 public task 信息、逐步 observation、历史 action 和 loss_bucket |
| `valid_tasks.jsonl` | 64 个 proxy 验证任务，用于本地评估与调参（≠ 正式 hidden world） |
| `public_tasks.jsonl` | 16 个小规模 proxy 调试任务 |
| `test_tasks.jsonl` | 32 个正式任务 public view，只含服务别名、拓扑、容量、成本、流量预测、观测空间和动作空间 |

### 观测说明

观测由 bucket 化指标、脱敏事件和可选诊断探针结果组成：

- 服务指标字段可能缺失
- 观测可能带有 `stale_steps`
- 事件可能延迟或丢包
- 事件字段为脱敏标识，不提供自然语言根因文本
- 诊断探针结果可能带噪声和延迟

> 具体字段、动作参数和合法取值以 `agent_api.md` 和 `action_schema.json` 为准。

---

## 评测说明

### 提交格式

- 提交文件名建议为 `NS-2026-08-answer.zip`
- zip 根目录必须直接包含且**只能包含** `agent.py`，不得多包一层目录
- 除 `agent.py` 外，不得提交其他源码文件、配置文件、数据文件、模型权重、checkpoint、策略权重、查表文件、预计算动作表、缓存 rollout、依赖包、wheel、site-packages 或大型中间文件

**agent.py 必须定义：**

```python
class Agent:
    def reset(self, task_info):
        ...

    def act(self, observation):
        return {"type": "NOOP"}
```

- 每个任务开始时调用一次 `reset(task_info)`
- 随后每分钟调用一次 `act(observation)`，agent 必须返回一个动作对象
- 每步最多一个动作
- 导入失败、缺少 `agent.py`、缺少 `Agent.act`、返回非法动作、运行异常或超时，可能导致低分、0 分或评测失败

---

### 动作集合

| 动作 | 参数 | 说明 |
|---|---|---|
| `SCALE` | `target`, `value` | 调整副本数 |
| `ROLLBACK_RELEASE` | `target` | 回滚目标服务发布 |
| `ROLLBACK_CONFIG` | `target` | 回滚目标服务配置 |
| `RESTART` | `target` | 重启目标服务 |
| `CIRCUIT_BREAK` | `target` | 对目标服务启用熔断 |
| `SET_TIMEOUT` | `value` | 设置超时，取值：100, 250, 500, 750, 1000, 1500 |
| `WARM_CACHE` | `target` | 预热缓存服务 |
| `REBUILD_INDEX` | `target` | 对存储服务执行索引维护 |
| `DRAIN_QUEUE` | `target` | 排空队列 |
| `THROTTLE` | `value` | 限流，范围 0 ~ 60 |
| `DIAGNOSTIC_PROBE` | `target` | 诊断探针（结果延迟出现，可能带噪声） |
| `NOOP` | 无 | 空动作 |

---

### 评分机制

隐藏评分器在 CPU 上逐步运行私有世界。每个 `test_tasks.jsonl` 正式任务只给出 public view；评测时由私有黑箱遥测世界产生逐分钟 observation，并调用 `agent.py` 决策。

**评分维度：**

| 维度 | 说明 |
|---|---|
| SLO 损失 | 错误、延迟、饱和、队列和依赖压力越低越好 |
| 恢复效率 | 越早把事故影响压低越好 |
| 资源成本 | 盲目扩容会持续扣分 |
| 动作成本 | 回滚、重启、索引维护、限流和诊断探针等均有成本 |
| 动作冷却与副作用 | 短时间内重复同类同目标动作会被阻断并扣分；部分动作会产生短期副作用 |
| 错误干预 | 对无关服务做高风险操作会被惩罚 |
| 稳健性 | 策略需要适应部分可观测、噪声、延迟反馈和任务差异 |

- 总分范围：**0 ~ 1500**
- 本题为探索型 Hard 任务，不承诺公开边界 exact 1500/1500
- 正式平台只返回总分

---

## 平台评测资源与安全边界

### 运行环境

- Docker sandbox（CPU-only Python judge）
- 操作系统：Linux x86_64
- Python 版本：**3.12.12**
- 不启用远程算力评测，不申请 GPU/显存，不提供 CUDA
- 不调用隐藏外部模型或 API
- 禁止联网（无公网、DNS 或内网访问）

### 安全约束

- 正式评测期间不得联网下载模型、安装依赖、调用外部 API、调用闭源模型服务、访问托管推理服务或探测平台网络
- 不要依赖 `pip install`、`apt install`、在线模型仓库、远程数据库或任何外部服务

### 可用第三方包

> 未列出的第三方包即使在当前镜像中偶然存在，也不作为题目承诺，选手不应依赖。

| 包 | 版本 |
|---|---|
| numpy | 2.3.2 |
| pandas | 2.3.1 |
| scipy | 1.16.0 |
| scikit-learn | 1.7.1 |
| scikit-image | 0.25.2 |
| opencv-python | 4.11.0.86 |
| pillow | 11.3.0 |
| joblib | 1.5.1 |
| networkx | 3.5 |
| tqdm | 4.67.1 |
| PyYAML | 6.0.2 |
| orjson | 3.11.3 |
| requests | 2.32.5 |

**不可用的包**：torch、tensorflow、xgboost、lightgbm（不得依赖这些包）

### 资源限制

> 若平台页面或组委会公告给出更新限制，以最新公告为准。

| 项目 | 限制 |
|---|---|
| 平台单次评测总超时 | 900 秒 |
| 提交 zip 大小 | 50 MB |
| 解压后总大小 | 1000 MB |
| 文件数量 | 2000 |
| 单文件大小 | 512 MB |
| 评测日志上限 | 512 KB |
| Docker sandbox CPU | 1 CPU |
| Docker sandbox 内存 | 2 GB |
| `Agent.reset` 单次调用超时 | 5 秒 |
| `Agent.act` 单次调用超时 | 2 秒 |
| agent 子进程 CPU 时间上限 | ~120 秒 |
| agent 子进程内存上限 | ~768 MiB |

### 运行约束

- 评测会限制常见科学计算库的线程数：`OMP_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`、`MKL_NUM_THREADS=1`
- 不建议在 `agent.py` 中使用多进程、多线程、长时间后台任务或大量文件句柄
- 每个任务会独立加载并运行一次 agent
- 可以在同一任务的 `reset` 和后续 `act` 之间保留内存状态
- 不要依赖跨任务、跨提交或跨评测的持久状态
- 本题不支持模型提交或大提交转存，不接受模型权重、checkpoint、大型中间文件、大型依赖包或缓存 rollout
- 最终提交不得依赖运行时下载资源
- 正式平台只返回总分，不生成选手可下载的评测附件，不公开逐任务 replay、隐藏私有状态、资源报告或评测细节

---

## 本地命令

```bash
# 格式检查
python tools/check_format.py example_submission

# 公开评测
python tools/run_public_eval.py example_submission --tasks public_tasks.jsonl

# 验证评测
python checker/checker.py --tasks valid_tasks.jsonl --submission example_submission

# 轨迹可视化（查看前 2 条）
python checker/visualize_trace.py train_traces.jsonl --limit 2
```
