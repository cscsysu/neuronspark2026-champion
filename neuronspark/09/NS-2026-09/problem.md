# NS-2026-09 观测之环：隐变量世界模型反事实预测

> **难度：Hard | 总分：1500 | 题型：探索型**

---

## 1. 任务介绍

环形观测站正在排查一组异常网格世界。巡检机器人只能看到身边很小的一块局部视野，传感器还会把许多事件压缩成粗粒度摘要；真正的动力学参数、场效应顺序和隐藏机制组合不会直接给出。

你需要扮演**世界模型工程师**，根据有限 probe 轨迹还原这些隐变量，并预测同一隐藏世界中多个反事实 query 的后果。

### 核心流程

每个测试 context 包含以下内容：
- 同一隐藏世界的完整初始地图、坐标锚点
- 若干 probe 轨迹、稀疏局部过程观测
- probe 终局完整观测、聚合事件传感器
- 多个共享同一隐藏 profile 的反事实 query 动作序列

目标：从公开校准观测中推断一致的世界动力学解释，并预测每个 query 执行后的终局各项指标。

### 你需要完成的工作

1. **学习世界机制** —— 从公开 train/valid 的 probe、query 和标签中学习世界机制与隐变量线索
2. **建立隐藏机制解释** —— 对 test 中每个 context 建立一致的隐藏机制解释，而不是孤立处理每个 query
3. **反事实 rollout** —— 对每个 query 的未来动作序列做反事实 rollout，输出终局网格、事件集合、事件时段、事件顺序和终局类型
4. **本地验证迭代** —— 在只获得平台总分反馈的条件下，通过本地验证、消融和可视化定位模型缺陷

### 关键约束

- 隐藏 profile 可能影响移动、场效应、实体交互、事件触发、状态转移和终局判定等多个方面
- 不同隐变量之间可能存在**组合效应**
- 同一 context 下多个 query 共享隐藏 profile 和初始隐状态，但未来动作不同，需要**跨 query 保持机制解释一致**
- 官方不承诺公开边界 exact 1500/1500，高分需要构建能恢复隐变量的世界模型

---

## 2. 数据说明

### 2.1 数据获取

在赛题页 "相关资源" 处下载数据包，或在 [ModelScope](https://modelscope.cn/datasets/SteamedFresh/NS-2026-AAA-data) 获取。

### 2.2 目录结构

```
NS-2026-09/
├── train.jsonl                 # 训练集
├── valid.jsonl                 # 验证集
├── test.jsonl                  # 测试集
├── frames/
│   ├── train/                  # train 的 probe/query 初始稀疏局部观测帧 (PNG)
│   ├── valid/
│   └── test/
├── label_schema.json           # 标签格式定义
├── DATA_NOTICE.md              # 数据注意事项
├── example_submission/
│   └── results.json            # 提交格式示例
└── tools/
    ├── baseline.py             # 基线代码
    ├── check_format.py         # 格式校验工具
    └── visualize_rollout.py    # 轨迹可视化工具
```

### 2.3 数据规模

| Split | Context 数 | Query 目标数 | 标签状态 |
|-------|-----------|-------------|---------|
| train | 720 | 1440 | 公开每个 query 的完整 label |
| valid | 160 | 320 | 公开每个 query 的完整 label |
| test  | 320 | 640 | **隐藏** 每个 query 终局目标 |

### 2.4 字段详解

#### Context 级字段（每条记录对应一个 context）

| 字段 | 类型 | 含义 |
|------|------|------|
| `id` | string | context 级样本 ID |
| `observation_mode` | string | 固定为 `sparse_egocentric_5x5_plus_event_sensors` |
| `coordinate_system` | object | 坐标系说明：x = 列号，y = 行号，原点在左上角 |
| `initial_full_grid` | list[string] | 该 context 的完整初始地图和实体位置（**test 也公开**） |
| `initial_entities` | object | 从 `initial_full_grid` 解析出的初始实体坐标 |
| `context_episodes` | list[object] | 同一隐藏世界 profile 下的 **probe 轨迹** |
| `queries` | list[object] | 同 context 下多个反事实 **query**，共享同一套隐藏 dynamics profile |

#### `context_episodes[ ]` — Probe 轨迹字段

| 字段 | 类型 | 含义 |
|------|------|------|
| `actions` | list[string] | probe 动作序列 |
| `observations` | list[object] | probe 观测序列；部分项含 `local_view`，部分为 sensor-only 观测 |
| `observed_final_full_grid` | list[string] | probe 动作执行后的**公开终局完整地图** |
| `observed_final_events` | object | probe 的公开终局事件集合 |
| `observed_final_event_timeline` | object | probe 的公开事件首次发生时段 |
| `observed_final_event_order` | list[string] | probe 的公开前三个首次发生事件顺序 |
| `observed_final_terminal` | string | probe 的公开终局类型 |

#### `queries[ ]` — Query 字段

| 字段 | 类型 | 含义 |
|------|------|------|
| `query_id` | string | 提交 `results.json` 时必须使用的预测 ID |
| `initial_observation` | object | query 初始稀疏局部观测（始终包含 `local_view`） |
| `initial_full_grid` | list[string] | query 对应的完整初始地图，与 context 的 `initial_full_grid` 对齐 |
| `initial_entities` | object | query 初始实体坐标 |
| `future_actions` | list[string] | **需要预测**的反事实未来动作序列 |
| `query_horizon` | int | query rollout 步数 |
| `label` | object | 仅 train/valid 提供：query 终局标签 |

### 2.5 重要说明

- **公开可见的校准信息**：`initial_full_grid` 和 probe 终局完整观测
- **稀疏化过程观测**：`local_view` 是 5×5 局部视野，sensor-only 观测只提供 sensor、事件增量、`event_timeline_so_far`、`event_order_so_far` 等字段
- `frames/` 中的 PNG 是 probe/query 的初始稀疏局部观测帧，完整地图以 JSON 文本字段提供
- 公开包**不包含**：完整逐步结构化状态、test query 终局答案、隐藏 profile、生成 seed、评分程序或内部答案材料

---

## 3. 评测说明

### 3.1 提交格式

- 提交文件：压缩为 `NS-2026-09-answer.zip`，包内只包含 `results.json`
- `results.json` 是一个 JSON list，每个 test query 必须出现且只能出现一次
- **id 使用 `test.jsonl` 中 `queries[].query_id`**，不是 context 级 id

#### 示例

```json
[
  {
    "id": "wmli_t_0000_xxxxxxxx_q0",
    "final_grid": ["############", "#..A.......#", "..."],
    "events": {
      "goal_reached": false,
      "collision": true,
      "hazard": false,
      "box_on_goal": false,
      "key_collected": true,
      "portal_used": false
    },
    "event_timeline": {
      "goal_reached": "never",
      "collision": "early",
      "hazard": "never",
      "box_on_goal": "never",
      "key_collected": "mid",
      "portal_used": "never"
    },
    "event_order": ["collision", "key_collected", "none"],
    "terminal": "blocked"
  }
]
```

### 3.2 格式要求

| 字段 | 要求 |
|------|------|
| `id` | 必须与 `test.jsonl` 中 `queries[].query_id` 完全一致 |
| `final_grid` | 12 行 × 12 列，字符集合以 `label_schema.json` 为准 |
| `events` | 包含 `label_schema.json` 中列出的**全部事件键**，取值为 `bool` |
| `event_timeline` | 包含全部事件键，取值：`never` / `early` / `mid` / `late`（事件首次发生的时段） |
| `event_order` | 长度为 3 的 list，每项为事件键或 `"none"`（前三个首次发生事件的顺序，不足用 `"none"` 补齐） |
| `terminal` | 取值：`goal` / `hazard` / `blocked` / `active` / `timeout` |

> 缺失样本、重复样本、未知样本或非法字段 → 本次提交 **0 分**。

### 3.3 得分计算（总分 1500）

| 评分项 | 分值 | 说明 |
|--------|------|------|
| 动态网格准确率 | 225 | 只在提交或隐藏答案出现动态实体/门/钥匙的位置比较 `final_grid` |
| 关键实体位置准确率 | 225 | 从提交网格解析 A/B/O/K 的终局位置，与隐藏答案比较；双方都不存在的实体不计分 |
| 事件向量准确率 | 250 | 六个事件键必须**整体一致**才计为该样本事件正确 |
| 事件时序准确率 | 250 | 对非 `never` 的事件比较首次发生时段 `early`/`mid`/`late` |
| 事件顺序准确率 | 250 | 比较前三个首次发生事件的顺序，`none` 只作补位 |
| 终局类型准确率 | 200 | 比较 `terminal` |
| 格式合法性 | 100 | 完整、合法、可评分 |

**总分公式：**

```
score = 100
      + 225 × dynamic_grid_accuracy
      + 225 × entity_accuracy
      + 250 × event_vector_accuracy
      + 250 × event_timeline_accuracy
      + 250 × event_order_accuracy
      + 200 × terminal_accuracy
```

> 平台正式反馈只展示**总分**。
