# NS-2026-09 观测之环：隐变量世界模型反事实预测 — Writeup

## 1. 方法概述

本方案采用**离散规则求解器 + 隐变量 profile 搜索**的方法。核心思路是：

1. 手工建模 12×12 网格世界的所有公开机制（移动、墙壁、冰面、箭头、传送门、门/钥匙、推箱/推球、球漂移等）
2. 将不确定的动力学参数组合成离散 Profile（约 280 个候选）
3. 对每个 test context，用其 5 条 probe 轨迹的公开终局观测作为校准目标，遍历所有候选 Profile，选择与 probe 终局匹配度最高的一个
4. 使用选定 Profile 对该 context 下所有 query 的 future_actions 执行 rollout，输出终局预测

## 2. 任务理解

- **输入**：12×12 网格世界初始地图 + 5 条 probe 校准轨迹（含动作序列和公开终局）+ 2 个 query 动作序列
- **输出**：每个 query 的 final_grid、events（6 个布尔事件）、event_timeline（事件首次发生时段）、event_order（前 3 个事件顺序）、terminal（终局类型）
- **核心难点**：世界的动力学参数（隐变量 profile）不直接给出，需要从 probe 观测中反推

### 关键隐变量维度

| 隐变量 | 候选值 | 含义 |
|--------|--------|------|
| push_box | True/False | 智能体能否推动箱子 B |
| push_orb | True/False | 智能体能否推动球 O |
| orb_mode | none/drift_U/D/L/R | 球的自主漂移方向 |
| orb_arrow_mode | none/one | 球是否受箭头场影响 |
| orb_ice_mode | none/one | 球是否受冰面影响 |
| box_arrow_mode | none/one | 箱子是否受箭头场影响 |
| orb_terrain_timing | after/before | 球的场效应在漂移前还是后 |

## 3. 关键改进

### 3.1 portal_used 事件触发修正（+3.1 分 valid 提升）

**发现**：通过分析训练数据，发现 4 个案例中智能体最终停留在 P（传送门）格子上，但 `portal_used=False`。这表明 `portal_used` 事件仅在**实际发生传送**时才触发，而非"走上 P 格子"即触发。

**修复**：从 `handle_cell_events()` 中移除 `portal_used` 标记，仅在 `apply_portal()` 中传送成功时标记。

```python
# 修复前（原始代码）：
def handle_cell_events(self, entity, step):
    if entity == "A":
        ...
        elif ch == "P":
            self.mark_event("portal_used", step)  # 走上P就标记

# 修复后：
def handle_cell_events(self, entity, step):
    if entity == "A":
        ...
        # 移除 portal_used 标记，仅在 apply_portal 中标记
```

**验证**：该修复使 valid 分数从 1434.74 提升至 1437.86（+3.13），且网格预测完全不变（0 差异），纯事件准确率提升。

### 3.2 Profile 搜索与匹配函数

Profile 匹配使用加权评分函数，权重经过反复调优：

```python
score += 8.0 * grid_cell_accuracy    # 动态网格格子匹配
score += 2.0 * entity_position_match  # 实体位置匹配（每个实体）
score += 8.0 * event_order_exact      # 事件顺序精确匹配
score += 0.5 * event_vector_match     # 事件向量匹配
score += 0.25 * grid_exact_match      # 全网格精确匹配 bonus
```

### 3.3 世界模型机制实现

完整实现的机制包括：

- **基础移动**：U/D/L/R/WAIT，墙壁阻挡，门锁（需钥匙）
- **实体交互**：推箱子、推球（可配置）
- **场效应**：箭头（单步/链式）、冰面（单步/滑行）
- **传送门**：pair 模式传送，事件标记修正
- **球漂移**：自主移动，方向/周期/反弹可配置
- **事件系统**：6 类事件触发、时序分桶（early/mid/late）、顺序记录
- **终局判定**：goal > hazard > blocked > timeout 优先级

## 4. 验证与复现

### 4.1 本地验证分数

| 数据集 | 分数 | 备注 |
|--------|------|------|
| valid.jsonl | 1437.86 | 修复后 |
| test（平台） | 1404.92 | 最终提交 |

### 4.2 各维度准确率（test 最终提交）

| 指标 | 准确率 |
|------|--------|
| dynamic_grid_accuracy | 0.9208 |
| entity_accuracy | 0.9453 |
| event_vector_accuracy | 0.9297 |
| event_timeline_accuracy | 0.9298 |
| event_order_accuracy | 0.9194 |
| terminal_accuracy | 0.9516 |

### 4.3 消融实验

| 配置 | Valid 分数 | 说明 |
|------|-----------|------|
| 基础规则（无 profile 搜索） | ~995 | 仅基本移动 + 碰撞 |
| + Profile 搜索（280 候选） | 1434.74 | 核心提升 |
| + portal_used 修正 | 1437.86 | 事件准确率提升 |
| + 扩展 profile（portal_cooldown, ice_slide） | 1428.91 | 过拟合，回退 |

### 4.4 失败案例分析

- **约 36/160 valid context 无完美 profile 匹配**：其中 20 个涉及传送门交互（bounce-back 行为），43 个涉及未知的 agent 路径偏移
- **Portal cooldown 尝试**：发现传送门冷却是一个真实机制（训练数据验证 23 个 context 受益），但 profile 选择无法可靠区分需要/不需要冷却的 context，导致净效果为负
- **Ice slide 扩展**：仅修复 7/223 个不完美 context，收益过低

## 5. Probe-to-Query 反事实可视化

### 示例 1: wmli_v_0000 — 传送门交互

```
初始地图:                    Probe 0 终局（portal_used=True）:
############                ############
#..##H#....#                #..##H#....#
#..#..v....#                #..#..v....#
#.^..I..KP.#                #.^..I..KPA#   ← A传送后停在P旁
#..#.......#                #..#.......#
#.I.P.<..A.#  → A在(9,5)   #.I.P.<....#
#.<..I.<.D>#                #.<..I.<.D>#
...                         ...

Query q0 预测（collision=True, portal_used=True）:
A沿不同路径到达传送门，被传送后碰墙 → terminal=blocked
```

### 示例 2: wmli_v_0006 — 球漂移 + 推动

```
初始: O在(4,4), A在(1,1), orb_mode=drift_L
Probe: O向左漂移3步到(1,4)后被墙阻挡
Query: A向下推O → O被推到(4,5)后继续左漂 → 最终O在(1,5)
```

### 示例 3: wmli_tr_0002 — 冰面滑行

```
初始: A在(1,6), I在(2,5)和(4,9)
A向上移动到冰面(2,5) → 滑行一格到(2,4)
后续动作到达传送门 → 传送到(7,9) → 事件顺序: [portal_used, collision, none]
```

### 示例 4: wmli_v_0034 — 事件顺序差异

```
Profile: push_box=True, orb_mode=drift_U
Probe: collision在early → 用于校准profile匹配
Query: 不同路径 → collision在mid, key_collected在late
→ event_order: [collision, key_collected, none]
```

### 示例 5: wmli_v_0013 — 球箭头交互

```
orb_arrow_mode=one, orb_terrain_timing=before
O在箭头'^'上 → 先受箭头推动向上，再按漂移方向移动
Probe验证: O最终位置匹配 before timing
Query: O漂移路径影响A的可达区域
```

## 6. 复现说明

### 6.1 环境

- Python 3.10+（标准库即可，无额外依赖）
- 运行时间: valid ~30s, test ~60s（单核）

### 6.2 运行命令

```bash
# 本地验证
python solver_submit.py eval --tasks valid.jsonl

# 生成提交
python solver_submit.py predict --tasks test.jsonl --out results.json

# 打包
zip NS-2026-09-answer.zip results.json
```

### 6.3 文件结构

```
09/
├── solver_submit.py    # 完整求解器（单文件，~450行）
└── writeup.md          # 本文档
```

### 6.4 随机种子

本方案为**纯确定性**规则求解，无随机种子。给定相同输入，输出完全确定。

## 7. AI 使用声明

- **AI 协助等级**: A1
- **使用工具**: Claude (Anthropic) 用于代码编写、调试辅助和分析
- **最终推理**: 由本地 Python 脚本 `solver_submit.py` 完成，AI 工具未接触完整 test 输入生成最终答案
- **test 推理过程**: `python solver_submit.py predict --tasks test.jsonl --out results.json`，纯本地确定性执行

## 8. 提交信息

- **最终提交文件**: `NS-2026-09-answer.zip`
- **SHA256**: `bdf15ebf1ee84e5f4b08598d8752af8c8b7cebd9a3209e935a4627585b3deb5f`
- **平台得分**: 1404.92 / 1500
- **提交时间**: 2026/5/27 13:22:21
