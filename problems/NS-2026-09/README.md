# NS-2026-09 观测之环：隐变量世界模型反事实预测

> 难度：Hard | 总分：1500 | 题型：探索型（来源：[problem.md](../../neuronspark/09/NS-2026-09/problem.md)）

## 1. 题目介绍

- **背景**：12×12 网格世界，巡检机器人只有 5×5 局部视野 + 粗粒度事件传感器；真实动力学参数（推箱/推球、球漂移、箭头/冰面场效应及其先后顺序等）构成隐藏 profile，不直接给出。
- **输入**（每个 context）：完整初始地图与实体坐标、5 条 probe 轨迹（动作序列、稀疏局部观测/sensor-only 观测、**probe 终局完整地图**、终局事件/时段/顺序/终局类型），以及 2 个共享同一隐藏 profile 的反事实 query（`future_actions`）。
- **输出**：每个 query 的 `final_grid`（12×12）、`events`（6 个布尔事件：goal_reached / collision / hazard / box_on_goal / key_collected / portal_used）、`event_timeline`（never/early/mid/late）、`event_order`（前 3 个首次发生事件，不足补 `none`）、`terminal`（goal/hazard/blocked/active/timeout）。
- **数据规模**：train 720 context / 1440 query（有标签）；valid 160 / 320（有标签）；test 320 / 640（隐藏）。
- **评分**：

```
score = 100(格式) + 225×动态网格准确率 + 225×实体(A/B/O/K)位置准确率
      + 250×事件向量整体准确率 + 250×事件时段准确率 + 250×事件顺序准确率 + 200×终局类型准确率
```

提交 `NS-2026-09-answer.zip`，包内只含 `results.json`（JSON list，id 用 `queries[].query_id`）；缺失/重复/非法样本整次 0 分。

## 2. 我们的做法

**最终方案：手写离散规则模拟器 + 隐变量 profile 枚举搜索（纯标准库、确定性、无训练）。**

1. **规则世界模型**：手工实现全部公开机制——U/D/L/R/WAIT 移动与墙阻挡、门 D 需钥匙 K、推箱 B / 推球 O、箭头 `^v<>`、冰面 I、传送门 P（成对传送）、球自主漂移、6 类事件触发与 early/mid/late 分桶、前三事件顺序记录、终局优先级 goal > hazard > blocked > timeout。
2. **隐变量离散化**：把不确定的动力学参数组合成候选 profile，共 **280 个**（`default_profiles()`）：
   `push_box`×`push_orb`（2×2）× `orb_arrow_mode`/`orb_ice_mode`/`box_arrow_mode`（none/one）× `orb_terrain_timing`（after/before，仅在球受场效应时区分）× `orb_mode`（none / drift_U/D/L/R）。
3. **系统辨识**：对每个 context，用全部候选 profile 重放 5 条 probe，与公开 probe 终局比对，加权打分选最优 profile（同一 context 的所有 query 共用，保证跨 query 解释一致）：
   `8.0×动态格匹配 + 2.0×每个实体位置匹配 + 8.0×事件顺序精确匹配 + 0.5×事件向量匹配 + 0.25×整图精确匹配`。
4. **反事实 rollout**：用选中的 profile 对每个 query 的 `future_actions` 模拟，直接输出终局各字段。

### 迭代过程

| 阶段 | 位置 | 内容 / 结论 |
|------|------|------|
| 早期探索（5-25） | [neuronspark2026/NS-2026-09/](../../neuronspark/neuronspark2026/NS-2026-09/) | `sim.py` → `sim2.py`（G/H 只触发事件不终止 rollout）+ `predict.py`~`predict_v5.py` 的 knob 搜索；后期叠加 sklearn GradientBoosting 分类器（事件/时段/顺序/终局，`*.pkl`）与各类 picker/sweep/analyze 脚本，产出 `results_v10`~`results_v42.json` 等。该路线未进入最终提交，分数未记录 |
| 基础求解器 | [wmli_solver.py](../../neuronspark/09/NS-2026-09/wmli_solver.py) | 规则模拟器 + profile 网格搜索（支持 `--profile` / `--fit-profiles`） |
| 加 profile 搜索 | — | writeup 消融：仅基础规则 valid ≈995 → 280 候选 profile 搜索 valid 1434.74 |
| 扩展 profile（回退） | [solver_v2.py](../../neuronspark/09/NS-2026-09/solver_v2.py)、[solver_final.py](../../neuronspark/09/NS-2026-09/solver_final.py) | 加入 portal_cooldown、agent ice_slide、box_ice 等；valid 降到 1428.91（过拟合：cooldown 机制真实存在但无法可靠区分哪些 context 需要；ice_slide 只修复 7/223 个不完美 context），**回退** |
| **最终版** | [solver_submit.py](../../neuronspark/09/NS-2026-09/solver_submit.py) | 280 profile + **portal_used 修正**：训练集中有 4 例 A 停在 P 上但 `portal_used=False`，改为仅在实际发生传送时标记；网格预测 0 差异，valid 1434.74 → 1437.86 |

**残余误差**（writeup）：约 36/160 个 valid context 找不到完美匹配 probe 的 profile，主要涉及传送门 bounce-back 行为和未知的 agent 路径偏移。

## 3. 结果

| 数据集 | 分数 | 来源 |
|--------|------|------|
| valid（本地 `eval`） | 1437.86 | writeup §4.1 |
| **test（平台，最终提交）** | **1404.92 / 1500** | writeup §8，提交时间 2026/5/27 13:22:21 |

test 各维度准确率（writeup §4.2）：dynamic_grid 0.9208 / entity 0.9453 / event_vector 0.9297 / event_timeline 0.9298 / event_order 0.9194 / terminal 0.9516。

## 4. 目录与文件说明

主工作目录 [neuronspark/09/NS-2026-09/](../../neuronspark/09/NS-2026-09/)：

| 文件 | 作用 |
|------|------|
| [solver_submit.py](../../neuronspark/09/NS-2026-09/solver_submit.py) | **最终提交脚本**（单文件，含模拟器、profile 搜索、本地评分）；与 writeup 目录中的副本内容一致 |
| [NS-2026-09-answer.zip](../../neuronspark/09/NS-2026-09/NS-2026-09-answer.zip) | 最终提交包（仅含 `results.json`），与 [neuronspark2026-answer/NS-2026-09-answer.zip](../../neuronspark/neuronspark2026-answer/NS-2026-09-answer.zip) 内容相同 |
| [results.json](../../neuronspark/09/NS-2026-09/results.json) | 最终 test 预测（640 条），即 zip 内文件 |
| [wmli_solver.py](../../neuronspark/09/NS-2026-09/wmli_solver.py) / [solver_v2.py](../../neuronspark/09/NS-2026-09/solver_v2.py) / [solver_final.py](../../neuronspark/09/NS-2026-09/solver_final.py) | 中间版本（基础版 / 扩展 profile 版 / 严格切换扩展 profile 版），均未作为最终提交 |
| [problem.md](../../neuronspark/09/NS-2026-09/problem.md)、[DATA_NOTICE.md](../../neuronspark/09/NS-2026-09/DATA_NOTICE.md)、[label_schema.json](../../neuronspark/09/NS-2026-09/label_schema.json) | 题面、数据说明、标签字符集/取值定义 |
| `train.jsonl` / `valid.jsonl` / `test.jsonl`、`frames/`、[example_submission/](../../neuronspark/09/NS-2026-09/example_submission/) | 官方数据（PNG 帧为 probe/query 初始局部视野，本方案未使用） |
| [tools/](../../neuronspark/09/NS-2026-09/tools/) | 官方工具：`baseline.py`、`check_format.py`、`visualize_rollout.py` |

其他位置：

- neuronspark2026/2026-09.zip<sup>†</sup>：原始题目数据包。
- [neuronspark2026/NS-2026-09/](../../neuronspark/neuronspark2026/NS-2026-09/)：早期探索目录（`sim*.py`、`predict*.py`、`train_*.py`、`analyze_*.py`、`*.pkl`、`results_*.json`），其中的 `results.json` 与最终提交**不同**。
- [neuronspark2026/writeup_submit/09/](../../neuronspark/neuronspark2026/writeup_submit/09/)：writeup + 提交版 `solver_submit.py` 副本。

## 5. 如何运行

- 环境：Python 3.10+，仅标准库（最终方案无第三方依赖、无随机性）；writeup 记录单核运行约 valid 30s / test 60s。早期目录的 `train_*.py` 等需要 numpy + scikit-learn。
- 脚本无硬编码绝对路径，`--tasks` / `--out` 默认相对当前目录，建议在数据目录下运行：

```bash
cd neuronspark/09/NS-2026-09

# 本地验证（打印 profile 数、总分和 6 项准确率）
python solver_submit.py eval --tasks valid.jsonl

# 生成 test 预测
python solver_submit.py predict --tasks test.jsonl --out results.json

# 格式检查（官方工具）
python tools/check_format.py results.json --tasks test.jsonl

# 打包（包内只能有 results.json）
zip NS-2026-09-answer.zip results.json
```

可视化某条样本：`python tools/visualize_rollout.py --tasks train.jsonl --id <context_id>`。

## 6. 参考

- [writeup.md](../../neuronspark/neuronspark2026/writeup_submit/09/writeup.md)：方法、消融、失败案例、probe→query 可视化示例、AI 使用声明（A1）、提交 SHA256。
- [problem.md](../../neuronspark/09/NS-2026-09/problem.md)：完整题面与评分细则。

---

† 标注的文件为比赛原始数据或过程记录，按数据协议未包含在公开仓库中，请从官方渠道获取题目数据包。
