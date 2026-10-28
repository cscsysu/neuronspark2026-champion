# NS-2026-01 智慧饭堂：人流与备餐预测

难度 Easy ｜ 总分 500 ｜ 题型：表格时间序列回归 ｜ AI 协助等级 A1 ｜ 队伍 claude-opus-4.8
（元信息来自题面，记录在 conversation-2026-05-27-01.txt<sup>†</sup>）

## 1. 题目介绍

- **背景**：校园饭堂要提前备餐，需要预测每个饭点、每个区域的就餐人数。人数有明显的星期/饭点/区域周期，考试周、节假日、天气、活动和菜单变化会带来局部异常。题面提示要"阅读 data.md 中的公开公式"（数据包里实际没有 data.md）。
- **数据**：一共 650 个连续日期 × 3 饭点（breakfast/lunch/dinner）× 6 区域（A01–A06）
  - `train.csv`：前 560 天，10080 行，列为 `date,meal,canteen_area,volume`
  - `test.csv`：后 90 天（2026-03-16 ~ 2026-06-13），1620 行，不含 volume
  - `calendar.csv`：weekday / is_weekend / semester_week / is_holiday / is_exam_week / is_makeup_day / campus_event_level
  - `weather.csv`（按 date+meal）：**demand_index**、weather、temperature、feels_like、humidity、wind_speed、rain_level
  - `menu.csv`（按 date+meal+area）：menu_type（7 类）、menu_popularity、is_promotion
  - 三张特征表都覆盖测试期
- **输出**：`results.csv`，列为 `date,meal,canteen_area,volume`，必须覆盖全部测试三元组，不能重复，volume 必须是有限非负数；打包成 `NS-2026-01-answer.zip`，压缩包里只能有 results.csv。
- **评分**：
  ```
  score = 200·max(0, 1-MAE/1000) + 150·max(0, 1-RMSE/1500) + 150·max(0, 1-MAPE/0.5)
  ```
  格式不合法直接 0 分。

## 2. 我们的做法

**核心发现**：`demand_index` 和 volume 高度相关（整体 r≈0.68，按 meal×area 分组后 r 在 0.78~0.91）。volume 大致等于 `demand_index × Σ(调节系数)`，也就是乘法结构，节假日、考试周、菜单、星期都是作用在 demand_index 上的乘数。

**最终方案：Two-Stage + 5-seed**（来源：writeup）
1. **Stage1：Ridge 线性乘法模型**。每个 (meal, area) 组单独训练一个 Ridge(alpha=1.0)，特征是 demand_index 与各调节因子的交互：`di`、`di×menu_popularity`、`di×is_promotion`、`di×menu_type`（7 类 one-hot）、`di×weekday`（7 天 one-hot）、`di×is_holiday/is_exam_week/is_makeup_day/campus_event_level/is_weekend`、`di×temperature/humidity/rain_level`。只用 Stage1 时验证 MAE 就有 18.73。
2. **Stage2：LightGBM 拟合残差**（`volume - Stage1`）。特征包括日历、天气、菜单热度/促销、meal/area 编码、demand_index、月份、星期；参数为 MAE loss、lr=0.03、num_leaves=63、feature/bagging_fraction=0.8，early stopping 100 轮。
3. Stage2 用 seed 42/123/456/789/1024 各训练一次取平均；**最终预测 = Stage1 + mean(Stage2)**，再 clip 到 ≥0。
4. **验证切分**：严格按时间。训练子集为 2024-09-02~2025-12-15（470 天），验证集为 2025-12-16~2026-03-15（90 天，和测试集等长）。

**迭代过程**
| 版本 | 做法 | 为什么改 |
|---|---|---|
| V1 `solve.py` | 纯 LightGBM（MAE loss），39 个特征：日历/天气/菜单、sin/cos 周期编码、组内历史均值、最近 N 周均值 | baseline |
| `solve_v2~v5.py` | 在 V1 基础上加 lag/统计特征、多目标 LGB 融合（权重用 DE/网格搜索）、提速版 | 本地分数一直在 463~465 附近，没有突破；这几版没有线上分数记录 |
| V2 Two-Stage | Stage1 只用 Ridge 拟合 `di×菜单特征`，Stage2 用 LGB 拟合残差（5000 轮） | 把 demand_index 的线性比例关系交给线性模型，MAE 从 ~50 降到 ~23 |
| V3 | Stage1 加入 weekday/holiday/exam/event/天气等交互 | 对应"乘法公式"，验证 MAE 降到 ~10.4 |
| V4（最终） | V3 + Stage2 5-seed 平均 | 降低方差 |

误差分析（writeup）：正常日验证 MAE 12.24，活动日 16.12，节假日 65.43。节假日样本少（训练集只有 33 天），是主要误差来源。

## 3. 结果

| 版本 | 本地 Val | 线上 | 来源 |
|---|---|---|---|
| V1 纯 LightGBM | 465.2 | 470.76（MAE 49.99 / RMSE 73.01 / MAPE 0.0398） | writeup + 对话日志 |
| V2 Two-Stage（简化 Stage1） | 484.9 | 482.82（MAE 27.38 / RMSE 47.78 / MAPE 0.0231） | writeup + 对话日志 |
| V3 强化 Stage1 | 492.7 | 未记录 | writeup |
| **V4 最终（5-seed）** | **492.9**（MAE 10.15 / RMSE 17.16 / MAPE 0.0113） | **489.92 / 500** | writeup |

> 注意 SHA256 不一致：writeup 第 7 节写的 `a4824020…6c69` 实际是 **V2** 包的哈希（对话日志里 V2 打包时输出的就是这个值）。仓库里 [neuronspark2026-answer/NS-2026-01-answer.zip](../../neuronspark/neuronspark2026-answer/NS-2026-01-answer.zip) 的哈希也是它，所以那是 V2 包（线上 482.82）。V4 最终包是 [2026-01/NS-2026-01-answer.zip](../../neuronspark/neuronspark2026/2026-01/NS-2026-01-answer.zip)，SHA256 为 `b2f2448d13139d9d0956636e9091fb80a1bc0a3e5abb7c46282d8d2749f07a53`。

## 4. 目录与文件说明

数据和代码目录：[neuronspark/neuronspark2026/2026-01/](../../neuronspark/neuronspark2026/2026-01/)

| 文件 | 作用 |
|---|---|
| train.csv<sup>†</sup> / test.csv<sup>†</sup> | 训练集 / 测试集 |
| calendar.csv<sup>†</sup> / weather.csv<sup>†</sup> / menu.csv<sup>†</sup> | 附加特征表 |
| [example_submission/results.csv](../../neuronspark/neuronspark2026/2026-01/example_submission/results.csv) | 官方示例提交 |
| [tools/check_format.py](../../neuronspark/neuronspark2026/2026-01/tools/check_format.py) | 官方格式检查脚本 |
| [solve.py](../../neuronspark/neuronspark2026/2026-01/solve.py) | V1 纯 LightGBM（线上 470.76） |
| [solve_v2.py](../../neuronspark/neuronspark2026/2026-01/solve_v2.py) | 5 个 LGB 配置 + differential_evolution 搜融合权重（较慢；脚本 import 了 Ridge 但没用到） |
| [solve_v3.py](../../neuronspark/neuronspark2026/2026-01/solve_v3.py) / [solve_v4.py](../../neuronspark/neuronspark2026/2026-01/solve_v4.py) / [solve_v5.py](../../neuronspark/neuronspark2026/2026-01/solve_v5.py) | 多模型融合 / 3 模型快速版 / 单模型极速版（都是纯 LGB） |
| [results.csv](../../neuronspark/neuronspark2026/2026-01/results.csv) | V4 最终预测（和下面这个 zip 里的内容一致） |
| [**NS-2026-01-answer.zip**](../../neuronspark/neuronspark2026/2026-01/NS-2026-01-answer.zip) | **最终提交包（V4，线上 489.92）** |
| [writeup_plots.png](../../neuronspark/neuronspark2026/2026-01/writeup_plots.png) | V2 阶段的预测/残差图 |
| [2026-01 2/](../../neuronspark/neuronspark2026/2026-01%202/) | 和 2026-01/ 完全相同的重复目录（`diff -r` 无差异） |
| 2026-01.zip<sup>†</sup> | 原始题目包（数据 + example_submission + tools） |
| [neuronspark2026-answer/NS-2026-01-answer.zip](../../neuronspark/neuronspark2026-answer/NS-2026-01-answer.zip) | V2 Two-Stage 包（线上 482.82），**不是**最终版 |
| [writeup_submit/01/](../../neuronspark/neuronspark2026/writeup_submit/01/) | writeup.md + prediction_analysis.png + feature_importance.png |

> **最终 Two-Stage（V2/V3/V4）的代码没有保存成文件**。根据对话日志，它是在 `2026-01/` 目录下用 `python3 -c "..."` 直接跑的。现有的 `solve*.py` 都是纯 LightGBM 版本，跑不出 489.92。要复现只能参考 [writeup](../../neuronspark/neuronspark2026/writeup_submit/01/writeup.md) 第 1.3 节和附录里的代码框架重新实现。

## 5. 如何运行

- 环境：Python 3.9+，依赖 `pandas numpy scikit-learn lightgbm`（solve_v2.py 还需要 `scipy`）。macOS 上 LightGBM 需要先 `brew install libomp`。CPU 即可，不需要 GPU。
- 所有脚本用相对路径读 `train.csv` 等文件，输出 `results.csv` 到当前目录，没有硬编码绝对路径，所以必须在数据目录下运行：

```bash
cd "neuronspark/neuronspark2026/2026-01"
pip install pandas numpy scikit-learn lightgbm scipy
python3 solve.py            # 或 solve_v5.py 等；会覆盖当前目录下的 results.csv
python3 tools/check_format.py results.csv --test-csv test.csv
zip NS-2026-01-answer.zip results.csv   # 会覆盖已有的最终包，请先备份
```

注意：运行任何 `solve*.py` 都会覆盖目录里现有的 V4 `results.csv`。如果只想验证最终包，可以解压到临时目录再检查：
`unzip -o NS-2026-01-answer.zip -d /tmp/ns01 && python3 tools/check_format.py /tmp/ns01/results.csv --test-csv test.csv`。

## 6. 参考

- Writeup：[writeup_submit/01/writeup.md](../../neuronspark/neuronspark2026/writeup_submit/01/writeup.md)（含图 [prediction_analysis.png](../../neuronspark/neuronspark2026/writeup_submit/01/prediction_analysis.png)、[feature_importance.png](../../neuronspark/neuronspark2026/writeup_submit/01/feature_importance.png)）
- 解题对话日志（题面、各版本线上分数、Two-Stage 内联代码的执行记录）：conversation-2026-05-27-01.txt<sup>†</sup>

---

† 标注的文件为比赛原始数据或过程记录，按数据协议未包含在公开仓库中，请从官方渠道获取题目数据包。
