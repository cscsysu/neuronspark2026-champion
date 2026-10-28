# NS-2026-04 分子炼金术：小分子性质预测

> 难度：Moderate ｜ 总分：800 ｜ 题型：单目标回归（SMILES → 连续值）｜ AI 协助等级：A1 ｜ 最终得分：**800 / 800**

## 1. 题目介绍

- **任务**：给定小分子的 SMILES 字符串，预测其物理化学性质 / 活性指标（连续值）。要求构建能跨 scaffold（Bemis-Murcko 骨架）泛化的分子表示与模型。
- **数据**（原始包 2026-04.zip<sup>†</sup>）：
  | 文件 | 规模 / 内容 |
  |---|---|
  | `train.csv` | 8500 条，列 `id,smiles,target` |
  | `test.csv` | 3000 条，列 `id,smiles` |
  | `features/hashed_smiles_{train,test}.npz` | 官方预提取的 2048 维 hashed SMILES token fingerprint（scipy 稀疏矩阵，由 `tools/make_features.py` 生成：1~4-gram token + 原子计数，blake2b 哈希，seed=202604） |
  | `scaffold_split.json` | 推荐的 scaffold-aware 划分：`train_ids`(8500) / `valid_ids`(1700)，`method` 为 RDKit-free 的近似 SMILES scaffold hash |
  | `tools/check_format.py` | 提交格式检查 |
- **目标分布**（writeup）：mean 1.2128，std 0.9370，min -1.5892，max 7.4558，右偏，少量 target > 5 的极端样本。
- **提交**：`NS-2026-04-answer.zip`，内只含 `results.csv`，表头恰为 `id,prediction`，覆盖全部测试 ID，预测值为有限数。
- **评分**：
  ```
  raw_score = 500*max(0, 1 - RMSE/0.80) + 250*max(0, Spearman) + 50
  若 RMSE <= 0.25 且 Spearman >= 0.96（gold-band）：score = 800，否则 score = raw_score
  ```
  两个阈值必须同时达到才跳满分，因此 RMSE 0.254 / Spearman 0.958 也只有 630 分。

## 2. 我们的做法

**最终方案（[submit_v6.py](../../neuronspark/neuronspark2026/submit_v6.py)）**：官方 2048 维 hashed fingerprint + 28 维正则解析的手工 SMILES 描述符（原子/杂原子/芳香/卤素计数、环数、分支、双三键、电荷、手性、比例、分子量代理、log 与交互项），共 2076 维；LightGBM **6 种配置 × 5 个种子 = 30 模型等权平均**，全量 8500 条训练、固定迭代数、不做早停。

| 配置 | 损失 | lr | 树数 | num_leaves | 特点 |
|---|---|---|---|---|---|
| A | MSE | 0.01 | 5000 | 63 | 低 lr 深度拟合 |
| B | Huber | 0.01 | 5000 | 63 | 对异常值鲁棒 |
| C | MSE | 0.03 | 2000 | 63 | 首版基准配置 |
| D | MSE + DART | 0.05 | 500 | 63 | drop_rate 0.1 防过拟合 |
| E | MSE | 0.005 | 8000 | 31 | 极低 lr + 强正则 |
| F | MAE (L1) | 0.01 | 5000 | 63 | 优化中位数，利于排序 |

种子：`42 + seed_idx*77 + config_idx*999`。不依赖 RDKit / GPU。

**关键结论**
1. 特征不是越多越好：加入 RDKit 描述符 / Morgan / MACCS 后线上反而变差，官方 hashed token fingerprint 对新 scaffold 更稳健。
2. 多 seed 只降方差，收益极小（v5 仅比首版好 0.001 RMSE）；**多损失函数集成**才是越过 gold-band 的关键。
3. 全量 8500 训练优于只用 scaffold-train 6800 条训练。
4. 踩坑：早期本地 "scaffold-val RMSE=0.0576" 是假象——`scaffold_split.json` 的 `train_ids` 包含了 `valid_ids`，验证集泄漏；正确做法是 train = train_ids − valid_ids（6800 / 1700）。

**迭代过程**（线上分数来自 对话日志<sup>†</sup> 中的 score details）

| 版本 / 脚本 | 改动 | 线上 RMSE | 线上 Spearman | 得分 |
|---|---|---|---|---|
| [solve_quick.py](../../neuronspark/neuronspark2026/solve_quick.py) | hashed + 简单描述符，单 LightGBM 2000 树 | 0.2547 | 0.9584 | 630 |
| [submit_fast.py](../../neuronspark/neuronspark2026/submit_fast.py) | + RDKit 全描述符/Morgan/MACCS（6521 维），30 模型，800 树 | 0.2613 | 0.9567 | 626 |
| [submit_v4.py](../../neuronspark/neuronspark2026/submit_v4.py) | Morgan + MACCS + hashed（6311 维），scaffold 早停定树数，30 模型 | 0.2765 | 0.9538 | 616 |
| [submit_v5.py](../../neuronspark/neuronspark2026/submit_v5.py) | 回到 hashed + 描述符，多 seed | 0.2537 | 0.9589 | 631 |
| **[submit_v6.py](../../neuronspark/neuronspark2026/submit_v6.py)** | **多 loss / 多 lr / DART 集成，6×5 模型** | 未记录（≤0.25） | 未记录（≥0.96） | **800** |

其余尝试：[solve_gpu.py](../../neuronspark/neuronspark2026/solve_gpu.py)（GPU 服务器版，RDKit + LightGBM/XGBoost + GNN 设想，日志中 LightGBM OOF RMSE=0.2438 / Spearman=0.9619，但该 OOF 未反映线上表现）、[submit_v2.py](../../neuronspark/neuronspark2026/submit_v2.py)、[submit_v3.py](../../neuronspark/neuronspark2026/submit_v3.py)（scaffold 早停 + 多配置，scaffold-val RMSE 0.263~0.267）、[submit_quick.py](../../neuronspark/neuronspark2026/submit_quick.py)（只用 6800 条训练，scaffold-val RMSE≈0.276）——均未成为最终方案。注意：v5 实际提交的是对话中临时缩减版（2000 树、5 seed），与仓库中 `submit_v5.py`（3500 树）参数不完全一致。

## 3. 结果

- **最终线上得分：800 / 800（gold-band 满分）**，提交时间 2026/5/23 16:31:07（来源：[writeup](../../neuronspark/neuronspark2026/writeup_submit/04/writeup.md)）。v6 的具体 RMSE/Spearman 未记录。
- 本地 scaffold-val（6800/1700）单配置 RMSE 约 0.263~0.267、Spearman 约 0.954~0.955，比线上测试更难。
- 最终包 SHA256：zip `db7c030f…11122cbf`，results.csv `b62e1251…65643c3b`（与 writeup 第 14 节一致，已核对）。

## 4. 目录与文件说明

以下路径均位于 [neuronspark/neuronspark2026/](../../neuronspark/neuronspark2026/)：

| 路径 | 说明 |
|---|---|
| 2026-04.zip<sup>†</sup> | 原始题目包（train/test/scaffold_split/features/tools/example_submission） |
| features/hashed_smiles_train.npz<sup>†</sup>、hashed_smiles_test.npz<sup>†</sup> | 官方 hashed fingerprint（与题包内一致） |
| [scaffold_split.json](../../neuronspark/neuronspark2026/scaffold_split.json) | 官方 scaffold 划分（与题包内一致） |
| **[submit_v6.py](../../neuronspark/neuronspark2026/submit_v6.py)** | **最终训练+推理+格式检查+打包脚本** |
| solve_quick.py / submit_fast.py / submit_v2~v5.py / submit_quick.py / solve_gpu.py | 历史迭代脚本（见上表） |
| **[neuronspark2026-answer/NS-2026-04-answer.zip](../../neuronspark/neuronspark2026/neuronspark2026-answer/NS-2026-04-answer.zip)** | **最终提交包**（与 [NS-2026-04-answer-v6.zip](../../neuronspark/neuronspark2026/NS-2026-04-answer-v6.zip) 字节一致） |
| [NS-2026-04-answer.zip](../../neuronspark/neuronspark2026/NS-2026-04-answer.zip)、[-v4](../../neuronspark/neuronspark2026/NS-2026-04-answer-v4.zip)、[-v5](../../neuronspark/neuronspark2026/NS-2026-04-answer-v5.zip)、[-quick](../../neuronspark/neuronspark2026/NS-2026-04-answer-quick.zip) | 历史提交包（同目录下的 NS-2026-04-answer.zip 不是最终版） |
| [submission/](../../neuronspark/neuronspark2026/submission/)、[submission_v4/](../../neuronspark/neuronspark2026/submission_v4/)、[submission_v5/](../../neuronspark/neuronspark2026/submission_v5/) | 历史 results.csv 输出（`submission_v6/` 目录为空） |
| [writeup_submit/04/writeup.md](../../neuronspark/neuronspark2026/writeup_submit/04/writeup.md) | 正式 writeup |
| conversation-2026-05-27-04.txt<sup>†</sup> | 完整解题对话日志（含题面与每次线上 score details） |

> 注意：`neuronspark/neuronspark2026/` 根目录下的 `train.csv`、`test.csv`、`tools/`、`example_submission/` 已被其他赛题（NS-2026-06 等）的数据覆盖，**不是本题数据**；本题的 csv 和 `tools/check_format.py` 需从 `2026-04.zip` 解压获得。

## 5. 如何运行

环境：Python 3.9+，CPU 即可（Apple M 系列约 10 分钟，内存 < 4 GB）。

```bash
pip install numpy scipy lightgbm      # 最终方案无需 RDKit / GPU

# 1) 在独立目录解压题包（勿解压到 neuronspark2026/ 根目录，会覆盖其他题文件）
mkdir -p /tmp/NS-2026-04 && cd /tmp/NS-2026-04
unzip "<仓库根>/neuronspark/neuronspark2026/2026-04.zip"
cp "<仓库根>/neuronspark/neuronspark2026/submit_v6.py" .

# 2) 训练 30 个模型 + 推理 + 格式检查 + 打包（脚本使用相对路径，必须在数据目录下运行）
python3 submit_v6.py
# 输出：submission_v6/results.csv 与 NS-2026-04-answer-v6.zip

# 3) 单独做格式检查（可选）
python3 tools/check_format.py submission_v6/results.csv --test-csv test.csv
```

上传前将 `NS-2026-04-answer-v6.zip` 重命名为 `NS-2026-04-answer.zip`。
其他脚本：`submit_fast/v2/v3/v4/quick.py` 需额外安装 `rdkit`（`solve_gpu.py` 还需 xgboost / torch）；`solve_quick.py` 硬编码了 `WORK_DIR = /Users/tristonchen/Desktop/neuronspark2026`，运行前需改为实际数据目录。

## 6. 参考

- Writeup：[writeup_submit/04/writeup.md](../../neuronspark/neuronspark2026/writeup_submit/04/writeup.md)（含特征 SHA256、失败案例分析、AI 使用声明）
- 题面与提交记录：conversation-2026-05-27-04.txt<sup>†</sup>
- 官方数据镜像（题面给出）：https://modelscope.cn/datasets/SteamedFresh/NS-2026-A-data

---

† 标注的文件为比赛原始数据或过程记录，按数据协议未包含在公开仓库中，请从官方渠道获取题目数据包。
