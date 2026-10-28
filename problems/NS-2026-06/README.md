# NS-2026-06 标注净化师：噪声标签清洗与鲁棒分类

> 题型：含噪弱标签的匿名特征 6 分类（表格数据）｜总分：1200｜难度：未记录｜线上最终得分：**1062 / 1200**（精确值 1061.7268）

## 1. 题目介绍

- **背景**：数据由公开的 20 Newsgroups 语料派生（见 [DATA_NOTICE.md](../../neuronspark/06/NS-2026-06/DATA_NOTICE.md)），原文经 TF-IDF/SVD 投影、标准化、随机旋转、扰动和取整，变成 96 维匿名稠密特征 `f000`–`f095`。不提供原文。
- **训练集** train.csv<sup>†</sup>：6015 行，给出 `weak_label`（弱规则/旧模型/标注组融合得到，含约 15%–25% **非均匀**噪声）、`weak_confidence`、`source_group`（10 个来源组 sg_00–sg_09）、`annotator_group`（6 个标注组）。
- **可信验证集** trusted_valid.csv<sup>†</sup>：1031 行，给出人工校验的真实 `label` 和 `source_group`，用于验证、校准和噪声分析。
- **测试集** test.csv<sup>†</sup>：1547 行，**只有 96 维特征**（没有 source_group / annotator_group）。
- **类别**（[label_map.json](../../neuronspark/06/NS-2026-06/label_map.json)）：`computer_hardware`、`system_software`、`mobility_vehicle`、`sports_recreation`、`medical_health`、`space_science`；其中 `computer_hardware`、`medical_health`、`space_science` 为高优先级类别（priority=true）。
- **噪声画像** [noise_profile_public.json](../../neuronspark/06/NS-2026-06/noise_profile_public.json)：公开各来源组的 top 弱标签计数和平均置信度。
- **输出**：`results.csv`，表头 `id,label`，id 必须与 test.csv 完全一致、无重复、标签合法（见 [tools/check_format.py](../../neuronspark/06/NS-2026-06/tools/check_format.py)），打包为 `NS-2026-06-answer.zip`。
- **评分**：官方 baseline 中的本地近似公式为 `900 × Macro F1 + 200 × 高优先级类别 micro 召回 + 100（格式分）`，满分 1200（见 [tools/baseline.py](../../neuronspark/06/NS-2026-06/tools/baseline.py) 的 `score()`）。

## 2. 我们的做法

**核心发现**：在 trusted_valid 上，每个 `source_group` 都对应**唯一**的真实标签（10 组 → 6 类，如 sg_03/sg_06 → computer_hardware，sg_00/sg_04 → mobility_vehicle）。用这个映射核对 train，发现 1312/6015（**21.81%**）行的 weak_label 与来源映射标签不一致，与题面噪声范围一致。所以 `source_group` 是比 `weak_label` 干净得多的监督信号。

**最终方案（来源组中转）**：

1. 从 trusted_valid 统计 `source_group → label` 映射（每组取多数标签）。
2. 用 train + trusted_valid 的 96 维特征训练 **10 类来源组分类器**：`StandardScaler + LogisticRegression(C=0.8, class_weight="balanced", lbfgs, max_iter=5000)`。
3. 对测试样本预测 10 个来源组概率，按映射**求和**得到 6 类标签概率。
4. 乘以在 trusted_valid OOF 上调出的固定类别倍率后取 argmax：hardware 1.30 / software 0.95 / vehicle 0.85 / sports 0.95 / medical 1.60 / space 1.05（抬高优先级类别，保护召回）。
5. `weak_label` 不作为训练目标，只用于噪声分析与疑似错标样本识别；`annotator_group` 测试集没有，只用于分析。
6. 随机种子统一为 `260606`。

**迭代过程**：

| 阶段 | 做法 | 文件 |
|---|---|---|
| 官方 baseline | weak_label + 置信度加权 LR，丢弃 18% 低一致性弱标签后重训 | [tools/baseline.py](../../neuronspark/06/NS-2026-06/tools/baseline.py) |
| 鲁棒训练尝试 | 可靠度打分 + 丢弃低可信弱标签（drop_ratio=0.18）+ 可信样本加权（gold_weight=6）+ LR/LinearSVC/ExtraTrees/XGBoost 融合 + 自动调类别倍率 | [robust_solution.py](../../neuronspark/06/NS-2026-06/robust_solution.py) |
| 噪声分析 | 按 source_group / annotator_group / confidence 统计弱标签与真实标签的分布差异，找到 sg→label 一一对应 | [analyze_noise.py](../../neuronspark/neuronspark2026/analyze_noise.py) |
| 最终方案 | 来源组分类器 + 概率映射 + 类别倍率 | [source_group_solution.py](../../neuronspark/06/NS-2026-06/source_group_solution.py) → [solver_06.py](../../neuronspark/06/NS-2026-06/writeup_06_submit/solver_06.py) |

Writeup 的说法：直接用 weak_label 训练线性模型时可信验证约 1000 分上下；也试过弱标签过滤、线性 SVM、树模型，最终来源组逻辑回归的本地 OOF 最稳，线上分数也最高。（中间版本没有单独的分数日志。）

另外，writeup 指出 `weak_confidence` 并不等于正确概率：有些错标样本的置信度依然很高（如 `trn_02671_21860c29`，conf=0.8927，弱标签 sports，来源映射判为 vehicle）。

## 3. 结果

| 来源 | Macro F1 | 优先级召回 | 分数 |
|---|---:|---:|---:|
| 线上（[writeup.md](../../neuronspark/06/NS-2026-06/writeup_06_submit/writeup.md)，2026/5/25 14:01:51 提交） | – | – | **1061.7268 / 1200**（页面显示 1062） |
| 本地 trusted_valid 5 折 OOF（[local_cv_and_generation.log](../../neuronspark/06/NS-2026-06/writeup_06_submit/logs/local_cv_and_generation.log)） | 0.89016 | 0.89573 | score_like 1080.29 |

各折 score_like：1101.16 / 1092.02 / 1070.47 / 1094.10 / 1043.76。OOF 整体 accuracy 0.8894；最主要的混淆是 system_software 被判成 computer_hardware（33 例）。格式检查结果：`OK: submission format is valid.`

## 4. 目录与文件说明

题目数据和代码在 [neuronspark/06/NS-2026-06/](../../neuronspark/06/NS-2026-06/)：

| 路径 | 说明 |
|---|---|
| train.csv<sup>†</sup> / trusted_valid.csv<sup>†</sup> / test.csv<sup>†</sup> / [label_map.json](../../neuronspark/06/NS-2026-06/label_map.json) / [noise_profile_public.json](../../neuronspark/06/NS-2026-06/noise_profile_public.json) / [DATA_NOTICE.md](../../neuronspark/06/NS-2026-06/DATA_NOTICE.md) | 官方数据（与原始题目包 2026-06.zip<sup>†</sup> 内容一致） |
| [tools/check_format.py](../../neuronspark/06/NS-2026-06/tools/check_format.py) / [tools/baseline.py](../../neuronspark/06/NS-2026-06/tools/baseline.py) | 官方格式检查与 baseline |
| [example_submission/results.csv](../../neuronspark/06/NS-2026-06/example_submission/results.csv) / [baseline_results.csv](../../neuronspark/06/NS-2026-06/baseline_results.csv) | 官方示例提交 / baseline 输出 |
| [robust_solution.py](../../neuronspark/06/NS-2026-06/robust_solution.py) | 中间版本：弱标签过滤 + 多模型融合 |
| [source_group_solution.py](../../neuronspark/06/NS-2026-06/source_group_solution.py) | 最终方案的原始脚本（产出 [results.csv](../../neuronspark/06/NS-2026-06/results.csv) 和 [NS-2026-06-answer.zip](../../neuronspark/06/NS-2026-06/NS-2026-06-answer.zip)） |
| [writeup_06_submit/](../../neuronspark/06/NS-2026-06/writeup_06_submit/) | **正式 writeup 提交目录**：[writeup.md](../../neuronspark/06/NS-2026-06/writeup_06_submit/writeup.md)、**[solver_06.py](../../neuronspark/06/NS-2026-06/writeup_06_submit/solver_06.py)（最终提交脚本）**、[run.sh](../../neuronspark/06/NS-2026-06/writeup_06_submit/run.sh)、requirements.txt、results.csv、NS-2026-06-answer.zip、analysis/、logs/、screenshots/ |
| [wp/](../../neuronspark/06/NS-2026-06/wp/) | 早期 writeup 包：[WRITEUP.md](../../neuronspark/06/NS-2026-06/wp/WRITEUP.md)、[ONLINE_SCORE.txt](../../neuronspark/06/NS-2026-06/wp/ONLINE_SCORE.txt)、[reproduce.sh](../../neuronspark/06/NS-2026-06/wp/reproduce.sh)、code/、submission/、analysis/、logs/（打包为 [06-wp.zip](../../neuronspark/06/NS-2026-06/06-wp.zip)） |
| writeup_06_submit/analysis/<sup>†</sup> | 噪声分析表：来源→标签映射、各来源组/标注组的不一致率、置信度统计、疑似错标样本 |
| [neuronspark2026/writeup_submit/06/](../../neuronspark/neuronspark2026/writeup_submit/06/) | 与 writeup_06_submit 相同的副本（solver_06.py 和 zip 的哈希一致） |
| [neuronspark2026/analyze_noise.py](../../neuronspark/neuronspark2026/analyze_noise.py) | 噪声模式分析脚本（neuronspark2026 根目录下的数据与本题数据相同） |
| [neuronspark2026-answer/NS-2026-06-answer.zip](../../neuronspark/neuronspark2026-answer/NS-2026-06-answer.zip) | **最终提交包**（只含 results.csv） |

**哈希核对**：最终提交 zip 的 sha256 为 `1278f498…caca96b`，与 writeup_06_submit/、writeup_submit/06/ 下的 zip 一致。NS-2026-06/ 根目录和 wp/submission/ 下的 zip（`99a4812b…`）里的 results.csv 完全相同（`15d66c81…`），只是 zip 元数据不同；solver_06.py 相比 source_group_solution.py 只改了 zip 内文件的时间戳和权限位，以便复现出平台上那个 zip 的哈希。

## 5. 如何运行

**环境**（[requirements.txt](../../neuronspark/06/NS-2026-06/writeup_06_submit/requirements.txt)，实测版本见 [dependency_versions.txt](../../neuronspark/06/NS-2026-06/writeup_06_submit/logs/dependency_versions.txt)）：Python 3.13.5，numpy 2.3.4，pandas 2.3.3，scipy 1.17.1，scikit-learn 1.8.0，xgboost 3.2.0（只有 robust_solution.py 用到，且可选）。只需 CPU。

脚本读取 `--data-dir` 下的 `train.csv`、`trusted_valid.csv`、`test.csv`、`label_map.json`。不加 `--predict-test` 时只跑 5 折 CV，不读测试集。脚本里没有硬编码的绝对路径；writeup 8.2 节写的 `/Users/tristonchen/Desktop/neuronspark2026/...` 是旧位置，现在应换成仓库内的路径。

```bash
# 方式 A：一键复现最终提交（DATA_DIR 默认是 run.sh 所在目录往上两级，即 neuronspark2026/，那里有数据和 tools/）
cd neuronspark/neuronspark2026/writeup_submit/06 && bash run.sh

# 方式 B：在 writeup_06_submit 下运行，需要显式指定数据目录（往上两级是 neuronspark/06/，那里没有数据）
cd neuronspark/06/NS-2026-06/writeup_06_submit && DATA_DIR=.. bash run.sh

# 方式 C：直接调用脚本 + 格式检查
cd neuronspark/06/NS-2026-06
python -u source_group_solution.py --n-splits 5 --predict-test     # 输出写到 data-dir 下的 results.csv / NS-2026-06-answer.zip
python tools/check_format.py results.csv --test-csv test.csv --label-map label_map.json

# 其他脚本
python tools/baseline.py --data-dir . --out baseline_results.csv   # 官方 baseline（--drop-ratio 默认 0.18）
python robust_solution.py [--predict-test] [--drop-ratio 0.18] [--gold-weight 6.0] [--n-splits 5]
cd ../../neuronspark2026 && python analyze_noise.py               # 用相对路径读取当前目录的 train.csv / trusted_valid.csv
```

注意：run.sh 会覆盖它所在目录的 `results.csv`、`NS-2026-06-answer.zip`、`logs/*.log` 和 `logs/hashes.txt`；[wp/reproduce.sh](../../neuronspark/06/NS-2026-06/wp/reproduce.sh)（`bash wp/reproduce.sh`）会覆盖 NS-2026-06/ 根目录的产物和 `wp/` 下的 code/、submission/、logs/。如果要保留存档，请先复制到别的地方再跑。

## 6. 参考

- 正式 writeup：[writeup_06_submit/writeup.md](../../neuronspark/06/NS-2026-06/writeup_06_submit/writeup.md)（副本：[writeup_submit/06/writeup.md](../../neuronspark/neuronspark2026/writeup_submit/06/writeup.md)）
- 早期 writeup：[wp/WRITEUP.md](../../neuronspark/06/NS-2026-06/wp/WRITEUP.md)，线上分数记录：[wp/ONLINE_SCORE.txt](../../neuronspark/06/NS-2026-06/wp/ONLINE_SCORE.txt)
- 疑似错标样本：suspected_noisy_samples.csv<sup>†</sup>
- AI 使用声明与 A1 合规风险说明：见 writeup.md 第 11 节

---

† 标注的文件为比赛原始数据或过程记录，按数据协议未包含在公开仓库中，请从官方渠道获取题目数据包。
