# NS-2026-06 标注净化师 Writeup

## 1. 方法概述

本题是 6 类英文讨论样本的匿名数值特征分类任务。训练集给出 `weak_label`，但弱标签存在非均匀噪声；`trusted_valid.csv` 给出人工校验标签，可以作为噪声分析、验证和校准依据；测试集只给出 96 维匿名特征。

最终提交采用的主方案不是直接学习 `weak_label`，而是利用可信验证集发现并验证 `source_group -> label` 的稳定映射，再训练一个从 96 维特征预测 `source_group` 的模型。测试阶段先得到每个测试样本属于各来源组的概率，再把来源组概率汇总成标签概率，并用可信验证集调出的类别倍率做最终校准。

最终线上分数为参赛者反馈的 1062/1200。本地 5 折 OOF 验证为：

| 指标 | 数值 |
|---|---:|
| Macro F1 | 0.89016 |
| 高优先级类别召回 | 0.89573 |
| score_like | 1080.29 |

完整日志见 `logs/local_cv_and_generation.log`。

## 2. 任务理解

赛题核心不是常规干净监督学习，而是含噪弱标签清洗。训练集的 `weak_label` 来自弱规则、旧模型或标注组融合，存在 15%-25% 左右的非均匀错标；可信验证集提供真实标签，是判断噪声模式和模型泛化的主要依据。

高分目标由 Macro F1、高优先级类别召回和格式分组成。因此方案需要同时保证类别均衡和 `computer_hardware`、`medical_health`、`space_science` 三个高优先级类别的召回。

## 3. 训练/验证策略

验证使用 `trusted_valid.csv` 做 5 折 StratifiedKFold。每一折中：

1. 将 4/5 trusted_valid 与全量 train 的 `source_group` 作为来源组监督。
2. 训练 `source_group` 分类器。
3. 在剩余 1/5 trusted_valid 上预测来源组概率。
4. 按可信验证集得到的 `source_group -> label` 映射，把来源概率汇总为标签概率。
5. 用固定类别倍率得到最终预测，计算 Macro F1、优先级召回和混淆矩阵。

可信验证集能代表隐藏测试目标的理由：

- 它与训练/测试使用同一 96 维匿名特征空间。
- 它覆盖所有 10 个 `source_group`，且样本量为 1031，足够用于稳定估计来源组与真实标签的关系。
- 其类别分布覆盖全部 6 类，并包含高优先级类别。
- 本地验证严格使用折外预测，避免在同一条可信样本上同时训练和评估。

同时也需要说明风险：测试集不提供 `source_group`，因此不能直接使用真实来源组作为测试特征。最终方案只使用 96 维特征预测来源组，再映射到标签。

## 4. 弱标签噪声分析

### 4.1 weak_label 与 source_group

可信验证集显示每个 `source_group` 与真实标签是一一稳定对应的：

| source_group | mapped_label |
|---|---|
| sg_00 | mobility_vehicle |
| sg_01 | system_software |
| sg_02 | space_science |
| sg_03 | computer_hardware |
| sg_04 | mobility_vehicle |
| sg_05 | medical_health |
| sg_06 | computer_hardware |
| sg_07 | system_software |
| sg_08 | sports_recreation |
| sg_09 | sports_recreation |

将训练集 `source_group` 映射出的标签与 `weak_label` 对比，估计训练弱标签不一致率为 21.8121%，共 1312/6015 行弱标签与来源映射标签不一致。这与题面给出的 15%-25% 噪声范围一致。

详细表格：

- `analysis/source_label_mapping.csv`
- `analysis/source_group_noise_summary.csv`
- `analysis/weak_label_counts.csv`

### 4.2 weak_confidence

`weak_confidence` 不能简单视为真实正确概率。疑似错标样本中仍存在较高 confidence 的错误弱标签，例如 `trn_02671_21860c29` 的 weak confidence 为 0.8927，但弱标签为 `sports_recreation`，来源映射判断为 `mobility_vehicle`。

各弱标签的 confidence 统计见：

- `analysis/weak_confidence_by_weak_label.csv`

### 4.3 annotator_group

`annotator_group` 只在训练集提供，测试集不可用，因此最终模型不把它作为测试特征。它用于噪声分析：按标注器组统计弱标签与来源映射标签的一致率，发现不同标注组的不一致率存在差异。统计见：

- `analysis/annotator_group_noise_summary.csv`

## 5. 关键改进

### 5.1 不直接相信 weak_label

初始尝试直接用 `weak_label` 训练线性模型，可信验证表现约在 1000 分上下。进一步将训练弱标签与可信验证中的来源标签结构对齐后，发现 `source_group` 是更稳定的监督信号。

最终模型没有把 `weak_label` 作为主训练目标，而是用 `source_group` 训练来源组分类器，然后经来源到标签映射得到最终标签。

### 5.2 来源组概率映射

模型输出 10 个来源组概率。由于多个来源组可能对应同一标签，例如 `sg_03` 和 `sg_06` 都对应 `computer_hardware`，所以将来源组概率按标签求和，得到 6 类标签概率。

### 5.3 类别校准

为兼顾 Macro F1 和高优先级召回，在 trusted_valid 的折外预测上调出固定类别倍率：

| label | multiplier |
|---|---:|
| computer_hardware | 1.30 |
| system_software | 0.95 |
| mobility_vehicle | 0.85 |
| sports_recreation | 0.95 |
| medical_health | 1.60 |
| space_science | 1.05 |

其中 `medical_health` 和 `computer_hardware` 的倍率更高，用于保护高优先级类别召回。

## 6. 模型选择

最终模型：

- `StandardScaler`
- `LogisticRegression(max_iter=5000, C=0.8, class_weight="balanced", solver="lbfgs", random_state=260606)`

选择原因：

- 96 维匿名数值特征对线性模型友好。
- 样本量较小，线性模型稳定且可复现。
- 来源组预测需要概率输出，逻辑回归比硬分类模型更方便做来源概率汇总。
- 本地 5 折中该方案优于直接弱标签训练和直接标签分类的多数组合。

## 7. 本地验证结果

5 折 OOF：

| fold | Macro F1 | Priority Recall | score_like |
|---:|---:|---:|---:|
| 1 | 0.91109 | 0.90588 | 1101.16 |
| 2 | 0.90119 | 0.90476 | 1092.02 |
| 3 | 0.88254 | 0.88095 | 1070.47 |
| 4 | 0.89821 | 0.92857 | 1094.10 |
| 5 | 0.85777 | 0.85882 | 1043.76 |
| OOF | 0.89016 | 0.89573 | 1080.29 |

每类 Precision/Recall/F1：

| label | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| computer_hardware | 0.8222 | 0.8894 | 0.8545 | 208 |
| system_software | 0.8877 | 0.7981 | 0.8405 | 208 |
| mobility_vehicle | 0.9124 | 0.8939 | 0.9031 | 198 |
| sports_recreation | 0.9469 | 0.9655 | 0.9561 | 203 |
| medical_health | 0.8696 | 0.9259 | 0.8969 | 108 |
| space_science | 0.9029 | 0.8774 | 0.8900 | 106 |

混淆矩阵，行是真实标签，列是预测标签，标签顺序为：

`computer_hardware, system_software, mobility_vehicle, sports_recreation, medical_health, space_science`

```text
[[185  16   4   1   2   0]
 [ 33 166   4   0   4   1]
 [  3   1 177   6   5   6]
 [  1   1   5 196   0   0]
 [  1   2   1   1 100   3]
 [  2   1   3   3   4  93]]
```

## 8. 疑似错标训练样本

处理方式：这些样本的 `weak_label` 与来源映射标签冲突，因此不把 weak label 作为最终训练目标；最终方案训练来源组分类器，并通过来源组映射与类别倍率得到预测。

| id | weak_label | 模型判断 | weak_confidence | source_group | annotator_group | 处理方式 |
|---|---|---|---:|---|---|---|
| trn_02671_21860c29 | sports_recreation | mobility_vehicle | 0.8927 | sg_00 | ag_01 | 不使用 weak label 作为目标 |
| trn_04245_9108d134 | sports_recreation | mobility_vehicle | 0.8615 | sg_04 | ag_00 | 不使用 weak label 作为目标 |
| trn_03032_9bd0ff8d | computer_hardware | system_software | 0.8538 | sg_01 | ag_01 | 不使用 weak label 作为目标 |
| trn_05679_7de754b4 | computer_hardware | mobility_vehicle | 0.8474 | sg_00 | ag_01 | 不使用 weak label 作为目标 |
| trn_05993_29518229 | space_science | sports_recreation | 0.8474 | sg_09 | ag_00 | 不使用 weak label 作为目标 |

完整列表见 `analysis/suspected_noisy_samples.csv`。

## 9. 复现命令

目录结构假设：

```text
NS-2026-06/
├── train.csv
├── trusted_valid.csv
├── test.csv
├── label_map.json
├── tools/check_format.py
├── source_group_solution.py
└── wp/
```

运行：

```bash
cd NS-2026-06
python -u source_group_solution.py --n-splits 5 --predict-test
python tools/check_format.py results.csv --test-csv test.csv --label-map label_map.json
```

或运行打包好的复现脚本：

```bash
bash wp/reproduce.sh
```

随机种子：

```text
SEED = 260606
StratifiedKFold shuffle=True random_state=260606
LogisticRegression random_state=260606
```

依赖版本见 `logs/dependency_versions.txt` 和 `requirements.txt`：

```text
python==3.13.5
numpy==2.3.4
pandas==2.3.3
scipy==1.17.1
scikit-learn==1.8.0
xgboost==3.2.0
```

最终文件 sha256：

```text
15d66c814e92af49f312a3f50b22ca6217d3bf28fb4a0ad4a789359b8b9996f7  results.csv
99a4812bfa9e469eed0431e7b6c29e2d0e23584373b32cf4b2357440c0a8c98c  NS-2026-06-answer.zip
```

格式检查：

```text
OK: submission format is valid.
```

## 10. 代码包说明

`wp/` 目录内容：

```text
wp/
├── WRITEUP.md
├── ONLINE_SCORE.txt
├── requirements.txt
├── reproduce.sh
├── analysis/
│   ├── annotator_group_noise_summary.csv
│   ├── noise_summary.csv
│   ├── source_group_noise_summary.csv
│   ├── source_label_mapping.csv
│   ├── suspected_noisy_samples.csv
│   ├── weak_confidence_by_weak_label.csv
│   └── weak_label_counts.csv
├── code/
│   └── source_group_solution.py
├── logs/
│   ├── dependency_versions.txt
│   ├── format_check.log
│   ├── hashes.txt
│   └── local_cv_and_generation.log
├── screenshots/
│   └── README.md
└── submission/
    ├── NS-2026-06-answer.zip
    └── results.csv
```

## 11. 证据截图

当前包内提供文本证据日志：

- 本地验证与生成日志：`logs/local_cv_and_generation.log`
- 格式检查日志：`logs/format_check.log`
- sha256：`logs/hashes.txt`

正式提交 writeup 前建议补充平台截图到 `screenshots/`：

- 提交记录截图，显示线上分数 1062/1200。
- 题目页/提交要求截图。
- 本地复现终端截图。

## 12. AI 使用声明

本次实验和提交生成使用了 AI Agent：Codex，模型为 GPT-5 系列 coding agent。AI 的用途包括：

- 阅读题面与本地数据字段。
- 解压数据包。
- 编写、运行并调试 `source_group_solution.py`。
- 读取 `train.csv`、`trusted_valid.csv`、`test.csv`。
- 生成最终 `results.csv` 与 `NS-2026-06-answer.zip`。
- 整理 writeup 和复现实验材料。

输入边界：AI 接触了公开训练集、可信验证集、测试输入特征、标签映射、官方检查脚本和本地运行日志；AI 未接触隐藏测试标签。

重要合规说明：题面 A1 规则写明，如果 AI 接触测试输入并生成最终答案，或 Agent 托管实验并生成提交，原则上视为违反等级规则。本次提交版本确实由 AI Agent 读取 `test.csv` 并生成最终提交文件，因此按题面 A1 规则存在明确合规风险，不应声明为完全 A1 合规。若作为获奖候选材料提交，应如实保留本声明，并由组委会按规则判断资格。
