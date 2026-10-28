# NS-2026-06 标注净化师：噪声标签清洗与鲁棒分类 Writeup

**队伍名称：** claude-opus-4.8  
**最终提交者：** chenshch23  
**最终提交时间：** 2026/5/25 14:01:51  
**线上得分：** 1061.726769082733 / 1200（页面显示 1062 / 1200）  
**最终提交文件：** `NS-2026-06-answer.zip`  
**最终提交压缩包 SHA256：** `1278f4983f9c1e47e3d64a1b208203b1dd8a6cb1737e28e58d24940a1caca96b`

---

## 1. 方法概述

本题是一个带系统性弱标签噪声的 6 类匿名特征分类任务。训练集只给出 `weak_label`，且存在约 15%-25% 非均匀噪声；`trusted_valid.csv` 给出可信人工标签；测试集只提供 96 维匿名数值特征。

最终方案的核心是：**不直接把 `weak_label` 当作真实标签训练，而是利用可信验证集发现 `source_group -> label` 的稳定映射，再从 96 维特征预测来源组，最后将来源组概率汇总为类别概率并做类别校准。**

具体流程：

1. 用 `trusted_valid.csv` 统计每个 `source_group` 对应的真实标签。
2. 使用 `train.csv + trusted_valid.csv` 中可见的来源组作为监督，训练来源组分类器。
3. 对测试样本预测 10 个来源组概率。
4. 将来源组概率按 `source_group -> label` 映射汇总成 6 类标签概率。
5. 使用可信验证集折外预测调出的类别倍率进行校准，提升 Macro F1 与高优先级类别召回。

最终模型为：

```text
StandardScaler
+ LogisticRegression(C=0.8, class_weight="balanced", solver="lbfgs", max_iter=5000)
```

随机种子固定为 `260606`。

---

## 2. 任务理解

题目强调“标注净化”，关键不是堆叠复杂模型，而是识别弱标签的系统性错误来源。训练数据包含：

- `weak_label`：含噪弱标签。
- `weak_confidence`：弱标注器置信度，但不等价于真实正确率。
- `source_group`：匿名来源域，训练集与可信验证集提供。
- `annotator_group`：匿名标注器组，仅训练集提供。

由于测试集没有 `source_group` 和 `annotator_group`，最终提交阶段不能直接使用真实来源组作为测试特征。因此本方案只在训练阶段利用来源组监督，测试阶段使用 96 维特征推断来源组。

评分由 Macro F1、高优先级类别召回和格式分组成。高优先级类别为：

```text
computer_hardware, medical_health, space_science
```

因此校准阶段额外保护这三类召回。

---

## 3. 训练/验证策略

### 3.1 可信验证集的使用

使用 `trusted_valid.csv` 做 5 折 StratifiedKFold。每折中：

1. 将 4/5 trusted_valid 与全量 train 的 `source_group` 用作来源组训练监督。
2. 在剩余 1/5 trusted_valid 上预测来源组概率。
3. 依据该折训练部分得到的来源组到标签映射，把来源概率汇总为标签概率。
4. 应用固定类别倍率，计算折外 Macro F1、优先级召回和混淆矩阵。

这样每个可信验证样本的预测均来自折外模型，避免直接在同一条可信样本上训练和评估。

### 3.2 为什么 trusted_valid 能代表隐藏测试目标

- trusted_valid 与 train/test 共享同一 96 维匿名特征空间。
- trusted_valid 覆盖所有 10 个 `source_group`。
- trusted_valid 覆盖全部 6 个真实类别，其中高优先级类别也有足够样本。
- 题面说明可信验证集用于本地验证、校准和噪声分析，隐藏测试目标与其同属公开数据生成流程。

### 3.3 验证风险控制

测试集不提供 `source_group`，所以本地验证时没有把真实 `source_group` 作为预测标签特征直接输入最终分类器，而是模拟测试条件：从匿名数值特征预测来源组，再映射为标签。

---

## 4. 弱标签噪声分析

### 4.1 source_group 与真实标签

可信验证集显示每个来源组对应唯一真实标签：

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

用该映射检查训练集弱标签，发现：

| 项目 | 数值 |
|---|---:|
| 训练样本数 | 6015 |
| weak_label 与来源映射一致 | 4703 |
| weak_label 与来源映射不一致 | 1312 |
| 估计弱标签噪声率 | 21.8121% |

这与题面 15%-25% 的噪声范围吻合。

详细统计文件：

- `analysis/source_label_mapping.csv`
- `analysis/source_group_noise_summary.csv`
- `analysis/noise_summary.csv`

### 4.2 weak_label

直接学习 `weak_label` 会被系统性噪声牵引。本题弱标签错标不是均匀随机错误，而是与来源组、标注组和类别混淆有关。因此最终模型没有把 `weak_label` 作为主监督目标，而是将其用于噪声分析和疑似错标样本识别。

弱标签类别计数见：

- `analysis/weak_label_counts.csv`

### 4.3 weak_confidence

`weak_confidence` 有参考价值，但不能直接视为标签正确概率。疑似错标样本中存在高置信度弱标签错误。例如：

- `trn_02671_21860c29`：weak confidence 为 0.8927，weak_label 为 `sports_recreation`，来源映射判断为 `mobility_vehicle`。
- `trn_03032_9bd0ff8d`：weak confidence 为 0.8538，weak_label 为 `computer_hardware`，来源映射判断为 `system_software`。

置信度统计见：

- `analysis/weak_confidence_by_weak_label.csv`

### 4.4 annotator_group

`annotator_group` 仅训练集提供，测试集不可用，因此不作为最终测试特征。我们按标注组统计弱标签与来源映射标签的不一致率，用于说明不同标注器组噪声程度不同。

统计见：

- `analysis/annotator_group_noise_summary.csv`

---

## 5. 样本过滤、重加权与标签校准

### 5.1 样本处理

最终方案没有将训练样本按 `weak_label` 训练分类器，因此没有依赖传统的弱标签过滤阈值。处理策略为：

- 若 `weak_label` 与来源映射标签一致，视作弱标签较可信，但仍不作为主训练目标。
- 若 `weak_label` 与来源映射标签冲突，标记为疑似错标样本。
- 模型训练目标为 `source_group`，而非 `weak_label`。

这样避免了高置信度但错误的弱标签污染最终目标。

### 5.2 模型选择

尝试过直接弱标签训练、弱标签过滤后训练、线性 SVM、树模型和来源组预测。最终来源组逻辑回归方案本地折外验证最稳定，且测试提交得分最高。

来源组逻辑回归优势：

- 对 96 维匿名连续特征稳定。
- 输出概率，便于将 10 个来源组概率合并为 6 类标签概率。
- 计算快、可复现，不需要 GPU。

### 5.3 类别校准

在可信验证集 OOF 概率上调出固定类别倍率：

| label | multiplier |
|---|---:|
| computer_hardware | 1.30 |
| system_software | 0.95 |
| mobility_vehicle | 0.85 |
| sports_recreation | 0.95 |
| medical_health | 1.60 |
| space_science | 1.05 |

其中 `computer_hardware`、`medical_health`、`space_science` 是高优先级类别，倍率略提高以保护召回。

---

## 6. 本地验证结果

5 折 OOF 日志见 `logs/local_cv_and_generation.log`。

| fold | Macro F1 | Priority Recall | score_like |
|---:|---:|---:|---:|
| 1 | 0.91109 | 0.90588 | 1101.16 |
| 2 | 0.90119 | 0.90476 | 1092.02 |
| 3 | 0.88254 | 0.88095 | 1070.47 |
| 4 | 0.89821 | 0.92857 | 1094.10 |
| 5 | 0.85777 | 0.85882 | 1043.76 |
| OOF | 0.89016 | 0.89573 | 1080.29 |

每类指标：

| label | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| computer_hardware | 0.8222 | 0.8894 | 0.8545 | 208 |
| system_software | 0.8877 | 0.7981 | 0.8405 | 208 |
| mobility_vehicle | 0.9124 | 0.8939 | 0.9031 | 198 |
| sports_recreation | 0.9469 | 0.9655 | 0.9561 | 203 |
| medical_health | 0.8696 | 0.9259 | 0.8969 | 108 |
| space_science | 0.9029 | 0.8774 | 0.8900 | 106 |

混淆矩阵，行是真实标签，列是预测标签，标签顺序为：

```text
computer_hardware, system_software, mobility_vehicle,
sports_recreation, medical_health, space_science
```

```text
[[185  16   4   1   2   0]
 [ 33 166   4   0   4   1]
 [  3   1 177   6   5   6]
 [  1   1   5 196   0   0]
 [  1   2   1   1 100   3]
 [  2   1   3   3   4  93]]
```

---

## 7. 疑似错标训练样本

以下样本的 `weak_label` 与来源映射判断冲突，作为疑似错标样本处理。最终训练不使用这些 weak label 作为目标。

| id | weak_label | 模型判断 | weak_confidence | source_group | annotator_group | 处理方式 |
|---|---|---|---:|---|---|---|
| trn_02671_21860c29 | sports_recreation | mobility_vehicle | 0.8927 | sg_00 | ag_01 | 不使用 weak label 作为目标 |
| trn_04245_9108d134 | sports_recreation | mobility_vehicle | 0.8615 | sg_04 | ag_00 | 不使用 weak label 作为目标 |
| trn_03032_9bd0ff8d | computer_hardware | system_software | 0.8538 | sg_01 | ag_01 | 不使用 weak label 作为目标 |
| trn_05679_7de754b4 | computer_hardware | mobility_vehicle | 0.8474 | sg_00 | ag_01 | 不使用 weak label 作为目标 |
| trn_05993_29518229 | space_science | sports_recreation | 0.8474 | sg_09 | ag_00 | 不使用 weak label 作为目标 |

完整表见 `analysis/suspected_noisy_samples.csv`。

---

## 8. 复现说明

### 8.1 环境

依赖见 `requirements.txt` 和 `logs/dependency_versions.txt`。

```text
python==3.13.5
numpy==2.3.4
pandas==2.3.3
scipy==1.17.1
scikit-learn==1.8.0
xgboost==3.2.0
```

本题数据规模较小，CPU 即可复现，不需要 GPU。

### 8.2 运行命令

假设目录结构为：

```text
neuronspark2026/
├── train.csv
├── trusted_valid.csv
├── test.csv
├── label_map.json
├── tools/check_format.py
└── writeup_submit/06/
    ├── solver_06.py
    ├── run.sh
    └── writeup.md
```

执行：

```bash
cd /Users/tristonchen/Desktop/neuronspark2026/writeup_submit/06
bash run.sh
```

`run.sh` 会完成：

1. 5 折本地 OOF 验证。
2. 读取测试特征生成 `results.csv`。
3. 打包 `NS-2026-06-answer.zip`。
4. 运行官方格式检查。
5. 输出 SHA256 到 `logs/hashes.txt`。

### 8.3 随机种子

```text
SEED = 260606
StratifiedKFold(n_splits=5, shuffle=True, random_state=260606)
LogisticRegression(random_state=260606)
```

### 8.4 最终文件 SHA256

平台上传的最终提交压缩包：

```text
1278f4983f9c1e47e3d64a1b208203b1dd8a6cb1737e28e58d24940a1caca96b  NS-2026-06-answer.zip
```

压缩包内 `results.csv`：

```text
15d66c814e92af49f312a3f50b22ca6217d3bf28fb4a0ad4a789359b8b9996f7  results.csv
```

格式检查：

```text
OK: submission format is valid.
```

---

## 9. 代码包说明

本目录内容：

```text
06/
├── writeup.md
├── solver_06.py
├── run.sh
├── requirements.txt
├── NS-2026-06-answer.zip
├── results.csv
├── analysis/
│   ├── annotator_group_noise_summary.csv
│   ├── noise_summary.csv
│   ├── source_group_noise_summary.csv
│   ├── source_label_mapping.csv
│   ├── suspected_noisy_samples.csv
│   ├── weak_confidence_by_weak_label.csv
│   └── weak_label_counts.csv
├── logs/
│   ├── dependency_versions.txt
│   ├── format_check.log
│   ├── hashes.txt
│   └── local_cv_and_generation.log
└── screenshots/
    └── README.md
```

`solver_06.py` 是完整可复现脚本。`run.sh` 是一键复现入口。

---

## 10. 证据截图

建议补充到 `screenshots/`：

- 平台提交记录截图，显示 2026/5/25 14:01:51 的 `1061.726769082733` 分提交。
- 题目页截图，显示 NS-2026-06、提交格式和 AI 协助等级。
- 本地运行 `bash run.sh` 的终端截图。

当前已提供文本证据：

- `logs/local_cv_and_generation.log`
- `logs/format_check.log`
- `logs/hashes.txt`

---

## 11. AI 使用声明

本次方案开发过程中使用了 AI 辅助工具，包括 Claude/Codex 类 coding agent，用途包括题面理解、代码编写、实验调试、日志整理和 writeup 草拟。

AI/Agent 接触的数据边界：

- 接触了公开训练集 `train.csv`。
- 接触了可信验证集 `trusted_valid.csv`。
- 接触了测试输入特征 `test.csv`。
- 未接触隐藏测试标签。

最终提交文件由本地 Python 脚本根据公开数据和测试输入特征生成。需要特别说明：题面 A1 规则写明，如果 AI 接触测试输入并生成最终答案，或 Agent 托管实验并生成提交，原则上视为违反等级规则。本次开发流程中 AI Agent 曾参与测试推理与提交文件生成，因此该提交存在 A1 合规风险。若作为获奖候选材料提交，应如实披露本声明，并接受组委会最终判定。
