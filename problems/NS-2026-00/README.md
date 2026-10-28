# NS-2026-00 星图识别员：校园票据分类

> 难度：Easy ｜ 总分：500 ｜ 题型：图像分类（10 类校园文档图片）｜ AI 协助等级：A1 ｜ 比赛时间：2026/5/23 08:00 – 2026/5/28 08:00

## 1. 题目介绍

**任务**：对校园事务系统中的文档图片做 10 类分类（NeuronSpark 2026 的入门题）。题面要求模型在拍摄设备、截图尺寸、轻微模糊、旋转、压缩、遮挡等变化下保持稳定；OCR、版面特征、图文融合、TTA 等只是可选的提分手段，不强制。

| label | 中文名 | label | 中文名 |
|---|---|---|---|
| invoice | 报销发票 | notice | 校园通知 |
| receipt | 消费小票 | handwritten_note | 手写便签 |
| schedule | 课程或日程表 | form | 申请或登记表 |
| poster | 社团活动海报 | meeting_minutes | 会议纪要 |
| lab_note | 实验记录 | grade_report | 成绩或学业报告 |

**数据**（来自 `2026-00.zip`）：

| 文件 | 内容 |
|---|---|
| `train.csv` | `id,image,label`，3600 张，10 类各 360 张 |
| `test.csv` | `id,image`，1200 张（无标签） |
| `images/train/`、`images/test/` | JPEG 图片，约 941×941 RGB |
| `label_map.json` | 10 个合法标签与中文名 |
| `example_submission/results.csv` | 示例提交 |
| `tools/check_format.py` | 格式检查工具 |

**提交格式**：`results.csv`，表头 `id,label`，覆盖 test.csv 全部 id 且每个只出现一次，label 必须是 10 个合法标签之一；压缩为 `NS-2026-00-answer.zip`，包内只能有 `results.csv`（不能带文件夹）。

**评分**：按 id 与隐藏标签对齐算 Accuracy，再按锚点线性插值换算成分数：

| Accuracy | 0.00 | 0.70 | 0.80 | 0.88 | 0.92 | ≥0.95 |
|---|---|---|---|---|---|---|
| 得分 | 0 | 100 | 200 | 300 | 400 | 500 |

（题面来源：比赛对话记录<sup>†</sup>里贴出的平台题目页，以及 [writeup](../../neuronspark/neuronspark2026/writeup_submit/00/writeup.md)。）

## 2. 我们的做法

**核心思路**：用 ImageNet 预训练的 **EfficientNet-B2**（`timm` 的 `efficientnet_b2.ra_in1k`）做迁移学习，把分类头换成 10 类线性层后微调整个网络。没有用 OCR、TTA 或模型集成。

- **输入**：260×260，ImageNet 均值/方差归一化。
- **数据增强（训练）**：Resize(292) → RandomCrop(260)、水平翻转、±10° 旋转、ColorJitter(0.2/0.2/0.1)。writeup 里说明了这些增强都不影响文字可读性（题目要求写这一点）。验证和测试只做 Resize + Normalize。
- **优化**：AdamW(wd=1e-4)，骨干和分类头分开设学习率（骨干 1e-5 / 分类头 1e-4），CosineAnnealingLR(T_max=15)，交叉熵损失，batch 32。
- **验证**：85%/15% 分层划分（`random_state=42`），验证集准确率最高时保存 `best_model.pth`。
- **设备**：Apple MPS。

**迭代过程**（来自对话记录）：
1. 一开始想用 `timm.create_model(..., pretrained=True)` 在线拉权重，但 HuggingFace / PyTorch 下载太慢，中途也试过换成 torchvision 的 EfficientNet-B0。
2. 最后从 hf-mirror 把 `efficientnet_b2.ra_in1k` 的 `model.safetensors` 下载到 `/tmp/efficientnet_b2.safetensors`，用 `safetensors` 在本地加载，改回 B2。
3. 跑完第 1 个 epoch 验证集准确率就到了 **99.63%（538/540）**。脚本随后提前退出（推测是 MPS 出错），最终提交用的就是第 1 个 epoch 保存的权重，**没有跑满 15 个 epoch**。
4. 用 `predict.py` 推理，过了 `check_format.py` 后打包提交，一次就拿到满分。

## 3. 结果

| 指标 | 数值 | 来源 |
|---|---|---|
| 验证集准确率（第 1 个 epoch） | 99.63%（538/540） | writeup / 对话记录 |
| 线上得分 | **500 / 500** | writeup（提交时间 2026/5/23 11:36:30）；对话记录中用户确认“这题满分” |
| 提交包 SHA256 | `97f0ecd3ed05197e7772c1f66cfee631643bf52a3ff0d6a62df5fa51bcd9e935` | writeup，和仓库里的 zip 一致 |

> 注意：writeup 第 5 节说预测分布是“每类 120”，但实际提交的 `results.csv` 分布是 notice 127 / lab_note 125 / invoice 122 / poster、handwritten_note、schedule、receipt 各 120 / grade_report 119 / form 117 / meeting_minutes 110。以文件为准。

## 4. 目录与文件说明

本题文件都在 `neuronspark/neuronspark2026/` 下（下面的链接都是相对本 README 的路径）：

| 路径 | 作用 |
|---|---|
| [train.py](../../neuronspark/neuronspark2026/train.py) | **训练脚本**：EfficientNet-B2 微调，输出 `best_model.pth` |
| [predict.py](../../neuronspark/neuronspark2026/predict.py) | **最终推理脚本**：加载 `best_model.pth`，生成 `results.csv` |
| 2026-00.zip<sup>†</sup> | 原始题目数据包（含 4800 张图片） |
| [2026-00-data/](../../neuronspark/neuronspark2026/2026-00-data/) | 解压后的工作目录（**仓库里没有 `images/`**，要从 zip 里重新解压） |
| 2026-00-data/best_model.pth<sup>†</sup> | 训练好的权重（31.3 MB，第 1 个 epoch） |
| [2026-00-data/results.csv](../../neuronspark/neuronspark2026/2026-00-data/results.csv) | 最终预测结果，和提交包里的内容完全相同 |
| [2026-00-data/tools/check_format.py](../../neuronspark/neuronspark2026/2026-00-data/tools/check_format.py) | 官方格式检查工具 |
| [2026-00-data/label_map.json](../../neuronspark/neuronspark2026/2026-00-data/label_map.json) | 合法标签表 |
| [neuronspark2026-answer/NS-2026-00-answer.zip](../../neuronspark/neuronspark2026-answer/NS-2026-00-answer.zip) | **最终提交包**（只有 `results.csv`） |
| [neuronspark2026/NS-2026-00-answer.zip](../../neuronspark/neuronspark2026/NS-2026-00-answer.zip)、[neuronspark2026/neuronspark2026-answer/NS-2026-00-answer.zip](../../neuronspark/neuronspark2026/neuronspark2026-answer/NS-2026-00-answer.zip) | 同一提交包的副本（SHA256 相同） |
| [writeup_submit/00/writeup.md](../../neuronspark/neuronspark2026/writeup_submit/00/writeup.md) | 提交的 Writeup |
| conversation-2026-05-27-00.txt<sup>†</sup> | 解题对话记录（含完整题面和过程） |

> `neuronspark2026/` 根目录下其他脚本和本题无关，例如 `classify.py` / `run_pro.py` 属于 NS-2026-03（Prompt Injection 分类）；根目录的 `results.csv`、`label_map.json` 也不是本题的。

## 5. 如何运行

**依赖**（来自 writeup）：Python 3.13、PyTorch（writeup 写的是 2.12.0）、torchvision、timm、pillow、pandas、scikit-learn、safetensors。

```bash
pip install torch torchvision timm pillow pandas scikit-learn safetensors
```

**1）准备数据**：把 `2026-00.zip` 解压到 `2026-00-data/`，主要是为了补上 `images/`：

```bash
cd "neuronspark/neuronspark2026"
unzip -o 2026-00.zip -d 2026-00-data
```

**2）准备预训练权重**：从 hf-mirror 下载 `timm/efficientnet_b2.ra_in1k` 的 `model.safetensors`，保存到 `/tmp/efficientnet_b2.safetensors`（`train.py` 里写死了这个路径，变量是 `MODEL_WEIGHTS`）。

**3）修改硬编码路径**：`train.py` 和 `predict.py` 里都写死了

```python
DATA_DIR = "/Users/tristonchen/Desktop/neuronspark2026/2026-00-data"
```

这是当时的原始路径，在本仓库里不存在。需要改成实际路径，比如 `<仓库根>/neuronspark/neuronspark2026/2026-00-data`。`best_model.pth` 和 `results.csv` 都会读写到 `DATA_DIR` 下面。

**4）训练 / 推理**（两个脚本都没有命令行参数）：

```bash
python3 train.py     # 输出 DATA_DIR/best_model.pth；有 MPS 用 MPS，否则用 CPU
python3 predict.py   # 输出 DATA_DIR/results.csv
```

如果只想复现提交结果，可以跳过训练，直接用仓库里现成的 `2026-00-data/best_model.pth` 跑 `predict.py`。

**5）格式检查与打包**（`check_format.py` 默认在当前目录找 `test.csv` / `label_map.json`，所以要在数据目录下执行）：

```bash
cd 2026-00-data
python3 tools/check_format.py results.csv   # 期望输出：OK: submission format is valid.
zip NS-2026-00-answer.zip results.csv       # 包内只能有 results.csv
```

## 6. 参考

- [Writeup：NS-2026-00 星图识别员：校园票据分类](../../neuronspark/neuronspark2026/writeup_submit/00/writeup.md)
- 解题对话记录（含完整题面）<sup>†</sup>
- 题面里给出的数据集地址：https://modelscope.cn/datasets/SteamedFresh/NS-2026-A-data

---

† 标注的文件为比赛原始数据或过程记录，按数据协议未包含在公开仓库中，请从官方渠道获取题目数据包。
