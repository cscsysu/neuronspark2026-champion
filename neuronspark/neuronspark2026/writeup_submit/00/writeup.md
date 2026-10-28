# NS-2026-00 星图识别员：校园票据分类 Writeup

## 队伍信息

- 队伍名称：claude-opus-4.8
- 提交者：chenshch23
- 最终得分：500 / 500
- 提交时间：2026/5/23 11:36:30

## 1. 方法概述

本方案采用 **EfficientNet-B2**（ImageNet 预训练）进行迁移学习，替换分类头后微调整个网络，完成 10 类校园文档图片分类任务。方案核心思路：利用预训练视觉模型已学习到的通用图像特征（纹理、边缘、色彩分布、版面结构），通过小学习率微调快速适配文档分类任务。

## 2. 数据分析

### 2.1 数据概况

| 项目 | 数值 |
|------|------|
| 训练集 | 3600 张图片 |
| 测试集 | 1200 张图片 |
| 类别数 | 10 类 |
| 类别分布 | 完全均衡，每类 360 张 |
| 图片尺寸 | 约 941×941 RGB |
| 图片格式 | JPEG |

### 2.2 类别分布

训练集各类别样本数完全均衡（各 360 张），无需额外的类别均衡采样策略。

```
360 schedule
360 receipt
360 poster
360 notice
360 meeting_minutes
360 lab_note
360 invoice
360 handwritten_note
360 grade_report
360 form
```

## 3. 模型与训练

### 3.1 模型架构

- **基础模型**：`timm.efficientnet_b2.ra_in1k`（EfficientNet-B2, ImageNet-1K 预训练）
- **分类头**：替换原始 1000 类线性层为 10 类线性层
- **参数量**：约 9.1M（EfficientNet-B2 骨干）

### 3.2 框架与环境

| 项目 | 详情 |
|------|------|
| 深度学习框架 | PyTorch 2.12.0 |
| 模型库 | timm |
| 计算设备 | Apple MPS (Metal Performance Shaders) |
| Python 版本 | 3.13.7 |
| 操作系统 | macOS (Apple Silicon ARM64) |

### 3.3 训练超参数

| 超参数 | 值 |
|--------|-----|
| 输入尺寸 | 260 × 260 |
| Batch Size | 32 |
| 优化器 | AdamW (weight_decay=1e-4) |
| 骨干网络学习率 | 1e-5 |
| 分类头学习率 | 1e-4 |
| 学习率调度 | CosineAnnealingLR (T_max=15) |
| 损失函数 | CrossEntropyLoss |
| 最大训练轮数 | 15 epochs |
| 训练/验证划分 | 85% / 15%，分层采样 (random_state=42) |

### 3.4 数据增强策略

训练时使用以下数据增强：

| 增强方法 | 参数 | 说明 |
|----------|------|------|
| Resize | (292, 292) | 先放大到略大于目标尺寸 |
| RandomCrop | 260 | 随机裁剪到模型输入尺寸 |
| RandomHorizontalFlip | p=0.5 | 随机水平翻转 |
| RandomRotation | ±10° | 小角度随机旋转 |
| ColorJitter | brightness=0.2, contrast=0.2, saturation=0.1 | 颜色抖动 |
| Normalize | mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225] | ImageNet标准化 |

**关于文字可读性的说明**：以上增强策略均不会破坏图片中文字的可读性。RandomCrop 裁剪幅度很小（从 292 裁至 260，仅裁去约 11% 的边缘像素）；RandomRotation 限制在 ±10° 以内，文字仍可辨认；ColorJitter 仅做轻微亮度/对比度/饱和度调整，不影响文字内容。这些增强旨在模拟实际拍摄中的轻微角度偏差和光照变化。

验证集和测试集仅使用 Resize + Normalize，无随机增强。

## 4. 训练结果

| 指标 | 值 |
|------|-----|
| Epoch 1 验证集准确率 | **99.63%** (538/540) |
| 保存的最佳模型 | best_model.pth (31.3 MB) |

模型在第 1 个 epoch 后即达到 99.63% 的验证集准确率，表明预训练 EfficientNet-B2 的视觉特征对文档分类任务具有极强的迁移能力，10 类文档在视觉外观上差异显著。

## 5. 推理

- 对 1200 张测试图片进行推理
- 使用验证集上表现最好的模型权重
- 推理时仅使用 Resize(260×260) + Normalize，无测试时增强(TTA)

### 预测分布

推理结果各类分布合理：

```
label         count
grade_report    120
receipt         120
poster          120
notice          120
meeting_minutes 120
lab_note        120
invoice         120
handwritten_note 120
schedule        120
form            120
```

## 6. 未使用的技术

- **未使用 OCR**：纯视觉特征已足够区分 10 类文档
- **未使用测试时增强 (TTA)**：单模型单次推理即可
- **未使用模型集成**：单模型即达到满分
- **未使用类别均衡采样**：数据本身完全均衡

## 7. 关键代码

### 训练脚本 (train.py)

```python
"""
NS-2026-00 校园票据分类 - 训练脚本
使用 timm EfficientNet-B2 预训练模型微调 (从本地权重加载)
"""
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm
from safetensors.torch import load_file
from sklearn.model_selection import train_test_split

# Config
DATA_DIR = "/Users/tristonchen/Desktop/neuronspark2026/2026-00-data"
MODEL_WEIGHTS = "/tmp/efficientnet_b2.safetensors"
BATCH_SIZE = 32
NUM_EPOCHS = 15
LR = 1e-4
IMG_SIZE = 260  # EfficientNet-B2 default input size
NUM_CLASSES = 10
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Label mapping
LABELS = [
    "invoice", "receipt", "schedule", "poster", "lab_note",
    "notice", "handwritten_note", "form", "meeting_minutes", "grade_report"
]
LABEL2IDX = {l: i for i, l in enumerate(LABELS)}
IDX2LABEL = {i: l for i, l in enumerate(LABELS)}

# Dataset
class DocImageDataset(Dataset):
    def __init__(self, df, data_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.data_dir, row["image"])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = LABEL2IDX[row["label"]]
        return img, label

# Transforms
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Load data
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
train_split, val_split = train_test_split(
    train_df, test_size=0.15, random_state=42, stratify=train_df["label"]
)

train_dataset = DocImageDataset(train_split, DATA_DIR, train_transform)
val_dataset = DocImageDataset(val_split, DATA_DIR, val_transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Model - load timm EfficientNet-B2 with local weights
model = timm.create_model("efficientnet_b2", pretrained=False, num_classes=1000)
state_dict = load_file(MODEL_WEIGHTS)
model.load_state_dict(state_dict, strict=False)
model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
model = model.to(DEVICE)

# Optimizer - different LR for backbone vs head
backbone_params = [p for n, p in model.named_parameters() if "classifier" not in n]
head_params = [p for n, p in model.named_parameters() if "classifier" in n]
optimizer = torch.optim.AdamW([
    {"params": backbone_params, "lr": LR * 0.1},
    {"params": head_params, "lr": LR},
], weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
criterion = nn.CrossEntropyLoss()

# Training loop
best_val_acc = 0.0
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    # Validation
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            _, preds = outputs.max(1)
            val_correct += preds.eq(labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total
    scheduler.step()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), os.path.join(DATA_DIR, "best_model.pth"))
```

### 推理脚本 (predict.py)

```python
"""
NS-2026-00 - 推理脚本
"""
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm

DATA_DIR = "/Users/tristonchen/Desktop/neuronspark2026/2026-00-data"
MODEL_PATH = os.path.join(DATA_DIR, "best_model.pth")
IMG_SIZE = 260
NUM_CLASSES = 10
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

LABELS = [
    "invoice", "receipt", "schedule", "poster", "lab_note",
    "notice", "handwritten_note", "form", "meeting_minutes", "grade_report"
]
IDX2LABEL = {i: l for i, l in enumerate(LABELS)}

class TestDataset(Dataset):
    def __init__(self, df, data_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.data_dir, row["image"])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, row["id"]

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
test_dataset = TestDataset(test_df, DATA_DIR, test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

model = timm.create_model("efficientnet_b2", pretrained=False, num_classes=NUM_CLASSES)
state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
model.load_state_dict(state_dict)
model = model.to(DEVICE)
model.eval()

all_ids, all_preds = [], []
with torch.no_grad():
    for imgs, ids in test_loader:
        imgs = imgs.to(DEVICE)
        outputs = model(imgs)
        _, preds = outputs.max(1)
        all_ids.extend(ids)
        all_preds.extend([IDX2LABEL[p.item()] for p in preds])

results_df = pd.DataFrame({"id": all_ids, "label": all_preds})
results_df.to_csv(os.path.join(DATA_DIR, "results.csv"), index=False)
```

## 8. 复现步骤

1. 解压数据集 `2026-00.zip` 到 `2026-00-data/` 目录
2. 安装依赖：`pip install torch torchvision timm pillow pandas scikit-learn safetensors`
3. 下载预训练权重：从 `hf-mirror.com` 下载 `timm/efficientnet_b2.ra_in1k` 的 `model.safetensors` 到 `/tmp/efficientnet_b2.safetensors`
4. 运行训练：`python3 train.py`
5. 运行推理：`python3 predict.py`
6. 使用 `tools/check_format.py` 验证格式
7. 打包提交：`zip NS-2026-00-answer.zip results.csv`

## 9. AI 协助声明

本题 AI 协助等级为 **A1**。

在本题解答过程中使用了 AI 辅助工具（Claude）进行：
- 代码编写（训练脚本、推理脚本）
- 方案设计（模型选择、超参数配置）
- 环境问题排查（网络下载、MPS 兼容性）

所有代码均在本地运行和验证，模型训练在本地 Apple MPS 设备上完成。

## 10. 提交文件 SHA256

```
NS-2026-00-answer.zip SHA256: 97f0ecd3ed05197e7772c1f66cfee631643bf52a3ff0d6a62df5fa51bcd9e935
```

## 11. 总结

本题作为 NeuronSpark 2026 的入门题，10 类校园文档图片在视觉外观上具有明显差异（发票有表格结构、海报有丰富色彩、手写便签有手写笔迹等）。使用 ImageNet 预训练的 EfficientNet-B2 进行迁移学习，仅需 1 个 epoch 即可达到 99.63% 的验证准确率，最终测试集得分满分 500 分。这说明预训练视觉特征对文档分类任务有极强的迁移性，无需 OCR 或复杂的多模态方案即可取得完美成绩。
