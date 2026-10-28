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

print(f"Using device: {DEVICE}")

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
print(f"Train: {len(train_split)}, Val: {len(val_split)}")

train_dataset = DocImageDataset(train_split, DATA_DIR, train_transform)
val_dataset = DocImageDataset(val_split, DATA_DIR, val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Model - load timm EfficientNet-B2 with local weights
print("Loading model...")
model = timm.create_model("efficientnet_b2", pretrained=False, num_classes=1000)
state_dict = load_file(MODEL_WEIGHTS)
model.load_state_dict(state_dict, strict=False)
# Replace classifier head for 10 classes
model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
model = model.to(DEVICE)
print("Model loaded successfully")

# Optimizer & Scheduler - different LR for backbone vs head
backbone_params = [p for n, p in model.named_parameters() if "classifier" not in n]
head_params = [p for n, p in model.named_parameters() if "classifier" in n]
optimizer = torch.optim.AdamW([
    {"params": backbone_params, "lr": LR * 0.1},
    {"params": head_params, "lr": LR},
], weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
criterion = nn.CrossEntropyLoss()

# Training
best_val_acc = 0.0
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

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

    train_acc = correct / total
    train_loss = running_loss / total

    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            _, preds = outputs.max(1)
            val_correct += preds.eq(labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total
    scheduler.step()

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), os.path.join(DATA_DIR, "best_model.pth"))
        print(f"  -> Saved best model (val_acc={val_acc:.4f})")

print(f"\nTraining complete. Best val acc: {best_val_acc:.4f}")
