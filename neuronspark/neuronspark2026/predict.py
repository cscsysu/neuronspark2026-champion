"""
NS-2026-00 - 推理脚本
对测试集生成预测结果
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

# Config
DATA_DIR = "/Users/tristonchen/Desktop/neuronspark2026/2026-00-data"
MODEL_PATH = os.path.join(DATA_DIR, "best_model.pth")
BATCH_SIZE = 32
IMG_SIZE = 260
NUM_CLASSES = 10
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# Label mapping
LABELS = [
    "invoice", "receipt", "schedule", "poster", "lab_note",
    "notice", "handwritten_note", "form", "meeting_minutes", "grade_report"
]
IDX2LABEL = {i: l for i, l in enumerate(LABELS)}

# Dataset
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

# Transform
test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Load test data
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
test_dataset = TestDataset(test_df, DATA_DIR, test_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Load model
print("Loading model...")
model = timm.create_model("efficientnet_b2", pretrained=False, num_classes=NUM_CLASSES)
state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
model.load_state_dict(state_dict)
model = model.to(DEVICE)
model.eval()
print("Model loaded")

# Inference
all_ids = []
all_preds = []

with torch.no_grad():
    for imgs, ids in test_loader:
        imgs = imgs.to(DEVICE)
        outputs = model(imgs)
        _, preds = outputs.max(1)
        all_ids.extend(ids)
        all_preds.extend([IDX2LABEL[p.item()] for p in preds])
        print(f"  Processed {len(all_ids)}/{len(test_df)}")

# Save results
results_df = pd.DataFrame({"id": all_ids, "label": all_preds})
results_path = os.path.join(DATA_DIR, "results.csv")
results_df.to_csv(results_path, index=False)
print(f"\nResults saved to {results_path}")
print(f"Total predictions: {len(results_df)}")
print(f"Label distribution:")
print(results_df["label"].value_counts())
