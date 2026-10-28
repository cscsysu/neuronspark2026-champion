#!/usr/bin/env python3
"""
NS-2026-05: Campus Sound Event Detection
Uses AST (Audio Spectrogram Transformer) for multi-label audio classification.
Model downloaded from ModelScope for faster access in China.
"""

import os
import csv
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import ASTFeatureExtractor, ASTModel, ASTConfig
import librosa
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')

def get_model_path():
    """Get model path - uses HF mirror via HF_ENDPOINT env var"""
    # HF_ENDPOINT should be set to https://hf-mirror.com before running
    model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"
    print(f"Using model: {model_name}")
    print(f"HF_ENDPOINT: {os.environ.get('HF_ENDPOINT', 'default')}")
    return model_name

# Config
DATA_ROOT = "/data2/css2025/NS-2026-05"
DEVICE = "cuda:0"  # mapped via CUDA_VISIBLE_DEVICES
BATCH_SIZE = 16
EPOCHS = 30
LR = 1e-4
NUM_LABELS = 9
SAMPLE_RATE = 16000
MAX_DURATION = 6.0  # seconds

LABEL_NAMES = ["footsteps", "keyboard", "door", "alarm", "vehicle", "rain", "applause", "glass_break", "ambient"]
LABEL2IDX = {name: i for i, name in enumerate(LABEL_NAMES)}
RARE_LABELS = ["alarm", "door", "glass_break", "vehicle"]


class AudioDataset(Dataset):
    def __init__(self, csv_path, data_root, feature_extractor, is_test=False):
        self.data_root = data_root
        self.feature_extractor = feature_extractor
        self.is_test = is_test
        self.samples = []
        
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append(row)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        row = self.samples[idx]
        audio_path = os.path.join(self.data_root, row['audio'])
        
        # Load audio
        waveform, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        
        # Pad or truncate to 6 seconds
        target_len = int(MAX_DURATION * SAMPLE_RATE)
        if len(waveform) < target_len:
            waveform = np.pad(waveform, (0, target_len - len(waveform)))
        else:
            waveform = waveform[:target_len]
        
        # Extract features using AST feature extractor
        inputs = self.feature_extractor(
            waveform, 
            sampling_rate=SAMPLE_RATE, 
            return_tensors="pt"
        )
        input_values = inputs.input_values.squeeze(0)  # [time, freq]
        
        if not self.is_test:
            # Multi-hot encoding
            labels = row['labels'].split('|')
            target = np.zeros(NUM_LABELS, dtype=np.float32)
            for label in labels:
                target[LABEL2IDX[label.strip()]] = 1.0
            return input_values, torch.tensor(target), row['id']
        else:
            return input_values, row['id']


class ASTClassifier(nn.Module):
    """AST-based multi-label classifier"""
    def __init__(self, model_path, num_labels=NUM_LABELS):
        super().__init__()
        self.ast = ASTModel.from_pretrained(model_path)
        hidden_size = self.ast.config.hidden_size  # 768
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_labels)
        )
    
    def forward(self, input_values):
        outputs = self.ast(input_values)
        # Use CLS token embedding
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        return logits


def compute_metrics(y_true, y_pred):
    """Compute macro F1, sample F1, rare recall"""
    # Macro F1
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Sample F1
    sample_f1 = f1_score(y_true, y_pred, average='samples', zero_division=0)
    
    # Rare recall
    rare_indices = [LABEL2IDX[l] for l in RARE_LABELS]
    rare_true = y_true[:, rare_indices]
    rare_pred = y_pred[:, rare_indices]
    # Per-class recall then average
    rare_recalls = []
    for i in range(len(rare_indices)):
        tp = (rare_true[:, i] * rare_pred[:, i]).sum()
        fn = (rare_true[:, i] * (1 - rare_pred[:, i])).sum()
        if tp + fn > 0:
            rare_recalls.append(tp / (tp + fn))
    rare_recall = np.mean(rare_recalls) if rare_recalls else 0.0
    
    return macro_f1, sample_f1, rare_recall


def find_optimal_thresholds(y_true, y_scores, num_labels=NUM_LABELS):
    """Find per-class optimal thresholds"""
    thresholds = np.zeros(num_labels)
    for i in range(num_labels):
        best_t = 0.5
        best_f1 = 0.0
        for t in np.arange(0.1, 0.9, 0.02):
            pred = (y_scores[:, i] >= t).astype(int)
            f1 = f1_score(y_true[:, i], pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        thresholds[i] = best_t
    return thresholds


def apply_constraints(predictions):
    """
    Apply label constraints:
    - ambient cannot co-occur with other labels
    - If no label predicted, default to ambient
    """
    result = predictions.copy()
    ambient_idx = LABEL2IDX['ambient']
    
    for i in range(len(result)):
        active = np.where(result[i] == 1)[0]
        if len(active) == 0:
            # No prediction -> ambient
            result[i, ambient_idx] = 1
        elif ambient_idx in active and len(active) > 1:
            # ambient with others -> remove ambient
            result[i, ambient_idx] = 0
    
    return result


def train():
    print("=" * 60)
    print("NS-2026-05: Campus Sound Event Detection with AST")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Data root: {DATA_ROOT}")
    
    # Load feature extractor
    print("\nLoading AST model from ModelScope...")
    model_path = get_model_path()
    feature_extractor = ASTFeatureExtractor.from_pretrained(model_path)
    
    # Load training data
    print("Loading training data...")
    train_csv = os.path.join(DATA_ROOT, "train.csv")
    
    # Read all labels for stratified split
    all_labels = []
    with open(train_csv, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_labels.append(row['labels'])
    
    # Use primary label for stratification
    primary_labels = [l.split('|')[0] for l in all_labels]
    
    # 5-fold cross validation for threshold tuning
    print("\nStarting training with 5-fold CV for threshold calibration...")
    
    full_dataset = AudioDataset(train_csv, DATA_ROOT, feature_extractor, is_test=False)
    
    # Train on full data, but first do a quick val split for thresholds
    n_samples = len(full_dataset)
    indices = list(range(n_samples))
    
    # Simple 80/20 split for threshold tuning
    np.random.seed(42)
    np.random.shuffle(indices)
    split = int(0.8 * n_samples)
    train_idx, val_idx = indices[:split], indices[split:]
    
    train_subset = torch.utils.data.Subset(full_dataset, train_idx)
    val_subset = torch.utils.data.Subset(full_dataset, val_idx)
    
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Initialize model
    print("Initializing AST model...")
    model = ASTClassifier(model_path).to(DEVICE)
    
    # Loss and optimizer
    # Use weighted BCE for rare class emphasis
    pos_weights = torch.ones(NUM_LABELS).to(DEVICE)
    for label in RARE_LABELS:
        pos_weights[LABEL2IDX[label]] = 2.0  # Higher weight for rare classes
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_val_score = 0.0
    best_model_state = None
    
    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss = 0.0
        for batch_idx, (inputs, targets, _) in enumerate(train_loader):
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)
        
        # Validate
        model.eval()
        all_targets = []
        all_scores = []
        with torch.no_grad():
            for inputs, targets, _ in val_loader:
                inputs = inputs.to(DEVICE)
                logits = model(inputs)
                scores = torch.sigmoid(logits).cpu().numpy()
                all_scores.append(scores)
                all_targets.append(targets.numpy())
        
        all_targets = np.vstack(all_targets)
        all_scores = np.vstack(all_scores)
        
        # Find optimal thresholds
        thresholds = find_optimal_thresholds(all_targets, all_scores)
        preds = (all_scores >= thresholds).astype(int)
        preds = apply_constraints(preds)
        
        macro_f1, sample_f1, rare_recall = compute_metrics(all_targets, preds)
        val_score = macro_f1 * 0.4 + sample_f1 * 0.25 + rare_recall * 0.25
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_train_loss:.4f} | "
              f"Macro-F1: {macro_f1:.4f} | Sample-F1: {sample_f1:.4f} | "
              f"Rare-Recall: {rare_recall:.4f} | Score: {val_score:.4f}")
        
        if val_score > best_val_score:
            best_val_score = val_score
            best_model_state = model.state_dict().copy()
            best_thresholds = thresholds.copy()
            print(f"  -> New best! Thresholds: {[f'{t:.2f}' for t in thresholds]}")
    
    print(f"\nBest validation score: {best_val_score:.4f}")
    print(f"Best thresholds: {dict(zip(LABEL_NAMES, [f'{t:.2f}' for t in best_thresholds]))}")
    
    # Now retrain on full data with best hyperparams
    print("\n" + "=" * 60)
    print("Retraining on full training data...")
    full_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    model = ASTClassifier(model_path).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for inputs, targets, _ in full_loader:
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        scheduler.step()
        if (epoch + 1) % 5 == 0:
            print(f"  Full train epoch {epoch+1}/{EPOCHS} | Loss: {train_loss/len(full_loader):.4f}")
    
    # Save model
    model_path = os.path.join(DATA_ROOT, "ast_model.pt")
    torch.save({
        'model_state': model.state_dict(),
        'thresholds': best_thresholds,
        'label_names': LABEL_NAMES,
    }, model_path)
    print(f"Model saved to {model_path}")
    
    # Predict on test set
    print("\n" + "=" * 60)
    print("Predicting on test set...")
    test_csv = os.path.join(DATA_ROOT, "test.csv")
    test_dataset = AudioDataset(test_csv, DATA_ROOT, feature_extractor, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    model.eval()
    all_ids = []
    all_scores = []
    with torch.no_grad():
        for inputs, ids in test_loader:
            inputs = inputs.to(DEVICE)
            logits = model(inputs)
            scores = torch.sigmoid(logits).cpu().numpy()
            all_scores.append(scores)
            all_ids.extend(ids)
    
    all_scores = np.vstack(all_scores)
    
    # Apply thresholds
    predictions = (all_scores >= best_thresholds).astype(int)
    predictions = apply_constraints(predictions)
    
    # Generate results.csv
    results_path = os.path.join(DATA_ROOT, "results.csv")
    with open(results_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'labels'])
        for i, sample_id in enumerate(all_ids):
            active_labels = [LABEL_NAMES[j] for j in range(NUM_LABELS) if predictions[i, j] == 1]
            if not active_labels:
                active_labels = ['ambient']
            writer.writerow([sample_id, '|'.join(active_labels)])
    
    print(f"Results saved to {results_path}")
    
    # Validate format
    print("\nValidating submission format...")
    os.system(f"python3 {DATA_ROOT}/tools/check_format.py {results_path} --test-csv {test_csv} --label-map {DATA_ROOT}/label_map.json")
    
    # Create submission zip
    import zipfile
    zip_path = os.path.join(DATA_ROOT, "NS-2026-05-answer.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(results_path, 'results.csv')
    print(f"\nSubmission zip: {zip_path}")
    print("Done!")


if __name__ == "__main__":
    train()
