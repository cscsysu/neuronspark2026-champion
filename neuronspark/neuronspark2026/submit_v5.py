"""
NS-2026-04 v5: 回归本源
用第1次拿630分的方案，加强版：更多树+多seed集成
第1次: hashed(2048) + 简单描述符(30) + 单模型2000树 = RMSE 0.254
这次: 同样特征 + 3000-4000树 + 20seed集成 = 目标RMSE < 0.25
"""
import csv
import json
import math
import os
import re
import warnings
import zipfile
from collections import Counter

import numpy as np
from scipy import sparse
import lightgbm as lgb

warnings.filterwarnings('ignore')

# 1. Load data
print("Loading data...")
with open("train.csv") as f:
    tr = list(csv.DictReader(f))
train_ids = [r["id"] for r in tr]
train_smiles = [r["smiles"] for r in tr]
y = np.array([float(r["target"]) for r in tr])

with open("test.csv") as f:
    te = list(csv.DictReader(f))
ids = [r["id"] for r in te]
test_smiles = [r["smiles"] for r in te]

print(f"Train: {len(tr)}, Test: {len(te)}")

# 2. Features: EXACTLY what worked before - hashed + simple descriptors
print("Building features...")
h_tr = sparse.load_npz("features/hashed_smiles_train.npz").toarray()
h_te = sparse.load_npz("features/hashed_smiles_test.npz").toarray()

TOKEN_RE = re.compile(r"Cl|Br|Si|Na|Li|Mg|Ca|Al|@@?|\[[^\]]+\]|[A-Za-z]|\d|\(|\)|=|#|-|\+|\\|/|\.|:")
ATOM_RE = re.compile(r"Cl|Br|Si|Na|Li|Mg|Ca|Al|[BCNOPSFIKH]|[bcnops]")


def compute_descriptors(smiles_list):
    features = []
    for smi in smiles_list:
        counts = Counter(ATOM_RE.findall(str(smi)))
        toks = TOKEN_RE.findall(str(smi))
        aromatic = sum(counts.get(x, 0) for x in ["c", "n", "o", "s", "p"])
        carbon = counts.get("C", 0) + counts.get("c", 0)
        nitrogen = counts.get("N", 0) + counts.get("n", 0)
        oxygen = counts.get("O", 0) + counts.get("o", 0)
        sulfur = counts.get("S", 0) + counts.get("s", 0)
        hetero = sum(v for k, v in counts.items() if k not in {"C", "c", "H"})
        halogen = counts.get("Cl", 0) + counts.get("Br", 0) + counts.get("F", 0) + counts.get("I", 0)
        total_atoms = sum(counts.values())
        n_rings = sum(ch.isdigit() for ch in smi)
        n_branches = smi.count("(")
        n_double = smi.count("=")
        n_triple = smi.count("#")
        n_charged = smi.count("+") + smi.count("-")
        n_chiral = smi.count("@")
        smi_len = len(smi)
        n_tokens = len(toks)
        carbon_ratio = carbon / max(total_atoms, 1)
        hetero_ratio = hetero / max(total_atoms, 1)
        aromatic_ratio = aromatic / max(total_atoms, 1)
        mw_proxy = (carbon * 12 + nitrogen * 14 + oxygen * 16 + sulfur * 32 + halogen * 35 + total_atoms)
        feat = [
            smi_len, n_tokens, total_atoms, carbon, nitrogen, oxygen, sulfur,
            hetero, aromatic, halogen, n_rings, n_branches, n_double, n_triple,
            n_charged, n_chiral, carbon_ratio, hetero_ratio, aromatic_ratio,
            mw_proxy, math.log1p(smi_len), math.log1p(total_atoms), math.log1p(mw_proxy),
            carbon * n_rings, aromatic * n_double, hetero * n_branches,
            total_atoms * n_rings, smi_len * halogen,
        ]
        features.append(feat)
    return np.array(features, dtype=np.float32)


desc_tr = compute_descriptors(train_smiles)
desc_te = compute_descriptors(test_smiles)

Xtr = np.hstack([h_tr, desc_tr])
Xte = np.hstack([h_te, desc_te])
print(f"Features: {Xtr.shape[1]} (same as first 630-score submission)")

# 3. Train on ALL 8500 with MORE trees and multi-seed
# First submission used 2000 trees and was still improving
# Now use 3500 trees + 20 seeds for better convergence + lower variance
print("\nTraining 20 models on full data (3500 trees each)...")

test_preds = np.zeros(len(ids))
n_models = 0

for seed in range(20):
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=3500,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.6,
        reg_alpha=0.1,
        reg_lambda=1.0,
        verbose=-1,
        random_state=42 + seed * 77,
    )
    model.fit(Xtr, y)
    test_preds += model.predict(Xte)
    n_models += 1
    if (seed + 1) % 5 == 0:
        print(f"  {seed+1}/20 models done")

test_preds /= n_models
print(f"Ensemble of {n_models} models")
print(f"Predictions: mean={test_preds.mean():.4f}, std={test_preds.std():.4f}")

# 4. Save
os.makedirs("submission_v5", exist_ok=True)
results_path = "submission_v5/results.csv"
with open(results_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "prediction"])
    for tid, p in zip(ids, test_preds):
        w.writerow([tid, f"{p:.6f}"])

import subprocess
result = subprocess.run(
    ["python3", "tools/check_format.py", results_path, "--test-csv", "test.csv"],
    capture_output=True, text=True
)
print(f"Format check: {result.stdout.strip()}")

with zipfile.ZipFile("NS-2026-04-answer-v5.zip", "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write(results_path, "results.csv")

print(f"\nCreated: NS-2026-04-answer-v5.zip")
print("Done!")
