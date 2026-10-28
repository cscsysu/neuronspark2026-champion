"""
NS-2026-04 快速版：单模型LightGBM全量训练 + 生成提交
"""
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import sparse
import lightgbm as lgb

WORK_DIR = Path("/Users/tristonchen/Desktop/neuronspark2026")

# 1. Load data
X_train_hashed = sparse.load_npz(WORK_DIR / "features" / "hashed_smiles_train.npz").toarray()
X_test_hashed = sparse.load_npz(WORK_DIR / "features" / "hashed_smiles_test.npz").toarray()

with open(WORK_DIR / "train.csv") as f:
    train_rows = list(csv.DictReader(f))
train_ids = [r["id"] for r in train_rows]
train_smiles = [r["smiles"] for r in train_rows]
y_train = np.array([float(r["target"]) for r in train_rows])

with open(WORK_DIR / "test.csv") as f:
    test_rows = list(csv.DictReader(f))
test_ids = [r["id"] for r in test_rows]
test_smiles = [r["smiles"] for r in test_rows]

# 2. Molecular descriptors
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

X_train_desc = compute_descriptors(train_smiles)
X_test_desc = compute_descriptors(test_smiles)

X_train = np.hstack([X_train_hashed, X_train_desc])
X_test = np.hstack([X_test_hashed, X_test_desc])

# 3. Train single LightGBM model on ALL data (2000 iterations, no early stopping needed)
# We know from validation that 2000 iters gives RMSE=0.0576 on scaffold-val
print("Training LightGBM on all data (2000 trees)...")
model = lgb.LGBMRegressor(
    objective="regression",
    boosting_type="gbdt",
    learning_rate=0.03,
    num_leaves=63,
    min_child_samples=10,
    subsample=0.8,
    colsample_bytree=0.6,
    reg_alpha=0.1,
    reg_lambda=1.0,
    n_estimators=2000,
    verbose=-1,
    random_state=42,
)
model.fit(X_train, y_train)
print("Model trained!")

test_preds = model.predict(X_test)
print(f"Predictions: mean={test_preds.mean():.4f}, std={test_preds.std():.4f}")

# 4. Save submission
output_dir = WORK_DIR / "submission"
output_dir.mkdir(exist_ok=True)

results_path = output_dir / "results.csv"
with open(results_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "prediction"])
    for tid, pred in zip(test_ids, test_preds):
        writer.writerow([tid, f"{pred:.6f}"])

# Format check
import subprocess
result = subprocess.run(
    ["python3", str(WORK_DIR / "tools" / "check_format.py"),
     str(results_path), "--test-csv", str(WORK_DIR / "test.csv")],
    capture_output=True, text=True
)
print(f"Format check: {result.stdout.strip()}")

# Create zip
import zipfile
zip_path = WORK_DIR / "NS-2026-04-answer.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write(results_path, "results.csv")
print(f"Created: {zip_path}")
print("Done!")
