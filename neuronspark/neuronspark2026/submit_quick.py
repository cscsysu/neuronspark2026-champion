"""
NS-2026-04 极简快速版 (5分钟)
Morgan FP + hashed features, 20个scaffold-aware模型集成
"""
import csv
import json
import math
import os
import warnings
import zipfile

import numpy as np
from scipy import sparse
import lightgbm as lgb

warnings.filterwarnings('ignore')

from rdkit import Chem, RDLogger
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
RDLogger.DisableLog('rdApp.*')

# 1. Load data
print("Loading data...")
with open("train.csv") as f:
    tr = list(csv.DictReader(f))
smiles_tr = [r["smiles"] for r in tr]
y = np.array([float(r["target"]) for r in tr])

with open("test.csv") as f:
    te = list(csv.DictReader(f))
ids = [r["id"] for r in te]
smiles_te = [r["smiles"] for r in te]

# 2. Features: Morgan FP (radius 2, 2048) + hashed SMILES
print("Building features...")
gen = GetMorganGenerator(radius=2, fpSize=2048)


def get_fp(slist):
    X = np.zeros((len(slist), 2048), dtype=np.float32)
    for i, s in enumerate(slist):
        mol = Chem.MolFromSmiles(s)
        if mol:
            X[i] = gen.GetFingerprintAsNumPy(mol)
    return X


Xtr = get_fp(smiles_tr)
Xte = get_fp(smiles_te)

h_tr = sparse.load_npz("features/hashed_smiles_train.npz").toarray()
h_te = sparse.load_npz("features/hashed_smiles_test.npz").toarray()

Xtr = np.hstack([Xtr, h_tr])
Xte = np.hstack([Xte, h_te])
print(f"Features: {Xtr.shape[1]}")

# 3. Scaffold split
with open("scaffold_split.json") as f:
    sc = json.load(f)

id2i = {r["id"]: i for i, r in enumerate(tr)}
valid_set = set(sc["valid_ids"])
train_idx = np.array([id2i[x] for x in id2i if x not in valid_set])
val_idx = np.array([id2i[x] for x in sc["valid_ids"] if x in id2i])

print(f"Scaffold: train={len(train_idx)}, val={len(val_idx)}")

# 4. Train 20 scaffold-aware models
print("\nTraining 20 scaffold-aware models...")
preds = np.zeros(len(ids))
val_preds = np.zeros(len(val_idx))

for seed in range(20):
    model = lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        n_estimators=3000,
        learning_rate=0.015,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.4,
        reg_alpha=0.1,
        reg_lambda=1.0,
        verbose=-1,
        random_state=42 + seed * 77,
        device="gpu",
        gpu_device_id=0,
    )
    model.fit(
        Xtr[train_idx], y[train_idx],
        eval_set=[(Xtr[val_idx], y[val_idx])],
        callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)]
    )
    preds += model.predict(Xte)
    val_preds += model.predict(Xtr[val_idx])
    print(f"  Model {seed+1}/20, best_iter={model.best_iteration_}")

preds /= 20
val_preds /= 20

# Validate
from scipy.stats import spearmanr
rmse = math.sqrt(np.mean((y[val_idx] - val_preds) ** 2))
sp, _ = spearmanr(y[val_idx], val_preds)
print(f"\nScaffold-val: RMSE={rmse:.4f}, Spearman={sp:.4f}")
if rmse <= 0.25 and sp >= 0.96:
    print("*** GOLD BAND on scaffold-val! ***")

# 5. Save
os.makedirs("submission_quick", exist_ok=True)
results_path = "submission_quick/results.csv"
with open(results_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "prediction"])
    for tid, p in zip(ids, preds):
        w.writerow([tid, f"{p:.6f}"])

import subprocess
result = subprocess.run(
    ["python3", "tools/check_format.py", results_path, "--test-csv", "test.csv"],
    capture_output=True, text=True
)
print(f"Format check: {result.stdout.strip()}")

with zipfile.ZipFile("NS-2026-04-answer-quick.zip", "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write(results_path, "results.csv")

print(f"\nCreated: NS-2026-04-answer-quick.zip")
print("Done!")
