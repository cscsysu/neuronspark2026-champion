"""
NS-2026-04 v4: 最优平衡版 (兼容旧版RDKit)
全量8500训练 + Morgan FP + hashed + 多seed集成
"""
import csv
import json
import math
import os
import warnings
import zipfile

import numpy as np
from scipy import sparse
from scipy.stats import spearmanr
import lightgbm as lgb

warnings.filterwarnings('ignore')

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, MACCSkeys

# 1. Load data
print("Loading data...")
with open("train.csv") as f:
    tr = list(csv.DictReader(f))
train_ids = [r["id"] for r in tr]
smiles_tr = [r["smiles"] for r in tr]
y = np.array([float(r["target"]) for r in tr])

with open("test.csv") as f:
    te = list(csv.DictReader(f))
ids = [r["id"] for r in te]
smiles_te = [r["smiles"] for r in te]

with open("scaffold_split.json") as f:
    sc = json.load(f)
id2i = {r["id"]: i for i, r in enumerate(tr)}
valid_set = set(sc["valid_ids"])
sc_train_idx = np.array([id2i[x] for x in id2i if x not in valid_set])
sc_val_idx = np.array([id2i[x] for x in sc["valid_ids"] if x in id2i])

print(f"Train: {len(tr)}, Test: {len(te)}")

# 2. Features: Morgan FP + MACCS + hashed
print("Building features...")


def build_fp(slist):
    n = len(slist)
    m2 = np.zeros((n, 2048), dtype=np.float32)
    m3 = np.zeros((n, 2048), dtype=np.float32)
    maccs = np.zeros((n, 167), dtype=np.float32)
    for i, s in enumerate(slist):
        mol = Chem.MolFromSmiles(s)
        if mol:
            fp2 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            arr2 = np.zeros(2048)
            DataStructs.ConvertToNumpyArray(fp2, arr2)
            m2[i] = arr2

            fp3 = AllChem.GetHashedMorganFingerprint(mol, 3, nBits=2048)
            for idx, val in fp3.GetNonzeroElements().items():
                m3[i, idx] = val

            fpm = MACCSkeys.GenMACCSKeys(mol)
            arrm = np.zeros(167)
            DataStructs.ConvertToNumpyArray(fpm, arrm)
            maccs[i] = arrm
    return m2, m3, maccs


m2_tr, m3_tr, mac_tr = build_fp(smiles_tr)
m2_te, m3_te, mac_te = build_fp(smiles_te)
print("  Fingerprints done")

h_tr = sparse.load_npz("features/hashed_smiles_train.npz").toarray()
h_te = sparse.load_npz("features/hashed_smiles_test.npz").toarray()

Xtr = np.hstack([m2_tr, m3_tr, mac_tr, h_tr])
Xte = np.hstack([m2_te, m3_te, mac_te, h_te])
print(f"Features: {Xtr.shape[1]}")

# 3. Find optimal iteration with scaffold split
print("\nCalibrating iterations on scaffold split...")
configs = [
    {"learning_rate": 0.015, "num_leaves": 63, "min_child_samples": 10,
     "subsample": 0.8, "colsample_bytree": 0.4, "reg_alpha": 0.1, "reg_lambda": 1.0},
    {"learning_rate": 0.02, "num_leaves": 95, "min_child_samples": 8,
     "subsample": 0.8, "colsample_bytree": 0.5, "reg_alpha": 0.05, "reg_lambda": 0.5},
    {"learning_rate": 0.01, "num_leaves": 127, "min_child_samples": 15,
     "subsample": 0.75, "colsample_bytree": 0.35, "reg_alpha": 0.3, "reg_lambda": 1.5},
]

opt_iters = []
for cfg_idx, cfg in enumerate(configs):
    model = lgb.LGBMRegressor(
        objective="regression", metric="rmse",
        n_estimators=5000, verbose=-1, random_state=42, **cfg
    )
    model.fit(
        Xtr[sc_train_idx], y[sc_train_idx],
        eval_set=[(Xtr[sc_val_idx], y[sc_val_idx])],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
    )
    val_pred = model.predict(Xtr[sc_val_idx])
    rmse = math.sqrt(np.mean((y[sc_val_idx] - val_pred) ** 2))
    sp, _ = spearmanr(y[sc_val_idx], val_pred)
    opt_iters.append(model.best_iteration_)
    print(f"  Config {cfg_idx}: iter={model.best_iteration_}, RMSE={rmse:.4f}, Sp={sp:.4f}")

# 4. Train on FULL 8500 data
print(f"\nTraining on ALL 8500 ({len(configs)} configs x 10 seeds = {len(configs)*10} models)...")

test_preds = np.zeros(len(ids))
n_models = 0

for cfg_idx, cfg in enumerate(configs):
    n_est = opt_iters[cfg_idx]
    for seed in range(10):
        model = lgb.LGBMRegressor(
            objective="regression", n_estimators=n_est,
            verbose=-1, random_state=42 + seed * 77 + cfg_idx * 999, **cfg
        )
        model.fit(Xtr, y)
        test_preds += model.predict(Xte)
        n_models += 1
    print(f"  Config {cfg_idx} done (10 seeds, {n_est} trees)")

test_preds /= n_models
print(f"Total: {n_models} models")
print(f"Predictions: mean={test_preds.mean():.4f}, std={test_preds.std():.4f}")

# 5. Save
os.makedirs("submission_v4", exist_ok=True)
results_path = "submission_v4/results.csv"
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

with zipfile.ZipFile("NS-2026-04-answer-v4.zip", "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write(results_path, "results.csv")

print(f"\nCreated: NS-2026-04-answer-v4.zip")
print("Done!")
