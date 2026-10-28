"""
NS-2026-04 v2: 正确的scaffold split + early stopping + 特征筛选
目标: RMSE<=0.25, Spearman>=0.96
"""
import csv
import json
import math
import os
import warnings
import zipfile
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

warnings.filterwarnings('ignore')

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, DataStructs, MACCSkeys
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
RDLogger.DisableLog('rdApp.*')

WORK_DIR = Path(".")
SEED = 42
np.random.seed(SEED)

# ============================================================
# 1. Load data
# ============================================================
print("=" * 60)
print("Loading data...")

with open(WORK_DIR / "train.csv") as f:
    train_rows = list(csv.DictReader(f))
train_ids = [r["id"] for r in train_rows]
train_smiles = [r["smiles"] for r in train_rows]
y_train = np.array([float(r["target"]) for r in train_rows])

with open(WORK_DIR / "test.csv") as f:
    test_rows = list(csv.DictReader(f))
test_ids = [r["id"] for r in test_rows]
test_smiles = [r["smiles"] for r in test_rows]

# Scaffold split - CORRECT usage: valid_ids should be EXCLUDED from training
with open(WORK_DIR / "scaffold_split.json") as f:
    scaffold = json.load(f)

train_id_to_idx = {tid: i for i, tid in enumerate(train_ids)}
valid_id_set = set(scaffold["valid_ids"])

# Proper split: train on samples NOT in valid_ids, validate on valid_ids
sc_train_idx = np.array([train_id_to_idx[tid] for tid in train_ids if tid not in valid_id_set])
sc_val_idx = np.array([train_id_to_idx[tid] for tid in scaffold["valid_ids"] if tid in train_id_to_idx])

print(f"Train: {len(train_ids)}, Test: {len(test_ids)}")
print(f"Scaffold split (correct): train={len(sc_train_idx)}, val={len(sc_val_idx)}")

# ============================================================
# 2. Feature Engineering
# ============================================================
print("\nBuilding features...")

gen2 = GetMorganGenerator(radius=2, fpSize=2048)
gen3 = GetMorganGenerator(radius=3, fpSize=2048)

desc_names = [name for name, _ in Descriptors.descList]
calc = MolecularDescriptorCalculator(desc_names)


def build_features(smiles_list, hashed_path):
    n = len(smiles_list)
    morgan2 = np.zeros((n, 2048), dtype=np.float32)
    morgan3 = np.zeros((n, 2048), dtype=np.float32)
    maccs = np.zeros((n, 167), dtype=np.float32)
    rdkit_desc = np.zeros((n, len(desc_names)), dtype=np.float32)

    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            morgan2[i] = gen2.GetFingerprintAsNumPy(mol)
            morgan3[i] = gen3.GetCountFingerprintAsNumPy(mol)
            fp = MACCSkeys.GenMACCSKeys(mol)
            arr = np.zeros(167)
            DataStructs.ConvertToNumpyArray(fp, arr)
            maccs[i] = arr
            try:
                d = calc.CalcDescriptors(mol)
                rdkit_desc[i] = [0.0 if (math.isnan(x) or math.isinf(x)) else x for x in d]
            except:
                pass
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{n} molecules processed")

    hashed = sparse.load_npz(hashed_path).toarray()
    return np.hstack([morgan2, morgan3, maccs, rdkit_desc, hashed])


X_train_full = build_features(train_smiles, WORK_DIR / "features" / "hashed_smiles_train.npz")
X_test_full = build_features(test_smiles, WORK_DIR / "features" / "hashed_smiles_test.npz")

# Remove constant/invalid columns
valid_cols = np.all(np.isfinite(X_train_full), axis=0) & (np.std(X_train_full, axis=0) > 1e-10)
X_train_full = np.nan_to_num(X_train_full[:, valid_cols], nan=0.0, posinf=0.0, neginf=0.0)
X_test_full = np.nan_to_num(X_test_full[:, valid_cols], nan=0.0, posinf=0.0, neginf=0.0)
print(f"Features after cleaning: {X_train_full.shape[1]}")

# ============================================================
# 3. Feature selection using scaffold split
# ============================================================
print("\nFeature selection via LightGBM importance...")

# Train a quick model to get feature importances
sel_model = lgb.LGBMRegressor(
    objective="regression", n_estimators=500, learning_rate=0.05,
    num_leaves=63, subsample=0.8, colsample_bytree=0.5, verbose=-1, random_state=SEED
)
sel_model.fit(
    X_train_full[sc_train_idx], y_train[sc_train_idx],
    eval_set=[(X_train_full[sc_val_idx], y_train[sc_val_idx])],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
)

# Select top features by importance
importances = sel_model.feature_importances_
top_k = min(4000, int(np.sum(importances > 0)))  # Keep features with non-zero importance
top_feat_idx = np.argsort(importances)[::-1][:top_k]

X_train = X_train_full[:, top_feat_idx]
X_test = X_test_full[:, top_feat_idx]
print(f"Selected top {top_k} features")

# Quick scaffold validation check
val_pred = sel_model.predict(X_train_full[sc_val_idx])
rmse_check = math.sqrt(np.mean((y_train[sc_val_idx] - val_pred) ** 2))
sp_check, _ = spearmanr(y_train[sc_val_idx], val_pred)
print(f"Quick scaffold-val check: RMSE={rmse_check:.4f}, Spearman={sp_check:.4f}")

# ============================================================
# 4. LightGBM with K-Fold CV + Early Stopping
# ============================================================
print("\n" + "=" * 60)
print("Training LightGBM with K-Fold CV + Early Stopping")
print("=" * 60)

configs = [
    {"learning_rate": 0.01, "num_leaves": 63, "min_child_samples": 10,
     "subsample": 0.8, "colsample_bytree": 0.4, "reg_alpha": 0.1, "reg_lambda": 1.0},
    {"learning_rate": 0.01, "num_leaves": 127, "min_child_samples": 20,
     "subsample": 0.7, "colsample_bytree": 0.3, "reg_alpha": 0.5, "reg_lambda": 2.0},
    {"learning_rate": 0.02, "num_leaves": 31, "min_child_samples": 5,
     "subsample": 0.85, "colsample_bytree": 0.5, "reg_alpha": 0.05, "reg_lambda": 0.5},
    {"learning_rate": 0.015, "num_leaves": 95, "min_child_samples": 15,
     "subsample": 0.75, "colsample_bytree": 0.45, "reg_alpha": 0.2, "reg_lambda": 1.5},
]

N_SEEDS = 5
N_FOLDS = 5

lgb_oof = np.zeros(len(y_train))
lgb_test = np.zeros(len(test_ids))
n_lgb = 0
best_iterations = []

for cfg_idx, cfg in enumerate(configs):
    for seed_i in range(N_SEEDS):
        seed = SEED + seed_i * 111 + cfg_idx * 1000
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

        for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train)):
            model = lgb.LGBMRegressor(
                objective="regression", metric="rmse",
                n_estimators=5000, verbose=-1, random_state=seed, **cfg
            )
            model.fit(
                X_train[tr_idx], y_train[tr_idx],
                eval_set=[(X_train[va_idx], y_train[va_idx])],
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
            )
            lgb_oof[va_idx] += model.predict(X_train[va_idx])
            lgb_test += model.predict(X_test)
            best_iterations.append(model.best_iteration_)
            n_lgb += 1

    print(f"  Config {cfg_idx+1}/{len(configs)} done ({N_SEEDS} seeds x {N_FOLDS} folds)")

lgb_oof /= (N_SEEDS * len(configs))
lgb_test /= n_lgb

rmse_lgb = math.sqrt(np.mean((y_train - lgb_oof) ** 2))
sp_lgb, _ = spearmanr(y_train, lgb_oof)
print(f"\nLightGBM OOF: RMSE={rmse_lgb:.4f}, Spearman={sp_lgb:.4f}")
print(f"Avg best iteration: {np.mean(best_iterations):.0f}")

# ============================================================
# 5. Ridge Regression (different view of data)
# ============================================================
print("\nTraining Ridge...")
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

ridge_oof = np.zeros(len(y_train))
ridge_test = np.zeros(len(test_ids))
n_ridge = 0

for alpha in [0.1, 1.0, 10.0, 50.0, 100.0]:
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train_sc)):
        model = Ridge(alpha=alpha)
        model.fit(X_train_sc[tr_idx], y_train[tr_idx])
        ridge_oof[va_idx] += model.predict(X_train_sc[va_idx])
        ridge_test += model.predict(X_test_sc)
        n_ridge += 1

ridge_oof /= 5  # 5 alphas
ridge_test /= n_ridge

rmse_ridge = math.sqrt(np.mean((y_train - ridge_oof) ** 2))
sp_ridge, _ = spearmanr(y_train, ridge_oof)
print(f"Ridge OOF: RMSE={rmse_ridge:.4f}, Spearman={sp_ridge:.4f}")

# ============================================================
# 6. Also train scaffold-aware models (train on 6800, predict all)
# ============================================================
print("\nTraining scaffold-aware models...")
scaffold_test = np.zeros(len(test_ids))
n_scaffold = 0

for cfg_idx, cfg in enumerate(configs):
    for seed_i in range(N_SEEDS):
        seed = SEED + seed_i * 111 + cfg_idx * 1000 + 99999
        model = lgb.LGBMRegressor(
            objective="regression", metric="rmse",
            n_estimators=5000, verbose=-1, random_state=seed, **cfg
        )
        model.fit(
            X_train[sc_train_idx], y_train[sc_train_idx],
            eval_set=[(X_train[sc_val_idx], y_train[sc_val_idx])],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
        )
        scaffold_test += model.predict(X_test)
        n_scaffold += 1

scaffold_test /= n_scaffold
print(f"  Scaffold-aware: {n_scaffold} models trained")

# Validate on scaffold val
sc_val_pred = np.zeros(len(sc_val_idx))
for cfg_idx, cfg in enumerate(configs):
    for seed_i in range(N_SEEDS):
        seed = SEED + seed_i * 111 + cfg_idx * 1000 + 99999
        model = lgb.LGBMRegressor(
            objective="regression", metric="rmse",
            n_estimators=5000, verbose=-1, random_state=seed, **cfg
        )
        model.fit(
            X_train[sc_train_idx], y_train[sc_train_idx],
            eval_set=[(X_train[sc_val_idx], y_train[sc_val_idx])],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
        )
        sc_val_pred += model.predict(X_train[sc_val_idx])
sc_val_pred /= n_scaffold

rmse_sc = math.sqrt(np.mean((y_train[sc_val_idx] - sc_val_pred) ** 2))
sp_sc, _ = spearmanr(y_train[sc_val_idx], sc_val_pred)
print(f"  Scaffold-val: RMSE={rmse_sc:.4f}, Spearman={sp_sc:.4f}")

# ============================================================
# 7. Ensemble optimization
# ============================================================
print("\n" + "=" * 60)
print("Optimizing ensemble...")

from scipy.optimize import minimize

oof_preds = np.column_stack([lgb_oof, ridge_oof])
test_preds_all = np.column_stack([lgb_test, ridge_test, scaffold_test])

# For test, we have 3 sources: lgb_cv, ridge_cv, scaffold_models
# For OOF, we only have lgb_cv and ridge_cv
# Optimize weights on OOF, then blend with scaffold

def neg_score(w):
    w = np.abs(w)
    w = w / w.sum()
    pred = oof_preds @ w
    rmse = math.sqrt(np.mean((y_train - pred) ** 2))
    sp, _ = spearmanr(y_train, pred)
    return -(500 * max(0, 1 - rmse / 0.80) + 250 * max(0, sp))

best_score = float('inf')
best_w = None
for _ in range(500):
    w0 = np.random.dirichlet(np.ones(2))
    res = minimize(neg_score, w0, method='Nelder-Mead', options={'maxiter': 3000})
    if res.fun < best_score:
        best_score = res.fun
        best_w = np.abs(res.x)
        best_w = best_w / best_w.sum()

print(f"OOF weights: LGB={best_w[0]:.3f}, Ridge={best_w[1]:.3f}")

# OOF blended prediction
oof_blend = oof_preds @ best_w
rmse_blend = math.sqrt(np.mean((y_train - oof_blend) ** 2))
sp_blend, _ = spearmanr(y_train, oof_blend)
print(f"OOF blend: RMSE={rmse_blend:.4f}, Spearman={sp_blend:.4f}")

# Final test: blend CV models with scaffold models
# Use 70% CV blend + 30% scaffold (scaffold models are more robust to distribution shift)
test_cv_blend = best_w[0] * lgb_test + best_w[1] * ridge_test
final_preds = 0.6 * test_cv_blend + 0.4 * scaffold_test

print(f"\nFinal predictions: mean={final_preds.mean():.4f}, std={final_preds.std():.4f}")

# Estimated score
raw_score = 500 * max(0, 1 - rmse_blend / 0.80) + 250 * max(0, sp_blend) + 50
if rmse_blend <= 0.25 and sp_blend >= 0.96:
    est_score = 800
else:
    est_score = raw_score
print(f"Estimated score (from OOF): {est_score:.1f}")

# ============================================================
# 8. Save submission
# ============================================================
print("\n" + "=" * 60)
os.makedirs("submission", exist_ok=True)
results_path = WORK_DIR / "submission" / "results.csv"
with open(results_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "prediction"])
    for tid, pred in zip(test_ids, final_preds):
        writer.writerow([tid, f"{pred:.6f}"])

import subprocess
result = subprocess.run(
    ["python3", "tools/check_format.py", str(results_path), "--test-csv", "test.csv"],
    capture_output=True, text=True
)
print(f"Format check: {result.stdout.strip()}")

zip_path = WORK_DIR / "NS-2026-04-answer.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write(results_path, "results.csv")

import hashlib
with open(results_path, "rb") as f:
    sha = hashlib.sha256(f.read()).hexdigest()
print(f"SHA256: {sha}")
print(f"Created: {zip_path}")
print("\nDone! Submit NS-2026-04-answer.zip")
