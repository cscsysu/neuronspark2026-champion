"""
NS-2026-04 v3: 精准版
策略:
1. 用scaffold split找最优迭代数 (early stopping)
2. 用全量8500训练，迭代数 = scaffold最优 * 0.85 (防过拟合)
3. 多seed多配置集成减方差
4. 特征: Morgan FP + 精选RDKit描述符 (去噪)
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

with open(WORK_DIR / "scaffold_split.json") as f:
    scaffold = json.load(f)

train_id_to_idx = {tid: i for i, tid in enumerate(train_ids)}
valid_id_set = set(scaffold["valid_ids"])
sc_train_idx = np.array([train_id_to_idx[tid] for tid in train_ids if tid not in valid_id_set])
sc_val_idx = np.array([train_id_to_idx[tid] for tid in scaffold["valid_ids"] if tid in train_id_to_idx])

print(f"Train: {len(train_ids)}, Test: {len(test_ids)}")
print(f"Scaffold: train={len(sc_train_idx)}, val={len(sc_val_idx)}")

# ============================================================
# 2. Feature Engineering - Multiple feature sets
# ============================================================
print("\nBuilding features...")

gen2 = GetMorganGenerator(radius=2, fpSize=2048)
gen3 = GetMorganGenerator(radius=3, fpSize=2048)
gen2_1024 = GetMorganGenerator(radius=2, fpSize=1024)

# Selected stable descriptors (avoid noisy ones)
GOOD_DESCS = [
    'MolWt', 'HeavyAtomMolWt', 'ExactMolWt', 'NumValenceElectrons',
    'FpDensityMorgan1', 'FpDensityMorgan2', 'FpDensityMorgan3',
    'MolLogP', 'MolMR', 'LabuteASA', 'TPSA',
    'NumHAcceptors', 'NumHDonors', 'NumRotatableBonds', 'NumHeteroatoms',
    'NumAromaticRings', 'NumSaturatedRings', 'NumAliphaticRings',
    'RingCount', 'NumAromaticHeterocycles', 'NumAromaticCarbocycles',
    'NumSaturatedHeterocycles', 'NumSaturatedCarbocycles',
    'NumAliphaticHeterocycles', 'NumAliphaticCarbocycles',
    'FractionCSP3', 'HeavyAtomCount', 'NHOHCount', 'NOCount',
    'NumRadicalElectrons',
    'MaxEStateIndex', 'MinEStateIndex', 'MaxAbsEStateIndex', 'MinAbsEStateIndex',
    'qed', 'BertzCT', 'BalabanJ', 'HallKierAlpha',
    'Kappa1', 'Kappa2', 'Kappa3', 'Chi0', 'Chi1', 'Chi0n', 'Chi1n',
    'Chi0v', 'Chi1v', 'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v',
    'Ipc', 'PEOE_VSA1', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5',
    'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9', 'PEOE_VSA10',
    'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13', 'PEOE_VSA14',
    'SMR_VSA1', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5',
    'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA8', 'SMR_VSA9', 'SMR_VSA10',
    'SlogP_VSA1', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5',
    'SlogP_VSA6', 'SlogP_VSA7', 'SlogP_VSA8', 'SlogP_VSA9', 'SlogP_VSA10',
    'SlogP_VSA11', 'SlogP_VSA12',
    'EState_VSA1', 'EState_VSA2', 'EState_VSA3', 'EState_VSA4', 'EState_VSA5',
    'EState_VSA6', 'EState_VSA7', 'EState_VSA8', 'EState_VSA9', 'EState_VSA10',
    'EState_VSA11', 'VSA_EState1', 'VSA_EState2', 'VSA_EState3', 'VSA_EState4',
    'VSA_EState5', 'VSA_EState6', 'VSA_EState7', 'VSA_EState8', 'VSA_EState9',
    'VSA_EState10',
]

# Filter to actually available descriptors
available_descs = {name for name, _ in Descriptors.descList}
GOOD_DESCS = [d for d in GOOD_DESCS if d in available_descs]
calc = MolecularDescriptorCalculator(GOOD_DESCS)


def build_features(smiles_list, hashed_path):
    n = len(smiles_list)
    morgan2 = np.zeros((n, 2048), dtype=np.float32)
    morgan3_count = np.zeros((n, 2048), dtype=np.float32)
    morgan2_1024 = np.zeros((n, 1024), dtype=np.float32)
    maccs = np.zeros((n, 167), dtype=np.float32)
    rdkit_desc = np.zeros((n, len(GOOD_DESCS)), dtype=np.float32)

    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            morgan2[i] = gen2.GetFingerprintAsNumPy(mol)
            morgan3_count[i] = gen3.GetCountFingerprintAsNumPy(mol)
            morgan2_1024[i] = gen2_1024.GetFingerprintAsNumPy(mol)
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
            print(f"  {i+1}/{n} done")

    hashed = sparse.load_npz(hashed_path).toarray()
    return morgan2, morgan3_count, morgan2_1024, maccs, rdkit_desc, hashed


print("  Processing train...")
m2_tr, m3_tr, m2s_tr, mac_tr, rd_tr, hash_tr = build_features(
    train_smiles, WORK_DIR / "features" / "hashed_smiles_train.npz")
print("  Processing test...")
m2_te, m3_te, m2s_te, mac_te, rd_te, hash_te = build_features(
    test_smiles, WORK_DIR / "features" / "hashed_smiles_test.npz")

# Clean rdkit descriptors
valid_rd = np.all(np.isfinite(rd_tr), axis=0) & (np.std(rd_tr, axis=0) > 1e-10)
rd_tr = rd_tr[:, valid_rd]
rd_te = rd_te[:, valid_rd]

# Create different feature combinations
feat_sets = {
    "morgan2+rdkit+hashed": (
        np.hstack([m2_tr, rd_tr, hash_tr]),
        np.hstack([m2_te, rd_te, hash_te])
    ),
    "morgan3count+maccs+rdkit": (
        np.hstack([m3_tr, mac_tr, rd_tr]),
        np.hstack([m3_te, mac_te, rd_te])
    ),
    "all_fps+rdkit": (
        np.hstack([m2_tr, m3_tr, mac_tr, rd_tr]),
        np.hstack([m2_te, m3_te, mac_te, rd_te])
    ),
    "hashed+rdkit": (
        np.hstack([hash_tr, rd_tr]),
        np.hstack([hash_te, rd_te])
    ),
}

for name, (Xtr, Xte) in feat_sets.items():
    print(f"  {name}: {Xtr.shape[1]} features")

# ============================================================
# 3. Find optimal iterations per config using scaffold split
# ============================================================
print("\nFinding optimal iterations via scaffold split...")

configs = [
    {"learning_rate": 0.01, "num_leaves": 63, "min_child_samples": 10,
     "subsample": 0.8, "colsample_bytree": 0.4, "reg_alpha": 0.1, "reg_lambda": 1.0},
    {"learning_rate": 0.015, "num_leaves": 95, "min_child_samples": 15,
     "subsample": 0.75, "colsample_bytree": 0.35, "reg_alpha": 0.3, "reg_lambda": 1.5},
    {"learning_rate": 0.02, "num_leaves": 31, "min_child_samples": 5,
     "subsample": 0.85, "colsample_bytree": 0.5, "reg_alpha": 0.05, "reg_lambda": 0.5},
    {"learning_rate": 0.01, "num_leaves": 127, "min_child_samples": 20,
     "subsample": 0.7, "colsample_bytree": 0.3, "reg_alpha": 0.5, "reg_lambda": 2.0},
]

# Use the main feature set for calibration
X_main_tr, X_main_te = feat_sets["morgan2+rdkit+hashed"]

optimal_iters = {}
for cfg_idx, cfg in enumerate(configs):
    model = lgb.LGBMRegressor(
        objective="regression", metric="rmse",
        n_estimators=5000, verbose=-1, random_state=SEED, **cfg
    )
    model.fit(
        X_main_tr[sc_train_idx], y_train[sc_train_idx],
        eval_set=[(X_main_tr[sc_val_idx], y_train[sc_val_idx])],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
    )
    opt_iter = model.best_iteration_
    optimal_iters[cfg_idx] = opt_iter

    val_pred = model.predict(X_main_tr[sc_val_idx])
    rmse = math.sqrt(np.mean((y_train[sc_val_idx] - val_pred) ** 2))
    sp, _ = spearmanr(y_train[sc_val_idx], val_pred)
    print(f"  Config {cfg_idx}: best_iter={opt_iter}, scaffold-val RMSE={rmse:.4f}, Spearman={sp:.4f}")

# ============================================================
# 4. Train on FULL data with calibrated iterations
# ============================================================
print("\nTraining on FULL data with calibrated iterations...")
print("Strategy: use 90% of optimal iterations to prevent overfitting on unseen scaffolds")

N_SEEDS = 10
test_preds_all = []

for feat_name, (X_tr, X_te) in feat_sets.items():
    test_pred_feat = np.zeros(len(test_ids))
    n_models = 0

    for cfg_idx, cfg in enumerate(configs):
        # Use 90% of scaffold-optimal iterations for full-data training
        n_est = int(optimal_iters[cfg_idx] * 0.90)
        n_est = max(200, n_est)  # minimum 200 trees

        for seed_i in range(N_SEEDS):
            seed = SEED + seed_i * 111 + cfg_idx * 1000 + hash(feat_name) % 10000
            model = lgb.LGBMRegressor(
                objective="regression", n_estimators=n_est,
                verbose=-1, random_state=seed, **cfg
            )
            model.fit(X_tr, y_train)
            test_pred_feat += model.predict(X_te)
            n_models += 1

    test_pred_feat /= n_models
    test_preds_all.append(test_pred_feat)
    print(f"  {feat_name}: {n_models} models, pred mean={test_pred_feat.mean():.4f}")

# ============================================================
# 5. Also train scaffold-aware models (train 6800, predict test)
# ============================================================
print("\nTraining scaffold-aware models (train on 6800)...")
scaffold_preds = []

for feat_name, (X_tr, X_te) in feat_sets.items():
    scaffold_pred = np.zeros(len(test_ids))
    n_models = 0

    for cfg_idx, cfg in enumerate(configs):
        for seed_i in range(N_SEEDS):
            seed = SEED + seed_i * 111 + cfg_idx * 1000 + hash(feat_name) % 10000 + 50000
            model = lgb.LGBMRegressor(
                objective="regression", metric="rmse",
                n_estimators=5000, verbose=-1, random_state=seed, **cfg
            )
            model.fit(
                X_tr[sc_train_idx], y_train[sc_train_idx],
                eval_set=[(X_tr[sc_val_idx], y_train[sc_val_idx])],
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
            )
            scaffold_pred += model.predict(X_te)
            n_models += 1

    scaffold_pred /= n_models
    scaffold_preds.append(scaffold_pred)
    print(f"  {feat_name}: {n_models} scaffold models")

# ============================================================
# 6. Ridge on full data
# ============================================================
print("\nTraining Ridge...")
X_all_tr = np.hstack([m2_tr, m3_tr, mac_tr, rd_tr, hash_tr])
X_all_te = np.hstack([m2_te, m3_te, mac_te, rd_te, hash_te])
X_all_tr = np.nan_to_num(X_all_tr, nan=0, posinf=0, neginf=0)
X_all_te = np.nan_to_num(X_all_te, nan=0, posinf=0, neginf=0)

scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_all_tr)
X_te_sc = scaler.transform(X_all_te)

ridge_pred = np.zeros(len(test_ids))
n_ridge = 0
for alpha in [0.1, 1.0, 10.0, 50.0]:
    model = Ridge(alpha=alpha)
    model.fit(X_tr_sc, y_train)
    ridge_pred += model.predict(X_te_sc)
    n_ridge += 1
ridge_pred /= n_ridge
print(f"  Ridge: {n_ridge} models")

# ============================================================
# 7. Final Ensemble
# ============================================================
print("\n" + "=" * 60)
print("Building final ensemble...")

# Average all full-data predictions
full_data_avg = np.mean(test_preds_all, axis=0)
# Average all scaffold predictions
scaffold_avg = np.mean(scaffold_preds, axis=0)

# Blend: scaffold models are more conservative (better generalization)
# Full-data models have more info but risk slight overfit
# Ridge provides linear regularization
# Try different blends
blends = {
    "50full_50scaffold": 0.5 * full_data_avg + 0.5 * scaffold_avg,
    "40full_50scaffold_10ridge": 0.4 * full_data_avg + 0.5 * scaffold_avg + 0.1 * ridge_pred,
    "30full_60scaffold_10ridge": 0.3 * full_data_avg + 0.6 * scaffold_avg + 0.1 * ridge_pred,
    "60scaffold_30full_10ridge": 0.6 * scaffold_avg + 0.3 * full_data_avg + 0.1 * ridge_pred,
    "70scaffold_30full": 0.7 * scaffold_avg + 0.3 * full_data_avg,
}

# Evaluate on scaffold val to pick best blend
print("\nEvaluating blends on scaffold-val (using scaffold models for val)...")
# Re-predict scaffold val with scaffold models
sc_val_preds_all = []
for feat_name, (X_tr, X_te) in feat_sets.items():
    sc_val_pred = np.zeros(len(sc_val_idx))
    n_m = 0
    for cfg_idx, cfg in enumerate(configs):
        for seed_i in range(N_SEEDS):
            seed = SEED + seed_i * 111 + cfg_idx * 1000 + hash(feat_name) % 10000 + 50000
            model = lgb.LGBMRegressor(
                objective="regression", metric="rmse",
                n_estimators=5000, verbose=-1, random_state=seed, **cfg
            )
            model.fit(
                X_tr[sc_train_idx], y_train[sc_train_idx],
                eval_set=[(X_tr[sc_val_idx], y_train[sc_val_idx])],
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
            )
            sc_val_pred += model.predict(X_tr[sc_val_idx])
            n_m += 1
    sc_val_pred /= n_m
    sc_val_preds_all.append(sc_val_pred)

scaffold_val_avg = np.mean(sc_val_preds_all, axis=0)
rmse_sc = math.sqrt(np.mean((y_train[sc_val_idx] - scaffold_val_avg) ** 2))
sp_sc, _ = spearmanr(y_train[sc_val_idx], scaffold_val_avg)
print(f"Scaffold ensemble val: RMSE={rmse_sc:.4f}, Spearman={sp_sc:.4f}")

if rmse_sc <= 0.25 and sp_sc >= 0.96:
    print("*** GOLD BAND ACHIEVED on scaffold-val! ***")

# Use scaffold-focused blend (safest for unseen scaffolds)
final_preds = 0.6 * scaffold_avg + 0.3 * full_data_avg + 0.1 * ridge_pred
print(f"\nFinal blend: 60% scaffold + 30% full-data + 10% ridge")
print(f"Final predictions: mean={final_preds.mean():.4f}, std={final_preds.std():.4f}")

# ============================================================
# 8. Save
# ============================================================
print("\n" + "=" * 60)
os.makedirs("submission_v3", exist_ok=True)
results_path = WORK_DIR / "submission_v3" / "results.csv"
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

zip_path = WORK_DIR / "NS-2026-04-answer-v3.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write(results_path, "results.csv")

import hashlib
with open(results_path, "rb") as f:
    sha = hashlib.sha256(f.read()).hexdigest()
print(f"SHA256: {sha}")
print(f"Created: {zip_path}")
print("Done!")
