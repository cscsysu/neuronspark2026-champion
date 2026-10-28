"""
NS-2026-04 分子炼金术：高分方案 (GPU服务器版)
目标: RMSE <= 0.25, Spearman >= 0.96 -> 满分800

方案:
1. RDKit Morgan fingerprint + 分子描述符
2. LightGBM + XGBoost 多seed集成
3. PyTorch GNN (AttentiveFP / GIN)
4. 最终加权集成

运行前安装:
pip install numpy scipy scikit-learn lightgbm xgboost pandas rdkit-pypi torch torch-geometric

Usage:
python solve_gpu.py
"""

import csv
import json
import math
import re
import os
import hashlib
from collections import Counter
from pathlib import Path
from typing import List

import numpy as np
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import lightgbm as lgb
import xgboost as xgb

# ============================================================
# Configuration
# ============================================================
WORK_DIR = Path(".")  # 在数据目录下运行
SEED = 42
N_SEEDS = 10  # 多seed集成
N_FOLDS = 5
USE_GNN = True  # 是否使用GNN

np.random.seed(SEED)

# ============================================================
# 1. Load data
# ============================================================
print("=" * 60)
print("Loading data...")
print("=" * 60)

with open(WORK_DIR / "train.csv") as f:
    train_rows = list(csv.DictReader(f))
train_ids = [r["id"] for r in train_rows]
train_smiles = [r["smiles"] for r in train_rows]
y_train = np.array([float(r["target"]) for r in train_rows])

with open(WORK_DIR / "test.csv") as f:
    test_rows = list(csv.DictReader(f))
test_ids = [r["id"] for r in test_rows]
test_smiles = [r["smiles"] for r in test_rows]

# Load scaffold split
with open(WORK_DIR / "scaffold_split.json") as f:
    scaffold = json.load(f)
train_id_to_idx = {tid: i for i, tid in enumerate(train_ids)}
sc_train_idx = np.array([train_id_to_idx[tid] for tid in scaffold["train_ids"] if tid in train_id_to_idx])
sc_val_idx = np.array([train_id_to_idx[tid] for tid in scaffold["valid_ids"] if tid in train_id_to_idx])

print(f"Train: {len(train_ids)}, Test: {len(test_ids)}")
print(f"Scaffold split: train={len(sc_train_idx)}, val={len(sc_val_idx)}")
print(f"Target: mean={y_train.mean():.4f}, std={y_train.std():.4f}")

# ============================================================
# 2. Feature Engineering with RDKit
# ============================================================
print("\n" + "=" * 60)
print("Feature Engineering with RDKit...")
print("=" * 60)

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, DataStructs, MACCSkeys
    from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
    from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
    # Suppress deprecation warnings
    RDLogger.DisableLog('rdApp.*')
    HAS_RDKIT = True
    print("RDKit available!")
except ImportError:
    HAS_RDKIT = False
    print("RDKit not available, using fallback features")


def get_morgan_fingerprints(smiles_list, radius=2, n_bits=2048):
    """Get Morgan fingerprints using new Generator API."""
    gen = GetMorganGenerator(radius=radius, fpSize=n_bits)
    fps = np.zeros((len(smiles_list), n_bits), dtype=np.float32)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = gen.GetFingerprintAsNumPy(mol)
            fps[i] = fp
    return fps


def get_morgan_count_fingerprints(smiles_list, radius=2, n_bits=2048):
    """Get Morgan count fingerprints using new Generator API."""
    gen = GetMorganGenerator(radius=radius, fpSize=n_bits)
    fps = np.zeros((len(smiles_list), n_bits), dtype=np.float32)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = gen.GetCountFingerprintAsNumPy(mol)
            fps[i] = fp
    return fps


def get_maccs_keys(smiles_list):
    """Get MACCS keys fingerprints (167 bits)."""
    fps = np.zeros((len(smiles_list), 167), dtype=np.float32)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = MACCSkeys.GenMACCSKeys(mol)
            arr = np.zeros(167)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fps[i] = arr
    return fps


def get_rdkit_descriptors(smiles_list):
    """Calculate 200+ RDKit molecular descriptors."""
    desc_names = [name for name, _ in Descriptors.descList]
    calc = MolecularDescriptorCalculator(desc_names)
    
    features = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            features.append([0.0] * len(desc_names))
        else:
            try:
                desc = calc.CalcDescriptors(mol)
                # Replace inf/nan with 0
                desc = [0.0 if (math.isnan(x) or math.isinf(x)) else x for x in desc]
                features.append(desc)
            except:
                features.append([0.0] * len(desc_names))
    
    return np.array(features, dtype=np.float32), desc_names


# Fallback: SMILES-based features (same as before)
TOKEN_RE = re.compile(r"Cl|Br|Si|Na|Li|Mg|Ca|Al|@@?|\[[^\]]+\]|[A-Za-z]|\d|\(|\)|=|#|-|\+|\\|/|\.|:")
ATOM_RE = re.compile(r"Cl|Br|Si|Na|Li|Mg|Ca|Al|[BCNOPSFIKH]|[bcnops]")

def compute_simple_descriptors(smiles_list):
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


# Build feature matrix
if HAS_RDKIT:
    print("Computing Morgan fingerprints (radius=2, 2048 bits)...")
    X_train_morgan = get_morgan_fingerprints(train_smiles, radius=2, n_bits=2048)
    X_test_morgan = get_morgan_fingerprints(test_smiles, radius=2, n_bits=2048)
    
    print("Computing Morgan count fingerprints (radius=3, 2048 bits)...")
    X_train_morgan3 = get_morgan_count_fingerprints(train_smiles, radius=3, n_bits=2048)
    X_test_morgan3 = get_morgan_count_fingerprints(test_smiles, radius=3, n_bits=2048)
    
    print("Computing MACCS keys...")
    X_train_maccs = get_maccs_keys(train_smiles)
    X_test_maccs = get_maccs_keys(test_smiles)
    
    print("Computing RDKit molecular descriptors...")
    X_train_rdkit, rdkit_names = get_rdkit_descriptors(train_smiles)
    X_test_rdkit, _ = get_rdkit_descriptors(test_smiles)
    
    # Remove constant/nan columns from rdkit descriptors
    valid_cols = []
    for i in range(X_train_rdkit.shape[1]):
        col = X_train_rdkit[:, i]
        if np.std(col) > 1e-10 and not np.any(np.isnan(col)):
            valid_cols.append(i)
    X_train_rdkit = X_train_rdkit[:, valid_cols]
    X_test_rdkit = X_test_rdkit[:, valid_cols]
    print(f"  Valid RDKit descriptors: {len(valid_cols)}")
    
    # Also load hashed features as additional signal
    X_train_hashed = sparse.load_npz(WORK_DIR / "features" / "hashed_smiles_train.npz").toarray()
    X_test_hashed = sparse.load_npz(WORK_DIR / "features" / "hashed_smiles_test.npz").toarray()
    
    # Simple descriptors
    X_train_simple = compute_simple_descriptors(train_smiles)
    X_test_simple = compute_simple_descriptors(test_smiles)
    
    # Combine all features
    X_train = np.hstack([X_train_morgan, X_train_morgan3, X_train_maccs, 
                         X_train_rdkit, X_train_hashed, X_train_simple])
    X_test = np.hstack([X_test_morgan, X_test_morgan3, X_test_maccs, 
                        X_test_rdkit, X_test_hashed, X_test_simple])
else:
    # Fallback: use provided hashed features + simple descriptors
    X_train_hashed = sparse.load_npz(WORK_DIR / "features" / "hashed_smiles_train.npz").toarray()
    X_test_hashed = sparse.load_npz(WORK_DIR / "features" / "hashed_smiles_test.npz").toarray()
    X_train_simple = compute_simple_descriptors(train_smiles)
    X_test_simple = compute_simple_descriptors(test_smiles)
    X_train = np.hstack([X_train_hashed, X_train_simple])
    X_test = np.hstack([X_test_hashed, X_test_simple])

print(f"Total feature matrix: train={X_train.shape}, test={X_test.shape}")

# Replace any remaining nan/inf
X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)


# ============================================================
# 3. Tree Model Ensemble (LightGBM + XGBoost)
# ============================================================
print("\n" + "=" * 60)
print("Training Tree Model Ensemble...")
print("=" * 60)

def eval_metrics(y_true, y_pred):
    rmse = math.sqrt(np.mean((y_true - y_pred) ** 2))
    sp, _ = spearmanr(y_true, y_pred)
    return rmse, sp


# --- LightGBM multi-seed, multi-fold ---
print("\n--- LightGBM Ensemble ---")
lgb_test_preds = np.zeros(len(test_ids))
lgb_oof_preds = np.zeros(len(train_ids))

lgb_params_list = [
    # Config 1: default
    {"learning_rate": 0.02, "num_leaves": 63, "min_child_samples": 10,
     "subsample": 0.8, "colsample_bytree": 0.5, "reg_alpha": 0.1, "reg_lambda": 1.0},
    # Config 2: deeper
    {"learning_rate": 0.02, "num_leaves": 127, "min_child_samples": 20,
     "subsample": 0.7, "colsample_bytree": 0.4, "reg_alpha": 0.5, "reg_lambda": 2.0},
    # Config 3: shallower, more regularized
    {"learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 5,
     "subsample": 0.85, "colsample_bytree": 0.6, "reg_alpha": 0.05, "reg_lambda": 0.5},
]

n_lgb_models = 0
for cfg_idx, lgb_cfg in enumerate(lgb_params_list):
    for seed_i in range(N_SEEDS):
        seed = SEED + seed_i * 111 + cfg_idx * 1000
        
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train)):
            model = lgb.LGBMRegressor(
                objective="regression",
                metric="rmse",
                boosting_type="gbdt",
                n_estimators=5000,
                verbose=-1,
                random_state=seed,
                **lgb_cfg
            )
            model.fit(
                X_train[tr_idx], y_train[tr_idx],
                eval_set=[(X_train[va_idx], y_train[va_idx])],
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
            )
            va_pred = model.predict(X_train[va_idx])
            lgb_oof_preds[va_idx] += va_pred
            lgb_test_preds += model.predict(X_test)
            n_lgb_models += 1
        
    print(f"  Config {cfg_idx}, seed batch {seed_i+1}/{N_SEEDS} done")

lgb_oof_preds /= (N_SEEDS * len(lgb_params_list))
lgb_test_preds /= n_lgb_models

rmse_lgb, sp_lgb = eval_metrics(y_train, lgb_oof_preds)
print(f"LightGBM OOF: RMSE={rmse_lgb:.4f}, Spearman={sp_lgb:.4f}")

# --- XGBoost multi-seed, multi-fold ---
print("\n--- XGBoost Ensemble ---")
xgb_test_preds = np.zeros(len(test_ids))
xgb_oof_preds = np.zeros(len(train_ids))

xgb_params_list = [
    {"learning_rate": 0.02, "max_depth": 6, "min_child_weight": 5,
     "subsample": 0.8, "colsample_bytree": 0.5, "reg_alpha": 0.1, "reg_lambda": 1.0},
    {"learning_rate": 0.02, "max_depth": 8, "min_child_weight": 10,
     "subsample": 0.7, "colsample_bytree": 0.4, "reg_alpha": 0.5, "reg_lambda": 2.0},
]

n_xgb_models = 0
for cfg_idx, xgb_cfg in enumerate(xgb_params_list):
    for seed_i in range(N_SEEDS):
        seed = SEED + seed_i * 111 + cfg_idx * 1000 + 50000
        
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train)):
            model = xgb.XGBRegressor(
                objective="reg:squarederror",
                n_estimators=5000,
                early_stopping_rounds=100,
                verbosity=0,
                random_state=seed,
                tree_method="hist",
                device="cuda",
                **xgb_cfg
            )
            model.fit(
                X_train[tr_idx], y_train[tr_idx],
                eval_set=[(X_train[va_idx], y_train[va_idx])],
                verbose=False
            )
            va_pred = model.predict(X_train[va_idx])
            xgb_oof_preds[va_idx] += va_pred
            xgb_test_preds += model.predict(X_test)
            n_xgb_models += 1
        
    print(f"  Config {cfg_idx}, seed batch {seed_i+1}/{N_SEEDS} done")

xgb_oof_preds /= (N_SEEDS * len(xgb_params_list))
xgb_test_preds /= n_xgb_models

rmse_xgb, sp_xgb = eval_metrics(y_train, xgb_oof_preds)
print(f"XGBoost OOF: RMSE={rmse_xgb:.4f}, Spearman={sp_xgb:.4f}")

# --- Ridge ---
print("\n--- Ridge Regression ---")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

ridge_oof_preds = np.zeros(len(train_ids))
ridge_test_preds = np.zeros(len(test_ids))
n_ridge = 0

for alpha in [1.0, 5.0, 10.0, 50.0, 100.0]:
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train_scaled)):
        model = Ridge(alpha=alpha)
        model.fit(X_train_scaled[tr_idx], y_train[tr_idx])
        ridge_oof_preds[va_idx] += model.predict(X_train_scaled[va_idx])
        ridge_test_preds += model.predict(X_test_scaled)
        n_ridge += 1

ridge_oof_preds /= 5  # 5 alphas
ridge_test_preds /= n_ridge

rmse_ridge, sp_ridge = eval_metrics(y_train, ridge_oof_preds)
print(f"Ridge OOF: RMSE={rmse_ridge:.4f}, Spearman={sp_ridge:.4f}")


# ============================================================
# 4. GNN Model (if PyTorch available)
# ============================================================
gnn_test_preds = None
gnn_oof_preds = None

if USE_GNN:
    print("\n" + "=" * 60)
    print("Training GNN Model...")
    print("=" * 60)
    
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.optim import Adam
        from torch.optim.lr_scheduler import CosineAnnealingLR
        
        # Try torch_geometric
        try:
            from torch_geometric.data import Data, DataLoader as PyGDataLoader
            from torch_geometric.nn import AttentiveFP, GINConv, global_add_pool, global_mean_pool
            HAS_PYG = True
        except ImportError:
            HAS_PYG = False
            print("torch_geometric not available, using simple MLP on fingerprints")
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        if HAS_PYG and HAS_RDKIT:
            # Build molecular graphs
            print("Building molecular graphs...")
            
            ATOM_FEATURES = {
                'atomic_num': list(range(1, 119)),
                'degree': [0, 1, 2, 3, 4, 5],
                'formal_charge': [-2, -1, 0, 1, 2],
                'num_hs': [0, 1, 2, 3, 4],
                'hybridization': [
                    Chem.rdchem.HybridizationType.SP,
                    Chem.rdchem.HybridizationType.SP2,
                    Chem.rdchem.HybridizationType.SP3,
                    Chem.rdchem.HybridizationType.SP3D,
                    Chem.rdchem.HybridizationType.SP3D2,
                ],
                'is_aromatic': [False, True],
            }
            
            def one_hot(value, choices):
                encoding = [0] * (len(choices) + 1)
                idx = choices.index(value) if value in choices else len(choices)
                encoding[idx] = 1
                return encoding
            
            def atom_features(atom):
                return (
                    one_hot(atom.GetAtomicNum(), ATOM_FEATURES['atomic_num']) +
                    one_hot(atom.GetDegree(), ATOM_FEATURES['degree']) +
                    one_hot(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']) +
                    one_hot(atom.GetTotalNumHs(), ATOM_FEATURES['num_hs']) +
                    one_hot(atom.GetHybridization(), ATOM_FEATURES['hybridization']) +
                    one_hot(atom.GetIsAromatic(), ATOM_FEATURES['is_aromatic'])
                )
            
            def bond_features(bond):
                bt = bond.GetBondType()
                return [
                    bt == Chem.rdchem.BondType.SINGLE,
                    bt == Chem.rdchem.BondType.DOUBLE,
                    bt == Chem.rdchem.BondType.TRIPLE,
                    bt == Chem.rdchem.BondType.AROMATIC,
                    bond.GetIsConjugated(),
                    bond.IsInRing(),
                ]
            
            def smiles_to_graph(smiles):
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    # Return a dummy graph
                    return Data(
                        x=torch.zeros(1, 139, dtype=torch.float),
                        edge_index=torch.zeros(2, 0, dtype=torch.long),
                        edge_attr=torch.zeros(0, 6, dtype=torch.float),
                    )
                
                # Atom features
                x = []
                for atom in mol.GetAtoms():
                    x.append(atom_features(atom))
                x = torch.tensor(x, dtype=torch.float)
                
                # Edge features
                edge_indices = []
                edge_attrs = []
                for bond in mol.GetBonds():
                    i = bond.GetBeginAtomIdx()
                    j = bond.GetEndAtomIdx()
                    bf = bond_features(bond)
                    edge_indices.extend([[i, j], [j, i]])
                    edge_attrs.extend([bf, bf])
                
                if edge_indices:
                    edge_index = torch.tensor(edge_indices, dtype=torch.long).t()
                    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
                else:
                    edge_index = torch.zeros(2, 0, dtype=torch.long)
                    edge_attr = torch.zeros(0, 6, dtype=torch.float)
                
                return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            
            # Build all graphs
            all_smiles = train_smiles + test_smiles
            all_graphs = []
            for i, smi in enumerate(all_smiles):
                g = smiles_to_graph(smi)
                all_graphs.append(g)
                if (i + 1) % 2000 == 0:
                    print(f"  Processed {i+1}/{len(all_smiles)} molecules")
            
            train_graphs = all_graphs[:len(train_smiles)]
            test_graphs = all_graphs[len(train_smiles):]
            
            # Set targets for train graphs
            for i, g in enumerate(train_graphs):
                g.y = torch.tensor([y_train[i]], dtype=torch.float)
            
            atom_feat_dim = train_graphs[0].x.shape[1]
            edge_feat_dim = 6
            
            print(f"  Atom feature dim: {atom_feat_dim}")
            print(f"  Edge feature dim: {edge_feat_dim}")
            
            # GNN Model
            class MolGNN(nn.Module):
                def __init__(self, atom_dim, edge_dim, hidden=256, heads=4, dropout=0.2):
                    super().__init__()
                    # Use GIN layers (simpler, often better for molecular tasks)
                    self.atom_encoder = nn.Linear(atom_dim, hidden)
                    
                    self.convs = nn.ModuleList()
                    self.bns = nn.ModuleList()
                    for _ in range(4):
                        mlp = nn.Sequential(
                            nn.Linear(hidden, hidden),
                            nn.ReLU(),
                            nn.Linear(hidden, hidden),
                        )
                        self.convs.append(GINConv(mlp))
                        self.bns.append(nn.BatchNorm1d(hidden))
                    
                    self.dropout = nn.Dropout(dropout)
                    self.head = nn.Sequential(
                        nn.Linear(hidden * 2, hidden),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden, hidden // 2),
                        nn.ReLU(),
                        nn.Linear(hidden // 2, 1),
                    )
                
                def forward(self, data):
                    x, edge_index, batch = data.x, data.edge_index, data.batch
                    x = self.atom_encoder(x)
                    
                    for conv, bn in zip(self.convs, self.bns):
                        x = conv(x, edge_index)
                        x = bn(x)
                        x = F.relu(x)
                        x = self.dropout(x)
                    
                    # Combine sum and mean pooling
                    x_sum = global_add_pool(x, batch)
                    x_mean = global_mean_pool(x, batch)
                    x = torch.cat([x_sum, x_mean], dim=-1)
                    
                    return self.head(x).squeeze(-1)
            
            # Train GNN with K-fold
            print("\nTraining GNN...")
            gnn_oof_preds = np.zeros(len(train_ids))
            gnn_test_preds = np.zeros(len(test_ids))
            n_gnn_models = 0
            
            N_GNN_SEEDS = 3
            N_GNN_FOLDS = 5
            EPOCHS = 150
            BATCH_SIZE = 128
            LR = 1e-3
            
            for gnn_seed in range(N_GNN_SEEDS):
                seed_val = SEED + gnn_seed * 777
                torch.manual_seed(seed_val)
                np.random.seed(seed_val)
                
                kf = KFold(n_splits=N_GNN_FOLDS, shuffle=True, random_state=seed_val)
                for fold, (tr_idx, va_idx) in enumerate(kf.split(train_graphs)):
                    tr_data = [train_graphs[i] for i in tr_idx]
                    va_data = [train_graphs[i] for i in va_idx]
                    
                    tr_loader = PyGDataLoader(tr_data, batch_size=BATCH_SIZE, shuffle=True)
                    va_loader = PyGDataLoader(va_data, batch_size=BATCH_SIZE, shuffle=False)
                    test_loader = PyGDataLoader(test_graphs, batch_size=BATCH_SIZE, shuffle=False)
                    
                    model = MolGNN(atom_feat_dim, edge_feat_dim, hidden=256, dropout=0.2).to(device)
                    optimizer = Adam(model.parameters(), lr=LR, weight_decay=1e-5)
                    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
                    
                    best_val_rmse = float('inf')
                    best_state = None
                    patience = 30
                    no_improve = 0
                    
                    for epoch in range(EPOCHS):
                        # Train
                        model.train()
                        for batch in tr_loader:
                            batch = batch.to(device)
                            optimizer.zero_grad()
                            pred = model(batch)
                            loss = F.mse_loss(pred, batch.y)
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            optimizer.step()
                        scheduler.step()
                        
                        # Validate
                        if (epoch + 1) % 5 == 0:
                            model.eval()
                            va_preds = []
                            with torch.no_grad():
                                for batch in va_loader:
                                    batch = batch.to(device)
                                    va_preds.append(model(batch).cpu().numpy())
                            va_preds = np.concatenate(va_preds)
                            va_rmse = math.sqrt(np.mean((y_train[va_idx] - va_preds) ** 2))
                            
                            if va_rmse < best_val_rmse:
                                best_val_rmse = va_rmse
                                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                                no_improve = 0
                            else:
                                no_improve += 5
                            
                            if no_improve >= patience:
                                break
                    
                    # Load best model and predict
                    model.load_state_dict(best_state)
                    model.eval()
                    
                    # OOF predictions
                    va_preds = []
                    with torch.no_grad():
                        for batch in va_loader:
                            batch = batch.to(device)
                            va_preds.append(model(batch).cpu().numpy())
                    gnn_oof_preds[va_idx] += np.concatenate(va_preds)
                    
                    # Test predictions
                    test_preds_fold = []
                    with torch.no_grad():
                        for batch in test_loader:
                            batch = batch.to(device)
                            test_preds_fold.append(model(batch).cpu().numpy())
                    gnn_test_preds += np.concatenate(test_preds_fold)
                    n_gnn_models += 1
                    
                    print(f"  Seed {gnn_seed}, Fold {fold}: best_val_rmse={best_val_rmse:.4f}")
            
            gnn_oof_preds /= N_GNN_SEEDS
            gnn_test_preds /= n_gnn_models
            
            rmse_gnn, sp_gnn = eval_metrics(y_train, gnn_oof_preds)
            print(f"\nGNN OOF: RMSE={rmse_gnn:.4f}, Spearman={sp_gnn:.4f}")
        
        else:
            # MLP on fingerprints as fallback
            print("Using MLP on fingerprints (no PyG)...")
            
            class MolMLP(nn.Module):
                def __init__(self, input_dim, hidden=512, dropout=0.3):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(input_dim, hidden),
                        nn.BatchNorm1d(hidden),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden, hidden // 2),
                        nn.BatchNorm1d(hidden // 2),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden // 2, hidden // 4),
                        nn.ReLU(),
                        nn.Linear(hidden // 4, 1),
                    )
                
                def forward(self, x):
                    return self.net(x).squeeze(-1)
            
            # Normalize features
            scaler_nn = StandardScaler()
            X_train_nn = scaler_nn.fit_transform(X_train).astype(np.float32)
            X_test_nn = scaler_nn.transform(X_test).astype(np.float32)
            
            gnn_oof_preds = np.zeros(len(train_ids))
            gnn_test_preds = np.zeros(len(test_ids))
            n_nn_models = 0
            
            EPOCHS = 200
            BATCH_SIZE = 256
            
            for nn_seed in range(3):
                seed_val = SEED + nn_seed * 777
                torch.manual_seed(seed_val)
                
                kf = KFold(n_splits=5, shuffle=True, random_state=seed_val)
                for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train_nn)):
                    X_tr = torch.tensor(X_train_nn[tr_idx]).to(device)
                    y_tr = torch.tensor(y_train[tr_idx], dtype=torch.float32).to(device)
                    X_va = torch.tensor(X_train_nn[va_idx]).to(device)
                    X_te = torch.tensor(X_test_nn).to(device)
                    
                    model = MolMLP(X_train_nn.shape[1], hidden=1024, dropout=0.3).to(device)
                    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
                    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
                    
                    best_val_rmse = float('inf')
                    best_state = None
                    
                    for epoch in range(EPOCHS):
                        model.train()
                        # Mini-batch training
                        perm = torch.randperm(len(X_tr))
                        for i in range(0, len(X_tr), BATCH_SIZE):
                            idx = perm[i:i+BATCH_SIZE]
                            optimizer.zero_grad()
                            pred = model(X_tr[idx])
                            loss = F.mse_loss(pred, y_tr[idx])
                            loss.backward()
                            optimizer.step()
                        scheduler.step()
                        
                        if (epoch + 1) % 10 == 0:
                            model.eval()
                            with torch.no_grad():
                                va_pred = model(X_va).cpu().numpy()
                            va_rmse = math.sqrt(np.mean((y_train[va_idx] - va_pred) ** 2))
                            if va_rmse < best_val_rmse:
                                best_val_rmse = va_rmse
                                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    
                    model.load_state_dict(best_state)
                    model.eval()
                    with torch.no_grad():
                        gnn_oof_preds[va_idx] += model(X_va).cpu().numpy()
                        gnn_test_preds += model(X_te).cpu().numpy()
                    n_nn_models += 1
                    print(f"  Seed {nn_seed}, Fold {fold}: val_rmse={best_val_rmse:.4f}")
            
            gnn_oof_preds /= 3
            gnn_test_preds /= n_nn_models
            
            rmse_gnn, sp_gnn = eval_metrics(y_train, gnn_oof_preds)
            print(f"\nMLP OOF: RMSE={rmse_gnn:.4f}, Spearman={sp_gnn:.4f}")
    
    except Exception as e:
        print(f"GNN/MLP training failed: {e}")
        import traceback
        traceback.print_exc()
        gnn_test_preds = None
        gnn_oof_preds = None


# ============================================================
# 5. Ensemble - Optimize weights using OOF predictions
# ============================================================
print("\n" + "=" * 60)
print("Optimizing Ensemble Weights...")
print("=" * 60)

# Collect all OOF predictions
oof_dict = {
    "lgb": lgb_oof_preds,
    "xgb": xgb_oof_preds,
    "ridge": ridge_oof_preds,
}
test_dict = {
    "lgb": lgb_test_preds,
    "xgb": xgb_test_preds,
    "ridge": ridge_test_preds,
}

if gnn_oof_preds is not None:
    oof_dict["gnn"] = gnn_oof_preds
    test_dict["gnn"] = gnn_test_preds

# Grid search for best ensemble weights
best_rmse = float('inf')
best_sp = 0
best_weights = None
best_pred = None

model_names = list(oof_dict.keys())
oof_matrix = np.column_stack([oof_dict[name] for name in model_names])
test_matrix = np.column_stack([test_dict[name] for name in model_names])

# Use scipy optimize
from scipy.optimize import minimize

def neg_score(weights):
    weights = np.abs(weights)
    weights = weights / weights.sum()
    pred = oof_matrix @ weights
    rmse = math.sqrt(np.mean((y_train - pred) ** 2))
    sp, _ = spearmanr(y_train, pred)
    # Optimize composite score
    score = 500 * max(0, 1 - rmse / 0.80) + 250 * max(0, sp)
    return -score

# Multiple random initializations
for _ in range(1000):
    w0 = np.random.dirichlet(np.ones(len(model_names)))
    res = minimize(neg_score, w0, method='Nelder-Mead', 
                   options={'maxiter': 5000, 'xatol': 1e-8})
    weights = np.abs(res.x)
    weights = weights / weights.sum()
    
    pred = oof_matrix @ weights
    rmse = math.sqrt(np.mean((y_train - pred) ** 2))
    sp, _ = spearmanr(y_train, pred)
    
    if rmse < best_rmse:
        best_rmse = rmse
        best_sp = sp
        best_weights = weights
        best_pred = pred

print(f"\nBest ensemble weights:")
for name, w in zip(model_names, best_weights):
    print(f"  {name}: {w:.4f}")
print(f"\nEnsemble OOF: RMSE={best_rmse:.4f}, Spearman={best_sp:.4f}")

# Compute final score
raw_score = 500 * max(0, 1 - best_rmse / 0.80) + 250 * max(0, best_sp) + 50
if best_rmse <= 0.25 and best_sp >= 0.96:
    score = 800
else:
    score = raw_score
print(f"Estimated score: {score:.1f} / 800")
print(f"Gold-band: RMSE<=0.25? {best_rmse<=0.25}, Spearman>=0.96? {best_sp>=0.96}")

# Final test predictions
final_test_preds = test_matrix @ best_weights

# Also check scaffold-val performance
sc_val_pred = best_pred[sc_val_idx] if len(sc_val_idx) > 0 else None
if sc_val_pred is not None:
    rmse_sc, sp_sc = eval_metrics(y_train[sc_val_idx], 
                                   oof_matrix[sc_val_idx] @ best_weights)
    print(f"\nScaffold-val: RMSE={rmse_sc:.4f}, Spearman={sp_sc:.4f}")


# ============================================================
# 6. Save Submission
# ============================================================
print("\n" + "=" * 60)
print("Saving submission...")
print("=" * 60)

output_dir = WORK_DIR / "submission"
output_dir.mkdir(exist_ok=True)

results_path = output_dir / "results.csv"
with open(results_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "prediction"])
    for tid, pred in zip(test_ids, final_test_preds):
        writer.writerow([tid, f"{pred:.6f}"])

print(f"Saved: {results_path}")
print(f"Predictions: mean={final_test_preds.mean():.4f}, std={final_test_preds.std():.4f}")

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

# SHA256 for reproducibility
with open(results_path, "rb") as f:
    sha = hashlib.sha256(f.read()).hexdigest()
print(f"SHA256: {sha}")
print(f"\nCreated: {zip_path}")
print("=" * 60)
print("DONE! Submit NS-2026-04-answer.zip")
print("=" * 60)
