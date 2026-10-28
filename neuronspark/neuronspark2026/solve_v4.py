"""
NS-2026-06 v4: 快速高效版 - 确保2分钟内完成
重点: 减少模型数量，保持核心优化逻辑
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, recall_score, classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(".")
SEED = 42
PRIORITY_LABELS = ["computer_hardware", "medical_health", "space_science"]
ALL_LABELS = ["computer_hardware", "medical_health", "mobility_vehicle", 
              "space_science", "sports_recreation", "system_software"]

# ========== Load ==========
train_df = pd.read_csv(DATA_DIR / "train.csv")
valid_df = pd.read_csv(DATA_DIR / "trusted_valid.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")
feat_cols = [c for c in train_df.columns if c.startswith("f")]

le = LabelEncoder()
le.fit(ALL_LABELS)

X_train = train_df[feat_cols].values
y_weak = le.transform(train_df["weak_label"].values)
weak_conf = train_df["weak_confidence"].values
source_groups = train_df["source_group"].values

X_valid = valid_df[feat_cols].values
y_valid = le.transform(valid_df["label"].values)
X_test = test_df[feat_cols].values
n_classes = len(ALL_LABELS)
n_train = len(X_train)
pi = [i for i, l in enumerate(le.classes_) if l in PRIORITY_LABELS]
pmask = np.isin(y_valid, pi)

print(f"Data: train={n_train}, valid={len(X_valid)}, test={len(X_test)}")

# ========== Iterative Cleaning (3 rounds, fast) ==========
print("\n[1] Iterative cleaning...")
current_labels = y_weak.copy()
current_weights = weak_conf.copy()

for it in range(3):
    oof = np.zeros((n_train, n_classes))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED+it*7)
    for fold, (ti, vi) in enumerate(skf.split(X_train, current_labels)):
        m = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                          subsample=0.8, colsample_bytree=0.7,
                          random_state=SEED+fold+it*5, verbosity=0)
        m.fit(X_train[ti], current_labels[ti], sample_weight=current_weights[ti])
        oof[vi] = m.predict_proba(X_train[vi])
    
    thresh = np.array([oof[current_labels==c, c].mean() if (current_labels==c).sum()>0 else 0.5 
                       for c in range(n_classes)])
    
    noisy = np.zeros(n_train, dtype=bool)
    for i in range(n_train):
        g = current_labels[i]
        if oof[i,g] < thresh[g]*(1-it*0.05) and np.max(np.delete(oof[i],g)) > oof[i,g]:
            noisy[i] = True
    
    ct = 0.55 - it*0.05
    for i in range(n_train):
        if noisy[i]:
            pc = np.argmax(oof[i])
            if oof[i,pc] > ct:
                current_labels[i] = pc
                current_weights[i] = 0.6*oof[i,pc] + 0.2*weak_conf[i]
                ct_count = ct  # just for counting
            else:
                current_weights[i] = 0.15
        else:
            current_weights[i] = 0.5 + 0.5*weak_conf[i]
    
    print(f"  R{it+1}: noisy={noisy.sum()}")

# ========== Train models for valid eval + test ==========
print("\n[2] Training models...")

# Train on train-only for valid evaluation
valid_preds = []
# XGB x3 configs
cfgs_xgb = [
    dict(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7, min_child_weight=3),
    dict(n_estimators=300, max_depth=7, learning_rate=0.07, subsample=0.85, colsample_bytree=0.75, min_child_weight=2),
    dict(n_estimators=500, max_depth=5, learning_rate=0.04, subsample=0.75, colsample_bytree=0.8, min_child_weight=4),
]
for ci, cfg in enumerate(cfgs_xgb):
    m = XGBClassifier(**cfg, reg_alpha=0.1, reg_lambda=1.0, random_state=SEED+ci, verbosity=0)
    m.fit(X_train, current_labels, sample_weight=current_weights)
    valid_preds.append(m.predict_proba(X_valid))
print("  XGB valid done")

# LGB x2
cfgs_lgb = [
    dict(n_estimators=300, max_depth=7, learning_rate=0.06, subsample=0.8, colsample_bytree=0.7, min_child_samples=8, num_leaves=63),
    dict(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.85, colsample_bytree=0.75, min_child_samples=5, num_leaves=47),
]
for ci, cfg in enumerate(cfgs_lgb):
    m = LGBMClassifier(**cfg, reg_alpha=0.1, reg_lambda=1.0, random_state=SEED+ci, verbose=-1)
    m.fit(X_train, current_labels, sample_weight=current_weights)
    valid_preds.append(m.predict_proba(X_valid))
print("  LGB valid done")

# Ensemble valid probs (equal weight)
val_ens = sum(valid_preds) / len(valid_preds)

# ========== Per-class scale optimization ==========
print("\n[3] Per-class scale tuning...")

def calc_score(probs, scales):
    adj = probs.copy()
    for k in range(n_classes):
        adj[:, k] *= scales[k]
    pred = np.argmax(adj, axis=1)
    macro = f1_score(y_valid, pred, average='macro')
    pr = recall_score(y_valid[pmask], pred[pmask], labels=pi, average='micro', zero_division=0)
    return 900*macro + 200*pr + 100

best_scales = np.ones(n_classes)
best_score = calc_score(val_ens, best_scales)

for _pass in range(5):
    improved = False
    for c in range(n_classes):
        for scale in np.arange(0.5, 2.5, 0.02):
            trial = best_scales.copy()
            trial[c] = scale
            s = calc_score(val_ens, trial)
            if s > best_score:
                best_score = s
                best_scales[c] = scale
                improved = True
    if not improved:
        break

print(f"  Scales: {dict(zip(ALL_LABELS, [f'{s:.2f}' for s in best_scales]))}")
print(f"  Val score: {best_score:.1f}")

# ========== Train on combined (train+valid) for test ==========
print("\n[4] Final test predictions...")
X_comb = np.vstack([X_train, X_valid])
y_comb = np.concatenate([current_labels, y_valid])
w_comb = np.concatenate([current_weights, np.full(len(X_valid), 4.0)])

test_preds = []
for ci, cfg in enumerate(cfgs_xgb):
    m = XGBClassifier(**cfg, reg_alpha=0.1, reg_lambda=1.0, random_state=SEED+ci, verbosity=0)
    m.fit(X_comb, y_comb, sample_weight=w_comb)
    test_preds.append(m.predict_proba(X_test))
print("  XGB test done")

for ci, cfg in enumerate(cfgs_lgb):
    m = LGBMClassifier(**cfg, reg_alpha=0.1, reg_lambda=1.0, random_state=SEED+ci, verbose=-1)
    m.fit(X_comb, y_comb, sample_weight=w_comb)
    test_preds.append(m.predict_proba(X_test))
print("  LGB test done")

test_ens = sum(test_preds) / len(test_preds)

# Apply scales
final_probs = test_ens.copy()
for k in range(n_classes):
    final_probs[:, k] *= best_scales[k]
final_pred = np.argmax(final_probs, axis=1)
test_labels = le.inverse_transform(final_pred)

# ========== Validation report ==========
val_adj = val_ens.copy()
for k in range(n_classes):
    val_adj[:, k] *= best_scales[k]
val_pred = np.argmax(val_adj, axis=1)
macro = f1_score(y_valid, val_pred, average='macro')
pr = recall_score(y_valid[pmask], val_pred[pmask], labels=pi, average='micro', zero_division=0)

print(f"\n{'='*50}")
print(f"  LOCAL SCORE: {900*macro+200*pr+100:.1f} / 1200")
print(f"  Macro F1: {macro:.5f}")
print(f"  Priority Recall: {pr:.5f}")
print(f"{'='*50}")
print(classification_report(le.inverse_transform(y_valid), le.inverse_transform(val_pred), digits=4))

for lbl in PRIORITY_LABELS:
    idx = le.transform([lbl])[0]
    mask = y_valid == idx
    if mask.sum() > 0:
        print(f"  {lbl}: recall={( val_pred[mask]==idx).mean():.4f}")

# ========== Save ==========
submission = pd.DataFrame({"id": test_df["id"], "label": test_labels})
submission.to_csv(DATA_DIR / "results.csv", index=False)
print(f"\nSaved results.csv ({len(submission)} rows)")
print(submission["label"].value_counts().sort_index())
