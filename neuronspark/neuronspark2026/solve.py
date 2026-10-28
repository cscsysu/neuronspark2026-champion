"""
NS-2026-06 解法：噪声标签清洗 + XGBoost/LightGBM 集成
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
print("Loading data...")
train_df = pd.read_csv(DATA_DIR / "train.csv")
valid_df = pd.read_csv(DATA_DIR / "trusted_valid.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")
feat_cols = [c for c in train_df.columns if c.startswith("f")]

le = LabelEncoder()
le.fit(ALL_LABELS)

X_train = train_df[feat_cols].values
y_weak = le.transform(train_df["weak_label"].values)
weak_conf = train_df["weak_confidence"].values

X_valid = valid_df[feat_cols].values
y_valid = le.transform(valid_df["label"].values)
X_test = test_df[feat_cols].values
n_classes = len(ALL_LABELS)

print(f"Train={len(X_train)}, Valid={len(X_valid)}, Test={len(X_test)}")

# ========== Step 1: OOF noise estimation ==========
print("\n[1/5] OOF noise estimation...")
oof_probs = np.zeros((len(X_train), n_classes))
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
for fold, (ti, vi) in enumerate(skf.split(X_train, y_weak)):
    m = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                      subsample=0.8, colsample_bytree=0.8, random_state=SEED+fold, verbosity=0)
    m.fit(X_train[ti], y_weak[ti], sample_weight=weak_conf[ti])
    oof_probs[vi] = m.predict_proba(X_train[vi])
print("  Done")

# ========== Step 2: Confident Learning ==========
print("[2/5] Identifying noisy labels...")
thresholds = np.array([oof_probs[y_weak==c, c].mean() if (y_weak==c).sum()>0 else 0.5 
                       for c in range(n_classes)])
noisy_mask = np.zeros(len(y_weak), dtype=bool)
for i in range(len(y_weak)):
    g = y_weak[i]
    if oof_probs[i, g] < thresholds[g] and np.max(np.delete(oof_probs[i], g)) > oof_probs[i, g]:
        noisy_mask[i] = True
print(f"  Noisy: {noisy_mask.sum()} ({noisy_mask.sum()/len(y_weak)*100:.1f}%)")

# ========== Step 3: Label correction + weighting ==========
print("[3/5] Correcting labels...")
corrected = y_weak.copy()
weights = np.ones(len(X_train))
for i in range(len(X_train)):
    if noisy_mask[i]:
        pc = np.argmax(oof_probs[i])
        pconf = oof_probs[i, pc]
        if pconf > 0.7:
            corrected[i] = pc
            weights[i] = 0.6 * pconf
        else:
            weights[i] = 0.2
    else:
        weights[i] = 0.5 + 0.5 * weak_conf[i]
print(f"  Relabeled: {(corrected != y_weak).sum()}")

# ========== Step 4: Validation ==========
print("[4/5] Evaluating on trusted valid...")
m_eval = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                       subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                       random_state=SEED, verbosity=0)
m_eval.fit(X_train, corrected, sample_weight=weights)
val_pred = m_eval.predict(X_valid)
val_probs = m_eval.predict_proba(X_valid)

macro = f1_score(y_valid, val_pred, average='macro')
pi = [i for i, l in enumerate(le.classes_) if l in PRIORITY_LABELS]
pmask = np.isin(y_valid, pi)
pr = recall_score(y_valid[pmask], val_pred[pmask], labels=pi, average='micro', zero_division=0)
base_score = 900*macro + 200*pr + 100
print(f"  Base score: {base_score:.1f} (macro_f1={macro:.4f}, priority_recall={pr:.4f})")

# Priority boost tuning on this single model's val_probs
best_score = base_score
best_boost = 1.0
for boost in np.arange(1.0, 1.5, 0.01):
    adj = val_probs.copy()
    adj[:, pi] *= boost
    pred = np.argmax(adj, axis=1)
    m_f1 = f1_score(y_valid, pred, average='macro')
    p_r = recall_score(y_valid[pmask], pred[pmask], labels=pi, average='micro', zero_division=0)
    s = 900*m_f1 + 200*p_r + 100
    if s > best_score:
        best_score = s
        best_boost = boost

print(f"  Priority boost: {best_boost:.2f}, boosted score: {best_score:.1f}")

# ========== Step 5: Final training + submission ==========
print("[5/5] Final training (train+valid) for submission...")
X_all = np.vstack([X_train, X_valid])
y_all = np.concatenate([corrected, y_valid])
w_all = np.concatenate([weights, np.full(len(y_valid), 3.0)])

# XGBoost x3 seeds
test_probs = np.zeros((len(X_test), n_classes))
for s in range(3):
    m = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                      reg_alpha=0.1, reg_lambda=1.0, random_state=SEED+s, verbosity=0)
    m.fit(X_all, y_all, sample_weight=w_all)
    test_probs += m.predict_proba(X_test)
test_probs /= 3
print("  XGBoost done")

# LightGBM x2 seeds (faster with fewer trees)
lgb_probs = np.zeros((len(X_test), n_classes))
for s in range(2):
    m = LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.08,
                       subsample=0.8, colsample_bytree=0.8, min_child_samples=10,
                       num_leaves=31, random_state=SEED+s, verbose=-1)
    m.fit(X_all, y_all, sample_weight=w_all)
    lgb_probs += m.predict_proba(X_test)
lgb_probs /= 2
print("  LightGBM done")

# Blend
final_probs = 0.6 * test_probs + 0.4 * lgb_probs

# Apply priority boost
final_probs[:, pi] *= best_boost
final_pred = np.argmax(final_probs, axis=1)
test_labels = le.inverse_transform(final_pred)

# Save
submission = pd.DataFrame({"id": test_df["id"], "label": test_labels})
submission.to_csv(DATA_DIR / "results.csv", index=False)
print(f"\n=== DONE ===")
print(f"Saved results.csv ({len(submission)} rows)")
print(f"Label distribution:")
print(submission["label"].value_counts().sort_index())
