"""
NS-2026-06 v3: 精简高效终极版
改进点:
1. 迭代清洗 3 轮（已验证有效）
2. 特征工程 (PCA + stats)
3. XGBoost 多配置集成 (不做重复valid训练，用最后一轮OOF评估)
4. Per-class threshold tuning
5. 伪标签
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, recall_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
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
print("Loading...")
train_df = pd.read_csv(DATA_DIR / "train.csv")
valid_df = pd.read_csv(DATA_DIR / "trusted_valid.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")
feat_cols = [c for c in train_df.columns if c.startswith("f")]

le = LabelEncoder()
le.fit(ALL_LABELS)

X_train_raw = train_df[feat_cols].values
y_weak = le.transform(train_df["weak_label"].values)
weak_conf = train_df["weak_confidence"].values
source_groups = train_df["source_group"].values

X_valid_raw = valid_df[feat_cols].values
y_valid = le.transform(valid_df["label"].values)
X_test_raw = test_df[feat_cols].values
n_classes = len(ALL_LABELS)
pi = [i for i, l in enumerate(le.classes_) if l in PRIORITY_LABELS]
pmask = np.isin(y_valid, pi)

# ========== Feature Engineering ==========
print("[FE] Features...")
X_all_raw = np.vstack([X_train_raw, X_valid_raw, X_test_raw])
pca = PCA(n_components=15, random_state=SEED)
X_pca = pca.fit_transform(X_all_raw)
X_stats = np.column_stack([
    X_all_raw.mean(axis=1), X_all_raw.std(axis=1),
    np.abs(X_all_raw).max(axis=1),
    np.percentile(X_all_raw, 25, axis=1),
    np.percentile(X_all_raw, 75, axis=1),
])
X_all = np.hstack([X_all_raw, X_pca, X_stats])
n_train, n_valid = len(X_train_raw), len(X_valid_raw)
X_train = X_all[:n_train]
X_valid = X_all[n_train:n_train+n_valid]
X_test = X_all[n_train+n_valid:]
print(f"  {X_train_raw.shape[1]} -> {X_all.shape[1]} features")

# ========== Iterative Cleaning ==========
print("\n[CLEAN] Iterative cleaning (3 rounds)...")
current_labels = y_weak.copy()
current_weights = weak_conf.copy()

for iteration in range(3):
    oof_probs = np.zeros((n_train, n_classes))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED+iteration*7)
    for fold, (ti, vi) in enumerate(skf.split(X_train, current_labels)):
        m = XGBClassifier(n_estimators=250, max_depth=6, learning_rate=0.08,
                          subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
                          random_state=SEED+fold+iteration*5, verbosity=0)
        m.fit(X_train[ti], current_labels[ti], sample_weight=current_weights[ti])
        oof_probs[vi] = m.predict_proba(X_train[vi])
    
    thresholds = np.array([oof_probs[current_labels==c, c].mean() 
                           if (current_labels==c).sum()>0 else 0.5 for c in range(n_classes)])
    
    noisy_mask = np.zeros(n_train, dtype=bool)
    for i in range(n_train):
        g = current_labels[i]
        if oof_probs[i, g] < thresholds[g] * (1.0 - iteration*0.05):
            max_other = np.max(np.delete(oof_probs[i], g))
            if max_other > oof_probs[i, g]:
                noisy_mask[i] = True
    
    conf_thresh = 0.55 - iteration * 0.05
    relabeled = 0
    for i in range(n_train):
        if noisy_mask[i]:
            pc = np.argmax(oof_probs[i])
            pconf = oof_probs[i, pc]
            if pconf > conf_thresh:
                current_labels[i] = pc
                current_weights[i] = 0.6 * pconf + 0.2 * weak_conf[i]
                relabeled += 1
            else:
                current_weights[i] = 0.15
        else:
            current_weights[i] = 0.5 + 0.5 * weak_conf[i]
    
    # Quick val check
    m_tmp = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.08, random_state=SEED, verbosity=0)
    m_tmp.fit(X_train, current_labels, sample_weight=current_weights)
    vp = m_tmp.predict(X_valid)
    macro = f1_score(y_valid, vp, average='macro')
    pr = recall_score(y_valid[pmask], vp[pmask], labels=pi, average='micro', zero_division=0)
    print(f"  R{iteration+1}: noisy={noisy_mask.sum()}, relabel={relabeled}, "
          f"macro={macro:.4f}, pr={pr:.4f}, score={900*macro+200*pr+100:.1f}")

# ========== Final OOF for validation-based tuning ==========
print("\n[OOF] Final OOF for threshold tuning...")
oof_val_probs = np.zeros((n_train, n_classes))
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED+99)
for fold, (ti, vi) in enumerate(skf.split(X_train, current_labels)):
    m = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
                      random_state=SEED+fold, verbosity=0)
    m.fit(X_train[ti], current_labels[ti], sample_weight=current_weights[ti])
    oof_val_probs[vi] = m.predict_proba(X_train[vi])

# ========== Train final models for test ==========
print("\n[TRAIN] Final ensemble for test predictions...")
X_combined = np.vstack([X_train, X_valid])
y_combined = np.concatenate([current_labels, y_valid])
w_combined = np.concatenate([current_weights, np.full(n_valid, 4.0)])

# Also train on train-only for proper valid evaluation
xgb_configs = [
    dict(n_estimators=600, max_depth=6, learning_rate=0.04, subsample=0.8, colsample_bytree=0.7, min_child_weight=3),
    dict(n_estimators=500, max_depth=7, learning_rate=0.05, subsample=0.85, colsample_bytree=0.75, min_child_weight=2),
    dict(n_estimators=700, max_depth=5, learning_rate=0.03, subsample=0.75, colsample_bytree=0.8, min_child_weight=4),
    dict(n_estimators=400, max_depth=8, learning_rate=0.06, subsample=0.9, colsample_bytree=0.65, min_child_weight=2),
]

lgb_configs = [
    dict(n_estimators=400, max_depth=7, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7, min_child_samples=8, num_leaves=63),
    dict(n_estimators=500, max_depth=6, learning_rate=0.04, subsample=0.85, colsample_bytree=0.75, min_child_samples=5, num_leaves=47),
]

test_preds_list = []
valid_preds_list = []

# XGBoost
print("  XGBoost...")
for ci, cfg in enumerate(xgb_configs):
    for s in range(2):
        # For test: train on combined
        m = XGBClassifier(**cfg, reg_alpha=0.1, reg_lambda=1.0, 
                          random_state=SEED+ci*10+s, verbosity=0)
        m.fit(X_combined, y_combined, sample_weight=w_combined)
        test_preds_list.append(m.predict_proba(X_test))
        # For valid: train on train only
        m2 = XGBClassifier(**cfg, reg_alpha=0.1, reg_lambda=1.0,
                           random_state=SEED+ci*10+s, verbosity=0)
        m2.fit(X_train, current_labels, sample_weight=current_weights)
        valid_preds_list.append(m2.predict_proba(X_valid))
print(f"    {len(test_preds_list)} models")

# LightGBM
print("  LightGBM...")
for ci, cfg in enumerate(lgb_configs):
    for s in range(2):
        m = LGBMClassifier(**cfg, reg_alpha=0.1, reg_lambda=1.0,
                           random_state=SEED+ci*10+s, verbose=-1)
        m.fit(X_combined, y_combined, sample_weight=w_combined)
        test_preds_list.append(m.predict_proba(X_test))
        m2 = LGBMClassifier(**cfg, reg_alpha=0.1, reg_lambda=1.0,
                            random_state=SEED+ci*10+s, verbose=-1)
        m2.fit(X_train, current_labels, sample_weight=current_weights)
        valid_preds_list.append(m2.predict_proba(X_valid))
print(f"    {len(test_preds_list)} models total")

# ========== Greedy Ensemble Selection ==========
print("\n[BLEND] Greedy ensemble selection...")

def score_probs(probs, scales=None):
    if scales is not None:
        probs = probs.copy()
        for k in range(n_classes):
            probs[:, k] *= scales[k]
    pred = np.argmax(probs, axis=1)
    macro = f1_score(y_valid, pred, average='macro')
    pr = recall_score(y_valid[pmask], pred[pmask], labels=pi, average='micro', zero_division=0)
    return 900*macro + 200*pr + 100, macro, pr

# Individual scores
n_models = len(valid_preds_list)
scores = [(score_probs(valid_preds_list[i])[0], i) for i in range(n_models)]
scores.sort(reverse=True)
print(f"  Best single: {scores[0][0]:.1f} (model {scores[0][1]})")

# Greedy forward
selected = [scores[0][1]]
ens_probs = valid_preds_list[selected[0]].copy()
best_s = scores[0][0]

for _ in range(min(n_models-1, 12)):
    best_next = None
    for i in range(n_models):
        if i in selected: continue
        trial = (ens_probs * len(selected) + valid_preds_list[i]) / (len(selected)+1)
        s, _, _ = score_probs(trial)
        if s > best_s:
            best_s = s
            best_next = i
    if best_next is None: break
    selected.append(best_next)
    ens_probs = sum(valid_preds_list[i] for i in selected) / len(selected)

print(f"  Selected {len(selected)} models, score: {best_s:.1f}")

val_ensemble = sum(valid_preds_list[i] for i in selected) / len(selected)
test_ensemble = sum(test_preds_list[i] for i in selected) / len(selected)

# ========== Per-class Scale Optimization ==========
print("\n[THRESH] Per-class scale optimization...")

best_scales = np.ones(n_classes)
best_total = best_s

for _pass in range(5):
    improved = False
    for c in range(n_classes):
        for scale in np.arange(0.5, 2.5, 0.02):
            trial_scales = best_scales.copy()
            trial_scales[c] = scale
            s, _, _ = score_probs(val_ensemble, trial_scales)
            if s > best_total:
                best_total = s
                best_scales[c] = scale
                improved = True
    if not improved:
        break

print(f"  Scales: {dict(zip(ALL_LABELS, [f'{s:.2f}' for s in best_scales]))}")
s_final, m_final, pr_final = score_probs(val_ensemble, best_scales)
print(f"  Score: {s_final:.1f} (macro={m_final:.4f}, pr={pr_final:.4f})")

# ========== Pseudo-labeling ==========
print("\n[PSEUDO] Pseudo-labeling...")

test_adj = test_ensemble.copy()
for k in range(n_classes):
    test_adj[:, k] *= best_scales[k]
test_adj_norm = test_adj / test_adj.sum(axis=1, keepdims=True)
test_conf = np.max(test_adj_norm, axis=1)
test_pred = np.argmax(test_adj_norm, axis=1)

pseudo_thresh = 0.90
pseudo_mask = test_conf > pseudo_thresh
print(f"  High-conf samples: {pseudo_mask.sum()} ({pseudo_mask.sum()/len(X_test)*100:.1f}%)")

if pseudo_mask.sum() > 100:
    X_pseudo = np.vstack([X_combined, X_test[pseudo_mask]])
    y_pseudo = np.concatenate([y_combined, test_pred[pseudo_mask]])
    w_pseudo = np.concatenate([w_combined, np.full(pseudo_mask.sum(), 2.0)])
    
    # Retrain top 4 XGB configs with pseudo
    pseudo_test_preds = []
    pseudo_valid_preds = []
    for ci in range(min(4, len(xgb_configs))):
        cfg = xgb_configs[ci]
        m = XGBClassifier(**cfg, reg_alpha=0.1, reg_lambda=1.0,
                          random_state=SEED+ci+200, verbosity=0)
        m.fit(X_pseudo, y_pseudo, sample_weight=w_pseudo)
        pseudo_test_preds.append(m.predict_proba(X_test))
        # Valid
        X_tp = np.vstack([X_train, X_test[pseudo_mask]])
        y_tp = np.concatenate([current_labels, test_pred[pseudo_mask]])
        w_tp = np.concatenate([current_weights, np.full(pseudo_mask.sum(), 2.0)])
        m2 = XGBClassifier(**cfg, reg_alpha=0.1, reg_lambda=1.0,
                           random_state=SEED+ci+200, verbosity=0)
        m2.fit(X_tp, y_tp, sample_weight=w_tp)
        pseudo_valid_preds.append(m2.predict_proba(X_valid))
    
    pseudo_val_ens = sum(pseudo_valid_preds) / len(pseudo_valid_preds)
    pseudo_test_ens = sum(pseudo_test_preds) / len(pseudo_test_preds)
    
    # Find best blend
    best_alpha = 0.0
    best_pseudo_score = best_total
    for alpha in np.arange(0.0, 0.5, 0.02):
        blended = (1-alpha)*val_ensemble + alpha*pseudo_val_ens
        s, _, _ = score_probs(blended, best_scales)
        if s > best_pseudo_score:
            best_pseudo_score = s
            best_alpha = alpha
    
    print(f"  Best blend alpha: {best_alpha:.2f}, score: {best_pseudo_score:.1f}")
    
    if best_alpha > 0:
        test_ensemble = (1-best_alpha)*test_ensemble + best_alpha*pseudo_test_ens
        val_ensemble = (1-best_alpha)*val_ensemble + best_alpha*pseudo_val_ens
        # Re-optimize scales
        for _pass in range(3):
            for c in range(n_classes):
                for scale in np.arange(0.5, 2.5, 0.02):
                    trial_scales = best_scales.copy()
                    trial_scales[c] = scale
                    s, _, _ = score_probs(val_ensemble, trial_scales)
                    if s > best_pseudo_score:
                        best_pseudo_score = s
                        best_scales[c] = scale
        print(f"  After re-opt: {best_pseudo_score:.1f}")

# ========== Output ==========
print("\n[OUTPUT] Generating submission...")

final_probs = test_ensemble.copy()
for k in range(n_classes):
    final_probs[:, k] *= best_scales[k]
final_pred = np.argmax(final_probs, axis=1)
test_labels = le.inverse_transform(final_pred)

# Final validation
s_out, m_out, pr_out = score_probs(val_ensemble, best_scales)
print(f"\n{'='*50}")
print(f"  LOCAL SCORE: {s_out:.1f} / 1200")
print(f"  Macro F1: {m_out:.5f} ({900*m_out:.1f} pts)")
print(f"  Priority Recall: {pr_out:.5f} ({200*pr_out:.1f} pts)")
print(f"{'='*50}")

val_pred_final = np.argmax(val_ensemble * best_scales, axis=1)
print(f"\nPer-class report:")
print(classification_report(le.inverse_transform(y_valid), le.inverse_transform(val_pred_final), digits=4))

print("Priority recall:")
for lbl in PRIORITY_LABELS:
    idx = le.transform([lbl])[0]
    mask = y_valid == idx
    if mask.sum() > 0:
        r = (val_pred_final[mask] == idx).mean()
        print(f"  {lbl}: {r:.4f} ({(val_pred_final[mask]==idx).sum()}/{mask.sum()})")

submission = pd.DataFrame({"id": test_df["id"], "label": test_labels})
submission.to_csv(DATA_DIR / "results.csv", index=False)
print(f"\nSaved results.csv ({len(submission)} rows)")
print(submission["label"].value_counts().sort_index())
