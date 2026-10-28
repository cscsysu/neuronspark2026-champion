"""
NS-2026-06 v10: SVM + Ensemble (based on GPT5.5 insight that RBF SVM >> XGBoost here)
Key: RBF SVM with C=1, gamma=0.01 on standardized features + source_group clean labels
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score, recall_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(".")
SEED = 42
PRIORITY_LABELS = ["computer_hardware", "medical_health", "space_science"]
ALL_LABELS = sorted(["computer_hardware", "medical_health", "mobility_vehicle",
                     "space_science", "sports_recreation", "system_software"])

print("=" * 60)
print("NS-2026-06 v10 - SVM + Ensemble")
print("=" * 60)

train_df = pd.read_csv(DATA_DIR / "train.csv")
valid_df = pd.read_csv(DATA_DIR / "trusted_valid.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")
feat_cols = [c for c in train_df.columns if c.startswith("f")]

le = LabelEncoder()
le.fit(ALL_LABELS)
n_classes = len(ALL_LABELS)

X_train = train_df[feat_cols].values
X_valid = valid_df[feat_cols].values
X_test = test_df[feat_cols].values
y_valid = valid_df["label"].values

# Source group -> true label
sg_map = {'sg_00': 'mobility_vehicle', 'sg_01': 'system_software', 'sg_02': 'space_science',
           'sg_03': 'computer_hardware', 'sg_04': 'mobility_vehicle', 'sg_05': 'medical_health',
           'sg_06': 'computer_hardware', 'sg_07': 'system_software',
           'sg_08': 'sports_recreation', 'sg_09': 'sports_recreation'}
y_train_clean = np.array([sg_map[sg] for sg in train_df["source_group"]])

n_train = len(X_train)
pi_labels = sorted(PRIORITY_LABELS)

print(f"Train: {n_train}, Valid: {len(X_valid)}, Test: {len(X_test)}")

# ==================== COMBINED TRAINING DATA ====================
print("\n[1] Preparing data...")
X_all_train = np.vstack([X_train, X_valid])
y_all_train = np.concatenate([y_train_clean, y_valid])

# Scale
scaler = StandardScaler()
X_all_train_sc = scaler.fit_transform(X_all_train)
X_test_sc = scaler.transform(X_test)

# Also scale train-only for validation
scaler_tr = StandardScaler()
X_train_sc = scaler_tr.fit_transform(X_train)
X_valid_sc = scaler_tr.transform(X_valid)

print(f"  Combined: {len(X_all_train)} samples with clean labels")

# ==================== TRAIN MODELS ====================
print("\n[2] Training models on all labeled data...")

test_preds = []
names = []

# RBF SVM (the key model - multiple configs)
print("  SVM RBF...")
svm_configs = [
    dict(C=1, gamma=0.01),
    dict(C=1, gamma='scale'),
    dict(C=2, gamma=0.01),
    dict(C=2, gamma='scale'),
    dict(C=4, gamma=0.01),
    dict(C=4, gamma=0.003),
    dict(C=0.5, gamma=0.003),
    dict(C=1, gamma=0.003),
    dict(C=8, gamma=0.01),
    dict(C=8, gamma='scale'),
]
for ci, cfg in enumerate(svm_configs):
    m = SVC(**cfg, class_weight='balanced', probability=True, random_state=SEED)
    m.fit(X_all_train_sc, y_all_train)
    test_preds.append(m.predict_proba(X_test_sc))
    names.append(f"SVM_C{cfg['C']}_g{cfg['gamma']}")
    print(f"    {names[-1]} done")

# LDA
print("  LDA...")
m = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
m.fit(X_all_train_sc, y_all_train)
test_preds.append(m.predict_proba(X_test_sc))
names.append("LDA")

# Ridge
print("  Ridge...")
for alpha in [0.1, 1.0, 10.0]:
    m = RidgeClassifier(alpha=alpha, class_weight='balanced')
    m.fit(X_all_train_sc, y_all_train)
    # RidgeClassifier doesn't have predict_proba, use decision function
    dec = m.decision_function(X_test_sc)
    # Softmax to get pseudo-probabilities
    from scipy.special import softmax
    test_preds.append(softmax(dec, axis=1))
    names.append(f"Ridge_a{alpha}")

# KNN
print("  KNN...")
for k in [5, 11, 21]:
    m = KNeighborsClassifier(n_neighbors=k, weights='distance')
    m.fit(X_all_train_sc, y_all_train)
    test_preds.append(m.predict_proba(X_test_sc))
    names.append(f"KNN_{k}")

# LogisticRegression
print("  LR...")
for C in [0.3, 1.0, 3.0]:
    m = LogisticRegression(max_iter=3000, C=C, class_weight='balanced', solver='lbfgs', random_state=SEED)
    m.fit(X_all_train_sc, y_all_train)
    test_preds.append(m.predict_proba(X_test_sc))
    names.append(f"LR_C{C}")

# XGBoost (raw features, not scaled)
print("  XGBoost...")
y_all_enc = le.transform(y_all_train)
for ci, cfg in enumerate([
    dict(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7, min_child_weight=3),
    dict(n_estimators=600, max_depth=5, learning_rate=0.04, subsample=0.75, colsample_bytree=0.8, min_child_weight=4),
    dict(n_estimators=400, max_depth=7, learning_rate=0.06, subsample=0.85, colsample_bytree=0.65, min_child_weight=2),
]):
    m = XGBClassifier(**cfg, reg_alpha=0.1, reg_lambda=1.0, random_state=SEED+ci, verbosity=0, tree_method='hist')
    m.fit(X_all_train, y_all_enc)
    proba = m.predict_proba(X_test)
    # Align to ALL_LABELS order
    aligned = np.zeros((len(X_test), n_classes))
    for i, c in enumerate(m.classes_):
        aligned[:, c] = proba[:, i]
    test_preds.append(aligned)
    names.append(f"XGB_{ci}")

# LightGBM
print("  LightGBM...")
for ci, cfg in enumerate([
    dict(n_estimators=400, max_depth=7, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7, min_child_samples=8, num_leaves=63),
    dict(n_estimators=500, max_depth=6, learning_rate=0.04, subsample=0.85, colsample_bytree=0.75, min_child_samples=5, num_leaves=47),
]):
    m = LGBMClassifier(**cfg, reg_alpha=0.1, reg_lambda=1.0, random_state=SEED+ci, verbose=-1)
    m.fit(X_all_train, y_all_enc)
    proba = m.predict_proba(X_test)
    aligned = np.zeros((len(X_test), n_classes))
    for i, c in enumerate(m.classes_):
        aligned[:, c] = proba[:, i]
    test_preds.append(aligned)
    names.append(f"LGB_{ci}")

# ExtraTrees
print("  ET + RF...")
for s in range(2):
    m = ExtraTreesClassifier(n_estimators=500, min_samples_leaf=2, random_state=SEED+s, class_weight='balanced', n_jobs=-1)
    m.fit(X_all_train, y_all_train)
    proba = m.predict_proba(X_test_sc)
    aligned = np.zeros((len(X_test), n_classes))
    for i, c in enumerate(m.classes_):
        aligned[:, ALL_LABELS.index(c)] = proba[:, i]
    test_preds.append(aligned)
    names.append(f"ET_{s}")

n_models = len(test_preds)
print(f"\n  Total: {n_models} models")

# ==================== ENSEMBLE ====================
print("\n[3] Equal-weight average...")

# Align all predictions to same label order
# SVM/LDA/KNN/LR output in the order of their classes_ attribute
# Need to check alignment
# For SVM with probability=True, classes_ follows alphabetical order which matches ALL_LABELS
# Let's verify and fix

# Re-do predictions with proper alignment
# Actually sklearn models trained with string labels will have classes_ in sorted order = ALL_LABELS
# XGB/LGB we already aligned manually above
# So all should be fine

test_ensemble = sum(test_preds) / n_models
final_pred_idx = np.argmax(test_ensemble, axis=1)

# Map indices to labels - need to know the label order
# For SVM/KNN/LDA/LR: classes_ will be sorted string labels = ALL_LABELS
# For XGB/LGB: we used le.transform so indices match le.classes_ = ALL_LABELS
# So final_pred_idx maps to ALL_LABELS
final_labels = [ALL_LABELS[i] for i in final_pred_idx]

# ==================== VALIDATION ====================
print("\n[4] Validation (train-only)...")

# Train SVM on train-only, eval on valid
val_preds = []
for cfg in [dict(C=1, gamma=0.01), dict(C=1, gamma='scale'), dict(C=2, gamma=0.01)]:
    m = SVC(**cfg, class_weight='balanced', probability=True, random_state=SEED)
    m.fit(X_train_sc, y_train_clean)
    val_preds.append(m.predict_proba(X_valid_sc))

for C in [0.3, 1.0, 3.0]:
    m = LogisticRegression(max_iter=3000, C=C, class_weight='balanced', solver='lbfgs', random_state=SEED)
    m.fit(X_train_sc, y_train_clean)
    val_preds.append(m.predict_proba(X_valid_sc))

m = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
m.fit(X_train_sc, y_train_clean)
val_preds.append(m.predict_proba(X_valid_sc))

val_ens = sum(val_preds) / len(val_preds)
val_pred_idx = np.argmax(val_ens, axis=1)
# Get label order from last model
val_classes = list(m.classes_)
val_pred_labels = [val_classes[i] for i in val_pred_idx]

macro = f1_score(y_valid, val_pred_labels, labels=ALL_LABELS, average='macro')
pr = recall_score(y_valid, val_pred_labels, labels=pi_labels, average='micro', zero_division=0)
total = 900*macro + 200*pr + 100

print(f"\n{'='*60}")
print(f"  *** ESTIMATED SCORE: {total:.1f} / 1200 ***")
print(f"  Macro F1:        {macro:.5f}")
print(f"  Priority Recall: {pr:.5f}")
print(f"{'='*60}")
print(classification_report(y_valid, val_pred_labels, labels=ALL_LABELS, digits=4))

# ==================== SAVE ====================
submission = pd.DataFrame({"id": test_df["id"], "label": final_labels})
submission.to_csv(DATA_DIR / "results.csv", index=False)
print(f"\nSaved results.csv ({len(submission)} rows)")
print(submission["label"].value_counts().sort_index())
print("\nDONE!")
