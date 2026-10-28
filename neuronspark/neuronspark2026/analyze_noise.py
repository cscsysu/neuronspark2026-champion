"""
Comprehensive Noise Pattern Analysis for Noisy Label Classification
Analyzes systematic mislabeling patterns across source_groups, annotator_groups,
and confidence levels to find actionable insights for model improvement.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Load data
train = pd.read_csv('train.csv')
valid = pd.read_csv('trusted_valid.csv')

feature_cols = [f'f{i:03d}' for i in range(96)]
classes = sorted(train['weak_label'].unique())
print(f"Classes: {classes}")
print(f"Train: {len(train)} samples, Valid: {len(valid)} samples")
print(f"Source groups in train: {sorted(train['source_group'].unique())}")
print(f"Annotator groups in train: {sorted(train['annotator_group'].unique())}")
print()

# ============================================================
# 1. TRUE label distribution per source_group (from valid)
# ============================================================
print("=" * 80)
print("1. TRUE LABEL DISTRIBUTION PER SOURCE_GROUP (from trusted_valid)")
print("=" * 80)
valid_dist = pd.crosstab(valid['source_group'], valid['label'], normalize='index')
valid_dist = valid_dist.reindex(columns=classes, fill_value=0)
print(valid_dist.round(3).to_string())
print()

# Also show counts
valid_counts = pd.crosstab(valid['source_group'], valid['label'])
print("Counts:")
print(valid_counts.to_string())
print()

# ============================================================
# 2. WEAK label distribution per source_group (from train)
# ============================================================
print("=" * 80)
print("2. WEAK LABEL DISTRIBUTION PER SOURCE_GROUP (from train)")
print("=" * 80)
train_dist = pd.crosstab(train['source_group'], train['weak_label'], normalize='index')
train_dist = train_dist.reindex(columns=classes, fill_value=0)
print(train_dist.round(3).to_string())
print()

# ============================================================
# 3. DISCREPANCY ANALYSIS: valid vs train distributions
# ============================================================
print("=" * 80)
print("3. DISCREPANCY: (weak_label distribution) - (true_label distribution)")
print("   Positive = over-represented in weak labels (other classes mislabeled AS this)")
print("   Negative = under-represented (this class mislabeled AS something else)")
print("=" * 80)

# Align on common source_groups
common_sgs = sorted(set(valid_dist.index) & set(train_dist.index))
discrepancy = train_dist.loc[common_sgs] - valid_dist.loc[common_sgs]
print(discrepancy.round(3).to_string())
print()

# Find largest discrepancies
print("TOP DISCREPANCIES (|diff| > 0.05):")
for sg in common_sgs:
    for cls in classes:
        diff = discrepancy.loc[sg, cls]
        if abs(diff) > 0.05:
            direction = "OVER-labeled" if diff > 0 else "UNDER-labeled"
            print(f"  {sg} / {cls}: {direction} by {diff:+.3f} "
                  f"(true: {valid_dist.loc[sg, cls]:.3f}, weak: {train_dist.loc[sg, cls]:.3f})")
print()

# ============================================================
# 4. MODEL-BASED ERROR ESTIMATION per (source_group, class)
# ============================================================
print("=" * 80)
print("4. ESTIMATED ERROR RATE PER (SOURCE_GROUP, CLASS)")
print("   Using GradientBoosting trained on trusted_valid to predict train labels")
print("=" * 80)

# Train model on valid data
le = LabelEncoder()
le.fit(classes)

X_valid = valid[feature_cols].values
y_valid = le.transform(valid['label'].values)

X_train = train[feature_cols].values
y_weak = le.transform(train['weak_label'].values)

# Train a strong model on valid
clf = GradientBoostingClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.1,
    subsample=0.8, random_state=42
)
clf.fit(X_valid, y_valid)

# Predict on train
train_pred = clf.predict(X_train)
train_pred_labels = le.inverse_transform(train_pred)
train['model_pred'] = train_pred_labels
train['disagree'] = (train['model_pred'] != train['weak_label']).astype(int)

# Overall disagreement rate
overall_disagree = train['disagree'].mean()
print(f"\nOverall disagreement rate (model vs weak_label): {overall_disagree:.3f} ({overall_disagree*100:.1f}%)")
print()

# Error rate per (source_group, weak_label class)
print("Estimated error rate per (source_group, weak_label):")
print("(Fraction where model prediction != weak_label)")
error_by_sg_class = train.groupby(['source_group', 'weak_label'])['disagree'].agg(['mean', 'count'])
error_pivot = error_by_sg_class['mean'].unstack(fill_value=0)
error_pivot = error_pivot.reindex(columns=classes, fill_value=0)
print(error_pivot.round(3).to_string())
print()

# Highlight high-error cells
print("HIGH ERROR CELLS (disagreement > 30%):")
for sg in error_pivot.index:
    for cls in error_pivot.columns:
        rate = error_pivot.loc[sg, cls]
        count = error_by_sg_class.loc[(sg, cls), 'count'] if (sg, cls) in error_by_sg_class.index else 0
        if rate > 0.30 and count >= 10:
            print(f"  {sg} / {cls}: {rate:.1%} error rate ({count} samples)")
print()

# ============================================================
# 5. ANNOTATOR GROUP CONFUSION PATTERNS
# ============================================================
print("=" * 80)
print("5. ANNOTATOR GROUP CONFUSION PATTERNS")
print("   For each annotator_group: what does the model think vs what the annotator labeled?")
print("=" * 80)

for ag in sorted(train['annotator_group'].unique()):
    subset = train[train['annotator_group'] == ag]
    disagree_subset = subset[subset['disagree'] == 1]
    
    if len(disagree_subset) < 5:
        continue
    
    print(f"\n--- {ag} (total: {len(subset)}, disagreements: {len(disagree_subset)}, "
          f"rate: {len(disagree_subset)/len(subset):.1%}) ---")
    
    # Confusion: weak_label -> model_pred (what annotator said -> what model thinks it really is)
    confusion = pd.crosstab(
        disagree_subset['weak_label'], 
        disagree_subset['model_pred'],
        margins=False
    )
    if len(confusion) > 0:
        print("Confusion (rows=weak_label, cols=model_pred) for DISAGREED samples:")
        print(confusion.to_string())
    
    # Top confusions
    print("  Top confusion pairs:")
    for (wl, mp), count in confusion.stack().sort_values(ascending=False).head(5).items():
        pct = count / len(disagree_subset) * 100
        print(f"    Labeled as '{wl}' but model says '{mp}': {count} ({pct:.1f}% of disagreements)")

print()

# ============================================================
# 6. CROSS-TABULATION: source_group x annotator_group noise
# ============================================================
print("=" * 80)
print("6. CROSS-TABULATION: NOISE RATE BY (SOURCE_GROUP x ANNOTATOR_GROUP)")
print("=" * 80)

cross_noise = train.groupby(['source_group', 'annotator_group'])['disagree'].agg(['mean', 'count'])
cross_pivot = cross_noise['mean'].unstack(fill_value=0)
cross_count = cross_noise['count'].unstack(fill_value=0)

print("\nDisagreement rate:")
print(cross_pivot.round(3).to_string())
print("\nSample counts:")
print(cross_count.to_string())

print("\nHIGHEST NOISE COMBINATIONS (rate > 30%, n >= 20):")
for sg in cross_pivot.index:
    for ag in cross_pivot.columns:
        rate = cross_pivot.loc[sg, ag]
        count = cross_count.loc[sg, ag]
        if rate > 0.30 and count >= 20:
            print(f"  {sg} x {ag}: {rate:.1%} disagreement ({count} samples)")

print("\nLOWEST NOISE COMBINATIONS (rate < 10%, n >= 20) — these are RELIABLE:")
for sg in cross_pivot.index:
    for ag in cross_pivot.columns:
        rate = cross_pivot.loc[sg, ag]
        count = cross_count.loc[sg, ag]
        if rate < 0.10 and count >= 20:
            print(f"  {sg} x {ag}: {rate:.1%} disagreement ({count} samples)")

print()

# ============================================================
# 7. CONFIDENCE ANALYSIS: confident-but-wrong patterns
# ============================================================
print("=" * 80)
print("7. CONFIDENCE ANALYSIS")
print("=" * 80)

# Confidence distribution for agree vs disagree
agree_conf = train[train['disagree'] == 0]['weak_confidence']
disagree_conf = train[train['disagree'] == 1]['weak_confidence']

print(f"\nConfidence stats when model AGREES with weak_label:")
print(f"  mean={agree_conf.mean():.4f}, median={agree_conf.median():.4f}, "
      f"std={agree_conf.std():.4f}")
print(f"Confidence stats when model DISAGREES with weak_label:")
print(f"  mean={disagree_conf.mean():.4f}, median={disagree_conf.median():.4f}, "
      f"std={disagree_conf.std():.4f}")

# Confidence bins analysis
bins = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
labels_bins = ['0-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0']
train['conf_bin'] = pd.cut(train['weak_confidence'], bins=bins, labels=labels_bins)

print("\nDisagreement rate by confidence bin:")
conf_analysis = train.groupby('conf_bin')['disagree'].agg(['mean', 'count'])
for bin_label, row in conf_analysis.iterrows():
    print(f"  {bin_label}: {row['mean']:.1%} disagreement ({int(row['count'])} samples)")

# Per source_group confidence analysis
print("\nConfidence vs disagreement per source_group (high-confidence errors):")
print("(Disagreement rate for samples with weak_confidence > 0.8)")
high_conf = train[train['weak_confidence'] > 0.8]
high_conf_errors = high_conf.groupby('source_group')['disagree'].agg(['mean', 'count'])
for sg, row in high_conf_errors.iterrows():
    if row['mean'] > 0.15:
        print(f"  {sg}: {row['mean']:.1%} of high-confidence labels are wrong ({int(row['count'])} samples)")

# The killer finding: confident-but-wrong by source_group
print("\n'CONFIDENT BUT WRONG' hotspots (conf > 0.85 AND model disagrees):")
confident_wrong = train[(train['weak_confidence'] > 0.85) & (train['disagree'] == 1)]
print(f"  Total confident-but-wrong samples: {len(confident_wrong)}")
cw_by_sg = confident_wrong.groupby(['source_group', 'weak_label', 'model_pred']).size().sort_values(ascending=False)
print("  Top patterns:")
for (sg, wl, mp), count in cw_by_sg.head(15).items():
    print(f"    {sg}: labeled '{wl}' (conf>0.85) but model says '{mp}' — {count} samples")

print()

# ============================================================
# 8. FEATURE-SPACE ANALYSIS: where does noise concentrate?
# ============================================================
print("=" * 80)
print("8. FEATURE-SPACE ANALYSIS")
print("=" * 80)

# PCA on train
pca = PCA(n_components=10, random_state=42)
X_train_pca = pca.fit_transform(X_train)

print(f"\nPCA variance explained (first 10): {pca.explained_variance_ratio_.round(3)}")
print(f"Cumulative: {pca.explained_variance_ratio_.cumsum().round(3)}")

# Compare PCA stats for noisy vs clean samples
clean_mask = train['disagree'] == 0
noisy_mask = train['disagree'] == 1

print(f"\nPCA component means for CLEAN vs NOISY samples:")
print(f"{'Component':<12} {'Clean mean':>12} {'Noisy mean':>12} {'Diff':>10}")
for i in range(5):
    clean_mean = X_train_pca[clean_mask, i].mean()
    noisy_mean = X_train_pca[noisy_mask, i].mean()
    print(f"  PC{i:<8} {clean_mean:>12.4f} {noisy_mean:>12.4f} {noisy_mean-clean_mean:>10.4f}")

# Per-class PCA analysis: are noisy samples at class boundaries?
print("\nDistance from class centroids (noisy vs clean):")
print("(Higher = further from centroid = more likely at decision boundary)")
for cls in classes:
    cls_mask = train['weak_label'] == cls
    cls_pca = X_train_pca[cls_mask]
    cls_clean = X_train_pca[cls_mask & clean_mask]
    cls_noisy = X_train_pca[cls_mask & noisy_mask]
    
    if len(cls_clean) < 5 or len(cls_noisy) < 5:
        continue
    
    centroid = cls_clean.mean(axis=0)  # Use clean samples as centroid
    
    clean_dists = np.sqrt(((cls_clean - centroid) ** 2).sum(axis=1))
    noisy_dists = np.sqrt(((cls_noisy - centroid) ** 2).sum(axis=1))
    
    print(f"  {cls:<25} clean_dist={clean_dists.mean():.3f}, noisy_dist={noisy_dists.mean():.3f}, "
          f"ratio={noisy_dists.mean()/clean_dists.mean():.2f}")

# Feature importance for detecting noise
print("\nTraining a noise detector (predict disagree from features):")
from sklearn.ensemble import RandomForestClassifier
noise_clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
noise_clf.fit(X_train, train['disagree'].values)
noise_importance = pd.Series(noise_clf.feature_importances_, index=feature_cols)
top_noise_features = noise_importance.sort_values(ascending=False).head(15)
print("Top features for detecting noisy labels:")
for feat, imp in top_noise_features.items():
    print(f"  {feat}: {imp:.4f}")

# Noise detection accuracy
from sklearn.model_selection import cross_val_score
scores = cross_val_score(noise_clf, X_train, train['disagree'].values, cv=5, scoring='roc_auc')
print(f"\nNoise detector AUC (5-fold CV): {scores.mean():.3f} +/- {scores.std():.3f}")
print("(Higher = noise is more predictable from features = systematic pattern)")

print()

# ============================================================
# SUMMARY OF ACTIONABLE FINDINGS
# ============================================================
print("=" * 80)
print("SUMMARY OF ACTIONABLE FINDINGS")
print("=" * 80)

print("\n--- A. Most Noisy Source Groups ---")
sg_noise = train.groupby('source_group')['disagree'].mean().sort_values(ascending=False)
for sg, rate in sg_noise.items():
    status = "HIGH NOISE" if rate > 0.25 else ("MODERATE" if rate > 0.15 else "LOW NOISE")
    print(f"  {sg}: {rate:.1%} disagreement [{status}]")

print("\n--- B. Most Noisy Annotator Groups ---")
ag_noise = train.groupby('annotator_group')['disagree'].mean().sort_values(ascending=False)
for ag, rate in ag_noise.items():
    status = "HIGH NOISE" if rate > 0.25 else ("MODERATE" if rate > 0.15 else "LOW NOISE")
    print(f"  {ag}: {rate:.1%} disagreement [{status}]")

print("\n--- C. Specific Mislabeling Patterns (most common confusions) ---")
confusions = train[train['disagree'] == 1].groupby(
    ['source_group', 'annotator_group', 'weak_label', 'model_pred']
).size().sort_values(ascending=False)

print("Top 20 specific mislabeling patterns:")
for (sg, ag, wl, mp), count in confusions.head(20).items():
    # Calculate rate within that sg/ag/wl combination
    subset = train[(train['source_group'] == sg) & 
                   (train['annotator_group'] == ag) & 
                   (train['weak_label'] == wl)]
    rate = count / len(subset) if len(subset) > 0 else 0
    print(f"  {sg} + {ag}: labels '{wl}' as '{wl}' but model says '{mp}' "
          f"— {count} samples ({rate:.0%} of that combination)")

print("\n--- D. Reliability Ranking ---")
print("Most reliable (source_group, annotator_group) combinations:")
reliable = cross_noise[cross_noise['count'] >= 20].sort_values('mean')
for (sg, ag), row in reliable.head(10).iterrows():
    print(f"  {sg} x {ag}: {row['mean']:.1%} disagreement ({int(row['count'])} samples)")

print("\n--- E. Confidence Calibration Issues ---")
print("Source groups where high confidence (>0.8) is LESS reliable:")
for sg in sorted(train['source_group'].unique()):
    sg_data = train[train['source_group'] == sg]
    high_conf_err = sg_data[sg_data['weak_confidence'] > 0.8]['disagree'].mean()
    low_conf_err = sg_data[sg_data['weak_confidence'] <= 0.8]['disagree'].mean()
    if high_conf_err > low_conf_err * 0.8 and high_conf_err > 0.15:
        print(f"  {sg}: high-conf error={high_conf_err:.1%}, low-conf error={low_conf_err:.1%} "
              f"— confidence is POORLY CALIBRATED here")

print("\n--- F. Class Pair Confusions (overall) ---")
confusion_pairs = train[train['disagree'] == 1].groupby(['weak_label', 'model_pred']).size()
confusion_pairs = confusion_pairs.sort_values(ascending=False)
print("Most confused class pairs (weak_label -> model thinks it's actually):")
for (wl, mp), count in confusion_pairs.head(10).items():
    total_wl = (train['weak_label'] == wl).sum()
    print(f"  '{wl}' -> '{mp}': {count} samples ({count/total_wl:.1%} of all '{wl}' labels)")

print("\n" + "=" * 80)
print("DONE. Use these findings to:")
print("1. Weight samples by reliability (down-weight high-noise sg/ag combos)")
print("2. Apply label correction for systematic confusions")
print("3. Use confidence-aware loss (don't trust high confidence in poorly-calibrated groups)")
print("4. Train separate models or apply different corrections per source_group")
print("=" * 80)
