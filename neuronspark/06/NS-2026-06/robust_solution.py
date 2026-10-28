import argparse
import csv
import hashlib
import json
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None


SEED = 260606
PRIORITY_LABELS = {"computer_hardware", "medical_health", "space_science"}


@dataclass(frozen=True)
class Config:
    drop_ratio: float = 0.18
    gold_weight: float = 6.0
    weak_weight_power: float = 1.0
    lr_c: float = 1.3
    svc_c: float = 0.6
    xgb_estimators: int = 260
    xgb_depth: int = 3
    n_splits: int = 5


def load_labels(data_dir: Path):
    with open(data_dir / "label_map.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return list(data["labels"].keys())


def feature_columns(df: pd.DataFrame):
    return [c for c in df.columns if c.startswith("f")]


def score(y_true, y_pred, labels):
    macro = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    priority_recall = recall_score(
        y_true,
        y_pred,
        labels=[c for c in labels if c in PRIORITY_LABELS],
        average="micro",
        zero_division=0,
    )
    return macro, priority_recall, 900 * macro + 200 * priority_recall + 100


def make_preprocessor(feature_cols):
    return StandardScaler()


def align_proba(classes, proba, labels):
    out = np.zeros((proba.shape[0], len(labels)), dtype=np.float64)
    for i, label in enumerate(classes):
        out[:, labels.index(label)] = proba[:, i]
    return out


def align_decision(classes, decision, labels):
    decision = np.asarray(decision)
    if decision.ndim == 1:
        decision = np.column_stack([-decision, decision])
    out = np.full((decision.shape[0], len(labels)), -20.0, dtype=np.float64)
    for i, label in enumerate(classes):
        out[:, labels.index(label)] = decision[:, i]
    return softmax(out / 1.8, axis=1)


def initial_reliability(train, feature_cols, labels):
    model = Pipeline(
        [
            ("prep", make_preprocessor(feature_cols)),
            (
                "clf",
                LogisticRegression(
                    max_iter=2500,
                    C=2.0,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=SEED,
                ),
            ),
        ]
    )
    model.fit(train[feature_cols], train["weak_label"], clf__sample_weight=train["weak_confidence"])
    proba = align_proba(model.classes_, model.predict_proba(train[feature_cols]), labels)
    weak_idx = np.array([labels.index(x) for x in train["weak_label"]])
    weak_prob = proba[np.arange(len(train)), weak_idx]
    pred = np.array(labels, dtype=object)[proba.argmax(axis=1)]
    consistent = (pred == train["weak_label"].to_numpy()).astype(float)

    ann_rel = train.assign(_consistent=consistent).groupby("annotator_group")["_consistent"].mean()
    src_rel = train.assign(_consistent=consistent).groupby("source_group")["_consistent"].mean()
    ann_factor = train["annotator_group"].map(ann_rel).to_numpy()
    src_factor = train["source_group"].map(src_rel).to_numpy()
    base = np.power(train["weak_confidence"].to_numpy(), 0.7) * np.power(np.clip(weak_prob, 1e-4, 1.0), 0.8)
    group = 0.55 + 0.30 * ann_factor + 0.15 * src_factor
    weights = base * group
    weights = weights / max(weights.mean(), 1e-8)
    return weak_prob, np.clip(weights, 0.10, 2.50), pred, proba


def make_training_frame(train, valid_part, feature_cols, cfg, weak_prob, weak_weights):
    keep_threshold = np.quantile(weak_prob, cfg.drop_ratio)
    keep = weak_prob >= keep_threshold
    x_weak = train.loc[keep, feature_cols].copy()
    y_weak = train.loc[keep, "weak_label"].copy()
    w_weak = weak_weights[keep] ** cfg.weak_weight_power

    x_gold = valid_part[feature_cols].copy()
    y_gold = valid_part["label"].copy()
    w_gold = np.full(len(valid_part), cfg.gold_weight, dtype=np.float64)

    x = pd.concat([x_weak, x_gold], ignore_index=True)
    y = pd.concat([y_weak, y_gold], ignore_index=True)
    w = np.concatenate([w_weak, w_gold])
    return x, y, w, int(keep.sum())


def fit_lr(x, y, w, feature_cols, cfg, seed):
    model = Pipeline(
        [
            ("prep", make_preprocessor(feature_cols)),
            (
                "clf",
                LogisticRegression(
                    max_iter=3000,
                    C=cfg.lr_c,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=seed,
                ),
            ),
        ]
    )
    model.fit(x, y, clf__sample_weight=w)
    return model


def fit_svc(x, y, w, feature_cols, cfg, seed):
    model = Pipeline(
        [
            ("prep", make_preprocessor(feature_cols)),
            (
                "clf",
                LinearSVC(
                    C=cfg.svc_c,
                    class_weight="balanced",
                    random_state=seed,
                    max_iter=12000,
                    dual="auto",
                ),
            ),
        ]
    )
    model.fit(x, y, clf__sample_weight=w)
    return model


def fit_extra_trees(x, y, w, feature_cols, seed):
    model = Pipeline(
        [
            ("prep", make_preprocessor(feature_cols)),
            (
                "clf",
                ExtraTreesClassifier(
                    n_estimators=420,
                    max_features="sqrt",
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(x, y, clf__sample_weight=w)
    return model


def fit_xgb(x, y, w, feature_cols, labels, cfg, seed):
    if XGBClassifier is None:
        return None, None
    prep = make_preprocessor(feature_cols)
    x_fit = prep.fit_transform(x)
    encoder = LabelEncoder().fit(labels)
    model = XGBClassifier(
        n_estimators=cfg.xgb_estimators,
        max_depth=cfg.xgb_depth,
        learning_rate=0.045,
        subsample=0.88,
        colsample_bytree=0.88,
        min_child_weight=2.0,
        reg_alpha=0.15,
        reg_lambda=3.5,
        objective="multi:softprob",
        eval_metric="mlogloss",
        num_class=len(labels),
        random_state=seed,
        n_jobs=max(1, min(4, os.cpu_count() or 1)),
        tree_method="hist",
    )
    model.fit(x_fit, encoder.transform(y), sample_weight=w, verbose=False)
    return (prep, model, encoder)


def predict_proba_bundle(bundle, x, feature_cols, labels):
    lr, svc, et, xgb_bundle = bundle
    probs = []

    probs.append(align_proba(lr.classes_, lr.predict_proba(x[feature_cols]), labels))

    svc_clf = svc.named_steps["clf"]
    svc_decision = svc.decision_function(x[feature_cols])
    probs.append(align_decision(svc_clf.classes_, svc_decision, labels))

    probs.append(align_proba(et.classes_, et.predict_proba(x[feature_cols]), labels))

    if xgb_bundle is not None:
        prep, model, encoder = xgb_bundle
        p = model.predict_proba(prep.transform(x[feature_cols]))
        probs.append(align_proba(encoder.inverse_transform(np.arange(len(encoder.classes_))), p, labels))

    weights = np.array([0.35, 0.25, 0.15, 0.25][: len(probs)], dtype=np.float64)
    weights = weights / weights.sum()
    return sum(w * p for w, p in zip(weights, probs))


def fit_bundle(train, valid_part, feature_cols, labels, cfg, weak_prob, weak_weights, seed):
    x, y, w, kept = make_training_frame(train, valid_part, feature_cols, cfg, weak_prob, weak_weights)
    lr = fit_lr(x, y, w, feature_cols, cfg, seed)
    svc = fit_svc(x, y, w, feature_cols, cfg, seed + 1)
    et = fit_extra_trees(x, y, w, feature_cols, seed + 2)
    xgb_bundle = fit_xgb(x, y, w, feature_cols, labels, cfg, seed + 3)
    return (lr, svc, et, xgb_bundle), kept


def predict_with_multipliers(proba, labels, multipliers):
    adjusted = proba * np.array([multipliers.get(label, 1.0) for label in labels])
    return np.array(labels, dtype=object)[adjusted.argmax(axis=1)]


def optimize_multipliers(y_true, proba, labels):
    grid = [0.75, 0.85, 0.95, 1.0, 1.05, 1.15, 1.30, 1.45]
    multipliers = {label: 1.0 for label in labels}
    best_pred = predict_with_multipliers(proba, labels, multipliers)
    best = score(y_true, best_pred, labels)[2]
    for _ in range(4):
        changed = False
        for label in labels:
            local_best = (best, multipliers[label])
            for value in grid:
                trial = dict(multipliers)
                trial[label] = value
                pred = predict_with_multipliers(proba, labels, trial)
                trial_score = score(y_true, pred, labels)[2]
                if trial_score > local_best[0] + 1e-9:
                    local_best = (trial_score, value)
            if local_best[1] != multipliers[label]:
                multipliers[label] = local_best[1]
                best = local_best[0]
                changed = True
        if not changed:
            break
    return multipliers


def run_cv(train, valid, feature_cols, labels, cfg):
    weak_prob, weak_weights, train_model_pred, _ = initial_reliability(train, feature_cols, labels)
    splitter = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=SEED)
    oof_proba = np.zeros((len(valid), len(labels)), dtype=np.float64)
    kept_counts = []

    for fold, (fit_idx, hold_idx) in enumerate(splitter.split(valid, valid["label"]), start=1):
        fit_gold = valid.iloc[fit_idx].reset_index(drop=True)
        hold_gold = valid.iloc[hold_idx].reset_index(drop=True)
        bundle, kept = fit_bundle(train, fit_gold, feature_cols, labels, cfg, weak_prob, weak_weights, SEED + fold * 101)
        oof_proba[hold_idx] = predict_proba_bundle(bundle, hold_gold, feature_cols, labels)
        kept_counts.append(kept)
        fold_pred = np.array(labels, dtype=object)[oof_proba[hold_idx].argmax(axis=1)]
        fold_score = score(hold_gold["label"], fold_pred, labels)
        print(
            f"fold={fold} kept={kept} macro_f1={fold_score[0]:.5f} "
            f"priority_recall={fold_score[1]:.5f} score_like={fold_score[2]:.2f}",
            flush=True,
        )

    raw_pred = np.array(labels, dtype=object)[oof_proba.argmax(axis=1)]
    raw_score = score(valid["label"], raw_pred, labels)
    multipliers = optimize_multipliers(valid["label"].to_numpy(), oof_proba, labels)
    tuned_pred = predict_with_multipliers(oof_proba, labels, multipliers)
    tuned_score = score(valid["label"], tuned_pred, labels)

    print("\nOOF raw:", f"macro_f1={raw_score[0]:.5f}", f"priority_recall={raw_score[1]:.5f}", f"score_like={raw_score[2]:.2f}")
    print("OOF tuned:", f"macro_f1={tuned_score[0]:.5f}", f"priority_recall={tuned_score[1]:.5f}", f"score_like={tuned_score[2]:.2f}")
    print("class multipliers:", json.dumps(multipliers, ensure_ascii=False, sort_keys=True))
    print("\nclassification report:")
    print(classification_report(valid["label"], tuned_pred, labels=labels, zero_division=0, digits=4))
    print("confusion matrix rows=true cols=pred labels=", labels)
    print(confusion_matrix(valid["label"], tuned_pred, labels=labels))

    suspicion = train.copy()
    suspicion["model_pred"] = train_model_pred
    suspicion["weak_prob"] = weak_prob
    suspicion["weight"] = weak_weights
    suspicion = suspicion[suspicion["model_pred"] != suspicion["weak_label"]].sort_values(["weak_prob", "weak_confidence"])
    print("\nlow-confidence suspected noisy weak labels:")
    cols = ["id", "weak_label", "model_pred", "weak_confidence", "source_group", "annotator_group", "weak_prob", "weight"]
    print(suspicion[cols].head(10).to_string(index=False))
    print(f"\nmean kept rows across folds: {np.mean(kept_counts):.1f}/{len(train)}")
    return multipliers, oof_proba


def write_submission(path, ids, labels):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "label"])
        writer.writeheader()
        for sid, label in zip(ids, labels):
            writer.writerow({"id": sid, "label": label})


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def make_answer_zip(csv_path, zip_path):
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, arcname="results.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--out", default="results.csv")
    parser.add_argument("--zip-out", default="NS-2026-06-answer.zip")
    parser.add_argument("--predict-test", action="store_true", help="Read test.csv and create results.csv plus answer zip.")
    parser.add_argument("--drop-ratio", type=float, default=0.18)
    parser.add_argument("--gold-weight", type=float, default=6.0)
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train = pd.read_csv(data_dir / "train.csv")
    valid = pd.read_csv(data_dir / "trusted_valid.csv")
    labels = load_labels(data_dir)
    feature_cols = feature_columns(train)
    cfg = Config(drop_ratio=args.drop_ratio, gold_weight=args.gold_weight, n_splits=args.n_splits)

    print("config:", cfg, flush=True)
    print("labels:", labels, flush=True)
    multipliers, _ = run_cv(train, valid, feature_cols, labels, cfg)

    if not args.predict_test:
        print("\nNot reading test.csv. Add --predict-test to generate the submission file.")
        return

    test = pd.read_csv(data_dir / "test.csv")
    weak_prob, weak_weights, _, _ = initial_reliability(train, feature_cols, labels)
    bundle, kept = fit_bundle(train, valid, feature_cols, labels, cfg, weak_prob, weak_weights, SEED + 999)
    test_proba = predict_proba_bundle(bundle, test, feature_cols, labels)
    test_pred = predict_with_multipliers(test_proba, labels, multipliers)
    out = data_dir / args.out
    zip_out = data_dir / args.zip_out
    write_submission(out, test["id"], test_pred)
    make_answer_zip(out, zip_out)
    print(f"\nwrote {out} with kept weak rows {kept}/{len(train)}")
    print(f"sha256(results.csv)={sha256_file(out)}")
    print(f"wrote {zip_out}")


if __name__ == "__main__":
    main()
