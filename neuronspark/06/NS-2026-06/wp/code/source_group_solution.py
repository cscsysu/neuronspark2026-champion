import argparse
import csv
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SEED = 260606
PRIORITY_LABELS = ["computer_hardware", "medical_health", "space_science"]
CLASS_MULTIPLIERS = {
    "computer_hardware": 1.30,
    "system_software": 0.95,
    "mobility_vehicle": 0.85,
    "sports_recreation": 0.95,
    "medical_health": 1.60,
    "space_science": 1.05,
}


def load_labels(data_dir):
    with open(data_dir / "label_map.json", "r", encoding="utf-8") as f:
        return list(json.load(f)["labels"].keys())


def feature_columns(df):
    return [c for c in df.columns if c.startswith("f")]


def score(y_true, y_pred, labels):
    macro = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    priority = recall_score(y_true, y_pred, labels=PRIORITY_LABELS, average="micro", zero_division=0)
    return macro, priority, 900 * macro + 200 * priority + 100


def source_to_label_map(valid):
    mapping = {}
    for source, part in valid.groupby("source_group"):
        counts = part["label"].value_counts()
        mapping[source] = counts.index[0]
    return mapping


def make_source_model():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=5000,
            C=0.8,
            class_weight="balanced",
            solver="lbfgs",
            random_state=SEED,
        ),
    )


def source_proba_to_label_proba(source_classes, source_proba, source_label_map, labels):
    label_proba = np.zeros((len(source_proba), len(labels)), dtype=np.float64)
    for i, source in enumerate(source_classes):
        label = source_label_map[source]
        label_proba[:, labels.index(label)] += source_proba[:, i]
    return label_proba


def predict_labels(label_proba, labels):
    multipliers = np.array([CLASS_MULTIPLIERS[label] for label in labels], dtype=np.float64)
    adjusted = label_proba * multipliers
    return np.array(labels, dtype=object)[adjusted.argmax(axis=1)]


def run_cv(train, valid, feature_cols, labels, n_splits):
    mapping = source_to_label_map(valid)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    oof = np.zeros((len(valid), len(labels)), dtype=np.float64)

    for fold, (fit_idx, hold_idx) in enumerate(splitter.split(valid, valid["label"]), start=1):
        source_train = pd.concat(
            [
                train[feature_cols + ["source_group"]],
                valid.iloc[fit_idx][feature_cols + ["source_group"]],
            ],
            ignore_index=True,
        )
        model = make_source_model()
        model.fit(source_train[feature_cols], source_train["source_group"])
        source_proba = model.predict_proba(valid.iloc[hold_idx][feature_cols])
        oof[hold_idx] = source_proba_to_label_proba(model.classes_, source_proba, mapping, labels)

        pred = predict_labels(oof[hold_idx], labels)
        fold_score = score(valid.iloc[hold_idx]["label"], pred, labels)
        print(
            f"fold={fold} macro_f1={fold_score[0]:.5f} "
            f"priority_recall={fold_score[1]:.5f} score_like={fold_score[2]:.2f}",
            flush=True,
        )

    pred = predict_labels(oof, labels)
    final_score = score(valid["label"], pred, labels)
    print(
        f"\nOOF tuned macro_f1={final_score[0]:.5f} "
        f"priority_recall={final_score[1]:.5f} score_like={final_score[2]:.2f}",
        flush=True,
    )
    print("source_to_label:", json.dumps(mapping, ensure_ascii=False, sort_keys=True))
    print("class_multipliers:", json.dumps(CLASS_MULTIPLIERS, ensure_ascii=False, sort_keys=True))
    print("\nclassification report:")
    print(classification_report(valid["label"], pred, labels=labels, zero_division=0, digits=4))
    print("confusion matrix rows=true cols=pred labels=", labels)
    print(confusion_matrix(valid["label"], pred, labels=labels))


def write_submission(path, ids, labels):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "label"])
        writer.writeheader()
        for sid, label in zip(ids, labels):
            writer.writerow({"id": sid, "label": label})


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def zip_answer(csv_path, zip_path):
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        info = zipfile.ZipInfo("results.csv", date_time=(2026, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o644 << 16
        with open(csv_path, "rb") as f:
            zf.writestr(info, f.read())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--out", default="results.csv")
    parser.add_argument("--zip-out", default="NS-2026-06-answer.zip")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--predict-test", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train = pd.read_csv(data_dir / "train.csv")
    valid = pd.read_csv(data_dir / "trusted_valid.csv")
    labels = load_labels(data_dir)
    feature_cols = feature_columns(train)

    run_cv(train, valid, feature_cols, labels, args.n_splits)

    if not args.predict_test:
        print("\nNot reading test.csv. Add --predict-test to create the final zip.")
        return

    test = pd.read_csv(data_dir / "test.csv")
    mapping = source_to_label_map(valid)
    source_train = pd.concat(
        [train[feature_cols + ["source_group"]], valid[feature_cols + ["source_group"]]],
        ignore_index=True,
    )
    model = make_source_model()
    model.fit(source_train[feature_cols], source_train["source_group"])
    label_proba = source_proba_to_label_proba(
        model.classes_,
        model.predict_proba(test[feature_cols]),
        mapping,
        labels,
    )
    pred = predict_labels(label_proba, labels)

    out_path = data_dir / args.out
    zip_path = data_dir / args.zip_out
    write_submission(out_path, test["id"], pred)
    zip_answer(out_path, zip_path)
    print(f"\nwrote {out_path}")
    print(f"sha256(results.csv)={sha256_file(out_path)}")
    print(f"wrote {zip_path}")


if __name__ == "__main__":
    main()
