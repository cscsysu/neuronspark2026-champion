import argparse
import csv
import os
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score


PRIORITY_LABELS = {"computer_hardware", "medical_health", "space_science"}


def read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_submission(path, ids, labels):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "label"])
        writer.writeheader()
        for sid, label in zip(ids, labels):
            writer.writerow({"id": sid, "label": label})


def feature_columns(rows):
    return [name for name in rows[0].keys() if name.startswith("f")]


def matrix(rows, columns):
    return np.array([[float(row[col]) for col in columns] for row in rows], dtype=np.float64)


def score(y_true, y_pred):
    labels = sorted(set(y_true) | set(y_pred))
    macro = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    priority_gold = [label for label in y_true if label in PRIORITY_LABELS]
    if not priority_gold:
        priority_recall = 1.0
    else:
        priority_recall = recall_score(
            y_true,
            y_pred,
            labels=sorted(PRIORITY_LABELS),
            average="micro",
            zero_division=0,
        )
    return macro, priority_recall, 900 * macro + 200 * priority_recall + 100


def train_initial_model(x_train, y_weak, weights):
    model = LogisticRegression(
        max_iter=1000,
        C=2.0,
        class_weight="balanced",
        solver="lbfgs",
        random_state=260606,
    )
    model.fit(x_train, y_weak, sample_weight=weights)
    return model


def main():
    parser = argparse.ArgumentParser(description="NS-2026-06 public robust-label baseline.")
    parser.add_argument("--data-dir", default=".", help="directory containing train.csv, trusted_valid.csv and test.csv")
    parser.add_argument("--out", default="results.csv", help="output results.csv path")
    parser.add_argument("--drop-ratio", type=float, default=0.18, help="fraction of low-consistency weak labels to drop")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_rows = read_csv(data_dir / "train.csv")
    valid_rows = read_csv(data_dir / "trusted_valid.csv")
    test_rows = read_csv(data_dir / "test.csv")
    cols = feature_columns(train_rows)

    x_train = matrix(train_rows, cols)
    y_weak = np.array([row["weak_label"] for row in train_rows])
    weak_weights = np.array([float(row["weak_confidence"]) for row in train_rows])

    x_valid = matrix(valid_rows, cols)
    y_valid = np.array([row["label"] for row in valid_rows])
    x_test = matrix(test_rows, cols)

    initial = train_initial_model(x_train, y_weak, weak_weights)
    valid_pred = initial.predict(x_valid)
    macro, priority_recall, local_score = score(y_valid, valid_pred)
    print(f"initial valid macro_f1={macro:.5f} priority_recall={priority_recall:.5f} score_like={local_score:.2f}")

    train_probs = initial.predict_proba(x_train)
    cls_to_idx = {name: i for i, name in enumerate(initial.classes_)}
    weak_prob = np.array([train_probs[i, cls_to_idx[label]] for i, label in enumerate(y_weak)])
    threshold = np.quantile(weak_prob, max(0.0, min(0.45, args.drop_ratio)))
    keep = weak_prob >= threshold

    x_final = np.vstack([x_train[keep], x_valid])
    y_final = np.concatenate([y_weak[keep], y_valid])
    final_weights = np.concatenate([
        weak_weights[keep] * (0.55 + weak_prob[keep]),
        np.full(len(y_valid), 2.0),
    ])

    final = LogisticRegression(
        max_iter=1200,
        C=1.3,
        class_weight="balanced",
        solver="lbfgs",
        random_state=260607,
    )
    final.fit(x_final, y_final, sample_weight=final_weights)
    final_valid_pred = final.predict(x_valid)
    macro, priority_recall, local_score = score(y_valid, final_valid_pred)
    print(f"final self-check macro_f1={macro:.5f} priority_recall={priority_recall:.5f} score_like={local_score:.2f}")
    print(f"kept weak-label train rows: {int(keep.sum())}/{len(keep)}")

    test_pred = final.predict(x_test)
    write_submission(Path(args.out), [row["id"] for row in test_rows], test_pred)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
