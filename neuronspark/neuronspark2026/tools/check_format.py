import argparse
import csv
import json
import os


def load_labels(path):
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    labels = data.get("labels", data) if isinstance(data, dict) else data
    if isinstance(labels, dict):
        return set(labels)
    return {item["name"] if isinstance(item, dict) else str(item) for item in labels}


def read_expected_ids(path):
    if not path:
        return None
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "id" not in reader.fieldnames:
            raise SystemExit("ERROR: test CSV must contain id column")
        return {row["id"].strip() for row in reader if row.get("id", "").strip()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default="results.csv")
    parser.add_argument("--test-csv", default=None)
    parser.add_argument("--label-map", default="label_map.json")
    args = parser.parse_args()

    csv_path = os.path.join(args.path, "results.csv") if os.path.isdir(args.path) else args.path
    legal_labels = load_labels(args.label_map)
    expected_ids = read_expected_ids(args.test_csv)

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["id", "label"]:
            raise SystemExit("ERROR: results.csv header must be id,label")
        seen = set()
        for row in reader:
            sid = row["id"].strip()
            if not sid:
                raise SystemExit("ERROR: empty id")
            if sid in seen:
                raise SystemExit(f"ERROR: duplicate id {sid}")
            seen.add(sid)
            label = row["label"].strip()
            if label == "":
                raise SystemExit(f"ERROR: empty label for {sid}")
            if legal_labels is not None and label not in legal_labels:
                raise SystemExit(f"ERROR: illegal label for {sid}: {label}")

    if not seen:
        raise SystemExit("ERROR: results.csv must not be empty")
    if expected_ids is not None and seen != expected_ids:
        missing = sorted(expected_ids - seen)[:5]
        extra = sorted(seen - expected_ids)[:5]
        raise SystemExit(f"ERROR: id mismatch; missing={missing}, extra={extra}")
    print("OK: submission format is valid.")


if __name__ == "__main__":
    main()
