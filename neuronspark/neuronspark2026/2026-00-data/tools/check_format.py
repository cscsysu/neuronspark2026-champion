import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Check NS-2026-00 results.csv format.")
    parser.add_argument("results_csv")
    parser.add_argument("--test-csv", default="test.csv")
    parser.add_argument("--label-map", default="label_map.json")
    args = parser.parse_args()

    labels = set(json.loads(Path(args.label_map).read_text(encoding="utf-8"))["labels"])
    with open(args.test_csv, "r", encoding="utf-8-sig", newline="") as f:
        expected_ids = {row["id"].strip() for row in csv.DictReader(f)}

    with open(args.results_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["id", "label"]:
            raise SystemExit("ERROR: results.csv header must be id,label")
        seen = set()
        for row in reader:
            sid = row["id"].strip()
            label = row["label"].strip()
            if sid in seen:
                raise SystemExit(f"ERROR: duplicate id {sid}")
            if sid not in expected_ids:
                raise SystemExit(f"ERROR: unknown id {sid}")
            if label not in labels:
                raise SystemExit(f"ERROR: illegal label {label}")
            seen.add(sid)
    missing = expected_ids - seen
    if missing:
        raise SystemExit(f"ERROR: missing {len(missing)} ids")
    print("OK: submission format is valid.")


if __name__ == "__main__":
    main()
