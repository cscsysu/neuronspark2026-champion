import argparse
import csv
import math


FIELDS = ["date", "meal", "canteen_area", "volume"]
TEST_FIELDS = ["date", "meal", "canteen_area"]


def read_rows(path, expected_fields):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != expected_fields:
            raise SystemExit(f"ERROR: {path} header must be {','.join(expected_fields)}")
        return list(reader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", nargs="?", default="results.csv")
    parser.add_argument("--test-csv")
    args = parser.parse_args()

    rows = read_rows(args.submission, FIELDS)
    if not rows:
        raise SystemExit("ERROR: empty submission")

    seen = set()
    for n, row in enumerate(rows, start=1):
        k = (row["date"].strip(), row["meal"].strip(), row["canteen_area"].strip())
        if k in seen:
            raise SystemExit(f"ERROR: duplicate key {k}")
        seen.add(k)
        try:
            value = float(row["volume"])
        except Exception:
            raise SystemExit(f"ERROR: invalid volume at row {n}")
        if not math.isfinite(value) or value < 0:
            raise SystemExit(f"ERROR: invalid volume at row {n}")

    if args.test_csv:
        test_rows = read_rows(args.test_csv, TEST_FIELDS)
        expected = {(r["date"].strip(), r["meal"].strip(), r["canteen_area"].strip()) for r in test_rows}
        extra = seen - expected
        missing = expected - seen
        if extra:
            raise SystemExit(f"ERROR: unknown key, e.g. {sorted(extra)[0]}")
        if missing:
            raise SystemExit(f"ERROR: missing {len(missing)} keys, e.g. {sorted(missing)[0]}")

    print("OK: submission format is valid.")


if __name__ == "__main__":
    main()
