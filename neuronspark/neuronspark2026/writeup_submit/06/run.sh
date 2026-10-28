#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${DATA_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

python -u "$SCRIPT_DIR/solver_06.py" \
  --data-dir "$DATA_DIR" \
  --n-splits 5 \
  --predict-test \
  --out "$SCRIPT_DIR/results.csv" \
  --zip-out "$SCRIPT_DIR/NS-2026-06-answer.zip" \
  | tee "$SCRIPT_DIR/logs/local_cv_and_generation.log"

python "$DATA_DIR/tools/check_format.py" "$SCRIPT_DIR/results.csv" \
  --test-csv "$DATA_DIR/test.csv" \
  --label-map "$DATA_DIR/label_map.json" \
  | tee "$SCRIPT_DIR/logs/format_check.log"

python - <<'PY' "$SCRIPT_DIR" | tee "$SCRIPT_DIR/logs/hashes.txt"
import hashlib
import sys
from pathlib import Path

base = Path(sys.argv[1])
for name in ["results.csv", "NS-2026-06-answer.zip", "solver_06.py"]:
    path = base / name
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    print(digest.hexdigest(), name)
PY
