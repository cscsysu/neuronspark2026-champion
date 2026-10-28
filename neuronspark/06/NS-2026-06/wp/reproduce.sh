#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python -u source_group_solution.py --n-splits 5 --predict-test | tee wp/logs/local_cv_and_generation.log
python tools/check_format.py results.csv --test-csv test.csv --label-map label_map.json | tee wp/logs/format_check.log

cp source_group_solution.py wp/code/source_group_solution.py
cp results.csv wp/submission/results.csv
cp NS-2026-06-answer.zip wp/submission/NS-2026-06-answer.zip

python - <<'PY' | tee wp/logs/hashes.txt
import hashlib

for path in [
    "results.csv",
    "NS-2026-06-answer.zip",
    "source_group_solution.py",
    "wp/submission/results.csv",
    "wp/submission/NS-2026-06-answer.zip",
    "wp/code/source_group_solution.py",
]:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    print(digest.hexdigest(), path)
PY
