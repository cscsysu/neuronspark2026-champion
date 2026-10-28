#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${DATA_DIR:-/Users/tristonchen/Desktop/neuronspark2026/NS-2026-07}"

mkdir -p "$SCRIPT_DIR/logs"

# 1) 解压最终提交包
mkdir -p "$SCRIPT_DIR/my_submission"
unzip -o "$SCRIPT_DIR/NS-2026-07-answer.zip" -d "$SCRIPT_DIR/my_submission" > /dev/null

# 2) 格式检查
python "$DATA_DIR/tools/check_format.py" "$SCRIPT_DIR/my_submission" \
  | tee "$SCRIPT_DIR/logs/format_check.log"

# 3) 公开 valid 评测
python "$DATA_DIR/tools/run_public_eval.py" "$SCRIPT_DIR/my_submission" \
  --tasks "$DATA_DIR/tasks/valid_tasks.jsonl" \
  | tee "$SCRIPT_DIR/logs/eval_valid.log"

# 4) 公开 test 评测
python "$DATA_DIR/tools/run_public_eval.py" "$SCRIPT_DIR/my_submission" \
  --tasks "$DATA_DIR/tasks/test_tasks.jsonl" \
  | tee "$SCRIPT_DIR/logs/eval_test.log"

# 5) SHA256
python - <<'PY' "$SCRIPT_DIR" | tee "$SCRIPT_DIR/logs/hashes.txt"
import hashlib
import sys
from pathlib import Path

base = Path(sys.argv[1])
for name in ["NS-2026-07-answer.zip", "agent.py"]:
    path = base / name
    if not path.exists():
        print(f"MISSING {name}")
        continue
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    print(f"{digest.hexdigest()}  {name}")
PY
