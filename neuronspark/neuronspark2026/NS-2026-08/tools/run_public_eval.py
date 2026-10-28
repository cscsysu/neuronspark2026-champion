from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run NS-2026-08 public proxy evaluation.")
    parser.add_argument("submission", help="Directory containing agent.py")
    parser.add_argument("--tasks", default=None)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    tasks = Path(args.tasks) if args.tasks else root / "public_tasks.jsonl"
    sys.path.insert(0, str(root / "checker"))
    from aiops_world import load_jsonl, score_agent
    result = score_agent(load_jsonl(tasks), args.submission, reveal=True)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("valid") else 2


if __name__ == "__main__":
    raise SystemExit(main())
