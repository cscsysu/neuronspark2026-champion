from __future__ import annotations

import argparse
import json
from pathlib import Path

from aiops_world import load_jsonl, score_agent


def main() -> int:
    parser = argparse.ArgumentParser(description="Run public NS-2026-08 proxy evaluation.")
    parser.add_argument("--tasks", default="public_tasks.jsonl")
    parser.add_argument("--submission", required=True, help="Directory containing agent.py")
    parser.add_argument("--summary", help="Optional JSON output path")
    args = parser.parse_args()
    tasks = load_jsonl(args.tasks)
    result = score_agent(tasks, args.submission, reveal=True)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.summary:
        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if result.get("valid") else 2


if __name__ == "__main__":
    raise SystemExit(main())
