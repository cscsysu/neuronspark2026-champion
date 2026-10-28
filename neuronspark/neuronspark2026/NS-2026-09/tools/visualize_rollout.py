
from __future__ import annotations

import argparse
import json


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="train.jsonl")
    parser.add_argument("--id", required=True)
    args = parser.parse_args()
    for row in read_jsonl(args.tasks):
        if row["id"] == args.id:
            print(json.dumps({
                "id": row["id"],
                "observation_mode": row["observation_mode"],
                "context_episodes": row["context_episodes"],
                "queries": row["queries"],
            }, ensure_ascii=False, indent=2))
            return 0
    raise SystemExit(f"id not found: {args.id}")


if __name__ == "__main__":
    raise SystemExit(main())
