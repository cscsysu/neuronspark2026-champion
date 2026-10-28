from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Print compact summaries for NS-2026-08 traces/tasks.")
    parser.add_argument("jsonl")
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()
    rows = []
    with Path(args.jsonl).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    for row in rows[:args.limit]:
        if "steps" in row:
            print(f"{row['trace_id']}: steps={len(row['steps'])} final_loss_bucket={row.get('final_loss_bucket')}")
            for step in row["steps"][:3]:
                obs = step["observation"]
                print(f"  t={obs['time']:02d} global={obs['global_health_bucket']} action={step['action']}")
        else:
            print(f"{row.get('task_id')}: horizon={row.get('horizon')} services={len(row.get('services', []))} has_proxy_world={'proxy_world' in row}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
