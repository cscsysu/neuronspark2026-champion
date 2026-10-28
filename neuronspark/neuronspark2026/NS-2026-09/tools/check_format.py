
from __future__ import annotations

import argparse
import json
import sys

GRID_SIZE = 12
EVENT_KEYS = ["goal_reached", "collision", "hazard", "box_on_goal", "key_collected", "portal_used"]
TIMELINE_VALUES = {"never", "early", "mid", "late"}
EVENT_ORDER_WIDTH = 3
EVENT_ORDER_VALUES = set(EVENT_KEYS + ["none"])
TERMINALS = {"goal", "hazard", "blocked", "active", "timeout"}
GRID_CHARS = set("#.ABOKDGHIP^v<>")


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def fail(msg: str) -> int:
    print(f"ERROR: {msg}", file=sys.stderr)
    return 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission")
    parser.add_argument("--tasks", default="test.jsonl")
    args = parser.parse_args()
    rows = read_jsonl(args.tasks)
    ids = {query["query_id"] for row in rows for query in row["queries"]}
    try:
        pred = json.load(open(args.submission, "r", encoding="utf-8"))
    except Exception as exc:
        return fail(f"cannot read submission: {exc}")
    if not isinstance(pred, list):
        return fail("top-level JSON must be list")
    seen = set()
    for item in pred:
        if not isinstance(item, dict):
            return fail("each item must be object")
        sid = item.get("id")
        if sid not in ids:
            return fail(f"unknown id: {sid}")
        if sid in seen:
            return fail(f"duplicate id: {sid}")
        seen.add(sid)
        grid = item.get("final_grid")
        if not isinstance(grid, list) or len(grid) != GRID_SIZE:
            return fail(f"invalid final_grid for {sid}")
        for row in grid:
            if not isinstance(row, str) or len(row) != GRID_SIZE or any(ch not in GRID_CHARS for ch in row):
                return fail(f"invalid final_grid row for {sid}")
        events = item.get("events")
        if not isinstance(events, dict):
            return fail(f"events must be object for {sid}")
        for key in EVENT_KEYS:
            if key not in events or not isinstance(events[key], bool):
                return fail(f"invalid event {key} for {sid}")
        timeline = item.get("event_timeline")
        if not isinstance(timeline, dict):
            return fail(f"event_timeline must be object for {sid}")
        for key in EVENT_KEYS:
            if key not in timeline or timeline[key] not in TIMELINE_VALUES:
                return fail(f"invalid event_timeline {key} for {sid}")
        order = item.get("event_order")
        if not isinstance(order, list) or len(order) != EVENT_ORDER_WIDTH:
            return fail(f"event_order must have {EVENT_ORDER_WIDTH} items for {sid}")
        for value in order:
            if value not in EVENT_ORDER_VALUES:
                return fail(f"invalid event_order for {sid}")
        if item.get("terminal") not in TERMINALS:
            return fail(f"invalid terminal for {sid}")
    missing = ids - seen
    if missing:
        return fail(f"missing ids, first={sorted(missing)[:3]}")
    print(f"OK: {len(pred)} predictions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
