
from __future__ import annotations

import argparse
import json

GRID_SIZE = 12
VIEW_RADIUS = 2
EVENT_KEYS = ["goal_reached", "collision", "hazard", "box_on_goal", "key_collected", "portal_used"]
EVENT_ORDER_WIDTH = 3


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def project_local_view(view):
    grid = [["." for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    for i in range(GRID_SIZE):
        grid[0][i] = "#"
        grid[GRID_SIZE - 1][i] = "#"
        grid[i][0] = "#"
        grid[i][GRID_SIZE - 1] = "#"
    cx = cy = GRID_SIZE // 2
    for vy, line in enumerate(view):
        for vx, ch in enumerate(line):
            if ch in {"A", "B", "O", "K", "D", "G", "H", "I", "P", "^", "v", "<", ">"}:
                x = cx + vx - VIEW_RADIUS
                y = cy + vy - VIEW_RADIUS
                if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
                    grid[y][x] = ch
    return ["".join(row) for row in grid]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="test.jsonl")
    parser.add_argument("--out", default="results.json")
    args = parser.parse_args()
    results = []
    for row in read_jsonl(args.tasks):
        for query in row["queries"]:
            results.append({
                "id": query["query_id"],
                "final_grid": project_local_view(query["initial_observation"]["local_view"]),
                "events": {key: False for key in EVENT_KEYS},
                "event_timeline": {key: "never" for key in EVENT_KEYS},
                "event_order": ["none"] * EVENT_ORDER_WIDTH,
                "terminal": "timeout",
            })
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"wrote {len(results)} predictions to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
