
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "simulator"))

from dexsim_core import evaluate_submission, read_jsonl


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("submission_dir", help="Directory containing agent.py")
    parser.add_argument("--tasks", default="tasks/valid_tasks.jsonl")
    parser.add_argument("--max-score", type=float, default=1500.0)
    args = parser.parse_args(argv)
    task_path = Path(args.tasks)
    if not task_path.is_absolute():
        task_path = Path.cwd() / task_path
    tasks = read_jsonl(task_path)
    result = evaluate_submission(args.submission_dir, tasks, max_score=args.max_score)
    result["task_file"] = str(task_path)
    result["public_split_policy"] = "Public task files, including test_tasks.jsonl, are local sanity/debug splits only and must not be used as leaderboard ranking feedback."
    result["note"] = "Public MuJoCo dev-simulator evaluation if mujoco is installed; otherwise lightweight proxy fallback. Public test is local sanity only and must not rank leaderboard submissions. Official hidden evaluation generates private procedural rollouts inside the judge, applies unseen object/environment/scene/physics/stage shifts and uses shortfall aggregation across withheld capability groups. Agent reset/act calls run in an isolated subprocess with per-call timeouts; optional reset_batch/act_batch may batch cross-task rollouts so GPU policies can amortize inference on the 4090D resource boundary."
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
