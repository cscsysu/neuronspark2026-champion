from __future__ import annotations

import argparse
import copy
import importlib.util
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate NS-2026-08 agent.py interface.")
    parser.add_argument("submission", help="Directory containing agent.py")
    parser.add_argument("--tasks", default=None, help="Task JSONL used for a smoke call; defaults to ../public_tasks.jsonl.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    tasks_path = Path(args.tasks) if args.tasks else root / "public_tasks.jsonl"
    sys.path.insert(0, str(root / "checker"))
    from aiops_world import load_jsonl, observe, init_runtime, task_public_view, validate_action

    agent_path = Path(args.submission) / "agent.py"
    if not agent_path.exists():
        print("ERROR: missing agent.py")
        return 2
    try:
        spec = importlib.util.spec_from_file_location("agent_check", agent_path)
        module = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(Path(args.submission).resolve()))
        spec.loader.exec_module(module)
        Agent = getattr(module, "Agent")
        agent = Agent()
        tasks = load_jsonl(tasks_path)
        task = tasks[0]
        if hasattr(agent, "reset"):
            agent.reset(copy.deepcopy(task_public_view(task)))
        obs = observe(task, init_runtime(task), 0)
        action = agent.act(copy.deepcopy(obs))
        ok, message, _ = validate_action(action, task, 0)
        if not ok:
            print(f"ERROR: invalid first action: {message}")
            return 2
        print("OK: agent.py defines Agent.reset/act-compatible interface.")
        return 0
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
