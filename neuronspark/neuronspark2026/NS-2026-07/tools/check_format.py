
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "simulator"))

from dexsim_core import AgentSubprocessProxy, DexSimEnv, read_jsonl, sanitize_task_info, validate_action


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("submission_dir", help="Directory containing agent.py")
    parser.add_argument("--tasks", default="tasks/train_tasks.jsonl")
    args = parser.parse_args(argv)
    data_root = ROOT
    task_path = Path(args.tasks)
    if not task_path.is_absolute():
        task_path = data_root / task_path
    tasks = read_jsonl(task_path)
    if not tasks:
        print("No tasks found", file=sys.stderr)
        return 2
    agent = None
    try:
        agent = AgentSubprocessProxy(args.submission_dir)
        sample = tasks[0]
        agent.reset(sanitize_task_info(sample))
        obs = DexSimEnv(sample, (sample.get("public_variants") or [{}])[0]).observation()
        action = agent.act(obs)
        _, err = validate_action(action)
        if err:
            print(f"Invalid action from Agent.act: {err}", file=sys.stderr)
            return 1
    except Exception as exc:
        print(f"Format check failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        if isinstance(agent, AgentSubprocessProxy):
            agent.close()
    print("OK: agent.py defines a loadable Agent and returns a legal action on a sample observation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
