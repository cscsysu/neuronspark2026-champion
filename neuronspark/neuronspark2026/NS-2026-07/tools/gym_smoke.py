#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve()
PUBLIC_ROOT = HERE.parents[1] if (HERE.parents[1] / "simulator" / "gymnasium_env.py").exists() else None
if PUBLIC_ROOT is not None:
    sys.path.insert(0, str(PUBLIC_ROOT / "simulator"))

from gymnasium_env import DexSimGymEnv  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke test the NS-2026-07 Gymnasium-style dev wrapper.")
    parser.add_argument("--tasks", default="tasks/valid_tasks.jsonl")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args(argv)

    env = DexSimGymEnv(
        task_file=args.tasks,
        task_index=args.index,
        observation_mode="compact" if args.compact else "public",
    )
    obs, info = env.reset(seed=0)
    total_reward = 0.0
    final_info = info
    for _ in range(args.steps):
        obs, reward, terminated, truncated, final_info = env.step(env.sample_action())
        total_reward += float(reward)
        if terminated or truncated:
            break
    result = {
        "task_id": info.get("task_id"),
        "sim_backend": final_info.get("sim_backend"),
        "observation_mode": "compact" if args.compact else "public",
        "steps_requested": args.steps,
        "total_reward": round(total_reward, 9),
        "terminated_or_truncated": bool(final_info.get("state", {}).get("done", False)),
        "final_metrics": final_info.get("metrics", {}),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
