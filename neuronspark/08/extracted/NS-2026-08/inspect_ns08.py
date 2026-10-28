from __future__ import annotations

import copy
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "checker"))

from aiops_world import (  # noqa: E402
    HORIZON,
    action_is_helpful,
    apply_action,
    init_runtime,
    load_jsonl,
    minute_loss,
    normalized_score,
    observe,
    rollout_policy,
    task_public_view,
    validate_action,
    world_state,
)


def load_agent(agent_path: Path):
    spec = importlib.util.spec_from_file_location("agent_mod", agent_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module.Agent


def run_task(agent_cls, task):
    agent = agent_cls()
    agent.reset(copy.deepcopy(task_public_view(task)))
    runtime = init_runtime(task)
    minute_losses = []
    for t in range(int(task.get("horizon", HORIZON))):
        obs = observe(task, runtime, t)
        action = agent.act(copy.deepcopy(obs))
        ok, _msg, clean = validate_action(action, task, t)
        if not ok:
            clean = {"type": "NOOP"}
        apply_action(task, runtime, clean, t)
        minute_losses.append(minute_loss(task, runtime, t))
    total = sum(minute_losses)
    total += runtime["resource_cost"] + runtime["action_cost"]
    total += runtime["wrong_action_penalty"] + runtime["cooldown_penalty"]
    result = {
        "valid": True,
        "total_loss": total,
        "slo_loss": sum(minute_losses),
        "resource_cost": runtime["resource_cost"],
        "action_cost": runtime["action_cost"],
        "wrong_action_penalty": runtime["wrong_action_penalty"],
        "cooldown_penalty": runtime["cooldown_penalty"],
        "actions": runtime["actions"],
        "minute_losses": minute_losses,
    }
    return result, normalized_score(task, result)


def print_task(agent_cls, task):
    result, frac = run_task(agent_cls, task)
    noop = rollout_policy(task, "noop")
    oracle = rollout_policy(task, "oracle")
    modes = (world_state(task) or {}).get("modes", [])
    print(f"\n=== {task['task_id']} frac={frac:.6f} total={result['total_loss']:.3f} noop={noop['total_loss']:.3f} oracle={oracle['total_loss']:.3f}")
    print(
        "components "
        f"slo={result['slo_loss']:.3f} res={result['resource_cost']:.3f} "
        f"act={result['action_cost']:.3f} wrong={result['wrong_action_penalty']:.3f} "
        f"cd={result['cooldown_penalty']:.3f}"
    )
    print("modes:")
    for m in modes:
        print(
            f"  t={m.get('start'):2d} {m.get('mode'):21s} "
            f"svc={m.get('service')} dep={m.get('dependency', m.get('service'))} "
            f"sev={float(m.get('severity', 0)):.2f} imp={float(m.get('impact', 0)):.2f}"
        )
    print("actions:")
    for a in result["actions"]:
        if a.get("type") == "NOOP":
            continue
        label = "help"
        if a.get("cooldown_blocked"):
            label = "cooldown"
        elif not action_is_helpful(a, modes):
            label = "wrong"
        val = f" v={a.get('value')}" if "value" in a else ""
        print(f"  t={a.get('time'):2d} {a.get('type'):18s} {a.get('target', ''):7s}{val:8s} {label}")
    peaks = sorted(enumerate(result["minute_losses"]), key=lambda kv: kv[1], reverse=True)[:8]
    print("top minute losses:", " ".join(f"{t}:{v:.1f}" for t, v in peaks))


def main() -> int:
    agent_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "submission" / "agent.py"
    tasks_arg = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "valid_tasks.jsonl"
    agent_cls = load_agent(agent_arg)
    tasks = load_jsonl(tasks_arg)
    if len(sys.argv) > 3:
        wanted = set(sys.argv[3:])
        tasks = [t for t in tasks if t.get("task_id") in wanted]

    rows = []
    action_stats = defaultdict(Counter)
    for task in tasks:
        result, frac = run_task(agent_cls, task)
        rows.append((frac, task, result))
        modes = (world_state(task) or {}).get("modes", [])
        for a in result["actions"]:
            at = a.get("type")
            if at == "NOOP":
                continue
            action_stats[at]["total"] += 1
            if a.get("cooldown_blocked"):
                action_stats[at]["cooldown"] += 1
            elif action_is_helpful(a, modes):
                action_stats[at]["help"] += 1
            else:
                action_stats[at]["wrong"] += 1

    print(f"agent={agent_arg}")
    print(f"tasks={tasks_arg} n={len(tasks)}")
    print("action stats:")
    for at, c in sorted(action_stats.items(), key=lambda kv: -kv[1]["total"]):
        print(f"  {at:18s} total={c['total']:4d} help={c['help']:4d} wrong={c['wrong']:3d} cd={c['cooldown']:3d}")
    print("\nworst tasks:")
    for frac, task, result in sorted(rows, key=lambda x: x[0])[:12]:
        print(
            f"  {task['task_id']:22s} frac={frac:.6f} total={result['total_loss']:.1f} "
            f"wrong={result['wrong_action_penalty']:.1f} act={result['action_cost']:.1f}"
        )
    if len(sys.argv) > 3:
        for _frac, task, _result in sorted(rows, key=lambda x: x[0]):
            print_task(agent_cls, task)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
