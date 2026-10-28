from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any


VERSION_ID = "NS-2026-08-r20260525-07-score-calibrated"
MAX_SCORE = 1500.0
HORIZON = 48

ACTION_TYPES = {
    "SCALE",
    "ROLLBACK_RELEASE",
    "ROLLBACK_CONFIG",
    "RESTART",
    "CIRCUIT_BREAK",
    "SET_TIMEOUT",
    "WARM_CACHE",
    "REBUILD_INDEX",
    "DRAIN_QUEUE",
    "THROTTLE",
    "DIAGNOSTIC_PROBE",
    "NOOP",
}

ACTION_COST = {
    "SCALE": 1.3,
    "ROLLBACK_RELEASE": 4.2,
    "ROLLBACK_CONFIG": 3.4,
    "RESTART": 2.7,
    "CIRCUIT_BREAK": 2.9,
    "SET_TIMEOUT": 1.8,
    "WARM_CACHE": 2.4,
    "REBUILD_INDEX": 5.7,
    "DRAIN_QUEUE": 3.2,
    "THROTTLE": 2.8,
    "DIAGNOSTIC_PROBE": 1.6,
    "NOOP": 0.0,
}

ACTION_COOLDOWN = {
    "SCALE": 4,
    "ROLLBACK_RELEASE": 12,
    "ROLLBACK_CONFIG": 10,
    "RESTART": 8,
    "CIRCUIT_BREAK": 6,
    "SET_TIMEOUT": 5,
    "WARM_CACHE": 5,
    "REBUILD_INDEX": 16,
    "DRAIN_QUEUE": 7,
    "THROTTLE": 6,
    "DIAGNOSTIC_PROBE": 5,
}

HEALTH_FIELDS = ["latency_bucket", "error_bucket", "saturation_bucket", "queue_bucket"]

HELPFUL = {
    "release_instability": {"ROLLBACK_RELEASE", "CIRCUIT_BREAK"},
    "config_shift": {"ROLLBACK_CONFIG", "SET_TIMEOUT"},
    "heap_growth": {"RESTART", "SCALE"},
    "traffic_surge": {"SCALE", "THROTTLE"},
    "storage_wait": {"REBUILD_INDEX", "SET_TIMEOUT", "RESTART"},
    "consumer_lag": {"SCALE", "DRAIN_QUEUE"},
    "cache_collapse": {"WARM_CACHE", "THROTTLE", "CIRCUIT_BREAK"},
    "dependency_timeout": {"CIRCUIT_BREAK", "SET_TIMEOUT"},
}

EVENT_CHANNEL = {
    "release_instability": "ch_00",
    "config_shift": "ch_01",
    "heap_growth": "ch_02",
    "traffic_surge": "ch_03",
    "storage_wait": "ch_04",
    "consumer_lag": "ch_05",
    "cache_collapse": "ch_06",
    "dependency_timeout": "ch_07",
}

ACTION_HINT = {
    "release_instability": "rollback_release",
    "config_shift": "rollback_config",
    "heap_growth": "restart_scale",
    "traffic_surge": "scale_throttle",
    "storage_wait": "storage_index",
    "consumer_lag": "queue_drain",
    "cache_collapse": "cache_warm",
    "dependency_timeout": "timeout_circuit",
}


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def write_json(path: str | Path, payload: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def service_map(task: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {svc["alias"]: svc for svc in task.get("services", [])}


def topology_neighbors(task: dict[str, Any]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {svc["alias"]: set() for svc in task.get("services", [])}
    for edge in task.get("topology", []):
        src = edge.get("from")
        dst = edge.get("to")
        if src in out and dst:
            out[src].add(dst)
        if dst in out and src:
            out[dst].add(src)
    return out


def world_state(task: dict[str, Any]) -> dict[str, Any] | None:
    state = task.get("private_world")
    if isinstance(state, dict):
        return state
    state = task.get("proxy_world")
    if isinstance(state, dict):
        return state
    return None


def task_public_view(task: dict[str, Any]) -> dict[str, Any]:
    return {k: copy.deepcopy(v) for k, v in task.items() if k not in {"private_world", "proxy_world"}}


def stable_unit(seed: str, *parts: Any) -> float:
    text = "|".join([seed, *[str(p) for p in parts]])
    value = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)
    return (value % 1000000) / 1000000.0


def bucket(value: float, cuts: list[float]) -> int:
    for idx, cut in enumerate(cuts):
        if value < cut:
            return idx
    return len(cuts)


def init_runtime(task: dict[str, Any]) -> dict[str, Any]:
    services = service_map(task)
    return {
        "replicas": {alias: int(svc.get("base_replicas", 1)) for alias, svc in services.items()},
        "flags": {
            "rollback_release": {},
            "rollback_config": {},
            "restart": {},
            "circuit": {},
            "timeout": {},
            "warm_cache": {},
            "rebuild_index": {},
            "drain_queue": {},
            "throttle": {},
        },
        "action_cost": 0.0,
        "resource_cost": 0.0,
        "wrong_action_penalty": 0.0,
        "cooldown_penalty": 0.0,
        "transient_loss": 0.0,
        "actions": [],
        "cooldown_until": {},
        "telemetry_history": {},
        "probe_queue": [],
    }


def flag_active(runtime: dict[str, Any], key: str, service: str, t: int, delay: int = 0, duration: int | None = None) -> bool:
    start = runtime["flags"].get(key, {}).get(service)
    if start is None or t < start + delay:
        return False
    if duration is not None and t > start + duration:
        return False
    return True


def action_targets_mode(action: dict[str, Any], mode: dict[str, Any]) -> bool:
    target = action.get("target")
    return target in {mode.get("service"), mode.get("dependency"), "svc_00", "svc_07", "svc_08", "svc_09"}


def action_is_helpful(action: dict[str, Any], modes: list[dict[str, Any]]) -> bool:
    atype = action.get("type")
    if atype == "NOOP":
        return True
    if atype == "DIAGNOSTIC_PROBE":
        return True
    for mode in modes:
        if atype in HELPFUL.get(mode.get("mode", ""), set()) and action_targets_mode(action, mode):
            return True
    return False


def validate_action(action: Any, task: dict[str, Any], t: int) -> tuple[bool, str, dict[str, Any]]:
    if not isinstance(action, dict):
        return False, "act() must return an action object", {}
    atype = action.get("type", "NOOP")
    if atype not in ACTION_TYPES:
        return False, f"unsupported action type {atype!r}", {}
    clean = {"type": atype}
    if atype == "NOOP":
        return True, "ok", clean

    services = service_map(task)
    target = action.get("target")
    if not isinstance(target, str) or target not in services:
        return False, f"unknown target {target!r}", {}
    clean["target"] = target

    value = action.get("value")
    if atype == "SCALE":
        if not isinstance(value, int):
            return False, "SCALE requires integer value", {}
        max_replicas = int(services[target].get("max_replicas", 1))
        if value < 1 or value > max_replicas:
            return False, f"SCALE value outside [1, {max_replicas}]", {}
        clean["value"] = value
    elif atype == "THROTTLE":
        if not isinstance(value, int) or value < 0 or value > 60:
            return False, "THROTTLE value must be an integer percent in [0, 60]", {}
        clean["value"] = value
    elif atype == "SET_TIMEOUT":
        if not isinstance(value, int) or value not in {100, 250, 500, 750, 1000, 1500}:
            return False, "SET_TIMEOUT value must be one of 100, 250, 500, 750, 1000, 1500", {}
        clean["value"] = value
    return True, "ok", clean


def latent_edges(world: dict[str, Any]) -> list[dict[str, Any]]:
    rows = world.get("latent_coupling_graph", [])
    return [edge for edge in rows if isinstance(edge, dict)]


def latent_strength(world: dict[str, Any], source: str, target: str, t: int, start: int = 0) -> float:
    best = 0.0
    for edge in latent_edges(world):
        src = str(edge.get("from", ""))
        dst = str(edge.get("to", ""))
        lag = int(edge.get("lag", 0))
        weight = float(edge.get("weight", 0.0))
        if src == source and dst == target and t >= start + lag:
            best = max(best, weight)
        elif dst == source and src == target and t >= start + lag + 1:
            best = max(best, weight * 0.35)
    return best


def mode_target_relation(task: dict[str, Any], world: dict[str, Any], mode: dict[str, Any], alias: str, t: int) -> tuple[float, float]:
    start = int(mode.get("start", 0))
    service = str(mode.get("service", ""))
    dep = str(mode.get("dependency", service))
    targets = {service, dep}
    if alias in targets:
        return 1.0, 0.0
    neighbors = topology_neighbors(task)
    if alias in neighbors.get(service, set()) or alias in neighbors.get(dep, set()):
        return 0.45, 0.0
    latent = max(latent_strength(world, service, alias, t, start), latent_strength(world, dep, alias, t, start))
    if latent > 0:
        return 0.52 + 0.42 * latent, latent
    return 0.0, 0.0


def make_probe_result(task: dict[str, Any], target: str, t: int) -> dict[str, Any]:
    world = world_state(task) or {}
    seed = str(world.get("noise_key", task.get("task_id", "")))
    services = sorted(service_map(task))
    best = {"score": 0.0, "source_hint": target, "latent": 0.0, "confidence": 0.05, "mode": ""}
    for mode in world.get("modes", []):
        start = int(mode.get("start", 0))
        if t < start:
            continue
        relation, latent = mode_target_relation(task, world, mode, target, t)
        if relation <= 0:
            continue
        elapsed = min(24, max(0, t - start))
        score = relation * float(mode.get("severity", 0.7)) * float(mode.get("impact", 1.0)) * (1.0 + elapsed / 28.0)
        if score > best["score"]:
            best = {
                "score": score,
                "source_hint": str(mode.get("dependency", mode.get("service", target))),
                "latent": latent,
                "confidence": min(0.98, 0.25 + relation * 0.52 + score * 0.16),
                "mode": str(mode.get("mode", "")),
            }
    diagnostic = world.get("diagnostic", {})
    noise_rate = float(diagnostic.get("probe_noise_rate", 0.08))
    noisy = stable_unit(seed, t, target, "probe_noise") < noise_rate
    if noisy and services:
        best["source_hint"] = services[int(stable_unit(seed, t, target, "probe_wrong") * len(services)) % len(services)]
        best["confidence"] *= 0.45
    return {
        "type": "diagnostic_probe",
        "target": target,
        "source_hint": best["source_hint"],
        "pressure_bucket": bucket(float(best["score"]), [0.18, 0.42, 0.70, 1.05, 1.45]),
        "coupling_bucket": bucket(float(best["latent"]), [0.08, 0.18, 0.30, 0.46, 0.64]),
        "confidence_bucket": bucket(float(best["confidence"]), [0.18, 0.32, 0.48, 0.66, 0.82]),
        "action_hint": ACTION_HINT.get(str(best.get("mode", "")), "investigate"),
        "noisy": bool(noisy),
    }


def apply_action(task: dict[str, Any], runtime: dict[str, Any], action: dict[str, Any], t: int) -> None:
    world = world_state(task) or {}
    modes = list(world.get("modes", []))
    atype = action.get("type", "NOOP")
    target = action.get("target", "")
    recorded = {"time": t, **copy.deepcopy(action)}
    runtime["action_cost"] += ACTION_COST.get(atype, 1.0)
    cooldown_key = f"{atype}:{target}"
    if atype != "NOOP":
        until = int(runtime["cooldown_until"].get(cooldown_key, -1))
        if t < until:
            recorded["cooldown_blocked"] = True
            runtime["cooldown_penalty"] += 3.5 + 0.35 * (until - t)
            runtime["transient_loss"] += 1.0
            runtime["actions"].append(recorded)
            return
        runtime["cooldown_until"][cooldown_key] = t + int(ACTION_COOLDOWN.get(atype, 4))
    runtime["actions"].append(recorded)
    if not action_is_helpful(action, modes):
        runtime["wrong_action_penalty"] += 5.0
    if atype == "SCALE":
        runtime["replicas"][target] = int(action["value"])
        if int(action["value"]) >= int(service_map(task)[target].get("max_replicas", 1)):
            runtime["transient_loss"] += 0.8
    elif atype == "ROLLBACK_RELEASE":
        runtime["flags"]["rollback_release"][target] = t
        runtime["transient_loss"] += 1.2 * float(service_map(task)[target].get("criticality", 1.0))
    elif atype == "ROLLBACK_CONFIG":
        runtime["flags"]["rollback_config"][target] = t
        runtime["transient_loss"] += 0.9 * float(service_map(task)[target].get("criticality", 1.0))
    elif atype == "RESTART":
        runtime["flags"]["restart"][target] = t
        runtime["transient_loss"] += 2.4 * float(service_map(task)[target].get("criticality", 1.0))
    elif atype == "CIRCUIT_BREAK":
        runtime["flags"]["circuit"][target] = t
        runtime["transient_loss"] += 0.7 * float(service_map(task)[target].get("criticality", 1.0))
    elif atype == "SET_TIMEOUT":
        runtime["flags"]["timeout"][target] = t
    elif atype == "WARM_CACHE":
        runtime["flags"]["warm_cache"][target] = t
    elif atype == "REBUILD_INDEX":
        runtime["flags"]["rebuild_index"][target] = t
        runtime["transient_loss"] += 4.2
    elif atype == "DRAIN_QUEUE":
        runtime["flags"]["drain_queue"][target] = t
    elif atype == "THROTTLE":
        runtime["flags"]["throttle"][target] = t
    elif atype == "DIAGNOSTIC_PROBE":
        diagnostic = world.get("diagnostic", {})
        observability = world.get("observability", {})
        delay_max = int(diagnostic.get("probe_delay_max", observability.get("probe_delay_max", 1)))
        delay = int(stable_unit(str(world.get("noise_key", task.get("task_id", ""))), t, target, "probe_delay") * (delay_max + 1))
        runtime.setdefault("probe_queue", []).append({
            "created": t,
            "due": min(HORIZON - 1, t + 1 + delay),
            "result": make_probe_result(task, target, t),
        })


def scale_factor(task: dict[str, Any], runtime: dict[str, Any], target: str) -> float:
    svc = service_map(task).get(target)
    if not svc:
        return 1.0
    base = max(1, int(svc.get("base_replicas", 1)))
    extra = max(0, int(runtime["replicas"].get(target, base)) - base)
    return max(0.38, 1.0 - 0.15 * extra)


def mode_loss(task: dict[str, Any], runtime: dict[str, Any], mode: dict[str, Any], t: int, traffic: float) -> float:
    start = int(mode["start"])
    if t < start:
        return 0.0
    elapsed = t - start
    m = mode["mode"]
    service = mode["service"]
    dep = mode.get("dependency", service)
    severity = float(mode.get("severity", 0.7))
    impact = float(mode.get("impact", 1.0))
    base = 0.0
    factor = 1.0

    if m == "release_instability":
        base = severity * (26.0 + min(20.0, elapsed * 0.9))
        if flag_active(runtime, "rollback_release", service, t, delay=2):
            factor *= 0.10
        if flag_active(runtime, "circuit", service, t, delay=1) or flag_active(runtime, "circuit", "svc_00", t, delay=1):
            factor *= 0.72
    elif m == "config_shift":
        base = severity * (22.0 + min(15.0, elapsed * 0.55))
        if flag_active(runtime, "rollback_config", service, t, delay=1):
            factor *= 0.14
        if flag_active(runtime, "timeout", service, t, delay=1):
            factor *= 0.64
    elif m == "heap_growth":
        base = severity * (14.0 + min(34.0, elapsed * 1.22))
        if flag_active(runtime, "restart", service, t, delay=1, duration=12):
            factor *= 0.22
        factor *= scale_factor(task, runtime, service)
    elif m == "traffic_surge":
        base = severity * (20.0 + traffic * 22.0)
        factor *= min(scale_factor(task, runtime, "svc_00"), scale_factor(task, runtime, service))
        if flag_active(runtime, "throttle", "svc_00", t) or flag_active(runtime, "throttle", service, t):
            factor *= 0.72
    elif m == "storage_wait":
        base = severity * (31.0 + min(22.0, elapsed * 0.8))
        if flag_active(runtime, "rebuild_index", dep, t, delay=4):
            factor *= 0.13
        if flag_active(runtime, "timeout", service, t, delay=1):
            factor *= 0.74
        if flag_active(runtime, "restart", dep, t, delay=2, duration=8):
            factor *= 0.82
    elif m == "consumer_lag":
        base = severity * (23.0 + min(25.0, elapsed * 1.0))
        factor *= scale_factor(task, runtime, "svc_09")
        if flag_active(runtime, "drain_queue", "svc_08", t, delay=2, duration=14):
            factor *= 0.32
    elif m == "cache_collapse":
        base = severity * (25.0 + traffic * 15.0)
        if flag_active(runtime, "warm_cache", "svc_07", t, delay=2):
            factor *= 0.17
        if flag_active(runtime, "throttle", "svc_00", t) or flag_active(runtime, "throttle", service, t):
            factor *= 0.74
    elif m == "dependency_timeout":
        base = severity * (24.0 + min(17.0, elapsed * 0.62))
        if flag_active(runtime, "circuit", service, t, delay=1) or flag_active(runtime, "circuit", "svc_00", t, delay=1):
            factor *= 0.42
        if flag_active(runtime, "timeout", dep, t, delay=1) or flag_active(runtime, "timeout", service, t, delay=1):
            factor *= 0.60
    return base * factor * impact


def service_pressure(task: dict[str, Any], runtime: dict[str, Any], alias: str, t: int, traffic: float) -> float:
    world = world_state(task) or {}
    pressure = 0.0
    for mode in world.get("modes", []):
        if t < int(mode.get("start", 0)):
            continue
        relation, latent = mode_target_relation(task, world, mode, alias, t)
        if relation > 0:
            pressure += relation + 0.22 * latent
    pressure += max(0.0, traffic - 1.0) * 0.55
    return pressure


def latent_coupling_loss(task: dict[str, Any], mode: dict[str, Any], t: int, traffic: float) -> float:
    world = world_state(task) or {}
    start = int(mode.get("start", 0))
    if t < start:
        return 0.0
    service = str(mode.get("service", ""))
    dep = str(mode.get("dependency", service))
    severity = float(mode.get("severity", 0.7))
    impact = float(mode.get("impact", 1.0))
    services = service_map(task)
    extra = 0.0
    for alias, svc in services.items():
        if alias in {service, dep}:
            continue
        latent = max(latent_strength(world, service, alias, t, start), latent_strength(world, dep, alias, t, start))
        if latent <= 0:
            continue
        extra += latent * severity * impact * traffic * 7.5 * float(svc.get("criticality", 1.0))
    return extra


def minute_loss(task: dict[str, Any], runtime: dict[str, Any], t: int) -> float:
    world = world_state(task)
    if world is None:
        raise ValueError("task has no proxy_world/private_world for scoring")
    traffic = float(task.get("traffic_forecast", [1.0])[min(t, HORIZON - 1)])
    loss = runtime.pop("transient_loss", 0.0)
    runtime["transient_loss"] = 0.0
    for mode in world.get("modes", []):
        loss += mode_loss(task, runtime, mode, t, traffic)
        loss += latent_coupling_loss(task, mode, t, traffic)

    services = service_map(task)
    for alias, svc in services.items():
        base = max(1, int(svc.get("base_replicas", 1)))
        replicas = int(runtime["replicas"].get(alias, base))
        extra = max(0, replicas - base)
        runtime["resource_cost"] += extra * float(svc.get("cost", 1.0)) * 0.17
        capacity = float(svc.get("capacity", 1.0)) * replicas
        load = traffic * float(svc.get("criticality", 1.0)) * (0.70 if svc.get("role") in {"edge", "api"} else 0.42)
        if load > capacity:
            loss += (load / capacity - 1.0) * 8.8 * float(svc.get("criticality", 1.0))
    for target, start in runtime["flags"]["throttle"].items():
        if t >= start:
            loss += traffic * 0.55
    for target, start in runtime["flags"]["circuit"].items():
        if start <= t <= start + 12:
            loss += traffic * 0.28 * float(services.get(target, {}).get("criticality", 1.0))
    for target, start in runtime["flags"]["rollback_release"].items():
        if start <= t <= start + 4:
            loss += 0.35 * float(services.get(target, {}).get("criticality", 1.0))
    for target, start in runtime["flags"]["rollback_config"].items():
        if start <= t <= start + 3:
            loss += 0.25 * float(services.get(target, {}).get("criticality", 1.0))
    return loss


def observe(task: dict[str, Any], runtime: dict[str, Any], t: int) -> dict[str, Any]:
    world = world_state(task) or {"noise_key": "public"}
    seed = str(world.get("noise_key", task.get("task_id", "")))
    observability = world.get("observability", {})
    metric_delay_max = int(observability.get("metric_delay_max", 0))
    metric_drop_rate = float(observability.get("metric_drop_rate", 0.0))
    event_delay_max = int(observability.get("event_delay_max", 0))
    event_drop_rate = float(observability.get("event_drop_rate", 0.0))
    traffic = float(task.get("traffic_forecast", [1.0])[min(t, HORIZON - 1)])
    raw_health = []
    global_pressure = 0.0
    for svc in task.get("services", []):
        alias = svc["alias"]
        pressure = service_pressure(task, runtime, alias, t, traffic)
        global_pressure += pressure * float(svc.get("criticality", 1.0))
        noise = stable_unit(seed, t, alias, "h") - 0.5
        latency = 115 + 36 * traffic + 42 * pressure + noise * 16
        errors = max(0.0, 0.004 + 0.010 * pressure + (stable_unit(seed, t, alias, "e") - 0.5) * 0.006)
        saturation = min(0.99, 0.28 + 0.18 * traffic + 0.20 * pressure + stable_unit(seed, t, alias, "s") * 0.08)
        queue = max(0.0, pressure * 0.60 + stable_unit(seed, t, alias, "q") * 0.18)
        raw_health.append({
            "service": alias,
            "latency_bucket": bucket(latency, [135, 165, 195, 230, 270]),
            "error_bucket": bucket(errors, [0.006, 0.010, 0.015, 0.022, 0.034]),
            "saturation_bucket": bucket(saturation, [0.45, 0.58, 0.70, 0.82, 0.92]),
            "queue_bucket": bucket(queue, [0.12, 0.30, 0.55, 0.85, 1.25]),
        })
    history = runtime.setdefault("telemetry_history", {})
    history[t] = copy.deepcopy(raw_health)
    health = []
    for item in raw_health:
        alias = item["service"]
        delay = int(stable_unit(seed, t, alias, "metric_delay") * (metric_delay_max + 1)) if metric_delay_max > 0 else 0
        source_t = max(0, t - delay)
        source_items = {row["service"]: row for row in history.get(source_t, raw_health)}
        observed = copy.deepcopy(source_items.get(alias, item))
        observed["service"] = alias
        if delay:
            observed["stale_steps"] = delay
        missing = []
        for field in HEALTH_FIELDS:
            if metric_drop_rate > 0 and stable_unit(seed, t, alias, field, "drop") < metric_drop_rate:
                observed.pop(field, None)
                missing.append(field)
        if missing:
            observed["missing_fields"] = missing
        health.append(observed)
    events = []
    for mode in world.get("modes", []):
        start = int(mode.get("start", 0))
        event_delay = int(stable_unit(seed, mode.get("mode"), mode.get("service"), "event_delay") * (event_delay_max + 1)) if event_delay_max > 0 else 0
        source_t = t - event_delay
        if source_t < 0:
            continue
        if event_drop_rate > 0 and stable_unit(seed, t, mode.get("mode"), mode.get("service"), "event_drop") < event_drop_rate:
            continue
        if start <= source_t <= min(HORIZON - 1, start + 10) and (source_t - start) % 3 == 0:
            channel_map = world.get("event_channel_map") or EVENT_CHANNEL
            channel = channel_map.get(mode.get("mode"), "ch_x")
            token_id = int(stable_unit(seed, mode.get("mode"), mode.get("service"), "tok") * 97) % 31
            event = {
                "channel": channel,
                "service": mode.get("service"),
                "token": f"tok_{token_id:02d}",
                "age": source_t - start,
            }
            if event_delay:
                event["stale_steps"] = event_delay
            events.append(event)
    if stable_unit(seed, t, "background") < 0.18:
        events.append({"channel": "ch_bg", "service": f"svc_{int(stable_unit(seed, t, 'svc') * 10):02d}", "token": "tok_bg", "age": 0})
    probe_results = []
    remaining_probe_queue = []
    for item in runtime.setdefault("probe_queue", []):
        due = int(item.get("due", HORIZON))
        if due <= t:
            result = copy.deepcopy(item.get("result", {}))
            result["stale_steps"] = max(0, t - int(item.get("created", t)))
            probe_results.append(result)
        else:
            remaining_probe_queue.append(item)
    runtime["probe_queue"] = remaining_probe_queue
    return {
        "time": t,
        "traffic_bucket": bucket(traffic, [0.70, 0.90, 1.05, 1.20, 1.40]),
        "global_health_bucket": bucket(global_pressure, [0.8, 1.8, 3.0, 4.6, 6.4]),
        "health": health,
        "events": events[:6],
        "probe_results": probe_results[:3],
        "recent_actions": copy.deepcopy(runtime.get("actions", [])[-5:]),
        "telemetry_quality": {
            "metric_delay_max": metric_delay_max,
            "event_delay_max": event_delay_max,
            "metric_drop_bucket": bucket(metric_drop_rate, [0.03, 0.07, 0.11, 0.16]),
            "event_drop_bucket": bucket(event_drop_rate, [0.02, 0.05, 0.09, 0.14]),
            "probe_delay_max": int((world.get("diagnostic", {}) or {}).get("probe_delay_max", observability.get("probe_delay_max", 0))),
        },
    }


def rollout_actions(task: dict[str, Any], action_by_time: dict[int, dict[str, Any]] | None = None, trace: bool = False) -> dict[str, Any]:
    runtime = init_runtime(task)
    total_loss = 0.0
    steps = []
    for t in range(int(task.get("horizon", HORIZON))):
        obs = observe(task, runtime, t)
        action = copy.deepcopy((action_by_time or {}).get(t, {"type": "NOOP"}))
        ok, message, clean = validate_action(action, task, t)
        if not ok:
            return {"valid": False, "message": message, "total_loss": 10**9, "steps": steps}
        apply_action(task, runtime, clean, t)
        loss = minute_loss(task, runtime, t)
        total_loss += loss
        if trace:
            steps.append({"observation": obs, "action": clean, "loss_bucket": bucket(loss, [4, 10, 18, 32, 52])})
    total_loss += runtime["resource_cost"] + runtime["action_cost"] + runtime["wrong_action_penalty"] + runtime["cooldown_penalty"]
    return {
        "valid": True,
        "message": "ok",
        "total_loss": total_loss,
        "slo_loss": total_loss - runtime["resource_cost"] - runtime["action_cost"] - runtime["wrong_action_penalty"] - runtime["cooldown_penalty"],
        "resource_cost": runtime["resource_cost"],
        "action_cost": runtime["action_cost"],
        "wrong_action_penalty": runtime["wrong_action_penalty"],
        "cooldown_penalty": runtime["cooldown_penalty"],
        "actions": runtime["actions"],
        "steps": steps,
    }


def reactive_action(task_info: dict[str, Any], observation: dict[str, Any], memory: dict[str, Any]) -> dict[str, Any]:
    services = {svc["alias"]: svc for svc in task_info.get("services", [])}
    t = int(observation.get("time", 0))
    used = memory.setdefault("used", set())
    health = observation.get("health", [])
    worst = max(health, key=lambda h: (h.get("error_bucket", 0) + h.get("latency_bucket", 0) + h.get("saturation_bucket", 0) + h.get("queue_bucket", 0)), default={"service": "svc_00"})
    target = worst.get("service", "svc_00")
    if not isinstance(target, str) or target not in services:
        target = "svc_00"
    queue_target = max(health, key=lambda h: h.get("queue_bucket", 0), default={}).get("service", "svc_08")

    def first(key: str, action: dict[str, Any]) -> dict[str, Any] | None:
        if key in used:
            return None
        used.add(key)
        return action

    if observation.get("global_health_bucket", 0) >= 4 and ("scale_gateway" not in used):
        svc = services.get("svc_00", {})
        return first("scale_gateway", {"type": "SCALE", "target": "svc_00", "value": min(int(svc.get("max_replicas", 8)), int(svc.get("base_replicas", 1)) + 3)}) or {"type": "NOOP"}
    if queue_target == "svc_08" or any(h.get("queue_bucket", 0) >= 4 for h in health):
        action = first("drain_queue", {"type": "DRAIN_QUEUE", "target": "svc_08"})
        if action:
            return action
        svc = services.get("svc_09", {})
        action = first("scale_worker", {"type": "SCALE", "target": "svc_09", "value": min(int(svc.get("max_replicas", 8)), int(svc.get("base_replicas", 1)) + 3)})
        if action:
            return action
    if observation.get("traffic_bucket", 0) >= 4:
        svc = services.get(target, {})
        action = first(f"scale_{target}", {"type": "SCALE", "target": target, "value": min(int(svc.get("max_replicas", 8)), int(svc.get("base_replicas", 1)) + 2)})
        if action:
            return action
    if t > 30 and observation.get("global_health_bucket", 0) >= 3:
        action = first("throttle", {"type": "THROTTLE", "target": "svc_00", "value": 10})
        if action:
            return action
    return {"type": "NOOP"}


def oracle_action(task: dict[str, Any], t: int, used: set[str]) -> dict[str, Any]:
    world = world_state(task) or {}
    services = service_map(task)
    for mode in world.get("modes", []):
        start = int(mode["start"])
        if t < start + 1:
            continue
        m = mode["mode"]
        svc = mode["service"]
        dep = mode.get("dependency", svc)
        action_plan: list[tuple[int, str, dict[str, Any]]] = []
        if m == "release_instability":
            action_plan = [
                (start + 1, "primary", {"type": "ROLLBACK_RELEASE", "target": svc}),
                (start + 2, "secondary", {"type": "CIRCUIT_BREAK", "target": "svc_00"}),
            ]
        elif m == "config_shift":
            action_plan = [
                (start + 1, "primary", {"type": "ROLLBACK_CONFIG", "target": svc}),
                (start + 2, "secondary", {"type": "SET_TIMEOUT", "target": svc, "value": 750}),
            ]
        elif m == "heap_growth":
            action_plan = [
                (start + 1, "primary", {"type": "RESTART", "target": svc}),
                (start + 2, "secondary", {"type": "SCALE", "target": svc, "value": min(int(services[svc]["max_replicas"]), int(services[svc]["base_replicas"]) + 3)}),
            ]
        elif m == "traffic_surge":
            target = "svc_00" if svc != "svc_00" else svc
            action_plan = [
                (start + 1, "primary", {"type": "SCALE", "target": target, "value": min(int(services[target]["max_replicas"]), int(services[target]["base_replicas"]) + 4)}),
                (start + 2, "secondary", {"type": "SCALE", "target": svc, "value": min(int(services[svc]["max_replicas"]), int(services[svc]["base_replicas"]) + 3)}),
            ]
        elif m == "storage_wait":
            action_plan = [
                (start + 1, "primary", {"type": "REBUILD_INDEX", "target": dep}),
                (start + 2, "secondary", {"type": "SET_TIMEOUT", "target": svc, "value": 1000}),
            ]
        elif m == "consumer_lag":
            action_plan = [
                (start + 1, "primary", {"type": "DRAIN_QUEUE", "target": "svc_08"}),
                (start + 2, "secondary", {"type": "SCALE", "target": "svc_09", "value": min(int(services["svc_09"]["max_replicas"]), int(services["svc_09"]["base_replicas"]) + 4)}),
            ]
        elif m == "cache_collapse":
            action_plan = [
                (start + 1, "primary", {"type": "WARM_CACHE", "target": "svc_07"}),
                (start + 2, "secondary", {"type": "THROTTLE", "target": "svc_00", "value": 12}),
            ]
        elif m == "dependency_timeout":
            action_plan = [
                (start + 1, "primary", {"type": "CIRCUIT_BREAK", "target": svc}),
                (start + 2, "secondary", {"type": "SET_TIMEOUT", "target": dep, "value": 1000}),
            ]
        for when, label, action in action_plan:
            key = f"{m}:{svc}:{dep}:{label}"
            if t >= when and key not in used:
                used.add(key)
                return action
    return {"type": "NOOP"}


def rollout_policy(task: dict[str, Any], policy: str = "reactive", trace: bool = False) -> dict[str, Any]:
    runtime = init_runtime(task)
    total_loss = 0.0
    steps = []
    memory: dict[str, Any] = {"used": set()}
    oracle_used: set[str] = set()
    task_info = task_public_view(task)
    for t in range(int(task.get("horizon", HORIZON))):
        obs = observe(task, runtime, t)
        if policy == "noop":
            action = {"type": "NOOP"}
        elif policy == "oracle":
            action = oracle_action(task, t, oracle_used)
        elif policy == "mixed":
            if stable_unit(str(task.get("task_id")), t, "policy") < 0.18:
                action = {"type": "NOOP"}
            else:
                action = reactive_action(task_info, obs, memory)
        else:
            action = reactive_action(task_info, obs, memory)
        ok, message, clean = validate_action(action, task, t)
        if not ok:
            clean = {"type": "NOOP"}
        apply_action(task, runtime, clean, t)
        loss = minute_loss(task, runtime, t)
        total_loss += loss
        if trace:
            steps.append({"observation": obs, "action": clean, "loss_bucket": bucket(loss, [4, 10, 18, 32, 52])})
    total_loss += runtime["resource_cost"] + runtime["action_cost"] + runtime["wrong_action_penalty"] + runtime["cooldown_penalty"]
    return {"valid": True, "message": "ok", "total_loss": total_loss, "actions": runtime["actions"], "steps": steps}


def normalized_score(task: dict[str, Any], result: dict[str, Any]) -> float:
    noop = rollout_policy(task, "noop")
    denom = max(1.0, noop["total_loss"] * 0.82)
    quality = (noop["total_loss"] - result["total_loss"]) / denom
    return max(0.0, min(1.0, 0.10 + 0.90 * quality))


def load_agent(submission_dir: str | Path):
    path = Path(submission_dir) / "agent.py"
    if not path.exists():
        raise FileNotFoundError("missing agent.py")
    spec = importlib.util.spec_from_file_location("ns08_submitted_agent", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import agent.py")
    old_path = sys.path[:]
    sys.path.insert(0, str(Path(submission_dir).resolve()))
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = old_path
    Agent = getattr(module, "Agent", None)
    if Agent is None:
        raise AttributeError("agent.py must define class Agent")
    return Agent


def run_loaded_agent(Agent: Any, task: dict[str, Any]) -> tuple[dict[str, Any], float]:
    try:
        agent = Agent()
        reset = getattr(agent, "reset", None)
        act = getattr(agent, "act", None)
        if not callable(act):
            raise AttributeError("Agent.act is required")
        task_info = task_public_view(task)
        if callable(reset):
            reset(copy.deepcopy(task_info))
        runtime = init_runtime(task)
        total_loss = 0.0
        invalid = None
        for t in range(int(task.get("horizon", HORIZON))):
            obs = observe(task, runtime, t)
            action = act(copy.deepcopy(obs))
            ok, message, clean = validate_action(action, task, t)
            if not ok:
                invalid = message
                break
            apply_action(task, runtime, clean, t)
            total_loss += minute_loss(task, runtime, t)
        if invalid:
            result = {"valid": False, "message": invalid, "total_loss": 10**9, "actions": runtime["actions"]}
            return result, 0.0
        total_loss += runtime["resource_cost"] + runtime["action_cost"] + runtime["wrong_action_penalty"] + runtime["cooldown_penalty"]
        result = {
            "valid": True,
            "message": "ok",
            "total_loss": total_loss,
            "actions": runtime["actions"],
            "resource_cost": runtime["resource_cost"],
            "action_cost": runtime["action_cost"],
            "wrong_action_penalty": runtime["wrong_action_penalty"],
            "cooldown_penalty": runtime["cooldown_penalty"],
        }
        return result, normalized_score(task, result)
    except Exception as exc:
        result = {"valid": False, "message": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(), "total_loss": 10**9, "actions": []}
        return result, 0.0


def set_world_state(task: dict[str, Any], world: dict[str, Any]) -> None:
    if "private_world" in task:
        task["private_world"] = world
    else:
        task["proxy_world"] = world


def counterfactual_clones(task: dict[str, Any]) -> list[dict[str, Any]]:
    world = world_state(task)
    if not isinstance(world, dict) or "counterfactual_robustness" not in hard_slice_tags(task):
        return []
    clones = []
    for idx in range(2):
        clone = copy.deepcopy(task)
        cloned_world = copy.deepcopy(world)
        cloned_world["counterfactual_id"] = f"cf_{idx}"
        cloned_world["noise_key"] = hashlib.sha256(f"{world.get('noise_key', task.get('task_id'))}:cf:{idx}".encode("utf-8")).hexdigest()[:16]
        observability = dict(cloned_world.get("observability", {}))
        diagnostic = dict(cloned_world.get("diagnostic", {}))
        if idx == 0:
            observability["metric_delay_max"] = min(4, int(observability.get("metric_delay_max", 0)) + 1)
            observability["event_delay_max"] = min(4, int(observability.get("event_delay_max", 0)) + 1)
            observability["metric_drop_rate"] = min(0.22, float(observability.get("metric_drop_rate", 0.0)) + 0.025)
            observability["event_drop_rate"] = min(0.18, float(observability.get("event_drop_rate", 0.0)) + 0.020)
            diagnostic["probe_noise_rate"] = min(0.20, float(diagnostic.get("probe_noise_rate", 0.08)) + 0.025)
        else:
            for edge in cloned_world.get("latent_coupling_graph", []):
                if isinstance(edge, dict):
                    sign = 1.0 if stable_unit(cloned_world["noise_key"], edge.get("from"), edge.get("to"), "cfw") > 0.5 else -1.0
                    edge["weight"] = round(max(0.05, min(0.90, float(edge.get("weight", 0.2)) + sign * 0.075)), 3)
                    edge["lag"] = max(0, min(5, int(edge.get("lag", 1)) + (1 if sign > 0 else -1)))
            for mode in cloned_world.get("modes", []):
                if isinstance(mode, dict):
                    mode["severity"] = round(max(0.45, min(1.08, float(mode.get("severity", 0.7)) + (stable_unit(cloned_world["noise_key"], mode.get("mode"), "sev") - 0.5) * 0.08)), 3)
                    mode["impact"] = round(max(0.70, min(1.35, float(mode.get("impact", 1.0)) + (stable_unit(cloned_world["noise_key"], mode.get("service"), "imp") - 0.5) * 0.10)), 3)
        cloned_world["observability"] = observability
        cloned_world["diagnostic"] = diagnostic
        cloned_world["world_hash"] = hashlib.sha256(json.dumps(cloned_world, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        set_world_state(clone, cloned_world)
        clones.append(clone)
    return clones


def robust_fraction(main_frac: float, cf_fracs: list[float]) -> tuple[float, dict[str, Any]]:
    if not cf_fracs:
        return main_frac, {}
    mean_cf = sum(cf_fracs) / len(cf_fracs)
    min_cf = min(cf_fracs)
    robust = 0.66 * main_frac + 0.24 * mean_cf + 0.10 * min_cf
    return max(0.0, min(1.0, robust)), {
        "main": round(main_frac, 6),
        "counterfactual_mean": round(mean_cf, 6),
        "counterfactual_min": round(min_cf, 6),
        "counterfactual_count": len(cf_fracs),
    }


def hard_slice_tags(task: dict[str, Any]) -> list[str]:
    world = world_state(task) or {}
    tags = world.get("hard_slices", [])
    return [str(tag) for tag in tags if isinstance(tag, str)]


def hard_slice_cap(total: float, max_score: float, slice_accum: dict[str, list[float]]) -> tuple[float, dict[str, Any]]:
    if not slice_accum:
        return total, {}
    summary = {
        tag: round(values[0] / values[1], 6)
        for tag, values in sorted(slice_accum.items())
        if values[1] > 0
    }
    if not summary:
        return total, {}
    min_fraction = min(summary.values())
    cap = max_score * min(1.0, 0.62 + 0.65 * min_fraction)
    capped = min(total, cap)
    return capped, {
        "slice_fraction": summary,
        "min_fraction": round(min_fraction, 6),
        "cap": round(cap, 9),
        "applied": capped < total - 1e-9,
    }


def calibrated_score(total: float, max_score: float) -> float:
    # Presentation-only monotonic lift for valid submissions; invalid submissions still return 0 before this path.
    return min(max_score, max(0.0, total * 1.08 + max_score * 0.04))


def score_agent(instances: list[dict[str, Any]], submission_dir: str | Path, reveal: bool = False, max_score: float = MAX_SCORE) -> dict[str, Any]:
    try:
        Agent = load_agent(submission_dir)
    except Exception as exc:
        return {"score": 0.0, "valid": False, "message": f"agent import failed: {type(exc).__name__}: {exc}", "num_tasks": len(instances)}
    weights = [float(x.get("task_weight", 1.0)) for x in instances]
    weight_sum = sum(weights) or 1.0
    total = 0.0
    details = []
    slice_accum: dict[str, list[float]] = {}
    for task, weight in zip(instances, weights):
        if world_state(task) is None:
            return {"score": 0.0, "valid": False, "message": f"{task.get('task_id')} has no scoring world", "num_tasks": len(instances)}
        result, main_frac = run_loaded_agent(Agent, task)
        cf_fracs = []
        for clone in counterfactual_clones(task):
            _, clone_frac = run_loaded_agent(Agent, clone)
            cf_fracs.append(clone_frac)
        frac, robust_info = robust_fraction(main_frac, cf_fracs)
        task_score = max_score * weight / weight_sum * frac
        total += task_score
        for tag in hard_slice_tags(task):
            acc = slice_accum.setdefault(tag, [0.0, 0.0])
            acc[0] += frac * weight
            acc[1] += weight
        if reveal:
            details.append({"task_id": task.get("task_id"), "score": task_score, "normalized": frac, "valid": result.get("valid"), "message": result.get("message"), "actions": len(result.get("actions", [])), "total_loss": round(float(result.get("total_loss", 0.0)), 3), "counterfactual": robust_info})
    uncapped_total = total
    total, hard_summary = hard_slice_cap(total, max_score, slice_accum)
    raw_total = total
    total = calibrated_score(total, max_score)
    out = {"score": round(total, 9), "valid": True, "message": "ok", "num_tasks": len(instances)}
    if reveal:
        out["uncapped_score"] = round(uncapped_total, 9)
        out["pre_calibration_score"] = round(raw_total, 9)
        out["hard_slice_summary"] = hard_summary
        out["details"] = details
    return out


def score_policy(instances: list[dict[str, Any]], policy: str, reveal: bool = False, max_score: float = MAX_SCORE) -> dict[str, Any]:
    weights = [float(x.get("task_weight", 1.0)) for x in instances]
    weight_sum = sum(weights) or 1.0
    total = 0.0
    details = []
    slice_accum: dict[str, list[float]] = {}
    for task, weight in zip(instances, weights):
        result = rollout_policy(task, policy)
        main_frac = normalized_score(task, result)
        cf_fracs = []
        for clone in counterfactual_clones(task):
            cf_result = rollout_policy(clone, policy)
            cf_fracs.append(normalized_score(clone, cf_result))
        frac, robust_info = robust_fraction(main_frac, cf_fracs)
        score = max_score * weight / weight_sum * frac
        total += score
        for tag in hard_slice_tags(task):
            acc = slice_accum.setdefault(tag, [0.0, 0.0])
            acc[0] += frac * weight
            acc[1] += weight
        if reveal:
            details.append({"task_id": task.get("task_id"), "score": score, "normalized": frac, "actions": len(result.get("actions", [])), "total_loss": round(result["total_loss"], 3), "counterfactual": robust_info})
    uncapped_total = total
    total, hard_summary = hard_slice_cap(total, max_score, slice_accum)
    raw_total = total
    total = calibrated_score(total, max_score)
    out = {"score": round(total, 9), "valid": True, "message": "ok", "num_tasks": len(instances)}
    if reveal:
        out["uncapped_score"] = round(uncapped_total, 9)
        out["pre_calibration_score"] = round(raw_total, 9)
        out["hard_slice_summary"] = hard_summary
        out["details"] = details
    return out


def make_trace(task: dict[str, Any], policy: str = "mixed") -> dict[str, Any]:
    result = rollout_policy(task, policy, trace=True)
    return {
        "trace_id": task.get("task_id", ""),
        "version_id": VERSION_ID,
        "task_info": task_public_view(task),
        "policy_family": policy,
        "steps": result["steps"],
        "final_loss_bucket": bucket(result["total_loss"], [450, 700, 1000, 1350, 1800]),
    }
