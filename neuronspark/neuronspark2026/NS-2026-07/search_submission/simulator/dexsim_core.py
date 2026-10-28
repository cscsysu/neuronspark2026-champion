
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import queue
import subprocess
import random
import sys
import threading
import traceback
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

try:
    import mujoco as _mujoco  # type: ignore
    import numpy as _np  # type: ignore
    _MUJOCO_AVAILABLE = True
except Exception:
    _mujoco = None
    _np = None
    _MUJOCO_AVAILABLE = False

ALLOWED_PRIMITIVES = {
    "brace",
    "push",
    "drag",
    "pivot",
    "roll",
    "lift_edge",
    "tap",
    "stabilize",
    "wait",
    "finish",
}
ALLOWED_FINGERS = {"thumb", "index", "middle", "ring", "pinky", "palm", "wrist"}
MAX_STEPS = 128
MIN_EFFECTIVE_TOOL_FORCE = 0.08
AGENT_STARTUP_TIMEOUT_SECONDS = 30.0
AGENT_RESET_TIMEOUT_SECONDS = 12.0
AGENT_RESET_BATCH_TIMEOUT_SECONDS = 20.0
AGENT_ACT_TIMEOUT_SECONDS = 2.0
AGENT_ACT_BATCH_TIMEOUT_SECONDS = 10.0
MAX_BATCH_ROLLOUTS = 64
NON_SUCCESS_BASE_SCORE = 0.05
NON_SUCCESS_OBJECTIVE_SCALE = 0.24
EARLY_FINISH_SCORE_CAP = 0.09
_AGENT_WORKER_CODE = r"""
import importlib.util
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

protocol_out = sys.stdout
sys.stdout = sys.stderr

def _send(payload):
    print(json.dumps(payload, ensure_ascii=False), file=protocol_out, flush=True)

def _load_agent():
    root = Path.cwd().resolve()
    agent_path = root / "agent.py"
    if not agent_path.exists() or not agent_path.is_file():
        raise RuntimeError("agent.py is required at submission root")
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("ns202607_user_agent", agent_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import agent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cls = getattr(module, "Agent", None)
    if cls is None:
        raise RuntimeError("agent.py must define class Agent")
    agent = cls()
    if not hasattr(agent, "reset") or not hasattr(agent, "act"):
        raise RuntimeError("Agent must provide reset(task_info) and act(observation)")
    supports_batch = hasattr(agent, "reset_batch") and hasattr(agent, "act_batch")
    return agent, supports_batch

try:
    agent, supports_batch = _load_agent()
    _send({"ok": True, "supports_batch": supports_batch})
    for line in sys.stdin:
        try:
            request = json.loads(line)
            cmd = request.get("cmd")
            if cmd == "close":
                break
            if cmd == "reset":
                agent.reset(request["task_info"])
                _send({"ok": True})
            elif cmd == "act":
                _send({"ok": True, "action": agent.act(request["observation"])})
            elif cmd == "reset_batch":
                if not supports_batch:
                    raise RuntimeError("Agent does not implement reset_batch/act_batch")
                agent.reset_batch(request["task_infos"])
                _send({"ok": True})
            elif cmd == "act_batch":
                if not supports_batch:
                    raise RuntimeError("Agent does not implement reset_batch/act_batch")
                _send({"ok": True, "actions": agent.act_batch(request["observations"])})
            else:
                _send({"ok": False, "error": "unknown agent worker command"})
        except Exception as exc:
            _send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
except Exception as exc:
    _send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    raise SystemExit(1)
"""


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def wrap_angle(value: float) -> float:
    while value > math.pi:
        value -= 2 * math.pi
    while value < -math.pi:
        value += 2 * math.pi
    return value


def vector_norm(vec: list[float]) -> float:
    return math.sqrt(float(vec[0]) ** 2 + float(vec[1]) ** 2)


def normalize(vec: list[float]) -> tuple[float, float]:
    norm = vector_norm(vec)
    if norm < 1e-9:
        return 0.0, 0.0
    return float(vec[0]) / norm, float(vec[1]) / norm


def effective_tool_normal(force: float, strength: float, contact: float) -> float:
    if force < MIN_EFFECTIVE_TOOL_FORCE:
        return 0.0
    return force * strength * (0.82 + 0.28 * clamp(contact, 0.0, 1.0))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def sanitize_task_info(task: dict[str, Any]) -> dict[str, Any]:
    blocked = {
        "private_rollouts",
        "hard_slice",
        "hidden_state",
        "official_score",
        "capability_tier",
        "benchmark_axes",
        "hidden_generalization_axes",
    }
    info = {k: v for k, v in task.items() if k not in blocked and k != "public_variants"}
    info["observation_contract"] = {
        "mode": "closed_loop",
        "max_steps": int(task.get("max_steps", MAX_STEPS)),
        "action_required_each_step": True,
        "modalities": [
            "low_dim_state",
            "finger_tactile",
            "contact_summary",
            "tactile_history",
            "tactile_heatmap_7x4",
            "tactile_image_7x8x8",
            "proxy_render_features",
            "vision_grid_16x16",
            "stage_context",
        ],
        "hidden_capability_layers": [
            "base_state_feedback",
            "tactile_slip_recovery",
            "cross_condition_generalization",
            "spatial_perception",
            "long_horizon_multi_stage",
        ],
        "network_during_evaluation": "disabled",
    }
    return info


def validate_action(action: Any) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(action, dict):
        return None, "action must be a dict"
    primitive = action.get("primitive")
    finger = action.get("finger", "palm")
    if primitive not in ALLOWED_PRIMITIVES:
        return None, "illegal primitive"
    if finger not in ALLOWED_FINGERS:
        return None, "illegal finger"
    try:
        force = float(action.get("force", 0.0))
    except Exception:
        return None, "force must be numeric"
    if not math.isfinite(force) or force < 0.0 or force > 1.0:
        return None, "force out of range"
    direction = action.get("direction", [0.0, 0.0])
    if not isinstance(direction, list) or len(direction) != 2:
        return None, "direction must be a length-2 list"
    try:
        dx = float(direction[0])
        dy = float(direction[1])
    except Exception:
        return None, "direction must be numeric"
    if not all(math.isfinite(v) for v in [dx, dy]) or vector_norm([dx, dy]) > 1.5:
        return None, "direction out of range"
    return {
        "primitive": primitive,
        "finger": finger,
        "force": force,
        "direction": [dx, dy],
    }, None


def _variant(variant: dict[str, Any] | None) -> dict[str, float]:
    v = dict(variant or {})
    return {
        "mass_scale": float(v.get("mass_scale", 1.0)),
        "friction_scale": float(v.get("friction_scale", 1.0)),
        "fragility_scale": float(v.get("fragility_scale", 1.0)),
        "contact_bias": float(v.get("contact_bias", 0.0)),
        "wrist_bias": float(v.get("wrist_bias", 0.0)),
        "start_dx": float(v.get("start_dx", 0.0)),
        "start_dy": float(v.get("start_dy", 0.0)),
        "start_dtheta": float(v.get("start_dtheta", 0.0)),
        "observation_noise": float(v.get("observation_noise", 0.010)),
        "actuator_scale": float(v.get("actuator_scale", 1.0)),
        "friction_anisotropy": float(v.get("friction_anisotropy", 0.0)),
        "surface_slope_x": float(v.get("surface_slope_x", 0.0)),
        "surface_slope_y": float(v.get("surface_slope_y", 0.0)),
        "pose_bias_x": float(v.get("pose_bias_x", 0.0)),
        "pose_bias_y": float(v.get("pose_bias_y", 0.0)),
        "pose_bias_theta": float(v.get("pose_bias_theta", 0.0)),
        "pose_dropout_rate": float(v.get("pose_dropout_rate", 0.0)),
        "tactile_dropout_rate": float(v.get("tactile_dropout_rate", 0.0)),
        "action_delay_steps": float(v.get("action_delay_steps", 0.0)),
        "contact_stiffness": float(v.get("contact_stiffness", 1.0)),
        "contact_damping": float(v.get("contact_damping", 1.0)),
    }


def _wait_action() -> dict[str, Any]:
    return {"primitive": "wait", "finger": "palm", "force": 0.0, "direction": [0.0, 0.0]}


class ProxyDexSimEnv:
    def __init__(self, task: dict[str, Any], variant: dict[str, Any] | None = None):
        self.task = task
        self.variant = _variant(variant)
        obj = task["object"]
        self.mass = max(0.2, float(obj["mass"]) * self.variant["mass_scale"])
        self.friction = clamp(float(obj["friction"]) * self.variant["friction_scale"], 0.08, 1.35)
        self.fragility = clamp(float(obj["fragility"]) * self.variant["fragility_scale"], 0.05, 1.0)
        self.compliance = float(obj.get("compliance", 0.35))
        self.size = float(obj.get("size", 0.12))
        start = task["start_pose"]
        self.x = float(start["x"]) + self.variant["start_dx"]
        self.y = float(start["y"]) + self.variant["start_dy"]
        self.theta = wrap_angle(float(start["theta"]) + self.variant["start_dtheta"])
        seed_text = f"{task['id']}::{json.dumps(variant or {}, sort_keys=True)}"
        seed = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:16], 16) & 0xFFFFFFFF
        self.rng = random.Random(seed)
        self.step_count = 0
        self.max_steps = int(task.get("max_steps", MAX_STEPS))
        self.contact = clamp(0.58 + 0.22 * self.compliance + self.variant["contact_bias"], 0.06, 0.96)
        self.damage = 0.0
        self.energy = 0.0
        self.slip_events = 0
        self.drop_events = 0
        self.tool_progress = 0.0
        self.sequence_progress = 0.0
        self.min_contact = self.contact
        self.max_shear = 0.0
        self.last_force = 0.0
        self.last_shear = 0.0
        self.last_slip_risk = 0.0
        self.last_action: dict[str, Any] | None = None
        self.vx = 0.0
        self.vy = 0.0
        self.vtheta = 0.0
        self.used_fingers: set[str] = set()
        self.tactile_history: list[list[float]] = []
        self.done = False
        self.finished = False
        self.invalid_reason: str | None = None
        self.reserved_used_before_tool = False
        self.action_delay_steps = max(0, int(round(self.variant["action_delay_steps"])))
        self._pending_actions: list[dict[str, Any]] = []
        self._pose_estimate_cache: dict[str, float] | None = None
        self.obstacle_contacts = 0
        self.stage_goals = self._normalize_stage_goals()
        self.stage_index = 0
        self.stage_completion_count = 0
        self.stage_reached_steps: list[int] = []

    def _noise(self, scale: float) -> float:
        return self.rng.uniform(-scale, scale)

    def _pose_estimate(self) -> dict[str, float]:
        n = self.variant["observation_noise"]
        dropout = clamp(self.variant["pose_dropout_rate"], 0.0, 0.95)
        if self._pose_estimate_cache is not None and self.rng.random() < dropout:
            return dict(self._pose_estimate_cache)
        estimate = {
            "x": round(self.x + self.variant["pose_bias_x"] + self._noise(n), 5),
            "y": round(self.y + self.variant["pose_bias_y"] + self._noise(n), 5),
            "theta": round(wrap_angle(self.theta + self.variant["pose_bias_theta"] + self._noise(n * 2.5)), 5),
        }
        self._pose_estimate_cache = estimate
        return dict(estimate)

    def _directional_friction(self, dir_x: float, dir_y: float) -> float:
        anisotropy = clamp(self.variant["friction_anisotropy"], -0.55, 0.55)
        directional = self.friction * (1.0 + anisotropy * (abs(dir_y) - abs(dir_x)))
        return clamp(directional, 0.05, 1.70)

    def _apply_scene_contacts(self) -> None:
        scene = self.task.get("scene_context", {})
        obstacles = scene.get("obstacles", []) if isinstance(scene, dict) else []
        for obstacle in obstacles:
            try:
                ox = float(obstacle.get("x", 0.0))
                oy = float(obstacle.get("y", 0.0))
                radius = float(obstacle.get("radius", 0.08)) + max(0.025, self.size * 0.22)
            except Exception:
                continue
            dx = self.x - ox
            dy = self.y - oy
            dist = math.sqrt(dx * dx + dy * dy)
            if 1e-6 < dist < radius:
                penetration = radius - dist
                push = penetration / dist
                self.x += dx * push * 0.70
                self.y += dy * push * 0.70
                self.contact = clamp(self.contact - 0.025 - 0.08 * penetration, 0.0, 1.0)
                self.damage += penetration * (0.035 + 0.080 * self.fragility)
                self.max_shear = max(self.max_shear, penetration * 3.0)
                self.obstacle_contacts += 1
        corridor = scene.get("corridor", {}) if isinstance(scene, dict) else {}
        try:
            width = float(corridor.get("width", 0.0))
            if width > 0.0:
                axis = corridor.get("axis", [1.0, 0.0])
                ax, ay = normalize(axis)
                lateral = -ay * self.x + ax * self.y - float(corridor.get("offset", 0.0))
                half = width * 0.5
                if abs(lateral) > half:
                    excess = abs(lateral) - half
                    sign = 1.0 if lateral >= 0.0 else -1.0
                    self.x -= (-ay) * sign * excess * 0.65
                    self.y -= ax * sign * excess * 0.65
                    self.contact = clamp(self.contact - 0.020 - 0.060 * excess, 0.0, 1.0)
                    self.damage += excess * (0.030 + 0.070 * self.fragility)
                    self.obstacle_contacts += 1
        except Exception:
            pass

    def _normalize_stage_goals(self) -> list[dict[str, Any]]:
        goals: list[dict[str, Any]] = []
        raw_goals = self.task.get("stage_goals", [])
        if isinstance(raw_goals, list):
            for raw in raw_goals:
                if not isinstance(raw, dict):
                    continue
                pose = raw.get("pose", raw)
                if not isinstance(pose, dict):
                    continue
                try:
                    goals.append(
                        {
                            "pose": {
                                "x": float(pose["x"]),
                                "y": float(pose["y"]),
                                "theta": float(pose["theta"]),
                            },
                            "tolerance": dict(raw.get("tolerance", {})),
                            "skill": str(raw.get("skill", "stage")),
                        }
                    )
                except Exception:
                    continue
        return goals

    def _final_target(self) -> dict[str, float]:
        target = self.task["target_pose"]
        return {"x": float(target["x"]), "y": float(target["y"]), "theta": float(target["theta"])}

    def _current_target(self) -> dict[str, float]:
        if not self.stage_goals:
            return self._final_target()
        idx = min(self.stage_index, len(self.stage_goals) - 1)
        return dict(self.stage_goals[idx]["pose"])

    def _stage_fraction(self) -> float:
        if not self.stage_goals:
            return 1.0
        return clamp(self.stage_completion_count / max(1, len(self.stage_goals)), 0.0, 1.0)

    def _update_stage_progress(self) -> None:
        if not self.stage_goals:
            return
        while self.stage_index < len(self.stage_goals):
            goal = self.stage_goals[self.stage_index]
            pose = goal["pose"]
            tolerance = goal.get("tolerance", {})
            dist_tol = float(tolerance.get("distance", 0.090))
            angle_tol = float(tolerance.get("angle", 0.38))
            dist = math.sqrt((self.x - float(pose["x"])) ** 2 + (self.y - float(pose["y"])) ** 2)
            angle = abs(wrap_angle(self.theta - float(pose["theta"])))
            if dist > dist_tol or angle > angle_tol:
                break
            self.stage_completion_count = max(self.stage_completion_count, self.stage_index + 1)
            if len(self.stage_reached_steps) < self.stage_completion_count:
                self.stage_reached_steps.append(self.step_count)
            if self.stage_index >= len(self.stage_goals) - 1:
                break
            self.stage_index += 1

    def _grid_coord(self, value: float, grid_size: int = 16, span: float = 0.92) -> int:
        scaled = (float(value) + span) / (2.0 * span)
        return int(clamp(math.floor(scaled * grid_size), 0, grid_size - 1))

    def _paint_disc(
        self,
        values: list[int],
        channel: int,
        x: float,
        y: float,
        radius: float,
        intensity: float,
        grid_size: int = 16,
    ) -> None:
        cx = self._grid_coord(x, grid_size)
        cy = self._grid_coord(y, grid_size)
        cell_radius = max(1, int(math.ceil(radius / 0.115)))
        base = channel * grid_size * grid_size
        peak = int(clamp(intensity, 0.0, 1.0) * 255)
        for gy in range(max(0, cy - cell_radius), min(grid_size, cy + cell_radius + 1)):
            for gx in range(max(0, cx - cell_radius), min(grid_size, cx + cell_radius + 1)):
                d = math.sqrt((gx - cx) ** 2 + (gy - cy) ** 2)
                if d <= cell_radius + 0.5:
                    value = int(peak * max(0.0, 1.0 - d / (cell_radius + 1.0)))
                    idx = base + gy * grid_size + gx
                    values[idx] = max(values[idx], value)

    def _vision_grid_16x16(self, pose_estimate: dict[str, float], target: dict[str, float]) -> dict[str, Any]:
        grid_size = 16
        values = [0] * (6 * grid_size * grid_size)
        final_target = self._final_target()
        self._paint_disc(values, 0, pose_estimate["x"], pose_estimate["y"], max(0.055, self.size * 0.55), 0.96, grid_size)
        self._paint_disc(values, 1, target["x"], target["y"], 0.060, 0.90, grid_size)
        self._paint_disc(values, 2, final_target["x"], final_target["y"], 0.052, 0.74, grid_size)
        scene = self.task.get("scene_context", {})
        if isinstance(scene, dict):
            for obstacle in scene.get("obstacles", []):
                try:
                    self._paint_disc(
                        values,
                        3,
                        float(obstacle.get("x", 0.0)),
                        float(obstacle.get("y", 0.0)),
                        float(obstacle.get("radius", 0.07)),
                        0.86,
                        grid_size,
                    )
                except Exception:
                    continue
            for distractor in scene.get("visual_distractors", []):
                try:
                    self._paint_disc(
                        values,
                        3,
                        float(distractor.get("x", 0.0)),
                        float(distractor.get("y", 0.0)),
                        float(distractor.get("radius", 0.04)),
                        float(distractor.get("intensity", 0.35)) * 0.55,
                        grid_size,
                    )
                except Exception:
                    continue
            corridor = scene.get("corridor", {})
            if isinstance(corridor, dict) and float(corridor.get("width", 0.0) or 0.0) > 0.0:
                ax, ay = normalize(corridor.get("axis", [1.0, 0.0]))
                half = float(corridor.get("width", 0.0)) * 0.5
                offset = float(corridor.get("offset", 0.0))
                base = 4 * grid_size * grid_size
                for gy in range(grid_size):
                    y = -0.92 + (gy + 0.5) * (1.84 / grid_size)
                    for gx in range(grid_size):
                        x = -0.92 + (gx + 0.5) * (1.84 / grid_size)
                        lateral = abs(-ay * x + ax * y - offset)
                        margin = abs(lateral - half)
                        if lateral <= half + 0.045:
                            values[base + gy * grid_size + gx] = max(values[base + gy * grid_size + gx], int(clamp(1.0 - margin / 0.18, 0.0, 1.0) * 210))
        contact_value = clamp(0.55 * self.contact + 0.25 * self.last_slip_risk + 0.20 * clamp(self.damage, 0.0, 1.0), 0.0, 1.0)
        self._paint_disc(values, 5, pose_estimate["x"], pose_estimate["y"], max(0.065, self.size * 0.65), contact_value, grid_size)
        return {
            "shape": [6, 16, 16],
            "encoding": "uint8_channels_first_flat",
            "channels": ["object", "current_target", "final_target", "obstacle_or_distractor", "corridor", "contact_slip_damage"],
            "values": values,
        }

    def _tactile_image_7x8x8(self, fingers: dict[str, dict[str, float]]) -> dict[str, Any]:
        names = ["thumb", "index", "middle", "ring", "pinky", "palm", "wrist"]
        values: list[int] = []
        for name in names:
            item = fingers[name]
            normal = float(item["normal"])
            shear = float(item["shear"])
            vibration = float(item["vibration"])
            contact = float(item["contact"])
            for gy in range(8):
                for gx in range(8):
                    radial = 1.0 - min(1.0, math.sqrt((gx - 3.5) ** 2 + (gy - 3.5) ** 2) / 5.0)
                    texture = 0.55 * normal * radial + 0.28 * shear * (gx / 7.0) + 0.22 * vibration * (gy / 7.0) + 0.25 * contact
                    values.append(int(clamp(texture / 1.8, 0.0, 1.0) * 255))
        return {
            "shape": [7, 8, 8],
            "encoding": "uint8_fingers_first_flat",
            "fingers": names,
            "values": values,
        }

    def observation(self) -> dict[str, Any]:
        n = self.variant["observation_noise"]
        self._update_stage_progress()
        target = self._current_target()
        final_target = self._final_target()
        pose_estimate = self._pose_estimate()
        normal_base = clamp(self.last_force * (0.85 + 0.35 * self.contact), 0.0, 1.5)
        shear_base = clamp(self.last_shear, 0.0, 2.0)
        fingers = {}
        tactile_dropout = clamp(self.variant["tactile_dropout_rate"], 0.0, 0.95)
        for i, finger in enumerate(["thumb", "index", "middle", "ring", "pinky", "palm", "wrist"]):
            contact_scale = 1.0 - 0.06 * i
            dropped = self.rng.random() < tactile_dropout
            finger_values = {
                "normal": round(clamp(normal_base * contact_scale + self._noise(0.018), 0.0, 1.8), 5),
                "shear": round(clamp(shear_base * (0.82 + 0.04 * i) + self._noise(0.018), 0.0, 2.2), 5),
                "vibration": round(clamp(0.03 + self.last_slip_risk * 0.72 + abs(self._noise(0.020)), 0.0, 1.0), 5),
                "contact": round(clamp(self.contact + self._noise(0.028), 0.0, 1.0), 5),
            }
            if dropped:
                finger_values = {
                    "normal": 0.0,
                    "shear": 0.0,
                    "vibration": round(clamp(0.015 + abs(self._noise(0.010)), 0.0, 1.0), 5),
                    "contact": round(clamp(finger_values["contact"] * 0.35, 0.0, 1.0), 5),
                }
            fingers[finger] = finger_values
        tactile_heatmap = [
            [
                fingers[finger]["normal"],
                fingers[finger]["shear"],
                fingers[finger]["vibration"],
                fingers[finger]["contact"],
            ]
            for finger in ["thumb", "index", "middle", "ring", "pinky", "palm", "wrist"]
        ]
        pose_delta = [
            round(float(target["x"]) - pose_estimate["x"], 5),
            round(float(target["y"]) - pose_estimate["y"], 5),
            round(wrap_angle(float(target["theta"]) - pose_estimate["theta"]), 5),
        ]
        final_pose_delta = [
            round(float(final_target["x"]) - pose_estimate["x"], 5),
            round(float(final_target["y"]) - pose_estimate["y"], 5),
            round(wrap_angle(float(final_target["theta"]) - pose_estimate["theta"]), 5),
        ]
        current_tactile_vector = [
            round(normal_base, 5),
            round(shear_base, 5),
            round(clamp(self.last_slip_risk, 0.0, 1.0), 5),
            round(clamp(self.contact, 0.0, 1.0), 5),
            round(clamp(self.damage, 0.0, 1.0), 5),
        ]
        return {
            "step": self.step_count,
            "remaining_steps": max(0, self.max_steps - self.step_count),
            "task_type": self.task["task_type"],
            "object_nominal": self.task.get("object_nominal", {}),
            "environment_context": self.task.get("environment_context", {}),
            "scene_context": self.task.get("scene_context", {}),
            "target_pose": target,
            "final_target_pose": final_target,
            "stage_context": {
                "enabled": bool(self.stage_goals),
                "current_stage_index": self.stage_index,
                "stage_count": len(self.stage_goals),
                "completion_fraction": round(self._stage_fraction(), 5),
                "current_stage_goal": target,
                "stage_reached_steps": list(self.stage_reached_steps[-4:]),
            },
            "low_dim_state": {
                "names": [
                    "dx_to_target",
                    "dy_to_target",
                    "dtheta_to_target",
                    "dx_to_final",
                    "dy_to_final",
                    "dtheta_to_final",
                    "vx",
                    "vy",
                    "vtheta",
                    "contact",
                    "slip_risk",
                    "damage",
                    "energy_fraction",
                    "stage_completion_fraction",
                ],
                "values": [
                    pose_delta[0],
                    pose_delta[1],
                    pose_delta[2],
                    final_pose_delta[0],
                    final_pose_delta[1],
                    final_pose_delta[2],
                    round(self.vx, 5),
                    round(self.vy, 5),
                    round(self.vtheta, 5),
                    round(clamp(self.contact, 0.0, 1.0), 5),
                    round(clamp(self.last_slip_risk, 0.0, 1.0), 5),
                    round(clamp(self.damage, 0.0, 1.0), 5),
                    round(clamp(self.energy / 72.0, 0.0, 1.0), 5),
                    round(self._stage_fraction(), 5),
                ],
            },
            "object_pose_estimate": {
                "x": pose_estimate["x"],
                "y": pose_estimate["y"],
                "theta": pose_estimate["theta"],
            },
            "object_velocity_estimate": {
                "vx": round(self.vx + self._noise(n * 0.8), 5),
                "vy": round(self.vy + self._noise(n * 0.8), 5),
                "vtheta": round(self.vtheta + self._noise(n * 1.2), 5),
            },
            "finger_tactile": fingers,
            "tactile_heatmap_7x4": tactile_heatmap,
            "tactile_image_7x8x8": self._tactile_image_7x8x8(fingers),
            "tactile_history": (self.tactile_history + [current_tactile_vector])[-6:],
            "proxy_render_features": {
                "pose_delta": pose_delta,
                "final_pose_delta": final_pose_delta,
                "object_size": round(self.size, 5),
                "target_ring_radius": round(max(0.05, 0.18 - self.size * 0.35), 5),
                "contact_ring": round(clamp(self.contact, 0.0, 1.0), 5),
                "tool_axis": self.task.get("tool_goal", {}).get("axis", [1.0, 0.0]),
                "scene_type": self.task.get("scene_context", {}).get("scene_type", "open") if isinstance(self.task.get("scene_context", {}), dict) else "open",
                "obstacle_count": len(self.task.get("scene_context", {}).get("obstacles", [])) if isinstance(self.task.get("scene_context", {}), dict) else 0,
                "stage_count": len(self.stage_goals),
                "current_stage_index": self.stage_index,
            },
            "vision_grid_16x16": self._vision_grid_16x16(pose_estimate, target),
            "contact_summary": {
                "coverage": round(clamp(self.contact + self._noise(0.018), 0.0, 1.0), 5),
                "min_contact_so_far": round(self.min_contact, 5),
                "slip_risk": round(clamp(self.last_slip_risk + self._noise(0.025), 0.0, 1.0), 5),
                "damage_risk": round(clamp(self.damage * 1.25 + self.fragility * max(0.0, self.last_force - 0.6), 0.0, 1.0), 5),
            },
            "last_action": self.last_action,
            "violation_flags": {
                "possible_drop": self.contact < 0.07 or self.drop_events > 0,
                "over_force_warning": self.last_force > float(self.task.get("resource_constraint", {}).get("max_peak_force", 0.8)),
                "resource_warning": self.reserved_used_before_tool,
                "obstacle_contact_warning": self.obstacle_contacts > 0,
            },
            "sensor_status": {
                "pose_dropout_rate_hint": round(clamp(self.variant["pose_dropout_rate"], 0.0, 1.0), 4),
                "tactile_dropout_rate_hint": round(clamp(self.variant["tactile_dropout_rate"], 0.0, 1.0), 4),
                "action_delay_steps_hint": self.action_delay_steps,
                "vision_grid_encoding": "uint8_channels_first_flat_6x16x16",
            },
        }

    def step(self, raw_action: Any) -> dict[str, Any]:
        if self.done:
            return {"done": True, "invalid": self.invalid_reason}
        action, err = validate_action(raw_action)
        if err:
            self.done = True
            self.invalid_reason = err
            return {"done": True, "invalid": err}
        assert action is not None
        primitive = action["primitive"]
        if primitive == "finish":
            self._update_stage_progress()
            self.done = True
            self.finished = True
            self.last_action = action
            return {"done": True, "invalid": None}
        if self.action_delay_steps > 0:
            self._pending_actions.append(action)
            if len(self._pending_actions) <= self.action_delay_steps:
                action = _wait_action()
            else:
                action = self._pending_actions.pop(0)
            primitive = action["primitive"]

        finger = action.get("finger", "palm")
        force = clamp(float(action.get("force", 0.0)) * self.variant["actuator_scale"], 0.0, 1.0)
        dir_x, dir_y = normalize(action.get("direction", [0.0, 0.0]))
        self.step_count += 1
        strength = {
            "thumb": 1.08,
            "index": 1.00,
            "middle": 0.95,
            "ring": 0.72,
            "pinky": 0.62,
            "palm": 1.18,
            "wrist": 1.28,
        }.get(finger, 1.0)
        wrist_conflict = self.variant["wrist_bias"] if finger == "wrist" or primitive == "pivot" else 0.0
        normal = force * strength * (0.82 + 0.28 * self.contact) * clamp(self.variant["contact_stiffness"], 0.55, 1.65)
        damping = clamp(self.variant["contact_damping"], 0.45, 1.65)
        shear = normal * (1.0 + 0.45 * (primitive in {"drag", "pivot", "roll"})) * (1.10 - 0.16 * damping) + abs(wrist_conflict)
        directional_friction = self._directional_friction(dir_x, dir_y)
        slip_margin = directional_friction * (0.65 + 0.35 * self.contact) - shear / (self.mass + 0.35)
        slip_risk = clamp(-slip_margin * 1.8 + 0.06, 0.0, 1.0)

        if primitive in {"push", "drag", "pivot", "roll", "lift_edge"}:
            self.used_fingers.add(finger)
        if primitive in {"brace", "stabilize"}:
            self.contact = clamp(self.contact + 0.075 + 0.05 * force, 0.0, 1.0)
            shear *= 0.55
            self.vx *= 0.55
            self.vy *= 0.55
            self.vtheta *= 0.60
        elif primitive == "wait":
            self.contact = clamp(self.contact - 0.010, 0.0, 1.0)
            self.vx *= 0.82
            self.vy *= 0.82
            self.vtheta *= 0.82
        else:
            if slip_margin < -0.05:
                self.slip_events += 1
                self.contact -= 0.045 + min(0.05, -slip_margin * 0.03)
            else:
                self.contact -= 0.014 * (1.0 - self.compliance)

        old_x, old_y, old_theta = self.x, self.y, self.theta
        if primitive == "push":
            mobility = max(0.0, normal - 0.12 * self.mass * directional_friction)
            gain = 0.030 * mobility / (self.mass ** 0.65)
            self.x += gain * dir_x
            self.y += gain * dir_y
            self.theta += 0.015 * normal * (-dir_y + 0.25 * dir_x) / (self.mass + self.size)
        elif primitive == "drag":
            mobility = max(0.0, normal - 0.08 * self.mass * directional_friction)
            gain = 0.023 * mobility / (self.mass ** 0.70)
            self.x += gain * dir_x
            self.y += gain * dir_y
            self.contact = clamp(self.contact - 0.018, 0.0, 1.0)
        elif primitive == "roll":
            gain = 0.020 * normal * (0.7 + self.compliance) / (self.mass ** 0.55)
            self.x += gain * dir_x
            self.y += gain * dir_y
            self.theta += 0.055 * normal * (1 if dir_x + dir_y >= 0 else -1)
        elif primitive == "pivot":
            sign = 1.0 if dir_x + dir_y >= 0 else -1.0
            self.theta += sign * 0.095 * normal / (self.mass + 0.2) + wrist_conflict * 0.010
            self.contact = clamp(self.contact - 0.012, 0.0, 1.0)
        elif primitive == "lift_edge":
            gain = 0.016 * normal / (self.mass ** 0.50)
            self.x += gain * dir_x
            self.y += gain * dir_y
            self.theta += 0.035 * normal
            self.contact = clamp(self.contact - 0.025 + 0.02 * self.compliance, 0.0, 1.0)
        elif primitive == "tap":
            tool_axis = self.task.get("tool_goal", {}).get("axis", [1.0, 0.0])
            ax, ay = normalize(tool_axis)
            align = max(0.0, ax * dir_x + ay * dir_y)
            tap_normal = effective_tool_normal(force, strength, self.contact)
            self.tool_progress += align * tap_normal * (0.055 + 0.035 * self.contact)
            if finger in set(self.task.get("resource_constraint", {}).get("reserve_fingers", [])):
                self.sequence_progress += align * tap_normal * 0.065

        reserved = set(self.task.get("resource_constraint", {}).get("reserve_fingers", []))
        if self.task["task_type"] == "resource_sequence" and reserved and self.used_fingers.issuperset(reserved) and self.tool_progress < 0.04:
            self.reserved_used_before_tool = True
        self.x += self.variant["surface_slope_x"] * (0.004 + 0.006 * max(0.0, 1.0 - self.contact))
        self.y += self.variant["surface_slope_y"] * (0.004 + 0.006 * max(0.0, 1.0 - self.contact))
        self._apply_scene_contacts()
        self._update_stage_progress()
        self.damage += max(0.0, normal - (1.05 - 0.72 * self.fragility)) * (0.010 + 0.020 * self.fragility)
        self.energy += force * force * (1.0 + 0.30 * (primitive in {"drag", "pivot", "lift_edge"}))
        self.contact = clamp(self.contact, 0.0, 1.0)
        self.min_contact = min(self.min_contact, self.contact)
        self.max_shear = max(self.max_shear, shear)
        if self.contact < 0.03 or abs(self.x) > 1.28 or abs(self.y) > 1.28:
            self.drop_events += 1
        self.theta = wrap_angle(self.theta)
        self.vx = self.x - old_x
        self.vy = self.y - old_y
        self.vtheta = wrap_angle(self.theta - old_theta)
        self.last_force = normal
        self.last_shear = shear
        self.last_slip_risk = slip_risk
        self.tactile_history.append([
            round(clamp(normal, 0.0, 1.5), 5),
            round(clamp(shear, 0.0, 2.0), 5),
            round(clamp(slip_risk, 0.0, 1.0), 5),
            round(clamp(self.contact, 0.0, 1.0), 5),
            round(clamp(self.damage, 0.0, 1.0), 5),
        ])
        self.tactile_history = self.tactile_history[-6:]
        self.last_action = action
        if self.step_count >= self.max_steps:
            self.done = True
        return {"done": self.done, "invalid": None}

    def state(self) -> dict[str, Any]:
        self._update_stage_progress()
        target = self._final_target()
        dist = math.sqrt((self.x - float(target["x"])) ** 2 + (self.y - float(target["y"])) ** 2)
        angle = abs(wrap_angle(self.theta - float(target["theta"])))
        return {
            "final_pose": {"x": self.x, "y": self.y, "theta": self.theta},
            "distance_error": dist,
            "angle_error": angle,
            "damage": self.damage,
            "energy": self.energy,
            "steps": self.step_count,
            "terminated_by_finish": self.finished,
            "slip_events": self.slip_events,
            "drop_events": self.drop_events,
            "obstacle_contacts": self.obstacle_contacts,
            "tool_progress": self.tool_progress,
            "sequence_progress": self.sequence_progress,
            "resource_violation": self.reserved_used_before_tool,
            "min_contact": self.min_contact,
            "max_shear": self.max_shear,
            "stage_completion_fraction": self._stage_fraction(),
            "stage_completion_count": self.stage_completion_count,
            "stage_count": len(self.stage_goals),
            "stage_reached_steps": list(self.stage_reached_steps),
            "invalid_reason": self.invalid_reason,
        }


def _mujoco_geom_spec(shape: str, size: float) -> tuple[str, str, float]:
    half = max(0.025, size * 0.32)
    if shape in {"cylinder", "tri_lobed_puck"}:
        return "cylinder", f"{size * 0.34:.5f} {size * 0.16:.5f}", size * 0.16
    if shape in {"irregular_capsule", "hooked_capsule"}:
        return "capsule", f"{size * 0.24:.5f} {size * 0.28:.5f}", size * 0.24
    if shape == "thin_plate":
        return "box", f"{size * 0.46:.5f} {size * 0.32:.5f} {size * 0.055:.5f}", size * 0.055
    if shape == "soft_pouch":
        return "ellipsoid", f"{size * 0.40:.5f} {size * 0.30:.5f} {size * 0.16:.5f}", size * 0.16
    return "box", f"{half:.5f} {size * 0.27:.5f} {size * 0.18:.5f}", size * 0.18


def _mujoco_scene_xml(task: dict[str, Any], variant: dict[str, Any]) -> str:
    obj = task["object"]
    mass = max(0.2, float(obj["mass"]) * float(variant.get("mass_scale", 1.0)))
    friction = clamp(float(obj["friction"]) * float(variant.get("friction_scale", 1.0)), 0.08, 1.35)
    size = float(obj.get("size", 0.12))
    shape = str(task.get("object_nominal", {}).get("shape", "rounded_box"))
    geom_type, geom_size, half_z = _mujoco_geom_spec(shape, size)
    env = task.get("environment_context", {})
    floor_friction = clamp(friction * float(env.get("friction_multiplier_nominal", 1.0)), 0.08, 1.6)
    start = task["start_pose"]
    z0 = max(0.025, half_z + 0.003)
    return f"""<mujoco model="ns202607_mujoco_devsim">
  <compiler angle="radian" coordinate="local"/>
  <option timestep="0.005" gravity="0 0 -9.81" integrator="Euler" cone="elliptic"/>
  <size nconmax="64" njmax="256"/>
  <default>
    <geom solref="0.018 1" solimp="0.90 0.96 0.001" condim="4"/>
  </default>
  <asset>
    <material name="floor_mat" rgba="0.42 0.44 0.40 1"/>
    <material name="object_mat" rgba="0.77 0.39 0.22 1"/>
  </asset>
  <worldbody>
    <light name="top" pos="0 0 1.6"/>
    <geom name="floor" type="plane" pos="0 0 0" size="1.6 1.6 0.04" material="floor_mat" friction="{floor_friction:.5f} 0.025 0.0002"/>
    <body name="object" pos="{float(start['x']):.5f} {float(start['y']):.5f} {z0:.5f}">
      <freejoint name="object_free"/>
      <geom name="object_geom" type="{geom_type}" size="{geom_size}" mass="{mass:.5f}" material="object_mat" friction="{friction:.5f} 0.030 0.0002"/>
      <site name="tactile_patch" pos="0 0 {half_z:.5f}" size="{max(0.01, size * 0.16):.5f}" rgba="0.1 0.6 0.9 0.35"/>
    </body>
  </worldbody>
</mujoco>"""


class MuJoCoDexSimEnv(ProxyDexSimEnv):
    def __init__(self, task: dict[str, Any], variant: dict[str, Any] | None = None):
        if not _MUJOCO_AVAILABLE:
            raise RuntimeError("MuJoCo is not available")
        super().__init__(task, variant)
        assert _mujoco is not None
        self.sim_backend = "mujoco"
        self._model = _mujoco.MjModel.from_xml_string(_mujoco_scene_xml(task, self.variant))
        self._data = _mujoco.MjData(self._model)
        self._body_id = _mujoco.mj_name2id(self._model, _mujoco.mjtObj.mjOBJ_BODY, "object")
        self._joint_id = _mujoco.mj_name2id(self._model, _mujoco.mjtObj.mjOBJ_JOINT, "object_free")
        self._qpos_adr = int(self._model.jnt_qposadr[self._joint_id])
        self._dof_adr = int(self._model.jnt_dofadr[self._joint_id])
        start = task["start_pose"]
        geom_type, geom_size, half_z = _mujoco_geom_spec(
            str(task.get("object_nominal", {}).get("shape", "rounded_box")),
            self.size,
        )
        del geom_type, geom_size
        self._data.qpos[self._qpos_adr:self._qpos_adr + 3] = [
            float(start["x"]) + self.variant["start_dx"],
            float(start["y"]) + self.variant["start_dy"],
            max(0.025, half_z + 0.003),
        ]
        yaw = wrap_angle(float(start["theta"]) + self.variant["start_dtheta"])
        self._data.qpos[self._qpos_adr + 3:self._qpos_adr + 7] = [
            math.cos(yaw * 0.5),
            0.0,
            0.0,
            math.sin(yaw * 0.5),
        ]
        _mujoco.mj_forward(self._model, self._data)
        self._sync_from_mujoco()

    def _sync_from_mujoco(self) -> None:
        assert _np is not None
        q = self._data.qpos
        v = self._data.qvel
        old_x, old_y, old_theta = self.x, self.y, self.theta
        self.x = float(q[self._qpos_adr])
        self.y = float(q[self._qpos_adr + 1])
        qw = float(q[self._qpos_adr + 3])
        qx = float(q[self._qpos_adr + 4])
        qy = float(q[self._qpos_adr + 5])
        qz = float(q[self._qpos_adr + 6])
        self.theta = wrap_angle(math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))
        self.vx = float(v[self._dof_adr])
        self.vy = float(v[self._dof_adr + 1])
        self.vtheta = float(v[self._dof_adr + 5])
        normal_force = 0.0
        tangential_force = 0.0
        for cidx in range(self._data.ncon):
            cf = _np.zeros(6, dtype=float)
            _mujoco.mj_contactForce(self._model, self._data, cidx, cf)
            normal_force += max(0.0, float(cf[0]))
            tangential_force += math.sqrt(float(cf[1]) ** 2 + float(cf[2]) ** 2)
        self.last_force = normal_force / max(1.0, 9.0 * self.mass)
        self.last_shear = tangential_force / max(1.0, 7.0 * self.mass)
        self.contact = clamp(0.10 + 0.10 * self._data.ncon + min(0.72, normal_force / max(2.0, 24.0 * self.mass)) + self.variant["contact_bias"], 0.0, 1.0)
        self.min_contact = min(self.min_contact, self.contact)
        self.max_shear = max(self.max_shear, self.last_shear)
        slip_margin = self.friction * (0.35 + 0.65 * self.contact) - tangential_force / max(1.0, normal_force + 1.0)
        self.last_slip_risk = clamp(-slip_margin * 1.45 + 0.08, 0.0, 1.0)
        if self.last_slip_risk > 0.68 and tangential_force > 0.25:
            self.slip_events += 1
        if self._data.qpos[self._qpos_adr + 2] < 0.004 or abs(self.x) > 1.28 or abs(self.y) > 1.28:
            self.drop_events += 1
        if not math.isfinite(self.x + self.y + self.theta):
            self.invalid_reason = "non_finite_sim_state"
            self.done = True
        if old_x != self.x or old_y != self.y or old_theta != self.theta:
            self.vx = self.x - old_x
            self.vy = self.y - old_y
            self.vtheta = wrap_angle(self.theta - old_theta)

    def _write_planar_pose_to_mujoco(self) -> None:
        self._data.qpos[self._qpos_adr] = self.x
        self._data.qpos[self._qpos_adr + 1] = self.y
        _mujoco.mj_forward(self._model, self._data)

    def observation(self) -> dict[str, Any]:
        obs = super().observation()
        obs["sim_backend"] = "mujoco"
        obs["simulator_source"] = self.task.get("simulation_source", {})
        return obs

    def step(self, raw_action: Any) -> dict[str, Any]:
        if self.done:
            return {"done": True, "invalid": self.invalid_reason}
        action, err = validate_action(raw_action)
        if err:
            self.done = True
            self.invalid_reason = err
            return {"done": True, "invalid": err}
        assert action is not None
        primitive = action["primitive"]
        if primitive == "finish":
            self._update_stage_progress()
            self.done = True
            self.finished = True
            self.last_action = action
            return {"done": True, "invalid": None}
        if self.action_delay_steps > 0:
            self._pending_actions.append(action)
            if len(self._pending_actions) <= self.action_delay_steps:
                action = _wait_action()
            else:
                action = self._pending_actions.pop(0)
            primitive = action["primitive"]
        finger = action.get("finger", "palm")
        force = clamp(float(action.get("force", 0.0)) * self.variant["actuator_scale"], 0.0, 1.0)
        dir_x, dir_y = normalize(action.get("direction", [0.0, 0.0]))
        self.step_count += 1
        strength = {
            "thumb": 1.08,
            "index": 1.00,
            "middle": 0.95,
            "ring": 0.72,
            "pinky": 0.62,
            "palm": 1.18,
            "wrist": 1.28,
        }.get(finger, 1.0)
        if primitive in {"push", "drag", "pivot", "roll", "lift_edge"}:
            self.used_fingers.add(finger)
        external = 0.0
        torque = 0.0
        stiffness = clamp(self.variant["contact_stiffness"], 0.55, 1.65)
        damping = clamp(self.variant["contact_damping"], 0.45, 1.65)
        directional_friction = self._directional_friction(dir_x, dir_y)
        if primitive in {"brace", "stabilize"}:
            self._data.qvel[self._dof_adr:self._dof_adr + 6] *= 0.60
            external = 1.0 * force * strength
        elif primitive == "wait":
            self._data.qvel[self._dof_adr:self._dof_adr + 6] *= 0.86
        elif primitive == "tap":
            external = 3.8 * force * strength
        elif primitive == "pivot":
            external = 3.4 * force * strength
            torque = 0.62 * force * (1.0 if dir_x + dir_y >= 0.0 else -1.0)
        elif primitive == "roll":
            external = 4.4 * force * strength
            torque = 0.42 * force * (1.0 if dir_x + dir_y >= 0.0 else -1.0)
        elif primitive == "lift_edge":
            external = 3.6 * force * strength
            torque = 0.30 * force
        else:
            external = 4.8 * force * strength
        if primitive == "drag":
            external *= 0.82
        external *= stiffness * clamp(1.10 - 0.10 * directional_friction, 0.82, 1.16)
        mass_limit = 2.6 + 1.2 * min(2.0, self.mass)
        external = clamp(external, 0.0, mass_limit)
        torque = clamp(torque, -0.55, 0.55)
        for _ in range(8):
            self._data.xfrc_applied[self._body_id, 0] = external * dir_x
            self._data.xfrc_applied[self._body_id, 1] = external * dir_y
            self._data.xfrc_applied[self._body_id, 5] = torque + (self.variant["wrist_bias"] if finger == "wrist" else 0.0)
            _mujoco.mj_step(self._model, self._data)
            qvel = self._data.qvel[self._dof_adr:self._dof_adr + 6]
            if not _np.isfinite(qvel).all():
                self.invalid_reason = "sim_unstable"
                self.done = True
                qvel[:] = 0.0
                break
            qvel[:] = _np.clip(qvel, -10.0, 10.0)
            qvel *= 0.975
        self._data.xfrc_applied[:] = 0.0
        self._sync_from_mujoco()
        self.x += self.variant["surface_slope_x"] * (0.004 + 0.006 * max(0.0, 1.0 - self.contact))
        self.y += self.variant["surface_slope_y"] * (0.004 + 0.006 * max(0.0, 1.0 - self.contact))
        self._apply_scene_contacts()
        self._update_stage_progress()
        self._write_planar_pose_to_mujoco()
        if primitive == "wait":
            action_normal = 0.0
        elif primitive in {"brace", "stabilize"}:
            action_normal = force * strength * 0.35 * (0.82 + 0.28 * self.contact)
        else:
            action_normal = force * strength * stiffness * (0.82 + 0.28 * self.contact)
        normal = clamp(action_normal, 0.0, 1.8)
        shear = clamp(self.last_shear * (1.10 - 0.16 * damping), 0.0, 2.2)
        if primitive == "tap":
            tool_axis = self.task.get("tool_goal", {}).get("axis", [1.0, 0.0])
            ax, ay = normalize(tool_axis)
            align = max(0.0, ax * dir_x + ay * dir_y)
            tap_normal = effective_tool_normal(force, strength, self.contact)
            self.tool_progress += align * tap_normal * (0.055 + 0.035 * self.contact)
            if finger in set(self.task.get("resource_constraint", {}).get("reserve_fingers", [])):
                self.sequence_progress += align * tap_normal * 0.065
        reserved = set(self.task.get("resource_constraint", {}).get("reserve_fingers", []))
        if self.task["task_type"] == "resource_sequence" and reserved and self.used_fingers.issuperset(reserved) and self.tool_progress < 0.04:
            self.reserved_used_before_tool = True
        self.damage += max(0.0, normal - (1.05 - 0.72 * self.fragility)) * (0.010 + 0.020 * self.fragility)
        self.energy += force * force * (1.0 + 0.30 * (primitive in {"drag", "pivot", "lift_edge"}))
        self.tactile_history.append([
            round(clamp(normal, 0.0, 1.5), 5),
            round(clamp(shear, 0.0, 2.0), 5),
            round(clamp(self.last_slip_risk, 0.0, 1.0), 5),
            round(clamp(self.contact, 0.0, 1.0), 5),
            round(clamp(self.damage, 0.0, 1.0), 5),
        ])
        self.tactile_history = self.tactile_history[-6:]
        self.last_action = action
        if self.step_count >= self.max_steps:
            self.done = True
        return {"done": self.done, "invalid": None}


class DexSimEnv(MuJoCoDexSimEnv if _MUJOCO_AVAILABLE else ProxyDexSimEnv):
    pass


def score_rollout(task: dict[str, Any], state: dict[str, Any]) -> dict[str, float | bool]:
    if state.get("invalid_reason"):
        return {
            "score01": 0.0,
            "success": False,
            "pose_score": 0.0,
            "objective": 0.0,
            "safety": 0.0,
            "contact": 0.0,
            "efficiency": 0.0,
        }
    dist = float(state["distance_error"])
    angle = float(state["angle_error"])
    pose_score = 0.66 * math.exp(-dist / 0.18) + 0.34 * math.exp(-angle / 0.55)
    tool_need = float(task.get("tool_goal", {}).get("progress", 0.0))
    if task["task_type"] == "tool_use":
        task_objective = min(1.0, float(state["tool_progress"]) / max(0.08, tool_need))
    elif task["task_type"] == "resource_sequence":
        task_objective = 0.72 * pose_score + 0.28 * min(1.0, float(state["sequence_progress"]) / max(0.08, tool_need))
    else:
        task_objective = pose_score
    has_stages = bool(task.get("stage_goals"))
    stage_fraction = clamp(float(state.get("stage_completion_fraction", 1.0 if not has_stages else 0.0)), 0.0, 1.0)
    if has_stages:
        task_objective = 0.70 * task_objective + 0.30 * stage_fraction
    safety = 1.0
    safety -= min(0.75, float(state["damage"]) * 1.85)
    safety -= min(0.50, float(state["drop_events"]) * 0.18)
    safety -= min(0.30, float(state.get("obstacle_contacts", 0.0)) * 0.035)
    safety -= 0.25 if state.get("resource_violation") else 0.0
    safety = clamp(safety, 0.0, 1.0)
    contact = clamp(0.45 + float(state["min_contact"]) * 0.85 - 0.035 * float(state["slip_events"]) - 0.018 * float(state.get("obstacle_contacts", 0.0)), 0.0, 1.0)
    energy_budget = max(72.0, float(task.get("max_steps", MAX_STEPS)) * 1.05)
    efficiency = clamp(1.0 - float(state["energy"]) / energy_budget - float(state["steps"]) / 320.0, 0.0, 1.0)
    success = (
        dist < 0.085
        and angle < 0.34
        and safety > 0.58
        and (not has_stages or stage_fraction >= 0.999)
        and (task["task_type"] != "tool_use" or task_objective > 0.72)
        and (task["task_type"] != "resource_sequence" or task_objective > 0.68)
    )
    score01 = 0.54 * (1.0 if success else 0.0) + 0.18 * task_objective + 0.12 * safety + 0.09 * contact + 0.07 * efficiency
    if not success:
        score01 = min(score01, NON_SUCCESS_BASE_SCORE + NON_SUCCESS_OBJECTIVE_SCALE * task_objective)
        if state.get("terminated_by_finish") and int(state.get("steps", 0)) <= 1:
            score01 = min(score01, EARLY_FINISH_SCORE_CAP)
    return {
        "score01": clamp(score01, 0.0, 1.0),
        "success": success,
        "pose_score": clamp(pose_score, 0.0, 1.0),
        "objective": clamp(task_objective, 0.0, 1.0),
        "safety": safety,
        "contact": contact,
        "efficiency": efficiency,
    }


def capability_tier_for_task(task: dict[str, Any]) -> str:
    tier = task.get("capability_tier")
    if tier in {"base_state_feedback", "tactile_slip_recovery", "cross_condition_generalization"}:
        return str(tier)
    axes = set(task.get("benchmark_axes", []))
    disturbances = task.get("known_disturbances", {})
    if axes.intersection(
        {
            "unseen_object_family",
            "unseen_environment",
            "multi_task_combo",
            "strong_private_physics_shift",
            "multi_stage_sequence",
            "spatial_perception",
            "visual_occlusion",
            "resource_bottleneck",
            "unseen_resource_constraint",
        }
    ):
        return "cross_condition_generalization"
    if any(bool(disturbances.get(key)) for key in ["low_friction_hint", "fragile_hint", "partial_contact_hint"]):
        return "tactile_slip_recovery"
    if task.get("task_type") in {"tool_use", "resource_sequence"}:
        return "tactile_slip_recovery"
    return "base_state_feedback"


def load_agent(submission_dir: str | Path) -> Any:
    root = Path(submission_dir)
    agent_path = root / "agent.py"
    if not agent_path.exists() or not agent_path.is_file():
        raise RuntimeError("agent.py is required at submission root")
    module_name = "ns202607_user_agent_" + str(abs(hash(str(agent_path.resolve()))))
    spec = importlib.util.spec_from_file_location(module_name, agent_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import agent.py")
    old_path = list(sys.path)
    sys.path.insert(0, str(root.resolve()))
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = old_path
    cls = getattr(module, "Agent", None)
    if cls is None:
        raise RuntimeError("agent.py must define class Agent")
    agent = cls()
    if not hasattr(agent, "reset") or not hasattr(agent, "act"):
        raise RuntimeError("Agent must provide reset(task_info) and act(observation)")
    return agent


class AgentSubprocessProxy:
    def __init__(self, submission_dir: str | Path):
        submission_root = Path(submission_dir).resolve()
        self._responses: queue.Queue[dict[str, Any]] = queue.Queue()
        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["NS202607_AGENT_WORKER"] = "1"
        env.pop("PYTHONPATH", None)
        self.proc = subprocess.Popen(
            [sys.executable, "-B", "-c", _AGENT_WORKER_CODE],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            bufsize=1,
            cwd=str(submission_root),
            env=env,
        )
        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader.start()
        hello = self._read_response(AGENT_STARTUP_TIMEOUT_SECONDS, "startup")
        if not hello.get("ok"):
            self.close()
            raise RuntimeError(str(hello.get("error", "agent worker failed to start")))
        self.supports_batch = bool(hello.get("supports_batch", False))

    def _read_stdout(self) -> None:
        if self.proc.stdout is None:
            self._responses.put({"ok": False, "error": "agent worker stdout is unavailable"})
            return
        try:
            for line in self.proc.stdout:
                try:
                    response = json.loads(line)
                    if isinstance(response, dict):
                        self._responses.put(response)
                    else:
                        self._responses.put({"ok": False, "error": "agent worker returned a non-dict response"})
                except Exception as exc:
                    self._responses.put({"ok": False, "error": f"invalid worker protocol: {type(exc).__name__}"})
        finally:
            self._responses.put({"ok": False, "error": "agent worker exited without response"})

    def _read_response(self, timeout_seconds: float, phase: str) -> dict[str, Any]:
        try:
            return self._responses.get(timeout=float(timeout_seconds))
        except queue.Empty:
            self.close()
            raise RuntimeError(f"agent worker {phase} timeout after {timeout_seconds:.1f}s")

    def _request(self, payload: dict[str, Any], timeout_seconds: float, phase: str) -> dict[str, Any]:
        if self.proc.stdin is None:
            raise RuntimeError("agent worker stdin is unavailable")
        self.proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()
        response = self._read_response(timeout_seconds, phase)
        if not response.get("ok"):
            raise RuntimeError(str(response.get("error", "agent worker request failed")))
        return response

    def reset(self, task_info: dict[str, Any]) -> None:
        self._request({"cmd": "reset", "task_info": task_info}, AGENT_RESET_TIMEOUT_SECONDS, "reset")

    def act(self, observation: dict[str, Any]) -> dict[str, Any]:
        response = self._request({"cmd": "act", "observation": observation}, AGENT_ACT_TIMEOUT_SECONDS, "act")
        action = response.get("action")
        if not isinstance(action, dict):
            raise RuntimeError("agent worker returned a non-dict action")
        return action

    def reset_batch(self, task_infos: list[dict[str, Any]]) -> None:
        self._request({"cmd": "reset_batch", "task_infos": task_infos}, AGENT_RESET_BATCH_TIMEOUT_SECONDS, "reset_batch")

    def act_batch(self, observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        response = self._request({"cmd": "act_batch", "observations": observations}, AGENT_ACT_BATCH_TIMEOUT_SECONDS, "act_batch")
        actions = response.get("actions")
        if not isinstance(actions, list) or not all(isinstance(action, dict) for action in actions):
            raise RuntimeError("agent worker returned invalid batched actions")
        return actions

    def close(self) -> None:
        try:
            if self.proc.stdin is not None and self.proc.poll() is None:
                self.proc.stdin.write(json.dumps({"cmd": "close"}) + "\n")
                self.proc.stdin.flush()
        except Exception:
            pass
        try:
            self.proc.terminate()
            self.proc.wait(timeout=1.0)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass


def _agent_worker_main(submission_dir: str | Path) -> int:
    protocol_out = sys.stdout
    sys.stdout = sys.stderr
    try:
        agent = load_agent(submission_dir)
        supports_batch = hasattr(agent, "reset_batch") and hasattr(agent, "act_batch")
        print(json.dumps({"ok": True, "supports_batch": supports_batch}), file=protocol_out, flush=True)
        for line in sys.stdin:
            try:
                request = json.loads(line)
                cmd = request.get("cmd")
                if cmd == "close":
                    return 0
                if cmd == "reset":
                    agent.reset(request["task_info"])
                    print(json.dumps({"ok": True}), file=protocol_out, flush=True)
                elif cmd == "act":
                    action = agent.act(request["observation"])
                    print(json.dumps({"ok": True, "action": action}, ensure_ascii=False), file=protocol_out, flush=True)
                elif cmd == "reset_batch":
                    if not supports_batch:
                        raise RuntimeError("Agent does not implement reset_batch/act_batch")
                    agent.reset_batch(request["task_infos"])
                    print(json.dumps({"ok": True}), file=protocol_out, flush=True)
                elif cmd == "act_batch":
                    if not supports_batch:
                        raise RuntimeError("Agent does not implement reset_batch/act_batch")
                    actions = agent.act_batch(request["observations"])
                    print(json.dumps({"ok": True, "actions": actions}, ensure_ascii=False), file=protocol_out, flush=True)
                else:
                    print(json.dumps({"ok": False, "error": "unknown agent worker command"}), file=protocol_out, flush=True)
            except Exception as exc:
                print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), file=protocol_out, flush=True)
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), file=protocol_out, flush=True)
        return 1
    return 0


def run_agent_episode(agent: Any, task: dict[str, Any], variant: dict[str, Any] | None = None) -> dict[str, Any]:
    env = DexSimEnv(task, variant)
    try:
        agent.reset(sanitize_task_info(task))
    except Exception as exc:
        state = env.state()
        state["invalid_reason"] = "reset_error"
        return {"metrics": score_rollout(task, state), "state": state, "error": f"reset_error: {type(exc).__name__}"}
    error = None
    while not env.done and env.step_count < env.max_steps:
        obs = env.observation()
        try:
            action = agent.act(obs)
        except Exception as exc:
            error = f"act_error: {type(exc).__name__}"
            env.invalid_reason = "act_error"
            env.done = True
            break
        result = env.step(action)
        if result.get("invalid"):
            error = str(result["invalid"])
            break
    state = env.state()
    metrics = score_rollout(task, state)
    return {"metrics": metrics, "state": state, "error": error}


def _agent_supports_batch(agent: Any) -> bool:
    if isinstance(agent, AgentSubprocessProxy):
        return bool(getattr(agent, "supports_batch", False))
    return hasattr(agent, "reset_batch") and hasattr(agent, "act_batch")


def _episode_error_result(env: DexSimEnv, task: dict[str, Any], error: str) -> dict[str, Any]:
    env.invalid_reason = error
    env.done = True
    state = env.state()
    state["invalid_reason"] = error
    return {"metrics": score_rollout(task, state), "state": state, "error": error}


def _batch_episode_error_result(
    env: DexSimEnv,
    task: dict[str, Any],
    task_index: int,
    variant_index: int,
    error: str,
) -> dict[str, Any]:
    result = _episode_error_result(env, task, error)
    result["task_index"] = task_index
    result["variant_index"] = variant_index
    return result


def _make_batch_task_info(
    task: dict[str, Any],
    task_index: int,
    variant_index: int,
    batch_index: int,
    batch_size: int,
) -> dict[str, Any]:
    info = dict(sanitize_task_info(task))
    info["batch_index"] = batch_index
    info["batch_size"] = batch_size
    info["task_batch_index"] = task_index
    info["variant_batch_index"] = variant_index
    info["rollout_id"] = f"task{task_index:05d}_variant{variant_index:03d}"
    return info


def run_agent_rollout_batch(
    agent: Any,
    rollout_specs: list[tuple[int, dict[str, Any], int, dict[str, Any]]],
    max_batch_rollouts: int = MAX_BATCH_ROLLOUTS,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    batch_size_limit = max(1, int(max_batch_rollouts))
    for chunk_start in range(0, len(rollout_specs), batch_size_limit):
        chunk = rollout_specs[chunk_start:chunk_start + batch_size_limit]
        envs = [DexSimEnv(task, variant) for _, task, _, variant in chunk]
        task_infos = [
            _make_batch_task_info(task, task_index, variant_index, local_idx, len(chunk))
            for local_idx, (task_index, task, variant_index, _) in enumerate(chunk)
        ]
        try:
            agent.reset_batch(task_infos)
        except Exception as exc:
            for local_idx, (task_index, task, variant_index, _) in enumerate(chunk):
                results.append(
                    _batch_episode_error_result(
                        envs[local_idx],
                        task,
                        task_index,
                        variant_index,
                        f"reset_batch_error: {type(exc).__name__}",
                    )
                )
            continue
        errors: list[str | None] = [None] * len(envs)
        while any((not env.done and env.step_count < env.max_steps) for env in envs):
            active_indices = [idx for idx, env in enumerate(envs) if not env.done and env.step_count < env.max_steps]
            observations = []
            for local_idx in active_indices:
                task_index, task, variant_index, _ = chunk[local_idx]
                obs = envs[local_idx].observation()
                obs["batch_index"] = local_idx
                obs["batch_size"] = len(chunk)
                obs["task_batch_index"] = task_index
                obs["variant_batch_index"] = variant_index
                obs["rollout_id"] = f"task{task_index:05d}_variant{variant_index:03d}"
                observations.append(obs)
            try:
                actions = agent.act_batch(observations)
            except Exception as exc:
                message = f"act_batch_error: {type(exc).__name__}"
                for local_idx in active_indices:
                    errors[local_idx] = message
                    envs[local_idx].invalid_reason = message
                    envs[local_idx].done = True
                break
            if len(actions) != len(active_indices):
                message = "act_batch_error: wrong_action_count"
                for local_idx in active_indices:
                    errors[local_idx] = message
                    envs[local_idx].invalid_reason = message
                    envs[local_idx].done = True
                break
            for local_idx, action in zip(active_indices, actions):
                result = envs[local_idx].step(action)
                if result.get("invalid"):
                    errors[local_idx] = str(result["invalid"])
        for local_idx, env in enumerate(envs):
            task_index, task, variant_index, _ = chunk[local_idx]
            state = env.state()
            metrics = score_rollout(task, state)
            results.append(
                {
                    "task_index": task_index,
                    "variant_index": variant_index,
                    "metrics": metrics,
                    "state": state,
                    "error": errors[local_idx],
                }
            )
    return results


def run_agent_task_batch(agent: Any, task: dict[str, Any], variants: list[dict[str, Any]]) -> list[dict[str, Any]]:
    specs = [(0, task, idx, variant) for idx, variant in enumerate(variants)]
    results = run_agent_rollout_batch(agent, specs, max_batch_rollouts=len(specs))
    return [{key: value for key, value in result.items() if key not in {"task_index", "variant_index"}} for result in results]


def score_agent(agent: Any, tasks: list[dict[str, Any]], max_score: float = 1500.0) -> dict[str, Any]:
    task_summaries = []
    type_scores: dict[str, list[float]] = {}
    tier_scores: dict[str, list[float]] = {}
    object_family_scores: dict[str, list[float]] = {}
    environment_scores: dict[str, list[float]] = {}
    action_error_count = 0
    use_batch = _agent_supports_batch(agent)
    batched_results_by_task: dict[int, list[dict[str, Any]]] = {}
    if use_batch:
        rollout_specs: list[tuple[int, dict[str, Any], int, dict[str, Any]]] = []
        for task_index, task in enumerate(tasks):
            variants = list(task.get("private_rollouts") or task.get("public_variants") or [{}])
            for variant_index, variant in enumerate(variants):
                rollout_specs.append((task_index, task, variant_index, variant))
        for result in run_agent_rollout_batch(agent, rollout_specs):
            batched_results_by_task.setdefault(int(result["task_index"]), []).append(result)
        for values in batched_results_by_task.values():
            values.sort(key=lambda item: int(item.get("variant_index", 0)))
    for task_index, task in enumerate(tasks):
        variants = task.get("private_rollouts") or task.get("public_variants") or [{}]
        variants = list(variants)
        rollout_scores = []
        successes = 0
        if use_batch:
            rollout_results = batched_results_by_task.get(task_index, [])
            if not rollout_results:
                rollout_results = [
                    _batch_episode_error_result(DexSimEnv(task, variant), task, task_index, idx, "batch_missing_result")
                    for idx, variant in enumerate(variants)
                ]
        else:
            rollout_results = [run_agent_episode(agent, task, variant) for variant in variants]
        for result in rollout_results:
            metrics = result["metrics"]
            score01 = float(metrics["score01"])
            rollout_scores.append(score01)
            successes += 1 if metrics.get("success") else 0
            if result.get("error"):
                action_error_count += 1
        robust = 0.72 * (sum(rollout_scores) / len(rollout_scores)) + 0.28 * min(rollout_scores)
        robust = clamp(robust, 0.0, 1.0)
        tier = capability_tier_for_task(task)
        object_family = str(task.get("object_nominal", {}).get("family", task.get("object_nominal", {}).get("shape", "unknown")))
        environment = str(task.get("environment_context", {}).get("surface_fixture", "unknown"))
        task_summaries.append({
            "task_type": task["task_type"],
            "capability_tier": tier,
            "object_family": object_family,
            "environment": environment,
            "score01": robust,
            "success_rate": successes / len(rollout_scores),
        })
        type_scores.setdefault(task["task_type"], []).append(robust)
        tier_scores.setdefault(tier, []).append(robust)
        object_family_scores.setdefault(object_family, []).append(robust)
        environment_scores.setdefault(environment, []).append(robust)
    raw_task_mean_score01 = sum(t["score01"] for t in task_summaries) / max(1, len(task_summaries))
    success_rate = sum(t["success_rate"] for t in task_summaries) / max(1, len(task_summaries))
    by_type = {
        key: round(sum(values) / len(values), 6)
        for key, values in sorted(type_scores.items())
    }
    by_tier = {
        key: round(sum(values) / len(values), 6)
        for key, values in sorted(tier_scores.items())
    }
    by_object_family = {
        key: round(sum(values) / len(values), 6)
        for key, values in sorted(object_family_scores.items())
    }
    by_environment = {
        key: round(sum(values) / len(values), 6)
        for key, values in sorted(environment_scores.items())
    }
    tier_values = list(by_tier.values()) or [0.0]
    tier_mean_score01 = sum(tier_values) / len(tier_values)
    tier_min_score01 = min(tier_values)
    final_score01 = clamp(0.58 * raw_task_mean_score01 + 0.24 * tier_mean_score01 + 0.18 * tier_min_score01, 0.0, 1.0)
    return {
        "score": round(max_score * final_score01, 9),
        "mean_score01": round(final_score01, 9),
        "raw_task_mean_score01": round(raw_task_mean_score01, 9),
        "mean_success_rate": round(success_rate, 9),
        "task_count": len(task_summaries),
        "action_error_count": action_error_count,
        "task_type_mean_score01": by_type,
        "capability_tier_mean_score01": by_tier,
        "capability_tier_min_score01": round(tier_min_score01, 9),
        "object_family_mean_score01": by_object_family,
        "environment_mean_score01": by_environment,
        "batched_rollout_evaluation": use_batch,
        "max_batch_rollouts": MAX_BATCH_ROLLOUTS if use_batch else 1,
        "aggregation_policy": "0.58 * task_mean + 0.24 * capability_tier_mean + 0.18 * worst_capability_tier",
    }


def evaluate_submission(submission_dir: str | Path, tasks: list[dict[str, Any]], max_score: float = 1500.0, isolate_agent: bool = True) -> dict[str, Any]:
    agent = None
    try:
        agent = AgentSubprocessProxy(submission_dir) if isolate_agent else load_agent(submission_dir)
        return score_agent(agent, tasks, max_score=max_score)
    except Exception as exc:
        return {
            "score": 0.0,
            "mean_score01": 0.0,
            "mean_success_rate": 0.0,
            "task_count": len(tasks),
            "action_error_count": len(tasks),
            "task_type_mean_score01": {},
            "capability_tier_mean_score01": {},
            "capability_tier_min_score01": 0.0,
            "object_family_mean_score01": {},
            "environment_mean_score01": {},
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        if isinstance(agent, AgentSubprocessProxy):
            agent.close()


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--agent-worker":
        raise SystemExit(_agent_worker_main(sys.argv[2]))
    raise SystemExit("dexsim_core.py is a library module")
