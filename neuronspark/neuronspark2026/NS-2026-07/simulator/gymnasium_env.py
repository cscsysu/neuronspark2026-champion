
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

try:
    import gymnasium as gym  # type: ignore
    from gymnasium import spaces  # type: ignore
    _GYM_AVAILABLE = True
except Exception:
    gym = None
    spaces = None
    _GYM_AVAILABLE = False

try:
    import numpy as np  # type: ignore
except Exception:
    np = None

try:
    from .dexsim_core import DexSimEnv, read_jsonl, sanitize_task_info, score_rollout
except Exception:
    from dexsim_core import DexSimEnv, read_jsonl, sanitize_task_info, score_rollout


ACTION_PRIMITIVES = ["brace", "stabilize", "push", "drag", "pivot", "roll", "lift_edge", "tap", "wait", "finish"]
FINGERS = ["thumb", "index", "middle", "ring", "pinky", "palm", "wrist"]


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _decode_action(action: dict[str, Any] | list[Any] | tuple[Any, ...]) -> dict[str, Any]:
    if isinstance(action, (list, tuple)):
        primitive_idx = int(action[0]) if len(action) > 0 else 8
        finger_idx = int(action[1]) if len(action) > 1 else 5
        force = float(action[2]) if len(action) > 2 else 0.0
        direction = action[3] if len(action) > 3 else [0.0, 0.0]
        action = {
            "primitive": primitive_idx,
            "finger": finger_idx,
            "force": force,
            "direction": direction,
        }
    payload = dict(action)
    primitive = payload.get("primitive", "wait")
    finger = payload.get("finger", "palm")
    if not isinstance(primitive, str):
        primitive_idx = int(primitive)
        primitive = ACTION_PRIMITIVES[max(0, min(len(ACTION_PRIMITIVES) - 1, primitive_idx))]
    if not isinstance(finger, str):
        finger_idx = int(finger)
        finger = FINGERS[max(0, min(len(FINGERS) - 1, finger_idx))]
    force = payload.get("force", 0.0)
    if isinstance(force, (list, tuple)):
        force = force[0] if force else 0.0
    direction = payload.get("direction", [0.0, 0.0])
    if hasattr(direction, "tolist"):
        direction = direction.tolist()
    if not isinstance(direction, (list, tuple)) or len(direction) < 2:
        direction = [0.0, 0.0]
    return {
        "primitive": str(primitive),
        "finger": str(finger),
        "force": _clamp(float(force), 0.0, 1.0),
        "direction": [float(direction[0]), float(direction[1])],
    }


def compact_observation(observation: dict[str, Any]) -> dict[str, Any]:
    low_dim = observation.get("low_dim_state", {}).get("values", [])
    tactile = observation.get("tactile_heatmap_7x4", [])
    tactile_image = observation.get("tactile_image_7x8x8", {}).get("values", [])
    vision_grid = observation.get("vision_grid_16x16", {}).get("values", [])
    if np is not None:
        return {
            "low_dim": np.asarray(low_dim, dtype=np.float32),
            "tactile_heatmap": np.asarray(tactile, dtype=np.float32),
            "tactile_image": np.asarray(tactile_image, dtype=np.float32).reshape(7, 8, 8) / 255.0,
            "vision_grid": np.asarray(vision_grid, dtype=np.float32).reshape(6, 16, 16) / 255.0,
        }
    return {
        "low_dim": list(low_dim),
        "tactile_heatmap": [list(row) for row in tactile],
        "tactile_image": list(tactile_image),
        "vision_grid": list(vision_grid),
    }


class _BaseEnv(gym.Env if _GYM_AVAILABLE else object):  # type: ignore[misc]
    metadata = {"render_modes": ["rgb_array", "none"], "render_fps": 8}


class DexSimGymEnv(_BaseEnv):
    """Gymnasium-style wrapper around the public NS-2026-07 dev simulator.

    The official contest still evaluates `agent.py` with `Agent.reset/act`.
    This wrapper is for local training, debugging and baseline development.
    """

    def __init__(
        self,
        task: dict[str, Any] | None = None,
        task_file: str | Path | None = None,
        task_index: int = 0,
        variant_index: int = 0,
        observation_mode: str = "public",
        render_mode: str | None = None,
    ):
        self.tasks = read_jsonl(task_file) if task_file is not None else ([task] if task is not None else [])
        if not self.tasks:
            raise ValueError("DexSimGymEnv requires task=... or task_file=...")
        self.task_index = int(task_index)
        self.variant_index = int(variant_index)
        self.observation_mode = observation_mode
        self.render_mode = render_mode or "none"
        self.rng = random.Random(0)
        self.env: DexSimEnv | None = None
        self.task: dict[str, Any] | None = None
        self.variant: dict[str, Any] | None = None
        self.last_score01 = 0.0
        if _GYM_AVAILABLE and np is not None:
            self.action_space = spaces.Dict({
                "primitive": spaces.Discrete(len(ACTION_PRIMITIVES)),
                "finger": spaces.Discrete(len(FINGERS)),
                "force": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                "direction": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
            })
            if observation_mode == "compact":
                self.observation_space = spaces.Dict({
                    "low_dim": spaces.Box(low=-float("inf"), high=float("inf"), shape=(14,), dtype=np.float32),
                    "tactile_heatmap": spaces.Box(low=0.0, high=2.5, shape=(7, 4), dtype=np.float32),
                    "tactile_image": spaces.Box(low=0.0, high=1.0, shape=(7, 8, 8), dtype=np.float32),
                    "vision_grid": spaces.Box(low=0.0, high=1.0, shape=(6, 16, 16), dtype=np.float32),
                })

    def _select_task(self, options: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any] | None]:
        opts = dict(options or {})
        if "task" in opts:
            task = opts["task"]
        else:
            idx = int(opts.get("task_index", self.task_index)) % len(self.tasks)
            task = self.tasks[idx]
        variants = task.get("public_variants") or [{}]
        variant = opts.get("variant")
        if variant is None:
            variant_idx = int(opts.get("variant_index", self.variant_index)) % len(variants)
            variant = variants[variant_idx]
        return task, variant

    def _format_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        if self.observation_mode == "compact":
            return compact_observation(observation)
        return observation

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):  # noqa: D401
        if seed is not None:
            self.rng.seed(seed)
        self.task, self.variant = self._select_task(options)
        self.env = DexSimEnv(self.task, self.variant)
        self.last_score01 = 0.0
        obs = self.env.observation()
        info = {
            "task_info": sanitize_task_info(self.task),
            "task_id": self.task.get("id"),
            "variant_id": (self.variant or {}).get("variant_id"),
            "sim_backend": obs.get("sim_backend", "proxy"),
        }
        return self._format_observation(obs), info

    def step(self, action: dict[str, Any] | list[Any] | tuple[Any, ...]):
        if self.env is None or self.task is None:
            self.reset()
        assert self.env is not None
        assert self.task is not None
        decoded = _decode_action(action)
        step_result = self.env.step(decoded)
        obs = self.env.observation()
        state = self.env.state()
        metrics = score_rollout(self.task, state)
        score01 = float(metrics["score01"])
        reward = score01 - self.last_score01
        if metrics.get("success"):
            reward += 0.2
        self.last_score01 = score01
        terminated = bool(self.env.done or metrics.get("success") or step_result.get("invalid"))
        truncated = bool(self.env.step_count >= self.env.max_steps and not terminated)
        info = {
            "metrics": metrics,
            "state": state,
            "decoded_action": decoded,
            "invalid": step_result.get("invalid"),
            "sim_backend": obs.get("sim_backend", "proxy"),
        }
        return self._format_observation(obs), float(reward), terminated, truncated, info

    def sample_action(self) -> dict[str, Any]:
        angle = self.rng.uniform(-3.14159, 3.14159)
        return {
            "primitive": self.rng.choice(["push", "drag", "pivot", "stabilize", "tap", "wait"]),
            "finger": self.rng.choice(FINGERS),
            "force": self.rng.uniform(0.15, 0.75),
            "direction": [__import__("math").cos(angle), __import__("math").sin(angle)],
        }

    def render(self):
        if self.env is None:
            return None
        return self.env.state()


def make_env(task_file: str | Path, task_index: int = 0, variant_index: int = 0, observation_mode: str = "public") -> DexSimGymEnv:
    return DexSimGymEnv(
        task_file=task_file,
        task_index=task_index,
        variant_index=variant_index,
        observation_mode=observation_mode,
    )
