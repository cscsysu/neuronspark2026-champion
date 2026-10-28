"""NS-2026-07 CEM Search Agent.

Uses Cross-Entropy Method (CEM) to search for optimal actions online.
At each step, samples many candidate actions, simulates forward using
a proxy environment, and picks the best action.

KEY INSIGHT: The simulator runs at 117K steps/sec on CPU.
With 2s per act() call, we have budget for 200K+ simulation steps.
This allows massive lookahead search that far exceeds any fixed rules.
"""
from __future__ import annotations

import copy
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

# Import simulator for lookahead
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
try:
    from simulator.dexsim_core import ProxyDexSimEnv, sanitize_task_info, score_rollout, validate_action
except ImportError:
    try:
        sys.path.insert(0, str(_ROOT / "simulator"))
        from dexsim_core import ProxyDexSimEnv, sanitize_task_info, score_rollout, validate_action
    except ImportError:
        ProxyDexSimEnv = None


def _norm2(x, y):
    return math.sqrt(x * x + y * y)

def _normalize(x, y):
    n = _norm2(x, y)
    if n < 1e-9:
        return 0.0, 0.0
    return x / n, y / n

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


ACTION_PRIMITIVES = ["brace", "push", "drag", "pivot", "roll", "lift_edge", "tap", "stabilize", "wait", "finish"]
FINGERS = ["thumb", "index", "middle", "ring", "pinky", "palm", "wrist"]

# Movement primitives and their best fingers
MOVE_PRIMITIVES = ["push", "drag", "roll", "lift_edge"]
ROTATE_PRIMITIVES = ["pivot", "roll"]


class Agent:
    """CEM-based online search agent.

    Strategy:
    1. Maintain a proxy environment that mirrors the real env state
    2. At each step, use CEM to search for best action sequence
    3. Execute only the first action from the best sequence
    4. Re-plan at next step (receding horizon)
    """

    def __init__(self):
        self._batch_agents = None

    def reset(self, task_info: dict) -> None:
        self.task_info = task_info
        self.task_type = task_info.get("task_type", "nonprehensile_relocation")
        self.tool_axis = task_info.get("tool_goal", {}).get("axis", [1.0, 0.0])
        self.tool_need = float(task_info.get("tool_goal", {}).get("progress", 0.0))
        self.reserved = task_info.get("resource_constraint", {}).get("reserve_fingers", [])
        self.max_peak_force = float(task_info.get("resource_constraint", {}).get("max_peak_force", 0.8))

        obj = task_info.get("object", {})
        self.fragility = float(obj.get("fragility", 0.0))
        self.mass = float(obj.get("mass", 1.0))
        self.obj_size = float(obj.get("size", 0.1))
        self.dmg_thresh = 1.05 - 0.72 * self.fragility

        # For CEM search
        self.rng = random.Random(42)
        self.step_num = 0
        self.taps_done = 0

        # Compute needed taps
        per_tap = 0.048
        if self.task_type == "tool_use":
            self.taps_needed = int(_clamp(math.ceil(max(0.08, self.tool_need) * 0.84 / per_tap), 9, 24))
        elif self.task_type == "resource_sequence":
            self.taps_needed = int(_clamp(math.ceil(max(0.08, self.tool_need) * 0.84 / per_tap), 9, 22))
        else:
            self.taps_needed = 0

    def reset_batch(self, task_infos: list[dict]) -> None:
        self._batch_agents = [Agent() for _ in task_infos]
        for a, t in zip(self._batch_agents, task_infos):
            a.reset(t)

    def act_batch(self, observations: list[dict]) -> list[dict]:
        if self._batch_agents is None:
            raise RuntimeError("reset_batch must be called before act_batch")
        return [self._batch_agents[int(o.get("batch_index", i))].act(o)
                for i, o in enumerate(observations)]

    def _create_proxy_env(self, observation: dict) -> ProxyDexSimEnv | None:
        """Create a proxy env that matches the current observed state."""
        if ProxyDexSimEnv is None:
            return None

        # Reconstruct task dict for proxy env
        task = dict(self.task_info)
        # Override start_pose with CURRENT estimated pose
        pose = observation.get("object_pose_estimate", {})
        task["start_pose"] = {
            "x": float(pose.get("x", 0.0)),
            "y": float(pose.get("y", 0.0)),
            "theta": float(pose.get("theta", 0.0)),
        }

        # Create env with default variant (no extreme physics)
        try:
            env = ProxyDexSimEnv(task, variant={"observation_noise": 0.001})
            # Sync state from observation
            env.step_count = int(observation.get("step", 0))
            state = observation["low_dim_state"]["values"]
            env.contact = float(state[9])
            env.damage = float(state[11]) * 0.5  # estimate
            env.energy = float(state[12]) * 72.0
            env.tool_progress = self.taps_done * 0.05
            # Set velocity
            vel = observation.get("object_velocity_estimate", {})
            env.vx = float(vel.get("vx", 0.0))
            env.vy = float(vel.get("vy", 0.0))
            env.vtheta = float(vel.get("vtheta", 0.0))
            return env
        except Exception:
            return None

    def _evaluate_action_sequence(self, env: ProxyDexSimEnv, actions: list[dict]) -> float:
        """Simulate action sequence and return score."""
        for action in actions:
            if env.done:
                break
            env.step(action)
        state = env.state()
        # Quick score heuristic (faster than full score_rollout)
        dist = state["distance_error"]
        angle = state["angle_error"]
        damage = state["damage"]
        stage_frac = state["stage_completion_fraction"]
        # Heuristic score
        pose_score = 0.66 * math.exp(-dist / 0.18) + 0.34 * math.exp(-angle / 0.55)
        safety = max(0.0, 1.0 - damage * 1.85 - state["drop_events"] * 0.18)
        score = 0.50 * pose_score + 0.20 * safety + 0.15 * stage_frac + 0.15 * (1.0 - dist)
        if state.get("terminated_by_finish") and dist < 0.085 and angle < 0.34 and safety > 0.58:
            score += 0.5  # success bonus
        return score

    def _generate_candidate_actions(self, obs: dict, n: int) -> list[dict]:
        """Generate diverse candidate actions for CEM sampling."""
        state = obs["low_dim_state"]["values"]
        dx, dy, dtheta = float(state[0]), float(state[1]), float(state[2])
        contact = float(state[9])
        damage_risk = float(obs.get("contact_summary", {}).get("damage_risk", 0.0))
        dist = _norm2(dx, dy)
        angle = abs(dtheta)

        # Target direction
        nx, ny = _normalize(dx, dy)
        # Rotation sign
        rot_sign = 1.0 if dtheta >= 0 else -1.0

        candidates = []
        safe_force = min(0.92, self.dmg_thresh / 1.15, self.max_peak_force + 0.05)
        if damage_risk > 0.2:
            safe_force *= 0.7

        # Smart candidates (likely good actions)
        # 1. Push toward target (various forces)
        for f in [0.5, 0.65, 0.75, 0.85, safe_force]:
            candidates.append({"primitive": "push", "finger": "palm", "force": min(f, safe_force), "direction": [nx, ny]})

        # 2. Push with slight angle variations
        for angle_off in [-0.3, -0.15, 0.15, 0.3]:
            cos_a, sin_a = math.cos(angle_off), math.sin(angle_off)
            rx = nx * cos_a - ny * sin_a
            ry = nx * sin_a + ny * cos_a
            candidates.append({"primitive": "push", "finger": "palm", "force": min(0.75, safe_force), "direction": [rx, ry]})

        # 3. Pivot for rotation
        candidates.append({"primitive": "pivot", "finger": "thumb", "force": min(0.8, safe_force), "direction": [rot_sign, 0.0]})
        candidates.append({"primitive": "pivot", "finger": "thumb", "force": min(0.6, safe_force), "direction": [rot_sign, 0.0]})

        # 4. Roll (move + rotate)
        candidates.append({"primitive": "roll", "finger": "palm", "force": min(0.7, safe_force), "direction": [nx, ny]})

        # 5. Brace/stabilize for contact
        candidates.append({"primitive": "brace", "finger": "palm", "force": 0.38, "direction": [0.0, 0.0]})
        candidates.append({"primitive": "stabilize", "finger": "palm", "force": 0.4, "direction": [0.0, 0.0]})

        # 6. Tap (for tool tasks)
        if self.taps_done < self.taps_needed:
            ax, ay = _normalize(float(self.tool_axis[0]), float(self.tool_axis[1]))
            finger = self.reserved[0] if (self.task_type == "resource_sequence" and self.reserved) else "index"
            candidates.append({"primitive": "tap", "finger": finger, "force": min(0.88, safe_force), "direction": [ax, ay]})

        # 7. Finish
        candidates.append({"primitive": "finish", "finger": "palm", "force": 0.0, "direction": [0.0, 0.0]})

        # 8. Random exploration
        while len(candidates) < n:
            prim = self.rng.choice(["push", "push", "push", "drag", "pivot", "roll", "brace"])
            finger = self.rng.choice(["palm", "palm", "thumb", "index"])
            force = self.rng.uniform(0.3, min(0.9, safe_force))
            angle_r = self.rng.uniform(-math.pi, math.pi)
            d = [math.cos(angle_r), math.sin(angle_r)]
            candidates.append({"primitive": prim, "finger": finger, "force": force, "direction": d})

        return candidates[:n]

    def _cem_search(self, obs: dict, horizon: int = 8, n_candidates: int = 80, n_elite: int = 15, n_iters: int = 3) -> dict:
        """CEM search for best first action."""
        env_base = self._create_proxy_env(obs)
        if env_base is None:
            return self._fallback_act(obs)

        # Generate initial candidates
        candidates = self._generate_candidate_actions(obs, n_candidates)

        best_action = candidates[0]
        best_score = -float('inf')

        for iteration in range(n_iters):
            scores = []
            for action in candidates:
                # Clone env and simulate
                env_copy = self._clone_env(env_base)
                if env_copy is None:
                    continue

                # Execute candidate action + follow-up with simple heuristic
                env_copy.step(action)
                # Simple follow-up: push toward target for remaining horizon
                state_vals = obs["low_dim_state"]["values"]
                follow_dx, follow_dy = float(state_vals[0]), float(state_vals[1])
                fnx, fny = _normalize(follow_dx, follow_dy)
                follow_action = {"primitive": "push", "finger": "palm", "force": 0.65, "direction": [fnx, fny]}
                for _ in range(horizon - 1):
                    if env_copy.done:
                        break
                    env_copy.step(follow_action)

                score = self._evaluate_action_sequence(env_copy, [])
                scores.append((score, action))

            if not scores:
                break

            # Sort and pick elite
            scores.sort(key=lambda x: x[0], reverse=True)
            if scores[0][0] > best_score:
                best_score = scores[0][0]
                best_action = scores[0][1]

            # For next iteration: mutate elite actions
            elite_actions = [s[1] for s in scores[:n_elite]]
            candidates = list(elite_actions)
            # Mutate elite
            for ea in elite_actions:
                for _ in range(3):
                    mutated = dict(ea)
                    mutated["force"] = _clamp(ea["force"] + self.rng.gauss(0, 0.1), 0.05, 1.0)
                    d = ea["direction"]
                    mutated["direction"] = [
                        _clamp(d[0] + self.rng.gauss(0, 0.15), -1.0, 1.0),
                        _clamp(d[1] + self.rng.gauss(0, 0.15), -1.0, 1.0),
                    ]
                    candidates.append(mutated)
            # Add fresh random candidates
            candidates.extend(self._generate_candidate_actions(obs, n_candidates - len(candidates)))
            candidates = candidates[:n_candidates]

        # Track taps
        if best_action["primitive"] == "tap":
            self.taps_done += 1

        return best_action

    def _clone_env(self, env: ProxyDexSimEnv) -> ProxyDexSimEnv | None:
        """Cheaply clone a proxy environment."""
        try:
            return copy.deepcopy(env)
        except Exception:
            return None

    def _fallback_act(self, obs: dict) -> dict:
        """Simple rule fallback when search is unavailable."""
        state = obs["low_dim_state"]["values"]
        dx, dy, dtheta = float(state[0]), float(state[1]), float(state[2])
        dx_f, dy_f = float(state[3]), float(state[4])
        contact = float(state[9])
        dist_f = _norm2(dx_f, dy_f)
        angle_f = abs(float(state[5]))

        if dist_f < 0.062 and angle_f < 0.24:
            return {"primitive": "finish", "finger": "palm", "force": 0.0, "direction": [0.0, 0.0]}
        if contact < 0.10:
            return {"primitive": "brace", "finger": "palm", "force": 0.38, "direction": [0.0, 0.0]}
        nx, ny = _normalize(dx, dy)
        return {"primitive": "push", "finger": "palm", "force": 0.70, "direction": [nx, ny]}

    def act(self, observation: dict) -> dict:
        self.step_num += 1
        state = observation["low_dim_state"]["values"]
        contact = float(state[9])
        dist_f = _norm2(float(state[3]), float(state[4]))
        angle_f = abs(float(state[5]))
        remaining = int(observation.get("remaining_steps", 128))

        # Quick safety checks (skip search for obvious actions)
        if contact < 0.05:
            return {"primitive": "stabilize", "finger": "palm", "force": 0.45, "direction": [0.0, 0.0]}

        # Use CEM search
        if ProxyDexSimEnv is not None:
            # Adjust search budget based on remaining steps
            if remaining <= 2:
                # Last steps: quick decision
                if dist_f < 0.13 and angle_f < 0.55:
                    return {"primitive": "finish", "finger": "palm", "force": 0.0, "direction": [0.0, 0.0]}
                nx, ny = _normalize(float(state[0]), float(state[1]))
                return {"primitive": "push", "finger": "palm", "force": 0.80, "direction": [nx, ny]}

            # CEM search with adaptive horizon
            horizon = min(12, remaining)
            return self._cem_search(observation, horizon=horizon, n_candidates=100, n_elite=20, n_iters=3)

        return self._fallback_act(observation)
