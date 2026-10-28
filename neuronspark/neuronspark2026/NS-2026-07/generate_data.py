#!/usr/bin/env python3
"""Generate training data by running the rule-based agent on train tasks.

Usage:
    python generate_data.py --tasks tasks/train_tasks.jsonl --output data/train_episodes.pt
    python generate_data.py --tasks tasks/train_tasks.jsonl --output data/train_episodes.pt --max-tasks 100  # quick test
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "simulator"))

from dexsim_core import DexSimEnv, sanitize_task_info, score_rollout, read_jsonl


# ============================================================
# Rule-based agent (v7 - our best version, inlined)
# ============================================================
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
PRIM2IDX = {p: i for i, p in enumerate(ACTION_PRIMITIVES)}
FINGER2IDX = {f: i for i, f in enumerate(FINGERS)}


class RuleAgent:
    """V7 rule agent for data generation."""

    def reset(self, task_info):
        self.task_type = task_info.get("task_type", "nonprehensile_relocation")
        self.tool_axis = task_info.get("tool_goal", {}).get("axis", [1.0, 0.0])
        self.tool_need = float(task_info.get("tool_goal", {}).get("progress", 0.0))
        self.reserved = task_info.get("resource_constraint", {}).get("reserve_fingers", [])
        self.max_peak_force = float(task_info.get("resource_constraint", {}).get("max_peak_force", 0.8))
        self.obstacles = task_info.get("scene_context", {}).get("obstacles", []) if isinstance(task_info.get("scene_context", {}), dict) else []
        self.corridor = task_info.get("scene_context", {}).get("corridor", {}) if isinstance(task_info.get("scene_context", {}), dict) else {}

        obj = task_info.get("object", {})
        self.fragility = float(obj.get("fragility", 0.0))
        self.mass = float(obj.get("mass", 1.0))
        self.obj_size = float(obj.get("size", 0.1))

        self.dmg_thresh = 1.05 - 0.72 * self.fragility
        self.safe_palm = min(0.92, self.dmg_thresh / 1.15)
        self.safe_index = min(0.95, self.dmg_thresh / 0.98)
        self.safe_thumb = min(0.92, self.dmg_thresh / 1.05)

        self.taps_done = 0
        per_tap = 0.048
        if self.task_type == "tool_use":
            target = max(0.08, self.tool_need) * 0.84
            self.taps_needed = int(_clamp(math.ceil(target / per_tap), 9, 24))
        elif self.task_type == "resource_sequence":
            target = max(0.08, self.tool_need) * 0.84
            self.taps_needed = int(_clamp(math.ceil(target / per_tap), 9, 22))
        else:
            self.taps_needed = 0

        self.phase = "brace"
        self.brace_done = 0
        self.stuck = 0
        self.last_dist = None

    def _force(self, base, role, damage_risk):
        if role == "palm":
            f = min(base, self.safe_palm)
        elif role == "tap":
            f = min(base, self.safe_index)
        elif role == "pivot":
            f = min(base, self.safe_thumb)
        else:
            f = min(base, self.safe_palm)
        if damage_risk > 0.15:
            f *= max(0.50, 1.0 - damage_risk * 0.55)
        f = min(f, self.max_peak_force + 0.06)
        return _clamp(f, 0.05, 1.0)

    def _avoid(self, dx, dy, obs):
        pose = obs.get("object_pose_estimate", {})
        ox, oy = float(pose.get("x", 0.0)), float(pose.get("y", 0.0))
        ax, ay = dx, dy
        for o in self.obstacles:
            try:
                cx, cy, r = float(o.get("x", 0)), float(o.get("y", 0)), float(o.get("radius", 0.08))
            except:
                continue
            ddx, ddy = ox - cx, oy - cy
            d = _norm2(ddx, ddy)
            danger = r + max(0.025, self.obj_size * 0.22) + 0.06
            if d < danger + 0.16 and d > 1e-6:
                ndx, ndy = _normalize(ddx, ddy)
                scale = (0.9 + (danger - d) * 5.0) if d < danger else (0.4 * (1.0 - (d - danger) / 0.16))
                ax += ndx * scale
                ay += ndy * scale
        if self.corridor:
            try:
                w = float(self.corridor.get("width", 0.0))
                if w > 0:
                    ca = self.corridor.get("axis", [1.0, 0.0])
                    cax, cay = _normalize(float(ca[0]), float(ca[1]))
                    lat = -cay * ox + cax * oy - float(self.corridor.get("offset", 0.0))
                    half = w * 0.5
                    margin = half - abs(lat)
                    if margin < 0.10:
                        sign = -1.0 if lat > 0 else 1.0
                        ax += (-cay) * sign * 0.5 * max(0.0, 1.0 - margin / 0.10)
                        ay += cax * sign * 0.5 * max(0.0, 1.0 - margin / 0.10)
                    if margin < 0.12:
                        along = ax * cax + ay * cay
                        perp = ax * (-cay) + ay * cax
                        perp *= 0.4
                        ax = along * cax + perp * (-cay)
                        ay = along * cay + perp * cax
            except:
                pass
        return _normalize(ax, ay)

    def act(self, observation):
        step = int(observation["step"])
        remaining = int(observation.get("remaining_steps", 128 - step))
        state = observation["low_dim_state"]["values"]
        dx, dy, dtheta = float(state[0]), float(state[1]), float(state[2])
        dx_f, dy_f, dtheta_f = float(state[3]), float(state[4]), float(state[5])
        contact = float(state[9])
        slip_risk = float(state[10])
        damage_risk = float(observation.get("contact_summary", {}).get("damage_risk", 0.0))
        dist = _norm2(dx, dy)
        angle = abs(dtheta)
        dist_f = _norm2(dx_f, dy_f)
        angle_f = abs(dtheta_f)
        sc = observation.get("stage_context", {})

        # Finish
        stages_ok = not sc.get("enabled", False) or float(sc.get("completion_fraction", 0)) >= 0.999
        tool_ok = self.task_type == "nonprehensile_relocation" or self.taps_done >= self.taps_needed * 0.72
        if dist_f < 0.062 and angle_f < 0.24 and stages_ok and tool_ok:
            return {"primitive": "finish", "finger": "palm", "force": 0.0, "direction": [0.0, 0.0]}
        if remaining <= 1:
            if dist_f < 0.13 and angle_f < 0.55:
                return {"primitive": "finish", "finger": "palm", "force": 0.0, "direction": [0.0, 0.0]}
            nfx, nfy = self._avoid(*_normalize(dx_f, dy_f), observation)
            return {"primitive": "push", "finger": "palm", "force": self._force(0.82, "palm", damage_risk), "direction": [nfx, nfy]}

        if contact < 0.06:
            return {"primitive": "stabilize", "finger": "palm", "force": 0.45, "direction": [0.0, 0.0]}

        if self.phase == "brace":
            self.brace_done += 1
            if (self.brace_done >= 1 and contact >= 0.22) or self.brace_done >= 2:
                self.phase = "tool" if self.taps_needed > 0 else "move"
            return {"primitive": "brace", "finger": "palm", "force": 0.38, "direction": [0.0, 0.0]}

        if contact < 0.13 or (slip_risk > 0.86 and contact < 0.26):
            return {"primitive": "brace", "finger": "palm", "force": 0.36, "direction": [0.0, 0.0]}

        if self.phase == "tool":
            if self.taps_done >= self.taps_needed:
                self.phase = "move"
            else:
                if contact < 0.20:
                    return {"primitive": "brace", "finger": "palm", "force": 0.33, "direction": [0.0, 0.0]}
                self.taps_done += 1
                ax, ay = _normalize(float(self.tool_axis[0]), float(self.tool_axis[1]))
                finger = self.reserved[0] if (self.task_type == "resource_sequence" and self.reserved) else "index"
                return {"primitive": "tap", "finger": finger, "force": self._force(0.88, "tap", damage_risk), "direction": [ax, ay]}

        # Move phase extras
        if self.taps_done < self.taps_needed and step % 4 == 0 and remaining > 10 and contact > 0.20:
            self.taps_done += 1
            ax, ay = _normalize(float(self.tool_axis[0]), float(self.tool_axis[1]))
            finger = self.reserved[0] if (self.task_type == "resource_sequence" and self.reserved) else "index"
            return {"primitive": "tap", "finger": finger, "force": self._force(0.85, "tap", damage_risk), "direction": [ax, ay]}

        if self.last_dist is not None and dist > 0.10:
            self.stuck = self.stuck + 1 if abs(dist - self.last_dist) < 0.003 else max(0, self.stuck - 2)
        self.last_dist = dist
        if self.stuck > 6:
            self.stuck = 0
            nx, ny = self._avoid(*_normalize(dx, dy), observation)
            return {"primitive": "drag", "finger": "index", "force": self._force(0.75, "palm", damage_risk), "direction": [nx, ny]}

        if angle > 0.36 and dist < 0.22:
            sign = 1.0 if dtheta >= 0 else -1.0
            f = 0.86 if self.mass > 1.2 else 0.80
            return {"primitive": "pivot", "finger": "thumb", "force": self._force(f, "pivot", damage_risk), "direction": [sign, 0.0]}

        if angle > 0.26 and dist > 0.12:
            nx, ny = self._avoid(*_normalize(dx, dy), observation)
            if (nx + ny >= 0) == (dtheta >= 0):
                return {"primitive": "roll", "finger": "palm", "force": self._force(0.74, "palm", damage_risk), "direction": [nx, ny]}
            else:
                sign = 1.0 if dtheta >= 0 else -1.0
                return {"primitive": "pivot", "finger": "thumb", "force": self._force(0.80, "pivot", damage_risk), "direction": [sign, 0.0]}

        if dist > 0.088:
            nx, ny = self._avoid(*_normalize(dx, dy), observation)
            base = 0.88 if self.mass > 1.3 else (0.82 if dist > 0.28 else (0.74 if dist > 0.15 else 0.65))
            return {"primitive": "push", "finger": "palm", "force": self._force(base, "palm", damage_risk), "direction": [nx, ny]}

        if angle > 0.12:
            sign = 1.0 if dtheta >= 0 else -1.0
            return {"primitive": "pivot", "finger": "thumb", "force": self._force(0.55, "pivot", damage_risk), "direction": [sign, 0.0]}

        tool_ok2 = self.task_type == "nonprehensile_relocation" or self.taps_done >= self.taps_needed * 0.65
        if dist_f < 0.065 and angle_f < 0.26 and stages_ok and tool_ok2:
            return {"primitive": "finish", "finger": "palm", "force": 0.0, "direction": [0.0, 0.0]}

        nx, ny = _normalize(dx, dy) if dist > 0.02 else _normalize(dx_f, dy_f)
        nx, ny = self._avoid(nx, ny, observation)
        return {"primitive": "push", "finger": "palm", "force": self._force(0.42, "palm", damage_risk), "direction": [nx, ny]}


def obs_to_features(obs, task_info):
    """Convert observation to fixed-size feature vector for training."""
    state = obs["low_dim_state"]["values"]
    # 14-dim low_dim_state
    low_dim = np.array(state, dtype=np.float32)

    # 7x4 tactile heatmap -> 28 dim
    heatmap = np.array(obs.get("tactile_heatmap_7x4", [[0]*4]*7), dtype=np.float32).flatten()

    # Contact summary: 4 dim
    cs = obs.get("contact_summary", {})
    contact_feat = np.array([
        float(cs.get("coverage", 0)),
        float(cs.get("min_contact_so_far", 0)),
        float(cs.get("slip_risk", 0)),
        float(cs.get("damage_risk", 0)),
    ], dtype=np.float32)

    # Task encoding: task_type(3) + fragility + mass + friction + tool_need + has_stages + obj_size + remaining_frac = 10
    task_type_vec = np.zeros(3, dtype=np.float32)
    tt = task_info.get("task_type", "nonprehensile_relocation")
    if tt == "nonprehensile_relocation":
        task_type_vec[0] = 1.0
    elif tt == "tool_use":
        task_type_vec[1] = 1.0
    else:
        task_type_vec[2] = 1.0

    obj = task_info.get("object", {})
    task_feat = np.array([
        float(obj.get("fragility", 0.0)),
        float(obj.get("mass", 1.0)) / 2.0,  # normalize
        float(obj.get("friction", 0.5)),
        float(task_info.get("tool_goal", {}).get("progress", 0.0)),
        1.0 if task_info.get("stage_goals") else 0.0,
        float(obj.get("size", 0.1)) * 5.0,
        float(obs.get("remaining_steps", 64)) / 128.0,
    ], dtype=np.float32)

    # Combine: 14 + 28 + 4 + 3 + 7 = 56 features
    features = np.concatenate([low_dim, heatmap, contact_feat, task_type_vec, task_feat])
    return features


def action_to_labels(action):
    """Convert action dict to training labels."""
    prim_idx = PRIM2IDX.get(action["primitive"], 8)  # default to wait
    finger_idx = FINGER2IDX.get(action["finger"], 5)  # default to palm
    force = float(action["force"])
    direction = [float(action["direction"][0]), float(action["direction"][1])]
    return prim_idx, finger_idx, force, direction


def generate_episodes(tasks, max_tasks=None, variants_per_task=3):
    """Generate training data by running rule agent on tasks."""
    agent = RuleAgent()
    all_features = []
    all_prim = []
    all_finger = []
    all_force = []
    all_direction = []

    if max_tasks:
        tasks = tasks[:max_tasks]

    total_episodes = 0
    total_successes = 0
    t0 = time.time()

    for task_idx, task in enumerate(tasks):
        variants = task.get("public_variants", [{}])[:variants_per_task]
        task_info = sanitize_task_info(task)

        for v in variants:
            env = DexSimEnv(task, v)
            agent.reset(task_info)

            episode_features = []
            episode_actions = []

            while not env.done and env.step_count < env.max_steps:
                obs = env.observation()
                action = agent.act(obs)

                # Collect data
                feat = obs_to_features(obs, task_info)
                labels = action_to_labels(action)
                episode_features.append(feat)
                episode_actions.append(labels)

                env.step(action)

            state = env.state()
            metrics = score_rollout(task, state)
            total_episodes += 1
            if metrics.get("success"):
                total_successes += 1

            # Store all steps (even from failed episodes - they contain useful partial behavior)
            for feat, (p, f, force, d) in zip(episode_features, episode_actions):
                all_features.append(feat)
                all_prim.append(p)
                all_finger.append(f)
                all_force.append(force)
                all_direction.append(d)

        if (task_idx + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"[{task_idx+1}/{len(tasks)}] episodes={total_episodes} success={total_successes}/{total_episodes} "
                  f"({total_successes/max(1,total_episodes)*100:.1f}%) samples={len(all_features)} "
                  f"elapsed={elapsed:.1f}s")

    print(f"\nDone! {total_episodes} episodes, {total_successes} successes ({total_successes/max(1,total_episodes)*100:.1f}%)")
    print(f"Total training samples: {len(all_features)}")

    # Convert to numpy arrays
    features = np.array(all_features, dtype=np.float32)
    prims = np.array(all_prim, dtype=np.int64)
    fingers = np.array(all_finger, dtype=np.int64)
    forces = np.array(all_force, dtype=np.float32)
    directions = np.array(all_direction, dtype=np.float32)

    return {
        "features": features,
        "primitives": prims,
        "fingers": fingers,
        "forces": forces,
        "directions": directions,
        "n_episodes": total_episodes,
        "n_successes": total_successes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="tasks/train_tasks.jsonl")
    parser.add_argument("--output", default="data/train_episodes.pt")
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--variants", type=int, default=3)
    args = parser.parse_args()

    tasks = read_jsonl(args.tasks)
    print(f"Loaded {len(tasks)} tasks from {args.tasks}")

    data = generate_episodes(tasks, max_tasks=args.max_tasks, variants_per_task=args.variants)

    # Save as numpy (compatible without torch)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(output_path).replace(".pt", ".npz"),
        features=data["features"],
        primitives=data["primitives"],
        fingers=data["fingers"],
        forces=data["forces"],
        directions=data["directions"],
    )
    print(f"Saved to {str(output_path).replace('.pt', '.npz')}")
    print(f"Feature shape: {data['features'].shape}")
    print(f"Episodes: {data['n_episodes']}, Successes: {data['n_successes']}")


if __name__ == "__main__":
    main()
