"""NS-2026-07: BC-trained MLP agent with rule safety wrapper.

The MLP learned from the rule agent running under extreme domain randomization.
It implicitly learns smooth force/direction adaptation that rules can't achieve.
Safety wrapper ensures no catastrophic failures.
"""
from __future__ import annotations
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ACTION_PRIMITIVES = ["brace", "push", "drag", "pivot", "roll", "lift_edge", "tap", "stabilize", "wait", "finish"]
FINGERS = ["thumb", "index", "middle", "ring", "pinky", "palm", "wrist"]
OBS_DIM = 56
N_PRIM = 10
N_FINGER = 7


def _norm2(x, y):
    return math.sqrt(x * x + y * y)

def _normalize(x, y):
    n = _norm2(x, y)
    if n < 1e-9: return 0.0, 0.0
    return x / n, y / n

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


class PolicyMLP(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, 128), nn.ReLU(),
        )
        self.ph = nn.Linear(128, N_PRIM)
        self.fh = nn.Linear(128, N_FINGER)
        self.foh = nn.Linear(128, 1)
        self.dh = nn.Linear(128, 2)

    def forward(self, x):
        h = self.net(x)
        return self.ph(h), self.fh(h), torch.sigmoid(self.foh(h)), torch.tanh(self.dh(h))


class Agent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = Path(__file__).parent / "model" / "policy.pt"
        self.model = None
        if model_path.exists():
            ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
            cfg = ckpt.get("config", {})
            self.model = PolicyMLP(cfg.get("obs_dim", OBS_DIM), cfg.get("hidden", 256)).to(self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.model.eval()
        self._batch_agents = None

    def reset(self, task_info: dict) -> None:
        self.task_info = task_info
        self.task_type = task_info.get("task_type", "nonprehensile_relocation")
        obj = task_info.get("object", {})
        self.fragility = float(obj.get("fragility", 0.0))
        self.mass = float(obj.get("mass", 1.0))
        self.dmg_thresh = 1.05 - 0.72 * self.fragility
        self.safe_force = min(0.92, self.dmg_thresh / 1.15)
        self.max_peak_force = float(task_info.get("resource_constraint", {}).get("max_peak_force", 0.8))
        self.taps_done = 0
        per_tap = 0.048
        if self.task_type == "tool_use":
            self.taps_needed = int(_clamp(math.ceil(max(0.08, float(task_info.get("tool_goal",{}).get("progress",0)))*0.84/per_tap), 9, 24))
        elif self.task_type == "resource_sequence":
            self.taps_needed = int(_clamp(math.ceil(max(0.08, float(task_info.get("tool_goal",{}).get("progress",0)))*0.84/per_tap), 9, 22))
        else:
            self.taps_needed = 0

    def reset_batch(self, task_infos):
        self._batch_agents = [Agent() for _ in task_infos]
        for a in self._batch_agents:
            a.model = self.model
            a.device = self.device
        for a, t in zip(self._batch_agents, task_infos):
            a.reset(t)

    def act_batch(self, observations):
        if self._batch_agents is None:
            raise RuntimeError("reset_batch must be called before act_batch")
        return [self._batch_agents[int(o.get("batch_index", i))].act(o)
                for i, o in enumerate(observations)]

    def _obs_to_features(self, obs):
        state = obs["low_dim_state"]["values"]
        low_dim = np.array(state, dtype=np.float32)
        heatmap = np.array(obs.get("tactile_heatmap_7x4", [[0]*4]*7), dtype=np.float32).flatten()
        cs = obs.get("contact_summary", {})
        contact_feat = np.array([
            float(cs.get("coverage", 0)), float(cs.get("min_contact_so_far", 0)),
            float(cs.get("slip_risk", 0)), float(cs.get("damage_risk", 0)),
        ], dtype=np.float32)
        task_type_vec = np.zeros(3, dtype=np.float32)
        tt = self.task_info.get("task_type", "nonprehensile_relocation")
        if tt == "nonprehensile_relocation": task_type_vec[0] = 1.0
        elif tt == "tool_use": task_type_vec[1] = 1.0
        else: task_type_vec[2] = 1.0
        obj = self.task_info.get("object", {})
        task_feat = np.array([
            float(obj.get("fragility", 0.0)), float(obj.get("mass", 1.0)) / 2.0,
            float(obj.get("friction", 0.5)), float(self.task_info.get("tool_goal", {}).get("progress", 0.0)),
            1.0 if self.task_info.get("stage_goals") else 0.0,
            float(obj.get("size", 0.1)) * 5.0, float(obs.get("remaining_steps", 64)) / 128.0,
        ], dtype=np.float32)
        return np.concatenate([low_dim, heatmap, contact_feat, task_type_vec, task_feat])

    def _safety_check(self, obs):
        """Rule-based safety overrides."""
        state = obs["low_dim_state"]["values"]
        dx_f, dy_f, dtheta_f = float(state[3]), float(state[4]), float(state[5])
        contact = float(state[9])
        dist_f = _norm2(dx_f, dy_f)
        angle_f = abs(dtheta_f)
        sc = obs.get("stage_context", {})
        remaining = int(obs.get("remaining_steps", 64))

        # Finish
        stages_ok = not sc.get("enabled", False) or float(sc.get("completion_fraction", 0)) >= 0.999
        tool_ok = self.task_type == "nonprehensile_relocation" or self.taps_done >= self.taps_needed * 0.70
        if dist_f < 0.060 and angle_f < 0.22 and stages_ok and tool_ok:
            return {"primitive": "finish", "finger": "palm", "force": 0.0, "direction": [0.0, 0.0]}

        if remaining <= 1 and dist_f < 0.13 and angle_f < 0.55:
            return {"primitive": "finish", "finger": "palm", "force": 0.0, "direction": [0.0, 0.0]}

        # Contact emergency
        if contact < 0.05:
            return {"primitive": "stabilize", "finger": "palm", "force": 0.45, "direction": [0.0, 0.0]}

        return None

    def act(self, observation: dict) -> dict:
        # Safety check first
        override = self._safety_check(observation)
        if override:
            if override["primitive"] == "tap":
                self.taps_done += 1
            return override

        if self.model is None:
            # Fallback
            state = observation["low_dim_state"]["values"]
            nx, ny = _normalize(float(state[0]), float(state[1]))
            return {"primitive": "push", "finger": "palm", "force": 0.65, "direction": [nx, ny]}

        # Model inference
        with torch.no_grad():
            feat = self._obs_to_features(observation)
            x = torch.tensor(feat, dtype=torch.float32, device=self.device).unsqueeze(0)
            prim_logits, finger_logits, force_pred, dir_pred = self.model(x)

            prim_idx = prim_logits.argmax(-1).item()
            finger_idx = finger_logits.argmax(-1).item()
            force = force_pred.item()
            direction = dir_pred[0].cpu().numpy().tolist()

        # Safety clamp
        force = min(force, self.safe_force, self.max_peak_force + 0.05)
        force = _clamp(force, 0.0, 1.0)

        # Track taps
        primitive = ACTION_PRIMITIVES[prim_idx]
        if primitive == "tap":
            self.taps_done += 1

        return {
            "primitive": primitive,
            "finger": FINGERS[finger_idx],
            "force": force,
            "direction": direction,
        }
