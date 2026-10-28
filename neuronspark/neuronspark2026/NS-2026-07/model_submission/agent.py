"""NS-2026-07 Model Agent with Rule Safety Wrapper.

Loads a trained GRU policy network and uses it for closed-loop control.
Falls back to rule-based safety for critical decisions (finish, emergency brace).
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ACTION_PRIMITIVES = ["brace", "push", "drag", "pivot", "roll", "lift_edge", "tap", "stabilize", "wait", "finish"]
FINGERS = ["thumb", "index", "middle", "ring", "pinky", "palm", "wrist"]
OBS_DIM = 56
HIDDEN_DIM = 256
NUM_PRIMITIVES = 10
NUM_FINGERS = 7


class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.primitive_head = nn.Sequential(nn.Linear(hidden_dim, 128), nn.ReLU(), nn.Linear(128, NUM_PRIMITIVES))
        self.finger_head = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, NUM_FINGERS))
        self.force_head = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 2))
        self.direction_head = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 4))
        self.value_head = nn.Sequential(nn.Linear(hidden_dim, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, obs, hidden=None):
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
        batch_size = obs.shape[0]
        if hidden is None:
            hidden = torch.zeros(1, batch_size, self.hidden_dim, device=obs.device)
        encoded = self.encoder(obs)
        gru_out, hidden_new = self.gru(encoded, hidden)
        features = gru_out[:, -1, :]
        prim_logits = self.primitive_head(features)
        finger_logits = self.finger_head(features)
        force_params = F.softplus(self.force_head(features)) + 1.0
        dir_params = self.direction_head(features)
        return prim_logits, finger_logits, force_params, dir_params, hidden_new


def _norm2(x, y):
    return math.sqrt(x * x + y * y)

def _normalize(x, y):
    n = _norm2(x, y)
    if n < 1e-9:
        return 0.0, 0.0
    return x / n, y / n

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


class Agent:
    """Model-based agent with rule safety wrapper."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = Path(__file__).parent / "model" / "policy.pt"
        self.model = None
        if model_path.exists():
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            config = checkpoint.get("config", {"obs_dim": OBS_DIM, "hidden_dim": HIDDEN_DIM})
            self.model = PolicyNetwork(config["obs_dim"], config["hidden_dim"]).to(self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()
        self.hidden = None
        self.task_info = None
        self._batch_agents = None

    def reset(self, task_info: dict) -> None:
        self.task_info = task_info
        self.hidden = None
        self.task_type = task_info.get("task_type", "nonprehensile_relocation")
        obj = task_info.get("object", {})
        self.fragility = float(obj.get("fragility", 0.0))
        self.mass = float(obj.get("mass", 1.0))
        self.dmg_thresh = 1.05 - 0.72 * self.fragility
        self.safe_force = min(0.92, self.dmg_thresh / 1.15)
        self.max_peak_force = float(task_info.get("resource_constraint", {}).get("max_peak_force", 0.8))

    def reset_batch(self, task_infos: list[dict]) -> None:
        self._batch_agents = [Agent.__new__(Agent) for _ in task_infos]
        for a, ti in zip(self._batch_agents, task_infos):
            a.__init__()
            a.model = self.model
            a.device = self.device
            a.reset(ti)

    def act_batch(self, observations: list[dict]) -> list[dict]:
        if self._batch_agents is None:
            raise RuntimeError("reset_batch must be called before act_batch")
        return [self._batch_agents[int(o.get("batch_index", i))].act(o)
                for i, o in enumerate(observations)]

    def _obs_to_features(self, obs: dict) -> np.ndarray:
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

    def _safety_override(self, obs: dict) -> dict | None:
        """Rule-based safety checks. Returns action if override needed, None otherwise."""
        state = obs["low_dim_state"]["values"]
        dx_f, dy_f, dtheta_f = float(state[3]), float(state[4]), float(state[5])
        contact = float(state[9])
        dist_f = _norm2(dx_f, dy_f)
        angle_f = abs(dtheta_f)
        sc = obs.get("stage_context", {})
        remaining = int(obs.get("remaining_steps", 64))

        # Finish check
        stages_ok = not sc.get("enabled", False) or float(sc.get("completion_fraction", 0)) >= 0.999
        if dist_f < 0.060 and angle_f < 0.22 and stages_ok:
            return {"primitive": "finish", "finger": "palm", "force": 0.0, "direction": [0.0, 0.0]}

        # Last step
        if remaining <= 1:
            if dist_f < 0.13 and angle_f < 0.55:
                return {"primitive": "finish", "finger": "palm", "force": 0.0, "direction": [0.0, 0.0]}

        # Contact emergency
        if contact < 0.05:
            return {"primitive": "stabilize", "finger": "palm", "force": 0.45, "direction": [0.0, 0.0]}

        return None

    def act(self, observation: dict) -> dict:
        # Safety override first
        override = self._safety_override(observation)
        if override is not None:
            return override

        # If no model, fallback to simple push
        if self.model is None:
            return self._fallback_act(observation)

        # Model inference
        with torch.no_grad():
            obs_feat = self._obs_to_features(observation)
            obs_tensor = torch.tensor(obs_feat, dtype=torch.float32, device=self.device).unsqueeze(0)
            prim_logits, finger_logits, force_params, dir_params, hidden_new = self.model(obs_tensor, self.hidden)
            self.hidden = hidden_new

            # Decode action (deterministic at test time)
            prim_idx = prim_logits.argmax(-1).item()
            finger_idx = finger_logits.argmax(-1).item()
            alpha = force_params[0, 0].item()
            beta_param = force_params[0, 1].item()
            force = alpha / (alpha + beta_param)
            direction = torch.tanh(dir_params[0, :2]).cpu().numpy()

        # Safety clamp force
        force = min(force, self.safe_force, self.max_peak_force + 0.05)
        force = max(0.0, min(1.0, force))

        # Clamp direction
        dx, dy = float(direction[0]), float(direction[1])
        n = _norm2(dx, dy)
        if n > 1.0:
            dx, dy = dx / n, dy / n

        return {
            "primitive": ACTION_PRIMITIVES[prim_idx],
            "finger": FINGERS[finger_idx],
            "force": force,
            "direction": [dx, dy],
        }

    def _fallback_act(self, obs: dict) -> dict:
        """Simple fallback when model not loaded."""
        state = obs["low_dim_state"]["values"]
        dx, dy = float(state[0]), float(state[1])
        contact = float(state[9])
        if contact < 0.15:
            return {"primitive": "brace", "finger": "palm", "force": 0.35, "direction": [0.0, 0.0]}
        nx, ny = _normalize(dx, dy)
        return {"primitive": "push", "finger": "palm", "force": 0.65, "direction": [nx, ny]}
