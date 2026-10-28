"""NS-2026-07: Hybrid Agent - Rule primitives + GRU force/direction.

The GRU model predicts optimal force and direction by learning from
successful trajectories under extreme domain randomization.
Rules handle discrete decisions (primitive, finger, finish, brace).
"""
from __future__ import annotations
import math, os
from pathlib import Path
from collections import deque

import numpy as np
import torch
import torch.nn as nn

def _norm2(x, y): return math.sqrt(x*x + y*y)
def _normalize(x, y):
    n = _norm2(x, y)
    return (x/n, y/n) if n > 1e-9 else (0.0, 0.0)
def _clamp(v, lo, hi): return max(lo, min(hi, v))

OBS_DIM = 56; HIDDEN_DIM = 128; SEQ_LEN = 8

class ForceDirectionGRU(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.1)
        self.force_head = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
        self.dir_head = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 2), nn.Tanh())
    def forward(self, x, hidden=None):
        encoded = self.encoder(x)
        gru_out, hidden_new = self.gru(encoded, hidden)
        last = gru_out[:, -1, :]
        return self.force_head(last), self.dir_head(last), hidden_new


class Agent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = Path(__file__).parent / "model" / "gru_policy.pt"
        self.model = None
        if model_path.exists():
            ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
            cfg = ckpt.get("config", {})
            self.model = ForceDirectionGRU(cfg.get("obs_dim", OBS_DIM), cfg.get("hidden_dim", HIDDEN_DIM)).to(self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.model.eval()
        self._batch_agents = None

    def reset(self, task_info):
        self.task_info = task_info
        self.task_type = task_info.get("task_type", "nonprehensile_relocation")
        self.tool_axis = task_info.get("tool_goal", {}).get("axis", [1.0, 0.0])
        self.tool_need = float(task_info.get("tool_goal", {}).get("progress", 0.0))
        self.reserved = task_info.get("resource_constraint", {}).get("reserve_fingers", [])
        self.max_peak_force = float(task_info.get("resource_constraint", {}).get("max_peak_force", 0.8))
        self.obstacles = task_info.get("scene_context", {}).get("obstacles", []) if isinstance(task_info.get("scene_context"), dict) else []
        self.corridor = task_info.get("scene_context", {}).get("corridor", {}) if isinstance(task_info.get("scene_context"), dict) else {}
        obj = task_info.get("object", {})
        self.fragility = float(obj.get("fragility", 0.0))
        self.mass = float(obj.get("mass", 1.0))
        self.obj_size = float(obj.get("size", 0.1))
        self.dmg_thresh = 1.05 - 0.72 * self.fragility
        self.safe_palm = min(0.92, self.dmg_thresh / 1.15)
        self.safe_thumb = min(0.92, self.dmg_thresh / 1.05)
        self.taps_done = 0
        per_tap = 0.048
        if self.task_type == "tool_use":
            self.taps_needed = int(_clamp(math.ceil(max(0.08, self.tool_need)*0.84/per_tap), 9, 24))
        elif self.task_type == "resource_sequence":
            self.taps_needed = int(_clamp(math.ceil(max(0.08, self.tool_need)*0.84/per_tap), 9, 22))
        else:
            self.taps_needed = 0
        self.phase = "brace"
        self.brace_done = 0
        self.force_boost = 1.0
        self.progress_history = []
        # GRU state
        self.obs_buffer = deque(maxlen=SEQ_LEN)
        self.gru_hidden = None

    def reset_batch(self, task_infos):
        self._batch_agents = [Agent() for _ in task_infos]
        for a in self._batch_agents:
            a.model = self.model; a.device = self.device
        for a, t in zip(self._batch_agents, task_infos):
            a.reset(t)

    def act_batch(self, observations):
        if self._batch_agents is None:
            raise RuntimeError("reset_batch must be called before act_batch")
        return [self._batch_agents[int(o.get("batch_index", i))].act(o) for i, o in enumerate(observations)]

    def _obs_to_features(self, obs):
        state = obs["low_dim_state"]["values"]
        low_dim = np.array(state, dtype=np.float32)
        heatmap = np.array(obs.get("tactile_heatmap_7x4", [[0]*4]*7), dtype=np.float32).flatten()
        cs = obs.get("contact_summary", {})
        cf = np.array([float(cs.get("coverage",0)), float(cs.get("min_contact_so_far",0)),
                       float(cs.get("slip_risk",0)), float(cs.get("damage_risk",0))], dtype=np.float32)
        tv = np.zeros(3, dtype=np.float32)
        tt = self.task_info.get("task_type", "nonprehensile_relocation")
        if tt == "nonprehensile_relocation": tv[0] = 1
        elif tt == "tool_use": tv[1] = 1
        else: tv[2] = 1
        obj = self.task_info.get("object", {})
        tf = np.array([float(obj.get("fragility",0)), float(obj.get("mass",1))/2,
                       float(obj.get("friction",0.5)), float(self.task_info.get("tool_goal",{}).get("progress",0)),
                       1.0 if self.task_info.get("stage_goals") else 0, float(obj.get("size",0.1))*5,
                       float(obs.get("remaining_steps",64))/128], dtype=np.float32)
        return np.concatenate([low_dim, heatmap, cf, tv, tf])

    def _model_predict(self, obs):
        """Get force and direction from GRU model."""
        feat = self._obs_to_features(obs)
        self.obs_buffer.append(feat)
        # Build sequence
        seq = list(self.obs_buffer)
        while len(seq) < SEQ_LEN:
            seq.insert(0, np.zeros(OBS_DIM, dtype=np.float32))
        seq_np = np.array(seq, dtype=np.float32)
        with torch.no_grad():
            x = torch.tensor(seq_np, dtype=torch.float32, device=self.device).unsqueeze(0)
            force_pred, dir_pred, self.gru_hidden = self.model(x, self.gru_hidden)
        return float(force_pred.item()), dir_pred[0].cpu().numpy().tolist()

    def _avoid(self, dx, dy, obs):
        pose = obs.get("object_pose_estimate", {})
        ox, oy = float(pose.get("x", 0)), float(pose.get("y", 0))
        ax, ay = dx, dy
        for o in self.obstacles:
            try: cx, cy, r = float(o["x"]), float(o["y"]), float(o.get("radius", 0.08))
            except: continue
            ddx, ddy = ox-cx, oy-cy
            d = _norm2(ddx, ddy)
            danger = r + max(0.025, self.obj_size*0.22) + 0.06
            if d < danger+0.16 and d > 1e-6:
                ndx, ndy = _normalize(ddx, ddy)
                scale = (0.9+(danger-d)*5) if d < danger else (0.4*(1-(d-danger)/0.16))
                ax += ndx*scale; ay += ndy*scale
        return _normalize(ax, ay)

    def act(self, observation):
        state = observation["low_dim_state"]["values"]
        dx, dy, dtheta = float(state[0]), float(state[1]), float(state[2])
        dx_f, dy_f, dtheta_f = float(state[3]), float(state[4]), float(state[5])
        contact = float(state[9]); slip_risk = float(state[10])
        damage_risk = float(observation.get("contact_summary", {}).get("damage_risk", 0))
        dist = _norm2(dx, dy); angle = abs(dtheta)
        dist_f = _norm2(dx_f, dy_f); angle_f = abs(dtheta_f)
        remaining = int(observation.get("remaining_steps", 128))
        sc = observation.get("stage_context", {})
        stages_ok = not sc.get("enabled") or float(sc.get("completion_fraction", 0)) >= 0.999
        tool_ok = self.task_type == "nonprehensile_relocation" or self.taps_done >= self.taps_needed*0.72

        # ===== SAFETY RULES (always override model) =====
        if dist_f < 0.062 and angle_f < 0.24 and stages_ok and tool_ok:
            return {"primitive": "finish", "finger": "palm", "force": 0.0, "direction": [0.0, 0.0]}
        if remaining <= 1 and dist_f < 0.13:
            return {"primitive": "finish", "finger": "palm", "force": 0.0, "direction": [0.0, 0.0]}
        if contact < 0.06:
            return {"primitive": "stabilize", "finger": "palm", "force": 0.60, "direction": [0.0, 0.0]}
        if self.phase == "brace":
            self.brace_done += 1
            if (self.brace_done >= 1 and contact >= 0.22) or self.brace_done >= 2:
                self.phase = "tool" if self.taps_needed > 0 else "move"
            return {"primitive": "brace", "finger": "palm", "force": 0.38, "direction": [0.0, 0.0]}
        if contact < 0.13 or (slip_risk > 0.86 and contact < 0.26):
            return {"primitive": "brace", "finger": "palm", "force": 0.36, "direction": [0.0, 0.0]}

        # ===== TOOL PHASE (rules) =====
        if self.phase == "tool":
            if self.taps_done >= self.taps_needed: self.phase = "move"
            else:
                if contact < 0.20:
                    return {"primitive": "brace", "finger": "palm", "force": 0.33, "direction": [0.0, 0.0]}
                self.taps_done += 1
                ax, ay = _normalize(float(self.tool_axis[0]), float(self.tool_axis[1]))
                finger = self.reserved[0] if (self.task_type == "resource_sequence" and self.reserved) else "index"
                return {"primitive": "tap", "finger": finger, "force": _clamp(0.88, 0, self.safe_palm), "direction": [ax, ay]}

        # Interleaved taps
        step = int(observation["step"])
        if self.taps_done < self.taps_needed and step % 4 == 0 and remaining > 10 and contact > 0.20:
            self.taps_done += 1
            ax, ay = _normalize(float(self.tool_axis[0]), float(self.tool_axis[1]))
            finger = self.reserved[0] if (self.task_type == "resource_sequence" and self.reserved) else "index"
            return {"primitive": "tap", "finger": finger, "force": _clamp(0.85, 0, self.safe_palm), "direction": [ax, ay]}

        # ===== MOVE PHASE: Use model for force/direction =====
        # Determine primitive using rules
        if angle > 0.36 and dist < 0.22:
            primitive = "pivot"; finger = "thumb"
        elif angle > 0.26 and dist > 0.12:
            nx, ny = self._avoid(*_normalize(dx, dy), observation)
            if (nx+ny >= 0) == (dtheta >= 0):
                primitive = "roll"; finger = "palm"
            else:
                primitive = "pivot"; finger = "thumb"
        elif dist > 0.088:
            primitive = "push"; finger = "palm"
        elif angle > 0.12:
            primitive = "pivot"; finger = "thumb"
        else:
            # Near target
            tool_ok2 = self.task_type == "nonprehensile_relocation" or self.taps_done >= self.taps_needed*0.65
            if dist_f < 0.065 and angle_f < 0.26 and stages_ok and tool_ok2:
                return {"primitive": "finish", "finger": "palm", "force": 0.0, "direction": [0.0, 0.0]}
            primitive = "push"; finger = "palm"

        # Get force and direction from MODEL (or fallback to rules)
        if self.model is not None:
            model_force, model_dir = self._model_predict(observation)
            # Safety clamp
            safe_limit = self.safe_palm if finger == "palm" else self.safe_thumb
            force = _clamp(model_force, 0.05, min(safe_limit, self.max_peak_force + 0.05))
            # Blend model direction with rule direction for safety
            rule_dx, rule_dy = _normalize(dx, dy) if dist > 0.02 else _normalize(dx_f, dy_f)
            rule_dx, rule_dy = self._avoid(rule_dx, rule_dy, observation)
            # 70% model, 30% rule (model leads, rule provides safety guardrail)
            direction = [0.7 * model_dir[0] + 0.3 * rule_dx, 0.7 * model_dir[1] + 0.3 * rule_dy]
            # Normalize
            dn = _norm2(direction[0], direction[1])
            if dn > 1e-6: direction = [direction[0]/dn, direction[1]/dn]
        else:
            # Fallback: pure rule
            rule_dx, rule_dy = _normalize(dx, dy) if dist > 0.02 else _normalize(dx_f, dy_f)
            rule_dx, rule_dy = self._avoid(rule_dx, rule_dy, observation)
            direction = [rule_dx, rule_dy]
            base = 0.88 if self.mass > 1.3 else (0.82 if dist > 0.28 else (0.74 if dist > 0.15 else 0.65))
            force = _clamp(min(base, self.safe_palm, self.max_peak_force+0.06), 0.05, 1.0)

        if primitive == "pivot":
            sign = 1.0 if dtheta >= 0 else -1.0
            direction = [sign, 0.0]

        return {"primitive": primitive, "finger": finger, "force": force, "direction": direction}
