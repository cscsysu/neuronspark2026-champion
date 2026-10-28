#!/usr/bin/env python3
"""
NS-2026-07: Hybrid GRU Training - ONE SCRIPT does everything.
Generates data → Trains GRU → Packages submission.

Run on server: CUDA_VISIBLE_DEVICES=2 python -u train_gru_hybrid.py

Total time: ~45-60 min on RTX 3090
"""
import json, math, os, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "simulator"))
from dexsim_core import DexSimEnv, sanitize_task_info, score_rollout, read_jsonl

# ================================================================
# Config
# ================================================================
OBS_DIM = 56
HIDDEN_DIM = 128
OUTPUT_DIM = 3  # force(1) + direction(2)
SEQ_LEN = 8  # GRU looks at last 8 steps

# ================================================================
# Rule Agent (v14 core - proven 321.9 on hidden)
# ================================================================
def _norm2(x, y): return math.sqrt(x*x + y*y)
def _normalize(x, y):
    n = _norm2(x, y)
    return (x/n, y/n) if n > 1e-9 else (0.0, 0.0)
def _clamp(v, lo, hi): return max(lo, min(hi, v))

class RuleAgent:
    """Simplified rule agent for data generation."""
    def reset(self, task_info):
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

    def _force(self, base, role, damage_risk, slip_risk=0.0):
        f = min(base, self.safe_palm if role != "pivot" else self.safe_thumb)
        if damage_risk > 0.15: f *= max(0.50, 1.0 - damage_risk * 0.55)
        if slip_risk > 0.35 and role != "tap": f *= max(0.55, 1.0 - (slip_risk-0.35)*0.55)
        f = min(f, self.max_peak_force + 0.06) * self.force_boost
        return _clamp(f, 0.05, 1.0)

    def _avoid(self, dx, dy, obs):
        pose = obs.get("object_pose_estimate", {})
        ox, oy = float(pose.get("x", 0)), float(pose.get("y", 0))
        ax, ay = dx, dy
        for o in self.obstacles:
            try:
                cx, cy, r = float(o["x"]), float(o["y"]), float(o.get("radius", 0.08))
            except: continue
            ddx, ddy = ox-cx, oy-cy
            d = _norm2(ddx, ddy)
            danger = r + max(0.025, self.obj_size*0.22) + 0.06
            if d < danger+0.16 and d > 1e-6:
                ndx, ndy = _normalize(ddx, ddy)
                scale = (0.9+(danger-d)*5) if d < danger else (0.4*(1-(d-danger)/0.16))
                ax += ndx*scale; ay += ndy*scale
        return _normalize(ax, ay)

    def act(self, obs):
        step = int(obs["step"])
        remaining = int(obs.get("remaining_steps", 128-step))
        state = obs["low_dim_state"]["values"]
        dx, dy, dtheta = float(state[0]), float(state[1]), float(state[2])
        dx_f, dy_f, dtheta_f = float(state[3]), float(state[4]), float(state[5])
        contact = float(state[9])
        slip_risk = float(state[10])
        damage_risk = float(obs.get("contact_summary", {}).get("damage_risk", 0))
        dist = _norm2(dx, dy); angle = abs(dtheta)
        dist_f = _norm2(dx_f, dy_f); angle_f = abs(dtheta_f)
        sc = obs.get("stage_context", {})
        stages_ok = not sc.get("enabled") or float(sc.get("completion_fraction", 0)) >= 0.999
        tool_ok = self.task_type == "nonprehensile_relocation" or self.taps_done >= self.taps_needed*0.72

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
        if self.phase == "tool":
            if self.taps_done >= self.taps_needed: self.phase = "move"
            else:
                if contact < 0.20:
                    return {"primitive": "brace", "finger": "palm", "force": 0.33, "direction": [0.0, 0.0]}
                self.taps_done += 1
                ax, ay = _normalize(float(self.tool_axis[0]), float(self.tool_axis[1]))
                finger = self.reserved[0] if (self.task_type == "resource_sequence" and self.reserved) else "index"
                return {"primitive": "tap", "finger": finger, "force": self._force(0.88, "tap", damage_risk), "direction": [ax, ay]}
        # Move phase taps
        if self.taps_done < self.taps_needed and step % 4 == 0 and remaining > 10 and contact > 0.20:
            self.taps_done += 1
            ax, ay = _normalize(float(self.tool_axis[0]), float(self.tool_axis[1]))
            finger = self.reserved[0] if (self.task_type == "resource_sequence" and self.reserved) else "index"
            return {"primitive": "tap", "finger": finger, "force": self._force(0.85, "tap", damage_risk), "direction": [ax, ay]}
        # Force adaptation
        self.progress_history.append(dist)
        if len(self.progress_history) > 10: self.progress_history.pop(0)
        if len(self.progress_history) >= 6 and self.progress_history[0] > 0.15:
            prog = self.progress_history[0] - self.progress_history[-1]
            if prog < 0.012*len(self.progress_history)*0.25 and prog >= -0.02:
                self.force_boost = min(1.50, self.force_boost + 0.06)
            elif prog > 0.012*len(self.progress_history)*0.5:
                self.force_boost = max(1.0, self.force_boost - 0.01)
        # Navigation
        if angle > 0.36 and dist < 0.22:
            sign = 1.0 if dtheta >= 0 else -1.0
            return {"primitive": "pivot", "finger": "thumb", "force": self._force(0.85, "pivot", damage_risk, slip_risk), "direction": [sign, 0.0]}
        if angle > 0.26 and dist > 0.12:
            nx, ny = self._avoid(*_normalize(dx, dy), obs)
            if (nx+ny >= 0) == (dtheta >= 0):
                return {"primitive": "roll", "finger": "palm", "force": self._force(0.74, "palm", damage_risk, slip_risk), "direction": [nx, ny]}
            else:
                sign = 1.0 if dtheta >= 0 else -1.0
                return {"primitive": "pivot", "finger": "thumb", "force": self._force(0.80, "pivot", damage_risk, slip_risk), "direction": [sign, 0.0]}
        if dist > 0.088:
            nx, ny = self._avoid(*_normalize(dx, dy), obs)
            base = 0.88 if self.mass > 1.3 else (0.82 if dist > 0.28 else (0.74 if dist > 0.15 else 0.65))
            return {"primitive": "push", "finger": "palm", "force": self._force(base, "palm", damage_risk, slip_risk), "direction": [nx, ny]}
        if angle > 0.12:
            sign = 1.0 if dtheta >= 0 else -1.0
            return {"primitive": "pivot", "finger": "thumb", "force": self._force(0.55, "pivot", damage_risk, slip_risk), "direction": [sign, 0.0]}
        tool_ok2 = self.task_type == "nonprehensile_relocation" or self.taps_done >= self.taps_needed*0.65
        if dist_f < 0.065 and angle_f < 0.26 and stages_ok and tool_ok2:
            return {"primitive": "finish", "finger": "palm", "force": 0.0, "direction": [0.0, 0.0]}
        nx, ny = _normalize(dx, dy) if dist > 0.02 else _normalize(dx_f, dy_f)
        nx, ny = self._avoid(nx, ny, obs)
        return {"primitive": "push", "finger": "palm", "force": self._force(0.42, "palm", damage_risk, slip_risk), "direction": [nx, ny]}


# ================================================================
# Feature extraction
# ================================================================
def obs_to_features(obs, task_info):
    state = obs["low_dim_state"]["values"]
    low_dim = np.array(state, dtype=np.float32)
    heatmap = np.array(obs.get("tactile_heatmap_7x4", [[0]*4]*7), dtype=np.float32).flatten()
    cs = obs.get("contact_summary", {})
    cf = np.array([float(cs.get("coverage",0)), float(cs.get("min_contact_so_far",0)),
                   float(cs.get("slip_risk",0)), float(cs.get("damage_risk",0))], dtype=np.float32)
    tv = np.zeros(3, dtype=np.float32)
    tt = task_info.get("task_type", "nonprehensile_relocation")
    if tt == "nonprehensile_relocation": tv[0] = 1
    elif tt == "tool_use": tv[1] = 1
    else: tv[2] = 1
    obj = task_info.get("object", {})
    tf = np.array([float(obj.get("fragility",0)), float(obj.get("mass",1))/2,
                   float(obj.get("friction",0.5)), float(task_info.get("tool_goal",{}).get("progress",0)),
                   1.0 if task_info.get("stage_goals") else 0, float(obj.get("size",0.1))*5,
                   float(obs.get("remaining_steps",64))/128], dtype=np.float32)
    return np.concatenate([low_dim, heatmap, cf, tv, tf])


def random_extreme_variant(rng):
    return {
        'mass_scale': rng.uniform(0.5, 2.0), 'friction_scale': rng.uniform(0.3, 1.8),
        'contact_bias': rng.uniform(-0.20, 0.15), 'observation_noise': rng.uniform(0.005, 0.10),
        'actuator_scale': rng.uniform(0.5, 1.5), 'friction_anisotropy': rng.uniform(-0.4, 0.4),
        'surface_slope_x': rng.uniform(-0.03, 0.03), 'surface_slope_y': rng.uniform(-0.03, 0.03),
        'pose_dropout_rate': rng.uniform(0.0, 0.5), 'tactile_dropout_rate': rng.uniform(0.0, 0.5),
        'action_delay_steps': rng.choice([0,0,0,1,1,2,2,3,4]),
        'contact_stiffness': rng.uniform(0.6, 1.6), 'contact_damping': rng.uniform(0.5, 1.5),
        'pose_bias_x': rng.uniform(-0.04, 0.04), 'pose_bias_y': rng.uniform(-0.04, 0.04),
    }


# ================================================================
# Phase 1: Generate data
# ================================================================
def generate_data(tasks, n_variants=5, max_tasks=6000):
    rng = random.Random(42)
    agent = RuleAgent()
    tasks = tasks[:max_tasks]

    # Store sequences for GRU training
    all_sequences = []  # list of episode sequences
    success_count = 0
    total_count = 0

    t0 = time.time()
    for ti, task in enumerate(tasks):
        task_info = sanitize_task_info(task)
        for vi in range(n_variants):
            variant = random_extreme_variant(rng)
            env = DexSimEnv(task, variant)
            agent.reset(task_info)

            episode_obs = []
            episode_force = []
            episode_dir = []

            while not env.done and env.step_count < env.max_steps:
                obs = env.observation()
                action = agent.act(obs)
                feat = obs_to_features(obs, task_info)
                episode_obs.append(feat)
                episode_force.append(float(action["force"]))
                episode_dir.append([float(action["direction"][0]), float(action["direction"][1])])
                env.step(action)

            state = env.state()
            metrics = score_rollout(task, state)
            total_count += 1

            # Keep ALL episodes but weight successes more
            if metrics.get("success") or float(metrics["score01"]) > 0.15:
                all_sequences.append({
                    "obs": np.array(episode_obs, dtype=np.float32),
                    "force": np.array(episode_force, dtype=np.float32),
                    "direction": np.array(episode_dir, dtype=np.float32),
                    "success": bool(metrics.get("success")),
                    "score": float(metrics["score01"]),
                })
                if metrics.get("success"):
                    success_count += 1

        if (ti+1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"  [{ti+1}/{len(tasks)}] episodes={total_count} kept={len(all_sequences)} "
                  f"success={success_count} time={elapsed:.0f}s", flush=True)

    elapsed = time.time() - t0
    print(f"Data generation: {total_count} episodes, {len(all_sequences)} kept, "
          f"{success_count} successes, {elapsed:.0f}s", flush=True)
    return all_sequences


# ================================================================
# Phase 2: Train GRU
# ================================================================
class ForceDirectionGRU(nn.Module):
    """GRU that predicts force and direction from observation sequence."""
    def __init__(self, obs_dim=OBS_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.1)
        self.force_head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()
        )
        self.dir_head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 2), nn.Tanh()
        )

    def forward(self, x, hidden=None):
        # x: (batch, seq_len, obs_dim)
        encoded = self.encoder(x)
        gru_out, hidden_new = self.gru(encoded, hidden)
        last = gru_out[:, -1, :]  # (batch, hidden)
        force = self.force_head(last)
        direction = self.dir_head(last)
        return force, direction, hidden_new


def prepare_training_data(sequences, seq_len=SEQ_LEN):
    """Convert episodes into fixed-length sequences for GRU training."""
    all_x = []  # (N, seq_len, obs_dim)
    all_force = []  # (N, 1)
    all_dir = []  # (N, 2)
    all_weight = []  # (N,) success episodes weighted higher

    for ep in sequences:
        obs = ep["obs"]
        force = ep["force"]
        direction = ep["direction"]
        weight = 2.0 if ep["success"] else 1.0

        # Create overlapping sequences
        for i in range(len(obs)):
            start = max(0, i - seq_len + 1)
            seq = obs[start:i+1]
            # Pad if shorter than seq_len
            if len(seq) < seq_len:
                pad = np.zeros((seq_len - len(seq), OBS_DIM), dtype=np.float32)
                seq = np.concatenate([pad, seq], axis=0)
            all_x.append(seq)
            all_force.append(force[i])
            all_dir.append(direction[i])
            all_weight.append(weight)

    return (np.array(all_x, dtype=np.float32),
            np.array(all_force, dtype=np.float32),
            np.array(all_dir, dtype=np.float32),
            np.array(all_weight, dtype=np.float32))


def train_model(sequences, epochs=50, batch_size=1024, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}", flush=True)

    # Prepare data
    print("Preparing sequences...", flush=True)
    X, Force, Dir, Weights = prepare_training_data(sequences)
    n = len(X)
    print(f"Training samples: {n}", flush=True)

    X_t = torch.tensor(X, device=device)
    F_t = torch.tensor(Force, device=device).unsqueeze(1)
    D_t = torch.tensor(Dir, device=device)
    W_t = torch.tensor(Weights, device=device)

    model = ForceDirectionGRU().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    t0 = time.time()
    for epoch in range(epochs):
        indices = torch.randperm(n, device=device)
        total_loss = 0; nb = 0

        for start in range(0, n, batch_size):
            idx = indices[start:start+batch_size]
            x_b = X_t[idx]
            f_b = F_t[idx]
            d_b = D_t[idx]
            w_b = W_t[idx]

            # Add noise augmentation to inputs
            if epoch < epochs * 0.8:
                noise = torch.randn_like(x_b) * 0.02
                x_b = x_b + noise

            force_pred, dir_pred, _ = model(x_b)

            loss_f = (w_b * (force_pred - f_b).pow(2).squeeze()).mean()
            loss_d = (w_b.unsqueeze(1) * (dir_pred - d_b).pow(2)).mean()
            loss = 2.0 * loss_f + 2.0 * loss_d

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item(); nb += 1

        scheduler.step()
        if (epoch+1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: loss={total_loss/nb:.4f} time={time.time()-t0:.0f}s", flush=True)

    print(f"Training complete in {time.time()-t0:.0f}s", flush=True)
    return model


# ================================================================
# Main
# ================================================================
def main():
    print("="*60, flush=True)
    print("NS-2026-07: GRU Hybrid Training", flush=True)
    print("="*60, flush=True)

    tasks = read_jsonl(str(ROOT / "tasks" / "train_tasks.jsonl"))
    print(f"Loaded {len(tasks)} tasks", flush=True)

    # Phase 1
    print("\n--- Phase 1: Data Generation ---", flush=True)
    sequences = generate_data(tasks, n_variants=5, max_tasks=6000)

    # Phase 2
    print("\n--- Phase 2: GRU Training ---", flush=True)
    model = train_model(sequences, epochs=50, batch_size=1024)

    # Phase 3: Save
    output_dir = ROOT / "gru_submission" / "model"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "gru_policy.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {"obs_dim": OBS_DIM, "hidden_dim": HIDDEN_DIM, "seq_len": SEQ_LEN},
    }, save_path)
    print(f"\nModel saved: {save_path} ({save_path.stat().st_size//1024} KB)", flush=True)
    print("Done! Now copy gru_submission/ to submission directory.", flush=True)


if __name__ == "__main__":
    main()
