#!/usr/bin/env python3
"""
NS-2026-07: Full-observation CNN+GRU training.
Uses tactile_image(7,8,8) + vision_grid(6,16,16) + low_dim(56).

This is what the 408-point competitor likely uses: rich spatial features.

Run: CUDA_VISIBLE_DEVICES=2 python -u train_full_cnn_gru.py
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
# Constants
# ================================================================
LOW_DIM = 56
TACTILE_SHAPE = (7, 8, 8)  # 448 values
VISION_SHAPE = (6, 16, 16)  # 1536 values
HIDDEN_DIM = 160
SEQ_LEN = 4

# ================================================================
# Rule Agent (same as before, for data generation)
# ================================================================
def _norm2(x, y): return math.sqrt(x*x + y*y)
def _normalize(x, y):
    n = _norm2(x, y)
    return (x/n, y/n) if n > 1e-9 else (0.0, 0.0)
def _clamp(v, lo, hi): return max(lo, min(hi, v))

class RuleAgent:
    """Simplified rule agent for data generation (same as train_gru_hybrid.py)."""
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
        self.phase = "brace"; self.brace_done = 0; self.force_boost = 1.0; self.progress_history = []

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
            try: cx, cy, r = float(o["x"]), float(o["y"]), float(o.get("radius", 0.08))
            except: continue
            ddx, ddy = ox-cx, oy-cy; d = _norm2(ddx, ddy)
            danger = r + max(0.025, self.obj_size*0.22) + 0.06
            if d < danger+0.16 and d > 1e-6:
                ndx, ndy = _normalize(ddx, ddy)
                scale = (0.9+(danger-d)*5) if d < danger else (0.4*(1-(d-danger)/0.16))
                ax += ndx*scale; ay += ndy*scale
        return _normalize(ax, ay)

    def act(self, obs):
        step = int(obs["step"]); remaining = int(obs.get("remaining_steps", 128-step))
        state = obs["low_dim_state"]["values"]
        dx, dy, dtheta = float(state[0]), float(state[1]), float(state[2])
        dx_f, dy_f, dtheta_f = float(state[3]), float(state[4]), float(state[5])
        contact = float(state[9]); slip_risk = float(state[10])
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
                if contact < 0.20: return {"primitive": "brace", "finger": "palm", "force": 0.33, "direction": [0.0, 0.0]}
                self.taps_done += 1
                ax, ay = _normalize(float(self.tool_axis[0]), float(self.tool_axis[1]))
                finger = self.reserved[0] if (self.task_type == "resource_sequence" and self.reserved) else "index"
                return {"primitive": "tap", "finger": finger, "force": self._force(0.88, "tap", damage_risk), "direction": [ax, ay]}
        if self.taps_done < self.taps_needed and step % 4 == 0 and remaining > 10 and contact > 0.20:
            self.taps_done += 1
            ax, ay = _normalize(float(self.tool_axis[0]), float(self.tool_axis[1]))
            finger = self.reserved[0] if (self.task_type == "resource_sequence" and self.reserved) else "index"
            return {"primitive": "tap", "finger": finger, "force": self._force(0.85, "tap", damage_risk), "direction": [ax, ay]}
        self.progress_history.append(dist)
        if len(self.progress_history) > 10: self.progress_history.pop(0)
        if len(self.progress_history) >= 6 and self.progress_history[0] > 0.15:
            prog = self.progress_history[0] - self.progress_history[-1]
            if prog < 0.012*len(self.progress_history)*0.25 and prog >= -0.02:
                self.force_boost = min(1.50, self.force_boost + 0.06)
            elif prog > 0.012*len(self.progress_history)*0.5:
                self.force_boost = max(1.0, self.force_boost - 0.01)
        if angle > 0.36 and dist < 0.22:
            sign = 1.0 if dtheta >= 0 else -1.0
            return {"primitive": "pivot", "finger": "thumb", "force": self._force(0.85, "pivot", damage_risk, slip_risk), "direction": [sign, 0.0]}
        if angle > 0.26 and dist > 0.12:
            nx, ny = self._avoid(*_normalize(dx, dy), obs)
            if (nx+ny >= 0) == (dtheta >= 0):
                return {"primitive": "roll", "finger": "palm", "force": self._force(0.74, "palm", damage_risk, slip_risk), "direction": [nx, ny]}
            else:
                return {"primitive": "pivot", "finger": "thumb", "force": self._force(0.80, "pivot", damage_risk, slip_risk), "direction": [1.0 if dtheta>=0 else -1.0, 0.0]}
        if dist > 0.088:
            nx, ny = self._avoid(*_normalize(dx, dy), obs)
            base = 0.88 if self.mass > 1.3 else (0.82 if dist > 0.28 else (0.74 if dist > 0.15 else 0.65))
            return {"primitive": "push", "finger": "palm", "force": self._force(base, "palm", damage_risk, slip_risk), "direction": [nx, ny]}
        if angle > 0.12:
            return {"primitive": "pivot", "finger": "thumb", "force": self._force(0.55, "pivot", damage_risk, slip_risk), "direction": [1.0 if dtheta>=0 else -1.0, 0.0]}
        tool_ok2 = self.task_type == "nonprehensile_relocation" or self.taps_done >= self.taps_needed*0.65
        if dist_f < 0.065 and angle_f < 0.26 and stages_ok and tool_ok2:
            return {"primitive": "finish", "finger": "palm", "force": 0.0, "direction": [0.0, 0.0]}
        nx, ny = _normalize(dx, dy) if dist > 0.02 else _normalize(dx_f, dy_f)
        nx, ny = self._avoid(nx, ny, obs)
        return {"primitive": "push", "finger": "palm", "force": self._force(0.42, "palm", damage_risk, slip_risk), "direction": [nx, ny]}


# ================================================================
# Feature extraction (FULL observations)
# ================================================================
def extract_full_obs(obs, task_info):
    """Extract all observation channels."""
    # Low-dim (56)
    state = obs["low_dim_state"]["values"]
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
    low_dim = np.concatenate([np.array(state, dtype=np.float32), heatmap, cf, tv, tf])

    # Tactile image (7, 8, 8) normalized to [0,1]
    ti_data = obs.get("tactile_image_7x8x8", {}).get("values", [0]*448)
    tactile_img = np.array(ti_data, dtype=np.float32).reshape(7, 8, 8) / 255.0

    # Vision grid (6, 16, 16) normalized to [0,1]
    vg_data = obs.get("vision_grid_16x16", {}).get("values", [0]*1536)
    vision_grid = np.array(vg_data, dtype=np.float32).reshape(6, 16, 16) / 255.0

    return low_dim, tactile_img, vision_grid


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
# CNN + GRU Model
# ================================================================
class FullObsCNNGRU(nn.Module):
    """CNN encoders for spatial data + GRU for temporal reasoning."""
    def __init__(self):
        super().__init__()
        # Vision encoder: (6, 16, 16) → 64
        self.vision_cnn = nn.Sequential(
            nn.Conv2d(6, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),  # → (32, 8, 8)
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),  # → (32, 4, 4)
            nn.Flatten(), nn.Linear(32*4*4, 64), nn.ReLU(),
        )
        # Tactile encoder: (7, 8, 8) → 32
        self.tactile_cnn = nn.Sequential(
            nn.Conv2d(7, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=2, padding=1), nn.ReLU(),  # → (16, 4, 4)
            nn.Flatten(), nn.Linear(16*4*4, 32), nn.ReLU(),
        )
        # Low-dim encoder: 56 → 64
        self.low_dim_enc = nn.Sequential(nn.Linear(56, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())
        # Fused: 64+32+64 = 160
        self.gru = nn.GRU(160, HIDDEN_DIM, batch_first=True, num_layers=2, dropout=0.1)
        # Output heads
        self.force_head = nn.Sequential(nn.Linear(HIDDEN_DIM, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
        self.dir_head = nn.Sequential(nn.Linear(HIDDEN_DIM, 64), nn.ReLU(), nn.Linear(64, 2), nn.Tanh())

    def forward(self, low_dim, tactile, vision, hidden=None):
        """
        low_dim: (batch, seq, 56)
        tactile: (batch, seq, 7, 8, 8)
        vision: (batch, seq, 6, 16, 16)
        """
        B, S = low_dim.shape[0], low_dim.shape[1]
        # Encode each timestep
        ld_enc = self.low_dim_enc(low_dim.reshape(B*S, -1)).reshape(B, S, 64)
        t_enc = self.tactile_cnn(tactile.reshape(B*S, 7, 8, 8)).reshape(B, S, 32)
        v_enc = self.vision_cnn(vision.reshape(B*S, 6, 16, 16)).reshape(B, S, 64)
        # Fuse
        fused = torch.cat([ld_enc, t_enc, v_enc], dim=-1)  # (B, S, 160)
        # GRU
        gru_out, hidden_new = self.gru(fused, hidden)
        last = gru_out[:, -1, :]
        force = self.force_head(last)
        direction = self.dir_head(last)
        return force, direction, hidden_new


# ================================================================
# Data Generation
# ================================================================
def generate_data(tasks, n_variants=4, max_tasks=2000):
    rng = random.Random(42)
    agent = RuleAgent()
    tasks = tasks[:max_tasks]
    all_episodes = []
    success_count = 0; total_count = 0
    t0 = time.time()

    for ti, task in enumerate(tasks):
        task_info = sanitize_task_info(task)
        for vi in range(n_variants):
            variant = random_extreme_variant(rng)
            env = DexSimEnv(task, variant)
            agent.reset(task_info)
            ep_low, ep_tac, ep_vis, ep_force, ep_dir = [], [], [], [], []
            while not env.done and env.step_count < env.max_steps:
                obs = env.observation()
                action = agent.act(obs)
                ld, tac, vis = extract_full_obs(obs, task_info)
                ep_low.append(ld); ep_tac.append(tac); ep_vis.append(vis)
                ep_force.append(float(action["force"]))
                ep_dir.append([float(action["direction"][0]), float(action["direction"][1])])
                env.step(action)
            state = env.state(); metrics = score_rollout(task, state)
            total_count += 1
            if metrics.get("success") or float(metrics["score01"]) > 0.15:
                all_episodes.append({
                    "low": ep_low, "tac": ep_tac, "vis": ep_vis,
                    "force": ep_force, "dir": ep_dir,
                    "success": bool(metrics.get("success")),
                })
                if metrics.get("success"): success_count += 1
        if (ti+1) % 500 == 0:
            print(f"  [{ti+1}/{len(tasks)}] eps={total_count} kept={len(all_episodes)} succ={success_count} t={time.time()-t0:.0f}s", flush=True)

    print(f"Data: {total_count} eps, {len(all_episodes)} kept, {success_count} successes, {time.time()-t0:.0f}s", flush=True)
    return all_episodes


# ================================================================
# Training
# ================================================================
def prepare_sequences(episodes, seq_len=SEQ_LEN):
    """Create training sequences from episodes."""
    all_ld, all_tac, all_vis, all_force, all_dir, all_w = [], [], [], [], [], []
    for ep in episodes:
        weight = 2.0 if ep["success"] else 1.0
        n = len(ep["low"])
        for i in range(n):
            start = max(0, i - seq_len + 1)
            # Get sequence
            ld_seq = ep["low"][start:i+1]
            tac_seq = ep["tac"][start:i+1]
            vis_seq = ep["vis"][start:i+1]
            # Pad
            pad_n = seq_len - len(ld_seq)
            if pad_n > 0:
                ld_seq = [np.zeros(LOW_DIM, dtype=np.float32)]*pad_n + ld_seq
                tac_seq = [np.zeros(TACTILE_SHAPE, dtype=np.float32)]*pad_n + tac_seq
                vis_seq = [np.zeros(VISION_SHAPE, dtype=np.float32)]*pad_n + vis_seq
            all_ld.append(np.array(ld_seq))
            all_tac.append(np.array(tac_seq))
            all_vis.append(np.array(vis_seq))
            all_force.append(ep["force"][i])
            all_dir.append(ep["dir"][i])
            all_w.append(weight)
    return (np.array(all_ld, dtype=np.float32), np.array(all_tac, dtype=np.float32),
            np.array(all_vis, dtype=np.float32), np.array(all_force, dtype=np.float32),
            np.array(all_dir, dtype=np.float32), np.array(all_w, dtype=np.float32))


def train(episodes, epochs=40, batch_size=512, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    print("Preparing sequences...", flush=True)
    LD, TAC, VIS, FORCE, DIR, W = prepare_sequences(episodes)
    n = len(LD)
    print(f"Training samples: {n}", flush=True)

    # Move to GPU in chunks (might be too large for all at once)
    model = FullObsCNNGRU().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    t0 = time.time()
    for epoch in range(epochs):
        indices = np.random.permutation(n)
        total_loss = 0; nb = 0
        for start in range(0, n, batch_size):
            idx = indices[start:start+batch_size]
            ld_b = torch.tensor(LD[idx], device=device)
            tac_b = torch.tensor(TAC[idx], device=device)
            vis_b = torch.tensor(VIS[idx], device=device)
            f_b = torch.tensor(FORCE[idx], device=device).unsqueeze(1)
            d_b = torch.tensor(DIR[idx], device=device)
            w_b = torch.tensor(W[idx], device=device)

            # Augmentation: add noise
            if epoch < epochs * 0.8:
                ld_b += torch.randn_like(ld_b) * 0.02
                tac_b += torch.randn_like(tac_b) * 0.02
                vis_b += torch.randn_like(vis_b) * 0.01

            force_pred, dir_pred, _ = model(ld_b, tac_b, vis_b)
            loss_f = (w_b * (force_pred - f_b).pow(2).squeeze()).mean()
            loss_d = (w_b.unsqueeze(1) * (dir_pred - d_b).pow(2)).mean()
            loss = 2.0 * loss_f + 2.0 * loss_d

            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item(); nb += 1

        scheduler.step()
        if (epoch+1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: loss={total_loss/nb:.4f} t={time.time()-t0:.0f}s", flush=True)

    print(f"Training done in {time.time()-t0:.0f}s", flush=True)
    return model


# ================================================================
# Main
# ================================================================
def main():
    print("="*60, flush=True)
    print("NS-2026-07: Full CNN+GRU Training", flush=True)
    print("="*60, flush=True)

    tasks = read_jsonl(str(ROOT / "tasks" / "train_tasks.jsonl"))
    print(f"Loaded {len(tasks)} tasks", flush=True)

    print("\n--- Phase 1: Data Generation ---", flush=True)
    episodes = generate_data(tasks, n_variants=4, max_tasks=2000)

    print("\n--- Phase 2: CNN+GRU Training ---", flush=True)
    model = train(episodes, epochs=40, batch_size=512)

    # Save
    output_dir = ROOT / "cnn_gru_submission" / "model"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "cnn_gru_policy.pt"
    torch.save({"model_state_dict": model.state_dict()}, save_path)
    print(f"\nModel saved: {save_path} ({save_path.stat().st_size//1024} KB)", flush=True)
    print("Done!", flush=True)


if __name__ == "__main__":
    main()
