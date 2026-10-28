#!/usr/bin/env python3
"""
NS-2026-07: Generate BC training data + Train MLP policy on CPU.
Complete pipeline - runs in ~20-30 min on Mac CPU.

Key innovation: trains with EXTREME domain randomization so the network
generalizes better than the rule teacher to unseen physics.
"""
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "simulator"))

from dexsim_core import DexSimEnv, sanitize_task_info, score_rollout, read_jsonl, load_agent

# =============================================================
# Constants
# =============================================================
ACTION_PRIMITIVES = ["brace", "push", "drag", "pivot", "roll", "lift_edge", "tap", "stabilize", "wait", "finish"]
FINGERS = ["thumb", "index", "middle", "ring", "pinky", "palm", "wrist"]
PRIM2IDX = {p: i for i, p in enumerate(ACTION_PRIMITIVES)}
FINGER2IDX = {f: i for i, f in enumerate(FINGERS)}
OBS_DIM = 56
N_PRIM = 10
N_FINGER = 7


# =============================================================
# Feature extraction (same as agent will use)
# =============================================================
def obs_to_features(obs, task_info):
    state = obs["low_dim_state"]["values"]
    low_dim = np.array(state, dtype=np.float32)
    heatmap = np.array(obs.get("tactile_heatmap_7x4", [[0]*4]*7), dtype=np.float32).flatten()
    cs = obs.get("contact_summary", {})
    contact_feat = np.array([
        float(cs.get("coverage", 0)), float(cs.get("min_contact_so_far", 0)),
        float(cs.get("slip_risk", 0)), float(cs.get("damage_risk", 0)),
    ], dtype=np.float32)
    task_type_vec = np.zeros(3, dtype=np.float32)
    tt = task_info.get("task_type", "nonprehensile_relocation")
    if tt == "nonprehensile_relocation": task_type_vec[0] = 1.0
    elif tt == "tool_use": task_type_vec[1] = 1.0
    else: task_type_vec[2] = 1.0
    obj = task_info.get("object", {})
    task_feat = np.array([
        float(obj.get("fragility", 0.0)),
        float(obj.get("mass", 1.0)) / 2.0,
        float(obj.get("friction", 0.5)),
        float(task_info.get("tool_goal", {}).get("progress", 0.0)),
        1.0 if task_info.get("stage_goals") else 0.0,
        float(obj.get("size", 0.1)) * 5.0,
        float(obs.get("remaining_steps", 64)) / 128.0,
    ], dtype=np.float32)
    return np.concatenate([low_dim, heatmap, contact_feat, task_type_vec, task_feat])


def random_extreme_variant(rng):
    return {
        'mass_scale': rng.uniform(0.5, 2.0),
        'friction_scale': rng.uniform(0.3, 1.8),
        'contact_bias': rng.uniform(-0.20, 0.15),
        'observation_noise': rng.uniform(0.005, 0.10),
        'actuator_scale': rng.uniform(0.5, 1.5),
        'friction_anisotropy': rng.uniform(-0.4, 0.4),
        'surface_slope_x': rng.uniform(-0.03, 0.03),
        'surface_slope_y': rng.uniform(-0.03, 0.03),
        'pose_dropout_rate': rng.uniform(0.0, 0.6),
        'tactile_dropout_rate': rng.uniform(0.0, 0.6),
        'action_delay_steps': rng.choice([0, 0, 0, 1, 1, 2, 2, 3, 4]),
        'contact_stiffness': rng.uniform(0.6, 1.6),
        'contact_damping': rng.uniform(0.5, 1.6),
        'pose_bias_x': rng.uniform(-0.04, 0.04),
        'pose_bias_y': rng.uniform(-0.04, 0.04),
        'pose_bias_theta': rng.uniform(-0.06, 0.06),
    }


# =============================================================
# Data generation
# =============================================================
def generate_data(tasks, agent, n_variants=3, max_tasks=None):
    rng = random.Random(42)
    if max_tasks:
        tasks = tasks[:max_tasks]

    all_obs = []
    all_prim = []
    all_finger = []
    all_force = []
    all_dir = []

    t0 = time.time()
    for ti, task in enumerate(tasks):
        task_info = sanitize_task_info(task)
        for vi in range(n_variants):
            variant = random_extreme_variant(rng)
            env = DexSimEnv(task, variant)
            agent.reset(task_info)
            while not env.done and env.step_count < env.max_steps:
                obs = env.observation()
                action = agent.act(obs)
                # Store
                feat = obs_to_features(obs, task_info)
                all_obs.append(feat)
                all_prim.append(PRIM2IDX.get(action["primitive"], 8))
                all_finger.append(FINGER2IDX.get(action["finger"], 5))
                all_force.append(float(action["force"]))
                d = action["direction"]
                all_dir.append([float(d[0]), float(d[1])])
                env.step(action)

        if (ti + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"  [{ti+1}/{len(tasks)}] samples={len(all_obs)} time={elapsed:.0f}s")

    elapsed = time.time() - t0
    print(f"Data generation complete: {len(all_obs)} samples in {elapsed:.0f}s")
    return {
        "obs": np.array(all_obs, dtype=np.float32),
        "prim": np.array(all_prim, dtype=np.int64),
        "finger": np.array(all_finger, dtype=np.int64),
        "force": np.array(all_force, dtype=np.float32),
        "direction": np.array(all_dir, dtype=np.float32),
    }


# =============================================================
# MLP Policy Network
# =============================================================
class PolicyMLP(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 128),
            nn.ReLU(),
        )
        self.prim_head = nn.Linear(128, N_PRIM)
        self.finger_head = nn.Linear(128, N_FINGER)
        self.force_head = nn.Linear(128, 1)
        self.dir_head = nn.Linear(128, 2)

    def forward(self, x):
        h = self.net(x)
        return self.prim_head(h), self.finger_head(h), torch.sigmoid(self.force_head(h)), torch.tanh(self.dir_head(h))


# =============================================================
# Training
# =============================================================
def train_model(data, epochs=30, batch_size=2048, lr=1e-3):
    device = torch.device("cpu")
    model = PolicyMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    n = len(data["obs"])
    obs_t = torch.tensor(data["obs"], device=device)
    prim_t = torch.tensor(data["prim"], device=device)
    finger_t = torch.tensor(data["finger"], device=device)
    force_t = torch.tensor(data["force"], device=device).unsqueeze(1)
    dir_t = torch.tensor(data["direction"], device=device)

    print(f"Training: {n} samples, {epochs} epochs, batch={batch_size}")
    t0 = time.time()

    for epoch in range(epochs):
        indices = torch.randperm(n, device=device)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, n, batch_size):
            idx = indices[start:start+batch_size]
            obs_b = obs_t[idx]
            prim_b = prim_t[idx]
            finger_b = finger_t[idx]
            force_b = force_t[idx]
            dir_b = dir_t[idx]

            prim_logits, finger_logits, force_pred, dir_pred = model(obs_b)

            loss_prim = F.cross_entropy(prim_logits, prim_b)
            loss_finger = F.cross_entropy(finger_logits, finger_b)
            loss_force = F.mse_loss(force_pred, force_b)
            loss_dir = F.mse_loss(dir_pred, dir_b)

            loss = loss_prim + 0.5 * loss_finger + 2.0 * loss_force + 2.0 * loss_dir

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        if (epoch + 1) % 5 == 0:
            avg = total_loss / n_batches
            elapsed = time.time() - t0
            print(f"  Epoch {epoch+1}/{epochs}: loss={avg:.4f} time={elapsed:.0f}s")

    print(f"Training complete in {time.time()-t0:.0f}s")
    return model


# =============================================================
# Main
# =============================================================
def main():
    print("=" * 60)
    print("NS-2026-07: BC Training Pipeline (CPU)")
    print("=" * 60)

    # Load tasks
    tasks = read_jsonl(str(ROOT / "tasks" / "train_tasks.jsonl"))
    print(f"Loaded {len(tasks)} training tasks")

    # Load rule agent as teacher
    agent = load_agent(str(ROOT / "my_submission"))
    print("Loaded rule agent (teacher)")

    # Generate training data with extreme domain randomization
    print("\n--- Phase 1: Data Generation ---")
    data = generate_data(tasks, agent, n_variants=3, max_tasks=6000)

    # Train model
    print("\n--- Phase 2: Model Training ---")
    model = train_model(data, epochs=30, batch_size=2048, lr=1e-3)

    # Save model
    output_dir = ROOT / "bc_submission" / "model"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "policy.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {"obs_dim": OBS_DIM, "hidden": 256, "n_prim": N_PRIM, "n_finger": N_FINGER},
    }, save_path)
    print(f"\nModel saved to {save_path} ({save_path.stat().st_size / 1024:.0f} KB)")

    # Quick eval
    print("\n--- Phase 3: Quick Evaluation ---")
    model.eval()
    val_tasks = read_jsonl(str(ROOT / "tasks" / "valid_tasks.jsonl"))[:20]
    # We'll need the agent to test...
    print("Use: python tools/run_public_eval.py bc_submission --tasks tasks/valid_tasks.jsonl")
    print("\nDone!")


if __name__ == "__main__":
    main()
