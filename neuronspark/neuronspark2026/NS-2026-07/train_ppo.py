#!/usr/bin/env python3
"""
NS-2026-07 PPO Training with GRU Policy + Extreme Domain Randomization.

This script trains a recurrent policy network using PPO to handle
extreme physics variations in the hidden evaluation.

Usage (on server with RTX 3090):
    pip install torch numpy scipy mujoco tqdm
    CUDA_VISIBLE_DEVICES=1 python train_ppo.py --steps 5000000 --output model_submission/model/

Architecture:
    Input (56 dim) → GRU (256 hidden) → MLP (256, 128) → Multi-head output
    - Primitive: 10-class categorical
    - Finger: 7-class categorical
    - Force: Beta distribution [0, 1]
    - Direction: Gaussian (2D)

Key: Extreme domain randomization during training ensures generalization
to hidden test's unseen physics parameters.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import deque
from multiprocessing import Process, Pipe
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "simulator"))

from dexsim_core import (
    DexSimEnv, sanitize_task_info, score_rollout, read_jsonl,
    ALLOWED_PRIMITIVES, ALLOWED_FINGERS, MAX_STEPS
)

# ============================================================
# Constants
# ============================================================
ACTION_PRIMITIVES = ["brace", "push", "drag", "pivot", "roll", "lift_edge", "tap", "stabilize", "wait", "finish"]
FINGERS = ["thumb", "index", "middle", "ring", "pinky", "palm", "wrist"]
OBS_DIM = 56  # 14 + 28 + 4 + 3 + 7
HIDDEN_DIM = 256
NUM_PRIMITIVES = 10
NUM_FINGERS = 7


# ============================================================
# Observation processing
# ============================================================
def obs_to_tensor(obs: dict, task_info: dict) -> np.ndarray:
    """Convert observation + task info to 56-dim feature vector."""
    state = obs["low_dim_state"]["values"]
    low_dim = np.array(state, dtype=np.float32)

    heatmap = np.array(obs.get("tactile_heatmap_7x4", [[0]*4]*7), dtype=np.float32).flatten()

    cs = obs.get("contact_summary", {})
    contact_feat = np.array([
        float(cs.get("coverage", 0)),
        float(cs.get("min_contact_so_far", 0)),
        float(cs.get("slip_risk", 0)),
        float(cs.get("damage_risk", 0)),
    ], dtype=np.float32)

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
        float(obj.get("mass", 1.0)) / 2.0,
        float(obj.get("friction", 0.5)),
        float(task_info.get("tool_goal", {}).get("progress", 0.0)),
        1.0 if task_info.get("stage_goals") else 0.0,
        float(obj.get("size", 0.1)) * 5.0,
        float(obs.get("remaining_steps", 64)) / 128.0,
    ], dtype=np.float32)

    return np.concatenate([low_dim, heatmap, contact_feat, task_type_vec, task_feat])


# ============================================================
# Extreme Domain Randomization
# ============================================================
def random_extreme_variant(rng: random.Random) -> dict:
    """Generate an extreme physics variant for training robustness."""
    return {
        "mass_scale": rng.uniform(0.5, 2.0),
        "friction_scale": rng.uniform(0.3, 1.8),
        "fragility_scale": rng.uniform(0.7, 1.5),
        "contact_bias": rng.uniform(-0.20, 0.15),
        "wrist_bias": rng.uniform(-0.05, 0.05),
        "start_dx": rng.uniform(-0.03, 0.03),
        "start_dy": rng.uniform(-0.03, 0.03),
        "start_dtheta": rng.uniform(-0.10, 0.10),
        "observation_noise": rng.uniform(0.005, 0.08),
        "actuator_scale": rng.uniform(0.6, 1.4),
        "friction_anisotropy": rng.uniform(-0.4, 0.4),
        "surface_slope_x": rng.uniform(-0.03, 0.03),
        "surface_slope_y": rng.uniform(-0.03, 0.03),
        "pose_bias_x": rng.uniform(-0.02, 0.02),
        "pose_bias_y": rng.uniform(-0.02, 0.02),
        "pose_bias_theta": rng.uniform(-0.05, 0.05),
        "pose_dropout_rate": rng.uniform(0.0, 0.6),
        "tactile_dropout_rate": rng.uniform(0.0, 0.6),
        "action_delay_steps": rng.choice([0, 0, 0, 1, 1, 2, 2, 3, 4]),
        "contact_stiffness": rng.uniform(0.6, 1.6),
        "contact_damping": rng.uniform(0.5, 1.6),
    }


# ============================================================
# Policy Network (GRU + Multi-head)
# ============================================================
class PolicyNetwork(nn.Module):
    """GRU-based policy for handling temporal dependencies."""

    def __init__(self, obs_dim=OBS_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # GRU for temporal reasoning
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        # Policy heads
        self.primitive_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, NUM_PRIMITIVES),
        )
        self.finger_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, NUM_FINGERS),
        )
        self.force_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # alpha, beta for Beta distribution
        )
        self.direction_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 4),  # mean_x, mean_y, log_std_x, log_std_y
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, obs, hidden=None):
        """
        obs: (batch, obs_dim) or (batch, seq_len, obs_dim)
        hidden: (1, batch, hidden_dim) or None
        """
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)  # (batch, 1, obs_dim)

        batch_size = obs.shape[0]
        if hidden is None:
            hidden = torch.zeros(1, batch_size, self.hidden_dim, device=obs.device)

        # Encode
        encoded = self.encoder(obs)  # (batch, seq, hidden)

        # GRU
        gru_out, hidden_new = self.gru(encoded, hidden)
        features = gru_out[:, -1, :]  # Last timestep: (batch, hidden)

        # Policy outputs
        prim_logits = self.primitive_head(features)
        finger_logits = self.finger_head(features)
        force_params = F.softplus(self.force_head(features)) + 1.0  # alpha, beta > 1
        dir_params = self.direction_head(features)

        # Value
        value = self.value_head(features).squeeze(-1)

        return prim_logits, finger_logits, force_params, dir_params, value, hidden_new

    def get_action(self, obs, hidden=None, deterministic=False):
        """Sample action from policy."""
        prim_logits, finger_logits, force_params, dir_params, value, hidden_new = self.forward(obs, hidden)

        # Primitive
        prim_dist = Categorical(logits=prim_logits)
        if deterministic:
            prim_action = prim_logits.argmax(-1)
        else:
            prim_action = prim_dist.sample()
        prim_log_prob = prim_dist.log_prob(prim_action)

        # Finger
        finger_dist = Categorical(logits=finger_logits)
        if deterministic:
            finger_action = finger_logits.argmax(-1)
        else:
            finger_action = finger_dist.sample()
        finger_log_prob = finger_dist.log_prob(finger_action)

        # Force (Beta distribution)
        alpha, beta = force_params[:, 0], force_params[:, 1]
        # Use mean for deterministic, sample otherwise
        if deterministic:
            force_action = alpha / (alpha + beta)
        else:
            # Reparameterized sample from Beta
            force_action = torch.distributions.Beta(alpha, beta).sample()
        force_log_prob = torch.distributions.Beta(alpha, beta).log_prob(force_action)

        # Direction (Gaussian)
        dir_mean = torch.tanh(dir_params[:, :2])
        dir_log_std = torch.clamp(dir_params[:, 2:], -2.0, 0.5)
        dir_std = dir_log_std.exp()
        dir_dist = Normal(dir_mean, dir_std)
        if deterministic:
            dir_action = dir_mean
        else:
            dir_action = dir_dist.sample()
        dir_action = torch.clamp(dir_action, -1.0, 1.0)
        dir_log_prob = dir_dist.log_prob(dir_action).sum(-1)

        # Total log prob
        log_prob = prim_log_prob + finger_log_prob + force_log_prob + dir_log_prob

        return {
            "primitive": prim_action,
            "finger": finger_action,
            "force": force_action,
            "direction": dir_action,
            "log_prob": log_prob,
            "value": value,
            "hidden": hidden_new,
        }

    def evaluate_actions(self, obs, hidden, actions):
        """Evaluate log probs and entropy for PPO update."""
        prim_logits, finger_logits, force_params, dir_params, value, _ = self.forward(obs, hidden)

        # Primitive
        prim_dist = Categorical(logits=prim_logits)
        prim_log_prob = prim_dist.log_prob(actions["primitive"])
        prim_entropy = prim_dist.entropy()

        # Finger
        finger_dist = Categorical(logits=finger_logits)
        finger_log_prob = finger_dist.log_prob(actions["finger"])
        finger_entropy = finger_dist.entropy()

        # Force
        alpha, beta = force_params[:, 0], force_params[:, 1]
        force_dist = torch.distributions.Beta(alpha, beta)
        force_log_prob = force_dist.log_prob(actions["force"].clamp(0.001, 0.999))
        force_entropy = force_dist.entropy()

        # Direction
        dir_mean = torch.tanh(dir_params[:, :2])
        dir_log_std = torch.clamp(dir_params[:, 2:], -2.0, 0.5)
        dir_std = dir_log_std.exp()
        dir_dist = Normal(dir_mean, dir_std)
        dir_log_prob = dir_dist.log_prob(actions["direction"]).sum(-1)
        dir_entropy = dir_dist.entropy().sum(-1)

        log_prob = prim_log_prob + finger_log_prob + force_log_prob + dir_log_prob
        entropy = prim_entropy + finger_entropy + force_entropy + dir_entropy

        return log_prob, entropy, value


# ============================================================
# Environment Wrapper
# ============================================================
class TrainEnv:
    """Single training environment with domain randomization."""

    def __init__(self, tasks, seed=0):
        self.tasks = tasks
        self.rng = random.Random(seed)
        self.task = None
        self.task_info = None
        self.env = None

    def reset(self):
        self.task = self.rng.choice(self.tasks)
        variant = random_extreme_variant(self.rng)
        self.task_info = sanitize_task_info(self.task)
        self.env = DexSimEnv(self.task, variant)
        obs = self.env.observation()
        return obs_to_tensor(obs, self.task_info)

    def step(self, action_dict):
        """Execute action, return (obs, reward, done, info)."""
        result = self.env.step(action_dict)
        obs = self.env.observation()
        state = self.env.state()
        metrics = score_rollout(self.task, state)

        # Shaped reward
        score01 = float(metrics["score01"])
        done = self.env.done or self.env.step_count >= self.env.max_steps

        # Dense reward signal
        reward = 0.0
        # Progress toward target
        dx = float(obs["low_dim_state"]["values"][0])
        dy = float(obs["low_dim_state"]["values"][1])
        dist = math.sqrt(dx*dx + dy*dy)
        reward -= dist * 0.1  # penalty for distance

        # Contact maintenance
        contact = float(obs["low_dim_state"]["values"][9])
        reward += contact * 0.05

        # Safety
        damage_risk = float(obs.get("contact_summary", {}).get("damage_risk", 0))
        reward -= damage_risk * 0.1

        # Success bonus
        if done and metrics.get("success"):
            reward += 5.0
        elif done:
            reward += score01 * 2.0  # partial credit

        obs_feat = obs_to_tensor(obs, self.task_info)
        return obs_feat, reward, done, {"score01": score01, "success": metrics.get("success", False)}


def decode_action(prim_idx, finger_idx, force, direction):
    """Convert network output to action dict."""
    return {
        "primitive": ACTION_PRIMITIVES[int(prim_idx)],
        "finger": FINGERS[int(finger_idx)],
        "force": float(max(0.0, min(1.0, force))),
        "direction": [float(direction[0]), float(direction[1])],
    }


# ============================================================
# PPO Training
# ============================================================
class PPOBuffer:
    """Storage for PPO rollout data."""

    def __init__(self, size, obs_dim, gamma=0.99, lam=0.95):
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.actions_prim = np.zeros(size, dtype=np.int64)
        self.actions_finger = np.zeros(size, dtype=np.int64)
        self.actions_force = np.zeros(size, dtype=np.float32)
        self.actions_dir = np.zeros((size, 2), dtype=np.float32)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.values = np.zeros(size, dtype=np.float32)
        self.log_probs = np.zeros(size, dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.float32)
        self.advantages = np.zeros(size, dtype=np.float32)
        self.returns = np.zeros(size, dtype=np.float32)
        self.gamma = gamma
        self.lam = lam
        self.ptr = 0
        self.size = size

    def store(self, obs, prim, finger, force, direction, reward, value, log_prob, done):
        i = self.ptr
        self.obs[i] = obs
        self.actions_prim[i] = prim
        self.actions_finger[i] = finger
        self.actions_force[i] = force
        self.actions_dir[i] = direction
        self.rewards[i] = reward
        self.values[i] = value
        self.log_probs[i] = log_prob
        self.dones[i] = done
        self.ptr += 1

    def compute_gae(self, last_value):
        """Compute GAE advantages."""
        n = self.ptr
        self.values[n - 1] if n > 0 else 0
        last_gae = 0
        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]
            next_non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + self.gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + self.gamma * self.lam * next_non_terminal * last_gae
            self.advantages[t] = last_gae
        self.returns[:n] = self.advantages[:n] + self.values[:n]

    def get(self):
        n = self.ptr
        # Normalize advantages
        adv = self.advantages[:n]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return {
            "obs": self.obs[:n],
            "prim": self.actions_prim[:n],
            "finger": self.actions_finger[:n],
            "force": self.actions_force[:n],
            "direction": self.actions_dir[:n],
            "log_probs": self.log_probs[:n],
            "advantages": adv,
            "returns": self.returns[:n],
        }

    def reset(self):
        self.ptr = 0


def train_ppo(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load tasks
    tasks = read_jsonl(args.tasks)
    print(f"Loaded {len(tasks)} training tasks")

    # Create environments
    num_envs = args.num_envs
    envs = [TrainEnv(tasks, seed=i) for i in range(num_envs)]

    # Create policy
    policy = PolicyNetwork(OBS_DIM, HIDDEN_DIM).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)

    # Training parameters
    total_steps = args.steps
    rollout_length = args.rollout_length
    batch_size = args.batch_size
    epochs_per_update = args.epochs
    clip_eps = 0.2
    value_coef = 0.5
    entropy_coef = 0.01
    max_grad_norm = 0.5

    # Logging
    episode_rewards = deque(maxlen=100)
    episode_successes = deque(maxlen=100)
    episode_scores = deque(maxlen=100)

    # Initialize envs
    obs_list = [env.reset() for env in envs]
    hidden_states = [None] * num_envs
    steps_done = 0
    updates = 0
    t0 = time.time()

    print(f"\nStarting PPO training: {total_steps} steps, {num_envs} envs, rollout={rollout_length}")
    print(f"Batch size: {batch_size}, Epochs: {epochs_per_update}")
    print("-" * 80)

    while steps_done < total_steps:
        # Collect rollouts
        buffer = PPOBuffer(rollout_length * num_envs, OBS_DIM)
        ep_rewards_batch = []
        ep_steps_batch = []

        policy.eval()
        with torch.no_grad():
            for step in range(rollout_length):
                obs_tensor = torch.tensor(np.array(obs_list), dtype=torch.float32, device=device)
                # Handle hidden states (some may be None after reset)
                if all(hs is None for hs in hidden_states):
                    h = None
                else:
                    h_list = [hs if hs is not None else torch.zeros(1, 1, HIDDEN_DIM, device=device)
                              for hs in hidden_states]
                    h = torch.cat(h_list, dim=1)  # (1, num_envs, hidden)

                action_out = policy.get_action(obs_tensor, h)

                for i in range(num_envs):
                    prim = action_out["primitive"][i].item()
                    finger = action_out["finger"][i].item()
                    force = action_out["force"][i].item()
                    direction = action_out["direction"][i].cpu().numpy()

                    action_dict = decode_action(prim, finger, force, direction)
                    next_obs, reward, done, info = envs[i].step(action_dict)

                    buffer.store(
                        obs_list[i], prim, finger, force, direction,
                        reward, action_out["value"][i].item(),
                        action_out["log_prob"][i].item(), float(done)
                    )

                    if done:
                        episode_scores.append(info["score01"])
                        episode_successes.append(1.0 if info["success"] else 0.0)
                        next_obs = envs[i].reset()
                        hidden_states[i] = None
                    else:
                        hidden_states[i] = action_out["hidden"][:, i:i+1, :]

                    obs_list[i] = next_obs

                steps_done += num_envs

        # Compute advantages
        with torch.no_grad():
            obs_tensor = torch.tensor(np.array(obs_list), dtype=torch.float32, device=device)
            h = None if hidden_states[0] is None else torch.cat([hs if hs is not None else torch.zeros(1, 1, HIDDEN_DIM, device=device) for hs in hidden_states], dim=1)
            _, _, _, _, last_values, _ = policy.forward(obs_tensor, h)
            buffer.compute_gae(last_values.cpu().numpy().mean())

        # PPO update
        data = buffer.get()
        dataset_size = len(data["obs"])

        policy.train()
        for epoch in range(epochs_per_update):
            indices = np.random.permutation(dataset_size)
            for start in range(0, dataset_size, batch_size):
                end = min(start + batch_size, dataset_size)
                idx = indices[start:end]

                obs_b = torch.tensor(data["obs"][idx], device=device)
                prim_b = torch.tensor(data["prim"][idx], device=device)
                finger_b = torch.tensor(data["finger"][idx], device=device)
                force_b = torch.tensor(data["force"][idx], device=device)
                dir_b = torch.tensor(data["direction"][idx], device=device)
                old_log_probs = torch.tensor(data["log_probs"][idx], device=device)
                advantages = torch.tensor(data["advantages"][idx], device=device)
                returns = torch.tensor(data["returns"][idx], device=device)

                actions = {"primitive": prim_b, "finger": finger_b, "force": force_b, "direction": dir_b}
                new_log_probs, entropy, values = policy.evaluate_actions(obs_b, None, actions)

                # PPO clipped objective
                ratio = (new_log_probs - old_log_probs).exp()
                clip_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
                policy_loss = -torch.min(ratio * advantages, clip_ratio * advantages).mean()

                value_loss = F.mse_loss(values, returns)
                entropy_loss = -entropy.mean()

                loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                optimizer.step()

        updates += 1
        buffer.reset()

        # Logging
        if updates % 10 == 0:
            elapsed = time.time() - t0
            fps = steps_done / elapsed
            avg_score = np.mean(episode_scores) if episode_scores else 0
            avg_success = np.mean(episode_successes) if episode_successes else 0
            print(f"[{steps_done:>8d}/{total_steps}] "
                  f"score={avg_score:.3f} success={avg_success:.1%} "
                  f"fps={fps:.0f} elapsed={elapsed:.0f}s")

        # Save periodically
        if updates % 50 == 0:
            save_path = Path(args.output) / "policy_checkpoint.pt"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": policy.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "steps": steps_done,
                "config": {"obs_dim": OBS_DIM, "hidden_dim": HIDDEN_DIM},
            }, save_path)

    # Final save
    save_path = Path(args.output) / "policy.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": policy.state_dict(),
        "config": {"obs_dim": OBS_DIM, "hidden_dim": HIDDEN_DIM},
    }, save_path)
    print(f"\nTraining complete! Model saved to {save_path}")
    print(f"Final: score={np.mean(episode_scores):.3f} success={np.mean(episode_successes):.1%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="tasks/train_tasks.jsonl")
    parser.add_argument("--output", default="model_submission/model/")
    parser.add_argument("--steps", type=int, default=5_000_000)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--rollout-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()
    train_ppo(args)


if __name__ == "__main__":
    main()
