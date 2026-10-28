# NS-2026-07 Training Data & Environment Guide

## Executive Summary

**NS-2026-07** is a dexterous hand manipulation control task where your neural network policy must:
- Manipulate objects (boxes, cylinders, deformable items) on a tactile-enabled surface
- Use multi-finger contact control with tactile feedback
- Handle three task types: nonprehensile relocation, tool use, and resource sequencing
- Operate within 128 timestep control horizons

**Available Training Data**: 1,200 weak demonstrations from scripted teacher policies
**Target Runtime**: PyTorch 2.11.0 + CUDA 12.8 on RTX 4090 D

---

## 1. Demonstration Data Structure (`weak_train_rollouts.jsonl`)

### File Statistics
- **Location**: `demonstrations/weak_train_rollouts.jsonl`
- **Format**: JSONL (1 episode per line, compact JSON)
- **Size**: ~97 MB
- **Episodes**: 1,200 episodes
- **Success Rate**: 9.58% (these are weak demonstrations!)
- **Policy Source**: `weak_stage_rule_teacher_v2` scripted teacher

### Episode-Level Structure

Each episode is a JSON object with:

```python
{
    # Task identification
    "task_id": "dex_tr_0000_f96ad66654",
    "variant_id": "proxy_0",
    "task_type": "tool_use",  # or nonprehensile_relocation, resource_sequence
    "policy": "weak_stage_rule_teacher_v2",
    
    # Task context (available at reset)
    "object_nominal": {
        "shape": "low_box",              # object geometry class
        "family": "prismatic",           # prismatic|curved|deformable_or_asymmetric|unknown
        "mass_bucket": "medium",         # light|medium|heavy
        "surface": "mixed",              # surface properties
        "fragility": "moderate",         # robust|moderate|fragile
        "size_m": 0.152                  # characteristic size in meters
    },
    
    "environment_context": {
        "surface_fixture": "micro_ridge_pad",  # contact surface type
        "workspace_layout": "edge_near_slot",  # scene configuration
        "friction_multiplier_nominal": 0.86,   # friction scaling
        "sensor_noise_hint": "medium",         # noise level
        "proxy_render_profile": "topdown_contact_grid_16x16"
    },
    
    # Episode trajectory (128 timesteps max)
    "steps": [
        # Step 0, 1, 2, ... 127
        {
            "step": 0,
            
            # OBSERVATION at this step (these are inputs to your policy!)
            "low_dim_state": [...14 floats...],
            "contact_summary": {...},
            "stage_context": {...},
            "tactile_heatmap_7x4": [...7x4 array...],
            
            # ACTION taken at this step (expert demonstration)
            "action": {
                "primitive": "brace",
                "finger": "palm",
                "force": 0.3,
                "direction": [0.0, 0.0]
            }
        },
        ...
    ],
    
    # Episode results
    "final_state": {...},
    "metrics": {
        "score01": 0.2632,           # overall success score [0,1]
        "success": false,            # boolean success
        "pose_score": 0.5962,        # target pose achievement
        "objective": 0.9630,         # task objective completion
        "safety": 0.25,              # safety/damage metric
        "contact": 0.5192,           # contact quality
        "efficiency": 0.1869         # energy efficiency
    }
}
```

---

## 2. Observation Space (Policy Inputs)

### 2.1 Low-Dimensional State (14 values)

The `low_dim_state` vector contains normalized state information:

```
Position 0-2:   Object position (x, y, angle) - normalized
Position 3-5:   Object position (x, y, angle) - duplicate? 
Position 6-8:   Velocity/rates (vx, vy, vang) - very small, close to 0
Position 9:     Contact metric (0.08-0.89) - likely normalized contact coverage
Position 10-11: [0 or small value]
Position 12-13: Progress metrics [0, 1] - stage/task completion fraction
```

**Example values from same episode**:
- Step 0:   `[-0.203, -0.087, -0.396, -0.203, -0.087, -0.396, 0, 0, 0, 0.081, 0, 0, 0, 1.0]`
- Step 32:  `[-0.150, -0.085, -0.065, -0.150, -0.085, -0.065, -0.0003, -0.0001, 0, 0.890, 0.076, 0.223, 0.308, 1.0]`
- Step 127: `[-0.132, -0.062, -0.082, -0.132, -0.062, -0.082, 0, 0, 0, 0.890, 0, 0.568, 0.770, 1.0]`

**Dimensions**: 14 floats

---

### 2.2 Contact Summary (4 values)

```python
"contact_summary": {
    "coverage": 0.0867,          # [0.075-1.0] spatial contact coverage
    "min_contact_so_far": 0.0814, # minimum contact value in episode
    "slip_risk": 0.0112,         # [0-0.194] slip precursor indicator
    "damage_risk": 0.0           # [0-1.0] damage/breakage risk
}
```

**Statistics across one episode**:
- Coverage: min=0.075, max=1.0, mean=0.852
- Slip risk: min=0, max=0.194, mean=0.051
- Damage risk: min=0, max=1.0, mean=0.754

---

### 2.3 Tactile Heatmap (7×4 grid)

```python
"tactile_heatmap_7x4": [
    [0.0, 0.0, 0.033, 0.086],      # 7 rows (finger/contact regions)
    [0.004, 0.008, 0.032, 0.069],  # 4 columns (spatial bins)
    ...
]
```

**Characteristics**:
- **Shape**: 7 rows (finger regions) × 4 columns (spatial bins)
- **Range**: [0, 2.5+] (unnormalized contact forces)
- **Meaning**: Per-finger contact heatmap showing normal forces

---

### 2.4 Stage Context (per-episode planning info)

```python
"stage_context": {
    "enabled": false,
    "current_stage_index": 0,
    "stage_count": 0,
    "completion_fraction": 1.0,
    "current_stage_goal": {"x": -0.197, "y": 0.049, "theta": -0.958},
    "stage_reached_steps": []
}
```

---

### 2.5 Vision Grid (Not in demonstration data)

⚠️ **Important**: The JSONL demonstrations do NOT include `vision_grid_16x16` or `tactile_image_7x8x8`.

From the agent API docs:
- `vision_grid_16x16`: 6 channels × 16×16 spatial tokens (can be regenerated from simulator)
- `tactile_image_7x8x8`: 7×8×8 tactile image tensor (can be regenerated from simulator)

These can be regenerated by running the same episodes through the MuJoCo simulator.

---

## 3. Action Space (Policy Outputs)

### 3.1 Action Schema

Each action is a dictionary with 4 required fields:

```python
action = {
    "primitive": str,          # 10 action primitives
    "finger": str,             # 7 possible fingers
    "force": float,            # [0.0, 1.0] normalized force
    "direction": [float, float] # 2D direction vector
}
```

### 3.2 Action Primitives (10 options)

```python
ACTION_PRIMITIVES = [
    "brace",       # 0 - brace/stabilize contact
    "stabilize",   # 1 - stabilize object
    "push",        # 2 - push manipulation
    "drag",        # 3 - dragging motion
    "pivot",       # 4 - rotate around point
    "roll",        # 5 - rolling motion
    "lift_edge",   # 6 - lift edge of object
    "tap",         # 7 - tap/percussion action
    "wait",        # 8 - wait/no action
    "finish"       # 9 - finish episode
]
```

**Usage in demonstrations** (1200 episodes):
- `push`: 66 instances (primary action)
- `brace`: 52 instances
- `tap`: 8 instances
- `pivot`: 2 instances

### 3.3 Finger Options (7 options)

```python
FINGERS = [
    "thumb",       # 0
    "index",       # 1
    "middle",      # 2
    "ring",        # 3
    "pinky",       # 4
    "palm",        # 5 - primary contact surface
    "wrist"        # 6
]
```

**Usage in demonstrations**:
- `palm`: 118 instances (dominant)
- `index`: 8 instances
- `thumb`: 2 instances

### 3.4 Force (continuous [0, 1])

**Statistics from demonstrations**:
- Min: 0.30
- Max: 0.94
- Mean: 0.582

### 3.5 Direction (continuous [x, y])

2D normalized direction vector (usually in XY plane).

**Statistics from demonstrations**:
- X range: [-1.0, 0.0], mean: -0.535
- Y range: [-0.513, 0.0], mean: -0.253

---

## 4. Manifest Metadata

```json
{
  "version_id": "NS-2026-07-r20260525-08-gpu-vision-stages",
  "file": "weak_train_rollouts.jsonl",
  "policy": "weak_stage_rule_teacher_v2",
  "source_split": "public train",
  "episode_count": 1200,
  "success_rate": 0.095833,
  "elapsed_seconds": 71.865,
  "sha256": "ff17f29024fe17c5c536aa21c2dc8d9f73840c0ec9efe8b8cfb9607d25ea5e32"
}
```

**Key Points**:
- Only 9.6% success rate → these are weak demonstrations for behavioral cloning
- Generated with scripted teacher policy
- Version pinned: `NS-2026-07-r20260525-08-gpu-vision-stages`

---

## 5. Gymnasium Environment Wrapper

The simulator provides a Gym-style wrapper at `simulator/gymnasium_env.py`:

### 5.1 Action Space

```python
action_space = spaces.Dict({
    "primitive": spaces.Discrete(10),           # 0-9
    "finger": spaces.Discrete(7),               # 0-6
    "force": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
    "direction": spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
})
```

### 5.2 Observation Space (Compact Mode)

```python
observation_space = spaces.Dict({
    "low_dim": spaces.Box(-∞, +∞, shape=(14,), dtype=np.float32),
    "tactile_heatmap": spaces.Box(0.0, 2.5, shape=(7,4), dtype=np.float32),
    "tactile_image": spaces.Box(0.0, 1.0, shape=(7,8,8), dtype=np.float32),
    "vision_grid": spaces.Box(0.0, 1.0, shape=(6,16,16), dtype=np.float32),
})
```

### 5.3 Reward Structure

```python
reward = (score01_new - score01_old)  # differential reward
if success:
    reward += 0.2  # bonus for task completion
```

### 5.4 Usage

```python
from simulator.gymnasium_env import DexSimGymEnv

env = DexSimGymEnv(
    task_file="tasks/valid_tasks.jsonl",
    task_index=0,
    observation_mode="compact"  # or "public"
)
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
```

---

## 6. Gym Environment Features

### 6.1 Does NOT Support Vectorized Envs

The wrapper is single-environment only. For batching, use the optional `Agent.act_batch()` interface directly.

### 6.2 Core Methods

- `reset(seed=None, options=None)` → obs, info
- `step(action)` → obs, reward, terminated, truncated, info
- `sample_action()` → random action
- `render()` → returns current state

### 6.3 Task Selection

```python
env.reset(options={
    "task": task_dict,           # or
    "task_index": 0,             # index into task file
    "variant": variant_dict,     # or
    "variant_index": 0           # index into public_variants
})
```

---

## 7. Training Data Statistics

### 7.1 Dataset Composition

```
Total episodes: 1,200
Success rate: 9.6%

Object types observed:
  - low_box (prismatic)
  - cylinder (curved)
  - Other shapes in larger dataset

Environment contexts:
  - micro_ridge_pad / edge_near_slot (common)
  - soft_damping_pad / tool_rail
  - Other surfaces

Mass buckets: light, medium, heavy
Friction ranges: 0.86-1.08
```

### 7.2 Episode Structure

- **Fixed horizon**: 128 steps max (includes early termination)
- **Step recording**: All steps recorded with observations + actions
- **Metrics**: Every episode includes 7 scoring metrics

### 7.3 Available Regenerable Data

High-dimensional observation tensors CAN be regenerated:

```python
# These are in the agent API but NOT in JSONL:
"vision_grid_16x16"      # 6×16×16 spatial vision grid (topdown)
"tactile_image_7x8x8"    # 7×8×8 tactile image tensor
```

Regenerate by running same tasks through MuJoCo simulator.

---

## 8. Runtime Requirements

### 8.1 Guaranteed Packages

```
torch==2.11.0+cu128
mujoco==3.3.7
numpy==2.4.6
scipy==1.17.1
tqdm==4.67.3
```

### 8.2 Optional (Dev Only)

```
gymnasium       # for Gym wrapper
```

### 8.3 Hardware Target

- **GPU**: 1× RTX 4090 D (24,564 MB VRAM)
- **RAM**: 16 GB extracted submission tree limit
- **Submission**: 8 GB zip size limit, 12 GB model files limit

### 8.4 Timeouts

- Agent startup: 30s
- `reset()`: 12s
- `act()`: 2s per timestep
- `act_batch()`: 10s for batch
- Total wall time: 2700s hidden

---

## 9. Key Training Insights from Manifest

1. **Weak demonstrations** → Good for behavioral cloning warm start, but not optimal
2. **Low success rate** (9.6%) → Policy needs to learn recovery strategies
3. **Scripted teacher** → Likely rule-based primitive selection; learn to improve
4. **Batch API opportunity** → 64-rollout batches on RTX 4090 for efficient training
5. **Multi-task distribution** → 3 task types, 4 task capabilities, diverse objects

---

## 10. Recommended Architecture Approach

Based on the data structure:

### Input Fusion
- **Low-dim state**: 14 floats (MLP branch)
- **Tactile heatmap**: 7×4 (small CNN)
- **Tactile image** (regenerated): 7×8×8 (CNN branch)
- **Vision grid** (regenerated): 6×16×16 (2D CNN backbone)

### Architecture Options
1. **Small CNN + MLP** (fits 12GB model budget)
   - ResNet-18 backbone for vision
   - 1D conv for tactile sequences
   - MLP fusion layer
   
2. **Transformer with vision** (if regenerating vision grids)
   - ViT or DeiT for spatial grids
   - Temporal tactile transformer
   
3. **RL fine-tuning on BC** (recommended)
   - Start with BC on demonstrations
   - Fine-tune with PPO/TRPO on live rollouts

### Action Decoding
- Separate heads for discrete (primitive, finger) and continuous (force, direction)
- Or: embed actions into latent space and decode

---

## 11. File Locations Quick Reference

```
/Users/tristonchen/Desktop/neuronspark2026/NS-2026-07/

demonstrations/
  ├── weak_train_rollouts.jsonl      # 1200 episodes for training
  └── manifest.json                   # metadata

simulator/
  └── gymnasium_env.py                # Gym wrapper

tasks/
  ├── valid_tasks.jsonl               # 360 local validation tasks
  └── test_tasks.jsonl                # 360 local test tasks

tools/
  ├── check_format.py                 # Validate submission
  ├── run_public_eval.py              # Evaluate agent
  └── render_replay.py                # Debug visualization

requirements-runtime.txt               # Official submission packages
agent_api.md                           # Submission interface spec
GETTING_STARTED.md                     # Setup guide
RUNTIME_ENVIRONMENT.md                 # Official runtime contract
```

---

## 12. Next Steps for Training

1. **Load and analyze** the demonstration data
2. **Regenerate vision/tactile images** from public simulator if needed
3. **Design architecture** fitting observation structure
4. **Implement behavioral cloning** baseline on 1200 episodes
5. **Evaluate on valid_tasks.jsonl** (360 tasks)
6. **Fine-tune with RL** if time permits (PPO/TRPO)
7. **Package agent.py** with model weights in submission

