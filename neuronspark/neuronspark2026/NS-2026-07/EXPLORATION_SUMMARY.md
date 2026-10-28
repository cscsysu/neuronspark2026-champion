# NS-2026-07 Complete Exploration Summary

**Date**: May 26, 2026  
**Task**: Dexterous Hand Manipulation Control with Neural Network Policy  
**Status**: ✓ Full exploration complete

---

## Executive Summary

You now have a **complete understanding** of the NS-2026-07 training data and environment. The demonstration dataset contains 1,200 episodes (~154K timesteps) from a weak scripted teacher policy that you can use for behavioral cloning or imitation learning.

### What You Need to Build

1. **Neural Network Policy**: Maps observations → actions
2. **Training Pipeline**: Load JSONL data, extract trajectories, train model
3. **Submission Package**: `agent.py` + model weights in zip file
4. **Evaluation**: Test on validation tasks before submission

---

## What Was Explored

### 1. Demonstration Data (`weak_train_rollouts.jsonl`)

**Structure**: 1,200 episodes, each with:
- Task metadata (object properties, environment context)
- 128 timesteps maximum per episode
- Observations at each step
- Expert actions at each step
- Final metrics and success flag

**Key Numbers**:
- Total episodes: 1,200
- Success rate: 9.6% (deliberately weak!)
- Total timesteps: ~154,000
- File size: 97 MB (JSONL format, 1 line per episode)

**Episode Life Cycle**:
```
{task_context} → step_0[obs→action] → step_1[obs→action] → ... → step_127 → {final_metrics}
```

### 2. Observation Structure (What Your Policy SEES)

Each timestep provides:

| Field | Dimension | Type | Purpose |
|-------|-----------|------|---------|
| `low_dim_state` | 14 | float32 | Normalized state (pose, velocity, progress) |
| `contact_summary` | 4 | float32 | Contact metrics (coverage, slip risk, damage risk) |
| `tactile_heatmap_7x4` | 7×4 | float32 | Per-finger contact forces |
| `stage_context` | dict | metadata | Planning/staging information |

**Sizes Across Dataset**:
- low_dim: 14 values, normalized [-∞, ∞]
- coverage: [0.075, 1.0], mean 0.852
- slip_risk: [0, 0.194], mean 0.051
- tactile: [0, 2.5+] unnormalized forces

### 3. Action Structure (What Your Policy OUTPUTS)

Each action is a dictionary with 4 fields:

```python
action = {
    'primitive': str,           # 10 discrete choices
    'finger': str,              # 7 discrete choices
    'force': float,             # [0.0, 1.0]
    'direction': [float, float] # 2D normalized vector
}
```

**Primitive Options (10)**:
- brace, stabilize, push, drag, pivot, roll, lift_edge, tap, wait, finish

**Finger Options (7)**:
- thumb, index, middle, ring, pinky, palm, wrist

**Continuous Values**:
- force: [0.30, 0.94], mean 0.582
- direction: mostly in negative quadrant ([-1, 0] × [-0.5, 0])

### 4. Gymnasium Environment Wrapper

**File**: `simulator/gymnasium_env.py` (222 lines)

**Features**:
- Wraps MuJoCo dev simulator
- Standard Gym API: `reset()`, `step()`, `sample_action()`
- Supports task/variant selection
- Automatic reward computation from metrics
- NO built-in vectorization (use `act_batch()` for batching)

**Observation Modes**:
- `"compact"`: Only low_dim + tactile_heatmap (small)
- `"public"`: Includes vision_grid and tactile_image (regenerable)

### 5. Runtime Environment

**Official Contract**:
- OS: Ubuntu 24.04.2 LTS
- Python: CPython 3.12.3
- GPU: 1× RTX 4090 D (24,564 MB VRAM)
- Packages: torch 2.11.0+cu128, mujoco 3.3.7, numpy, scipy

**Timeouts** (CRITICAL):
- Startup: 30s
- `reset()`: 12s per call
- `act()`: **2s per timestep** ← Main constraint!
- Total: 2700s wall time

**Limits**:
- Submission zip: 8 GB
- Extracted: 16 GB
- Model files: 12 GB
- Single file: 8 GB

---

## Documentation Files Created

### 1. `TRAINING_DATA_GUIDE.md` (14 KB, 507 lines)

**Comprehensive reference covering**:
- Full episode structure with examples
- Detailed observation space breakdown
- Action space with statistics
- Gymnasium wrapper API
- Training data statistics
- Runtime requirements
- Architecture recommendations

**When to use**: Deep understanding of data format, designing architecture

### 2. `CODE_EXAMPLES.md` (17 KB, 578 lines)

**Production-ready code patterns**:
- Loading demonstration data
- Extracting trajectories
- Encoding/decoding actions
- Building neural network model
- Training loop with loss computation
- Action decoding strategies
- Policy wrapper for Gym
- Evaluation harness

**When to use**: Implementing your training pipeline

### 3. `QUICK_REFERENCE.md` (6 KB, 201 lines)

**At-a-glance cheat sheet**:
- Episode structure diagram
- Observation/action tables
- Dataset stats
- Minimal model code
- Data loading snippet
- Gym usage example
- Timeouts and limits
- Training checklist

**When to use**: Quick lookup during development

### 4. `EXPLORATION_SUMMARY.md` (This file)

**Complete exploration report** with all findings and next steps.

---

## Key Findings & Recommendations

### Finding #1: Weak Demonstrations Are Intentional

The 9.6% success rate is **by design**. The weak teacher policy is meant to:
- Provide behavioral cloning warm-start
- Force your model to learn recovery strategies
- Simulate the challenge of learning from imperfect data

**Implication**: Don't expect BC alone to achieve high scores. Plan for RL fine-tuning.

### Finding #2: Two Primary Observation Modalities

The JSONL contains:
- **Low-dimensional**: 14 floats + 4-float contact summary + 7×4 tactile heatmap
- **NOT included**: vision_grid_16x16, tactile_image_7x8x8 (but can regenerate!)

**Implication**: Can train with compact observations OR regenerate full tensors from simulator.

### Finding #3: Multi-Head Action Decoding

Actions require separate branches:
- Discrete: primitive (10) and finger (7) → use CrossEntropy loss
- Continuous: force (1) and direction (2) → use MSE/cosine loss

**Implication**: Multi-task learning architecture recommended.

### Finding #4: No Built-in Vectorization

The Gym wrapper is single-environment only. For efficiency on RTX 4090:
- Use optional `Agent.act_batch()` interface
- Batch up to 64 rollouts
- Amortize inference overhead

**Implication**: Design model for batch inference from day one.

### Finding #5: Tight Per-Step Timeout

2 seconds per `act()` call includes:
- Observation preprocessing
- Forward pass
- Action decoding
- Network latency (if any)

**Implication**: Model must be lightweight. ResNet-18 level or smaller.

---

## Recommended Training Plan

### Phase 1: Behavioral Cloning (Days 1-2)

```
1. Load 1,200 episodes from JSONL
2. Extract ~154k (obs, action) pairs
3. Build multi-head fusion model
4. Train with supervised learning
   - Primitive: CrossEntropy
   - Finger: CrossEntropy
   - Force: MSE in [0,1]
   - Direction: Cosine similarity
5. Validate on tasks/valid_tasks.jsonl (360 tasks)
6. Expected BC accuracy: 40-60%
```

### Phase 2: Evaluation & Analysis (Days 2-3)

```
1. Run BC model on 36 validation tasks
2. Measure: score01, success%, episode_length
3. Identify failure modes
4. Analyze when it succeeds vs. fails
5. Plan RL improvements
```

### Phase 3: RL Fine-Tuning (Days 3-4, optional)

```
1. Initialize PPO/TRPO from BC weights
2. Use act_batch() for efficiency
3. Collect 64-rollout batches
4. Target: 2-3% improvement in success rate
```

### Phase 4: Packaging & Submission (Day 5)

```
1. Create agent.py with Agent class
2. Implement reset(task_info) and act(observation)
3. Load model weights at startup
4. Bundle everything in submission zip
5. Test with tools/check_format.py
6. Submit!
```

---

## Quick Start Code

### Loading and Inspecting Data

```python
import json
import numpy as np

# Load all 1,200 episodes
episodes = []
with open('demonstrations/weak_train_rollouts.jsonl') as f:
    for line in f:
        episodes.append(json.loads(line))

print(f"Loaded {len(episodes)} episodes")
print(f"Success rate: {sum(1 for e in episodes if e['metrics']['success']) / len(episodes):.2%}")

# Inspect first episode
ep = episodes[0]
print(f"Task: {ep['task_id']}, Steps: {len(ep['steps'])}, Success: {ep['metrics']['success']}")

# Look at first step
step = ep['steps'][0]
obs = {
    'low_dim': np.array(step['low_dim_state']),
    'contact': np.array([step['contact_summary'][k] for k in ['coverage', 'slip_risk', 'damage_risk', 'min_contact_so_far']]),
    'tactile': np.array(step['tactile_heatmap_7x4']),
}
action = step['action']
print(f"Observation shapes: {[(k, v.shape) for k, v in obs.items()]}")
print(f"Action: {action}")
```

### Building and Training

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Model definition (see CODE_EXAMPLES.md for full version)
class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_low = nn.Linear(14, 128)
        self.encoder_tactile = nn.Conv1d(7, 32, 2)
        self.fusion = nn.Linear(160, 256)
        self.heads = nn.ModuleDict({
            'primitive': nn.Linear(256, 10),
            'finger': nn.Linear(256, 7),
            'force': nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()),
            'direction': nn.Linear(256, 2),
        })
    
    def forward(self, low_dim, tactile):
        x1 = torch.relu(self.encoder_low(low_dim))
        x2 = torch.relu(self.encoder_tactile(tactile).flatten(1))
        x = torch.relu(self.fusion(torch.cat([x1, x2], 1)))
        return {k: v(x) for k, v in self.heads.items()}

# Training
device = torch.device('cuda')
model = Policy().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Extract and prepare data (see CODE_EXAMPLES.md for full extraction)
# ...

# Train loop
for epoch in range(20):
    for batch in train_loader:
        low_dim, tactile, prim, finger, force, direction = batch
        output = model(low_dim.to(device), tactile.to(device))
        
        loss = (
            nn.functional.cross_entropy(output['primitive'], prim.to(device)) +
            nn.functional.cross_entropy(output['finger'], finger.to(device)) +
            0.5 * nn.functional.mse_loss(output['force'], force.to(device)) +
            0.5 * nn.functional.mse_loss(
                nn.functional.normalize(output['direction'], p=2, dim=1),
                nn.functional.normalize(direction.to(device), p=2, dim=1)
            )
        )
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Evaluation

```python
from simulator.gymnasium_env import make_env

env = make_env(task_file="tasks/valid_tasks.jsonl", observation_mode="compact")

total_score = 0
successes = 0

for task_idx in range(36):  # Evaluate on 36 tasks
    obs, info = env.reset(options={'task_index': task_idx})
    
    for step in range(128):
        with torch.no_grad():
            low_dim = torch.from_numpy(obs['low_dim']).unsqueeze(0).to(device)
            tactile = torch.from_numpy(obs['tactile_heatmap']).unsqueeze(0).to(device)
            output = model(low_dim, tactile)
            
            action = {
                'primitive': PRIMITIVES[output['primitive'].argmax(1).item()],
                'finger': FINGERS[output['finger'].argmax(1).item()],
                'force': float(output['force'].item()),
                'direction': nn.functional.normalize(output['direction'], p=2, dim=1).squeeze().tolist(),
            }
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if info['metrics']['success']:
            successes += 1
            break
        if terminated or truncated:
            break
    
    total_score += info['metrics']['score01']

print(f"Average score: {total_score / 36:.3f}")
print(f"Success rate: {successes / 36:.1%}")
```

---

## Common Questions Answered

**Q: Why is the success rate only 9.6%?**  
A: The demonstrations are intentionally weak to test your ability to learn beyond the data.

**Q: Can I use vision_grid and tactile_image?**  
A: Yes, but you need to regenerate them from the simulator for training data.

**Q: How do I make it fast enough for 2s per step?**  
A: Use a small model (ResNet-18 or smaller), batch operations, and profile inference time.

**Q: Should I use BC or RL?**  
A: Start with BC for quick baseline (~1-2 days), then RL for improvement if time permits.

**Q: Does the Gym wrapper support async/vectorization?**  
A: No, but use `act_batch()` for batching multiple rollouts efficiently.

**Q: What's the optimal submission strategy?**  
A: BC baseline + RL fine-tuning, prioritize inference speed over model size.

---

## Files in Your Project

**New Documentation**:
```
TRAINING_DATA_GUIDE.md    ← Comprehensive reference (507 lines)
CODE_EXAMPLES.md          ← Implementation patterns (578 lines)
QUICK_REFERENCE.md        ← Cheat sheet (201 lines)
EXPLORATION_SUMMARY.md    ← This file
```

**Existing Resources**:
```
demonstrations/weak_train_rollouts.jsonl  ← 1,200 episodes
demonstrations/manifest.json              ← Metadata
simulator/gymnasium_env.py                ← Gym wrapper
tasks/valid_tasks.jsonl                   ← 360 validation tasks
tasks/test_tasks.jsonl                    ← 360 test tasks
requirements-runtime.txt                  ← Official packages
agent_api.md                              ← Submission API
GETTING_STARTED.md                        ← Setup guide
RUNTIME_ENVIRONMENT.md                    ← Runtime spec
```

---

## Next Steps

1. **Today**: Read `TRAINING_DATA_GUIDE.md` for full understanding
2. **Tomorrow**: Use `CODE_EXAMPLES.md` to build training pipeline
3. **This week**: Train BC model, evaluate, plan improvements
4. **Before submission**: Package with `agent.py`, validate format

---

## Summary Table

| Aspect | Value/Details |
|--------|---------------|
| **Episodes** | 1,200 |
| **Steps** | ~154,000 |
| **Success Rate** | 9.6% (weak) |
| **Observation Dims** | 14 (state) + 4 (contact) + 28 (tactile) + context |
| **Action Dims** | 10 (primitive) + 7 (finger) + 1 (force) + 2 (direction) |
| **Model Size Limit** | 12 GB |
| **Inference Timeout** | 2s per step |
| **Batch Size** | Up to 64 rollouts |
| **Recommended Start** | Behavioral cloning → RL |

---

## Exploration Complete ✓

You now have:
- ✓ Full understanding of data structure
- ✓ Complete code examples
- ✓ Architecture recommendations
- ✓ Training strategy
- ✓ Submission checklist

**Ready to build your policy!**

---

*Exploration completed: May 26, 2026*  
*Files created: TRAINING_DATA_GUIDE.md, CODE_EXAMPLES.md, QUICK_REFERENCE.md*  
*Total lines of documentation: 1,286*
