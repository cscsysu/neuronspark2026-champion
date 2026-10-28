# NS-2026-07 Quick Reference Card

## 📋 Episode Structure at a Glance

```
ONE EPISODE = {
  task_id, variant_id, task_type,
  object_nominal: {shape, family, mass_bucket, surface, fragility, size_m},
  environment_context: {surface_fixture, workspace_layout, friction_multiplier_nominal, sensor_noise_hint},
  
  steps: [128 maximum steps] {
    step: int,
    low_dim_state: [14 floats],           ← Policy INPUT
    contact_summary: {coverage, slip_risk, damage_risk, min_contact_so_far},
    stage_context: {enabled, stage_index, completion_fraction, goal},
    tactile_heatmap_7x4: [[7×4 matrix]],  ← Policy INPUT
    action: {primitive, finger, force, direction}  ← Teacher LABEL
  },
  
  final_state: {...},
  metrics: {score01, success, pose_score, objective, safety, contact, efficiency}
}
```

## 🎯 Observation at Each Step

| Field | Shape | Type | Range | Meaning |
|-------|-------|------|-------|---------|
| `low_dim_state` | (14,) | float32 | [-∞, ∞] | Normalized object pose + velocity + progress |
| `coverage` | scalar | float | [0.075, 1.0] | Contact surface coverage |
| `slip_risk` | scalar | float | [0, 0.194] | Slip precursor |
| `damage_risk` | scalar | float | [0, 1.0] | Breakage risk |
| `tactile_heatmap` | (7,4) | float32 | [0, 2.5+] | Per-finger contact forces |

## 🕹️ Action at Each Step

| Field | Type | Range | Encoding | |
|-------|------|-------|----------|--|
| `primitive` | str | 10 choices | `["brace", "stabilize", "push", "drag", "pivot", "roll", "lift_edge", "tap", "wait", "finish"]` | Discrete |
| `finger` | str | 7 choices | `["thumb", "index", "middle", "ring", "pinky", "palm", "wrist"]` | Discrete |
| `force` | float | [0.0, 1.0] | Continuous | Continuous |
| `direction` | [x, y] | [-1, 1] each | 2D vector | Continuous |

## 📊 Dataset Stats

| Metric | Value |
|--------|-------|
| Episodes | 1,200 |
| Steps per episode | ≤ 128 (variable) |
| Total steps | ~154,000 |
| Success rate | 9.6% |
| Policy source | Weak scripted teacher |

## 🏗️ Minimal Model Architecture

```python
class Policy(nn.Module):
    def __init__(self):
        # 14 → 128
        self.low_dim = nn.Linear(14, 128)
        # 7×4 → 128
        self.tactile = nn.Sequential(
            nn.Conv1d(7, 32, 2),
            nn.Flatten(),
            nn.Linear(32*3, 128)
        )
        # Fusion: 256 → 256
        self.fusion = nn.Linear(256, 256)
        # Outputs
        self.primitives = nn.Linear(256, 10)   # logits
        self.fingers = nn.Linear(256, 7)       # logits
        self.force = nn.Sequential(            # [0,1]
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.direction = nn.Linear(256, 2)     # unnormalized
    
    def forward(self, low_dim, tactile):
        x1 = F.relu(self.low_dim(low_dim))
        x2 = F.relu(self.tactile(tactile))
        x = F.relu(self.fusion(torch.cat([x1, x2], 1)))
        return {
            'primitive': self.primitives(x),
            'finger': self.fingers(x),
            'force': self.force(x),
            'direction': F.normalize(self.direction(x), p=2, dim=1)
        }
```

## 💾 Loading Data

```python
import json
episodes = []
with open('demonstrations/weak_train_rollouts.jsonl') as f:
    for line in f:
        episodes.append(json.loads(line))

# Extract one episode's trajectory
ep = episodes[0]
for step in ep['steps']:
    obs = {
        'low_dim': step['low_dim_state'],        # 14 floats
        'contact': step['contact_summary'],      # 4 floats
        'tactile': step['tactile_heatmap_7x4'],  # 7×4 matrix
    }
    action = step['action']  # {primitive, finger, force, direction}
```

## 🔄 Using Gym Wrapper

```python
from simulator.gymnasium_env import make_env

env = make_env(task_file="tasks/valid_tasks.jsonl", observation_mode="compact")
obs, info = env.reset()

# obs has keys: low_dim, tactile_heatmap, tactile_image, vision_grid (if public mode)
# info has: task_info, task_id, variant_id

action = {
    'primitive': 'push',
    'finger': 'palm',
    'force': 0.5,
    'direction': [-0.5, 0.0]
}

obs, reward, terminated, truncated, info = env.step(action)
# reward = score01_new - score01_old [+ 0.2 if success]
# metrics in info['metrics']
```

## 🎯 Reward Computation

```python
# Reward is already pre-computed from metrics:
reward = metrics['score01'] - prev_score01
if metrics['success']:
    reward += 0.2  # Success bonus
```

## ⚡ Training Checklist

- [ ] Load 1,200 episodes: `json.loads()` from JSONL
- [ ] Extract ~154k (obs, action) pairs
- [ ] Encode actions: primitive/finger → indices, direction → normalize
- [ ] Create PyTorch DataLoader (batch_size=32-64)
- [ ] Build multi-head model with fusion layer
- [ ] Train with CrossEntropy + MSE losses
- [ ] Evaluate on `tasks/valid_tasks.jsonl` (360 tasks)
- [ ] Export weights as `.pt` file
- [ ] Package with `agent.py` in submission zip

## ⏱️ Timeouts (Critical!)

| Operation | Timeout |
|-----------|---------|
| Agent startup | 30s |
| Agent.reset() | 12s |
| Agent.act() | **2s** ← Single step! |
| Agent.act_batch() | 10s |
| Total wall time | 2700s |

→ **Optimize inference to < 1s/step** for headroom

## 📦 Submission Limits

| Resource | Limit |
|----------|-------|
| Zip file | 8 GB |
| Extracted | 16 GB |
| Model files | 12 GB |
| Single file | 8 GB |
| File count | 8000 |

## 🔗 Key Files

```
demonstrations/weak_train_rollouts.jsonl  ← 1,200 episodes for training
demonstrations/manifest.json               ← Metadata
simulator/gymnasium_env.py                 ← Gym wrapper
tasks/valid_tasks.jsonl                    ← 360 local validation tasks
agent_api.md                               ← Submission spec
TRAINING_DATA_GUIDE.md                     ← Full reference
CODE_EXAMPLES.md                           ← Implementation patterns
```

## 🚀 Next: Build, Test, Submit

1. **Develop locally** using `simulator/gymnasium_env.py`
2. **Evaluate on valid_tasks.jsonl** before submission
3. **Create agent.py** with `Agent.reset()` and `Agent.act()`
4. **Bundle model weights** in submission zip
5. **Test format** with `tools/check_format.py`
6. **Submit** when ready

---

*Last updated: 2026-05-26*
