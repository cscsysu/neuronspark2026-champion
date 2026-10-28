# NS-2026-07 Code Examples & Implementation Details

## Loading and Exploring the Demonstration Data

### Basic Loading

```python
import json
import numpy as np

def load_demonstrations(filepath, limit=None):
    """Load demonstration episodes from JSONL file."""
    episodes = []
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            episode = json.loads(line)
            episodes.append(episode)
    return episodes

# Load all 1200 episodes
demos = load_demonstrations('demonstrations/weak_train_rollouts.jsonl')
print(f"Loaded {len(demos)} episodes")
print(f"Success rate: {sum(1 for d in demos if d['metrics']['success']) / len(demos):.2%}")
```

### Extracting Training Data

```python
def extract_trajectories(episodes):
    """Extract observations, actions, rewards from episodes."""
    observations = []
    actions = []
    rewards = []
    episode_lengths = []
    
    for episode in episodes:
        ep_obs = []
        ep_actions = []
        ep_rewards = []
        
        score_prev = 0.0
        for step in episode['steps']:
            # Observation
            obs = {
                'low_dim': np.array(step['low_dim_state'], dtype=np.float32),
                'contact': np.array([
                    step['contact_summary']['coverage'],
                    step['contact_summary']['slip_risk'],
                    step['contact_summary']['damage_risk'],
                    step['contact_summary']['min_contact_so_far']
                ], dtype=np.float32),
                'tactile': np.array(step['tactile_heatmap_7x4'], dtype=np.float32),
                'stage': step['stage_context'],
            }
            
            # Action (teacher)
            action = step['action']
            act_dict = {
                'primitive': action['primitive'],
                'finger': action['finger'],
                'force': float(action['force']),
                'direction': np.array(action['direction'], dtype=np.float32),
            }
            
            # Reward (from metrics)
            metrics = episode['metrics']
            score_new = metrics['score01']
            reward = score_new - score_prev
            if metrics['success']:
                reward += 0.2
            score_prev = score_new
            
            ep_obs.append(obs)
            ep_actions.append(act_dict)
            ep_rewards.append(reward)
        
        observations.append(ep_obs)
        actions.append(ep_actions)
        rewards.append(ep_rewards)
        episode_lengths.append(len(ep_obs))
    
    return observations, actions, rewards, episode_lengths

obs_all, acts_all, rews_all, ep_lens = extract_trajectories(demos)
print(f"Total steps: {sum(ep_lens)}")
print(f"Avg episode length: {np.mean(ep_lens):.1f}")
```

### Encoding Actions for Neural Networks

```python
ACTION_PRIMITIVES = ["brace", "stabilize", "push", "drag", "pivot", 
                     "roll", "lift_edge", "tap", "wait", "finish"]
FINGERS = ["thumb", "index", "middle", "ring", "pinky", "palm", "wrist"]

def encode_action(action_dict):
    """Encode action to numerical format."""
    primitive_idx = ACTION_PRIMITIVES.index(action_dict['primitive'])
    finger_idx = FINGERS.index(action_dict['finger'])
    force = action_dict['force']  # [0, 1]
    direction = action_dict['direction']  # [x, y]
    
    return {
        'primitive': primitive_idx,
        'finger': finger_idx,
        'force': np.array([force], dtype=np.float32),
        'direction': np.array(direction, dtype=np.float32),
    }

def decode_action(encoded):
    """Decode numerical action back to dict format."""
    return {
        'primitive': ACTION_PRIMITIVES[encoded['primitive']],
        'finger': FINGERS[encoded['finger']],
        'force': float(encoded['force'][0]),
        'direction': encoded['direction'].tolist(),
    }

# Example
action = acts_all[0][0]
encoded = encode_action(action)
print(f"Original: {action}")
print(f"Encoded: {encoded}")
print(f"Decoded: {decode_action(encoded)}")
```

---

## Using the Gymnasium Environment

### Basic Environment Setup

```python
from simulator.gymnasium_env import DexSimGymEnv, make_env
import numpy as np

# Create environment
env = make_env(
    task_file="tasks/valid_tasks.jsonl",
    task_index=0,
    variant_index=0,
    observation_mode="compact"
)

# Reset for new episode
obs, info = env.reset(seed=42)

print(f"Task ID: {info['task_info']['id']}")
print(f"Observation keys: {obs.keys()}")
print(f"Observation shapes:")
for key, val in obs.items():
    if isinstance(val, np.ndarray):
        print(f"  {key}: {val.shape}")

# Take one step
action = env.sample_action()
print(f"\nAction: {action}")

obs, reward, terminated, truncated, info = env.step(action)
print(f"Reward: {reward}")
print(f"Metrics: {info['metrics']}")
print(f"Done: {terminated or truncated}")
```

### Running Episodes

```python
def run_episode(env, policy_fn, max_steps=128):
    """Run one episode with given policy."""
    obs, info = env.reset()
    episode_data = {
        'task_id': info['task_id'],
        'steps': [],
        'total_reward': 0.0,
        'success': False,
    }
    
    for step_idx in range(max_steps):
        # Get action from policy
        action = policy_fn(obs, info)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        episode_data['steps'].append({
            'observation': obs,
            'action': action,
            'reward': reward,
            'metrics': info['metrics'],
        })
        episode_data['total_reward'] += reward
        
        if info['metrics'].get('success'):
            episode_data['success'] = True
            break
        
        if terminated or truncated:
            break
    
    return episode_data

# Example: random policy
def random_policy(obs, info):
    import random
    import math
    angle = random.uniform(-3.14159, 3.14159)
    return {
        'primitive': random.choice(['push', 'drag', 'pivot', 'stabilize', 'tap', 'wait']),
        'finger': random.choice(['thumb', 'index', 'middle', 'ring', 'pinky', 'palm', 'wrist']),
        'force': random.uniform(0.15, 0.75),
        'direction': [math.cos(angle), math.sin(angle)],
    }

env = make_env(task_file="tasks/valid_tasks.jsonl")
episode = run_episode(env, random_policy)
print(f"Episode length: {len(episode['steps'])}")
print(f"Total reward: {episode['total_reward']:.3f}")
print(f"Success: {episode['success']}")
```

---

## Building a Behavioral Cloning Model

### Simple Architecture

```python
import torch
import torch.nn as nn

class DexSimPolicy(nn.Module):
    """Simple multi-head policy for NS-2026-07."""
    
    def __init__(self, 
                 low_dim_size=14,
                 tactile_size=28,  # 7x4
                 hidden_size=256,
                 num_primitives=10,
                 num_fingers=7):
        super().__init__()
        
        # Low-dimensional state processing
        self.low_dim_net = nn.Sequential(
            nn.Linear(low_dim_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
        )
        
        # Tactile heatmap processing
        self.tactile_net = nn.Sequential(
            nn.Conv1d(7, 32, kernel_size=2),  # 7 rows, small conv
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, hidden_size // 2),
            nn.ReLU(),
        )
        
        # Fusion layer
        fusion_input = (hidden_size // 2) + (hidden_size // 2)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        
        # Discrete action heads
        self.primitive_head = nn.Linear(hidden_size, num_primitives)
        self.finger_head = nn.Linear(hidden_size, num_fingers)
        
        # Continuous action heads
        self.force_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # Force in [0, 1]
        )
        
        self.direction_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )
    
    def forward(self, low_dim, tactile):
        """
        Args:
            low_dim: (batch, 14)
            tactile: (batch, 7, 4)
        
        Returns:
            dict with action components
        """
        # Process observations
        low_dim_feat = self.low_dim_net(low_dim)
        tactile_feat = self.tactile_net(tactile)
        
        # Fuse
        combined = torch.cat([low_dim_feat, tactile_feat], dim=1)
        fused = self.fusion(combined)
        
        # Decode actions
        primitive_logits = self.primitive_head(fused)
        finger_logits = self.finger_head(fused)
        force = self.force_head(fused)
        direction = self.direction_head(fused)
        
        return {
            'primitive_logits': primitive_logits,
            'finger_logits': finger_logits,
            'force': force,
            'direction': direction,
        }

# Test forward pass
model = DexSimPolicy()
low_dim = torch.randn(4, 14)  # batch=4
tactile = torch.randn(4, 7, 4)
out = model(low_dim, tactile)
print(f"Primitive logits shape: {out['primitive_logits'].shape}")
print(f"Force shape: {out['force'].shape}")
print(f"Direction shape: {out['direction'].shape}")
```

### Training Loop

```python
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

def prepare_dataset(observations, actions, batch_size=32):
    """Prepare PyTorch dataset from demonstrations."""
    low_dims = []
    tactiles = []
    primitives = []
    fingers = []
    forces = []
    directions = []
    
    for ep_obs, ep_acts in zip(observations, actions):
        for obs, act in zip(ep_obs, ep_acts):
            low_dims.append(obs['low_dim'])
            tactiles.append(obs['tactile'])
            
            # Encode actions
            primitives.append(ACTION_PRIMITIVES.index(act['primitive']))
            fingers.append(FINGERS.index(act['finger']))
            forces.append(act['force'])
            directions.append(act['direction'])
    
    # Convert to tensors
    low_dims = torch.stack([torch.from_numpy(x) for x in low_dims])
    tactiles = torch.stack([torch.from_numpy(x) for x in tactiles])
    primitives = torch.tensor(primitives, dtype=torch.long)
    fingers = torch.tensor(fingers, dtype=torch.long)
    forces = torch.stack([torch.tensor([x]) for x in forces])
    directions = torch.stack([torch.from_numpy(x) for x in directions])
    
    dataset = TensorDataset(low_dims, tactiles, primitives, fingers, forces, directions)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for low_dim, tactile, prim, finger, force, direction in dataloader:
        low_dim = low_dim.to(device)
        tactile = tactile.to(device)
        prim = prim.to(device)
        finger = finger.to(device)
        force = force.to(device)
        direction = direction.to(device)
        
        # Forward pass
        output = model(low_dim, tactile)
        
        # Loss
        loss_prim = F.cross_entropy(output['primitive_logits'], prim)
        loss_finger = F.cross_entropy(output['finger_logits'], finger)
        loss_force = F.mse_loss(output['force'], force)
        
        # Direction: normalize before loss
        direction_norm = F.normalize(output['direction'], p=2, dim=1)
        direction_true = F.normalize(direction, p=2, dim=1)
        loss_direction = F.mse_loss(direction_norm, direction_true)
        
        loss = loss_prim + loss_finger + 0.5 * loss_force + 0.5 * loss_direction
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

# Training example
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DexSimPolicy().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Load data
train_loader = prepare_dataset(obs_all, acts_all, batch_size=32)

# Train
for epoch in range(10):
    loss = train_epoch(model, train_loader, optimizer, device)
    print(f"Epoch {epoch+1}: Loss = {loss:.4f}")
```

---

## Action Decoding Strategies

### Greedy Decoding

```python
def greedy_decode(model_output):
    """Greedy action decoding."""
    primitive_idx = torch.argmax(model_output['primitive_logits'], dim=1)
    finger_idx = torch.argmax(model_output['finger_logits'], dim=1)
    force = model_output['force'].squeeze(1)  # Already [0,1]
    direction = F.normalize(model_output['direction'], p=2, dim=1)
    
    return {
        'primitive': [ACTION_PRIMITIVES[p.item()] for p in primitive_idx],
        'finger': [FINGERS[f.item()] for f in finger_idx],
        'force': force.detach().cpu().numpy(),
        'direction': direction.detach().cpu().numpy(),
    }
```

### Temperature-Based Sampling

```python
def sample_decode(model_output, temperature=1.0):
    """Sample actions with temperature."""
    # Discrete sampling
    prim_dist = torch.distributions.Categorical(
        logits=model_output['primitive_logits'] / temperature
    )
    finger_dist = torch.distributions.Categorical(
        logits=model_output['finger_logits'] / temperature
    )
    
    primitive_idx = prim_dist.sample()
    finger_idx = finger_dist.sample()
    
    # Continuous
    force = model_output['force'].squeeze(1)
    direction = F.normalize(model_output['direction'], p=2, dim=1)
    
    return {
        'primitive': [ACTION_PRIMITIVES[p.item()] for p in primitive_idx],
        'finger': [FINGERS[f.item()] for f in finger_idx],
        'force': force.detach().cpu().numpy(),
        'direction': direction.detach().cpu().numpy(),
    }
```

---

## Interfacing with the Gym Environment

### Policy Wrapper

```python
class NeuralNetworkPolicy:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model = DexSimPolicy().to(device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
    
    def act(self, observation):
        """Convert Gym observation to action."""
        with torch.no_grad():
            low_dim = torch.from_numpy(observation['low_dim']).unsqueeze(0).to(self.device)
            tactile = torch.from_numpy(observation['tactile_heatmap']).unsqueeze(0).to(self.device)
            
            output = self.model(low_dim, tactile)
            
            # Decode (batch size 1)
            prim = ACTION_PRIMITIVES[output['primitive_logits'].argmax(1).item()]
            finger = FINGERS[output['finger_logits'].argmax(1).item()]
            force = output['force'].squeeze().item()
            direction = output['direction'].squeeze(0).cpu().numpy()
        
        return {
            'primitive': prim,
            'finger': finger,
            'force': float(force),
            'direction': direction.tolist(),
        }

# Usage
policy = NeuralNetworkPolicy('model_weights.pt')
env = make_env(task_file="tasks/valid_tasks.jsonl")

obs, info = env.reset()
for _ in range(128):
    action = policy.act(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

print(f"Success: {info['metrics']['success']}")
```

---

## Evaluation on Validation Tasks

```python
def evaluate_policy(policy, task_file, num_tasks=10, max_steps=128):
    """Evaluate policy on multiple tasks."""
    env = make_env(task_file=task_file)
    
    results = {
        'scores': [],
        'successes': [],
        'episode_lengths': [],
        'by_task_type': {}
    }
    
    for task_idx in range(num_tasks):
        obs, info = env.reset(options={'task_index': task_idx})
        task_type = info['task_info'].get('task_type', 'unknown')
        
        total_reward = 0.0
        success = False
        ep_length = 0
        
        for step in range(max_steps):
            action = policy.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            ep_length += 1
            
            if info['metrics']['success']:
                success = True
                break
            if terminated or truncated:
                break
        
        results['scores'].append(total_reward)
        results['successes'].append(success)
        results['episode_lengths'].append(ep_length)
        
        if task_type not in results['by_task_type']:
            results['by_task_type'][task_type] = {'scores': [], 'successes': []}
        results['by_task_type'][task_type]['scores'].append(total_reward)
        results['by_task_type'][task_type]['successes'].append(success)
    
    # Summary stats
    print(f"Average score: {np.mean(results['scores']):.3f}")
    print(f"Success rate: {np.mean(results['successes']):.1%}")
    print(f"Avg episode length: {np.mean(results['episode_lengths']):.1f}")
    
    for task_type, metrics in results['by_task_type'].items():
        print(f"\n{task_type}:")
        print(f"  Score: {np.mean(metrics['scores']):.3f}")
        print(f"  Success: {np.mean(metrics['successes']):.1%}")
    
    return results

# Evaluate
policy = NeuralNetworkPolicy('model_weights.pt')
eval_results = evaluate_policy(policy, "tasks/valid_tasks.jsonl", num_tasks=36)
```

