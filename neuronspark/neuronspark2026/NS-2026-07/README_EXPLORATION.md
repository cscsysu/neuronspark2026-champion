# NS-2026-07 Complete Data Exploration

This directory now contains **comprehensive documentation** of the NS-2026-07 training data and environment.

## 📚 Documentation Files

### Start Here 👈
- **`QUICK_REFERENCE.md`** - 1-page cheat sheet with all key information
  - Episode structure diagram
  - Observation/action tables
  - Dataset stats
  - Timeouts and limits
  - Training checklist

### For Deep Understanding
- **`TRAINING_DATA_GUIDE.md`** - Complete reference manual (507 lines)
  - Full episode structure details
  - Observation space breakdown with examples
  - Action space specifications
  - Gymnasium wrapper API documentation
  - Training recommendations

### For Implementation
- **`CODE_EXAMPLES.md`** - Production-ready code patterns (578 lines)
  - Data loading
  - Trajectory extraction
  - Neural network models
  - Training loops
  - Evaluation harness

### For Planning
- **`EXPLORATION_SUMMARY.md`** - Complete exploration report
  - All findings and recommendations
  - Phased training plan
  - FAQ and troubleshooting
  - Next steps checklist

---

## 🎯 Quick Facts

| Aspect | Details |
|--------|---------|
| **Training Data** | 1,200 episodes (~154K timesteps) |
| **Success Rate** | 9.6% (intentionally weak) |
| **Observation** | 14 dims + 4 contact + 7×4 tactile |
| **Actions** | 10 primitives + 7 fingers + force + direction |
| **Model Limit** | 12 GB |
| **Per-Step Timeout** | 2 seconds ⚠️ CRITICAL |
| **Recommended Approach** | Behavioral cloning → RL fine-tuning |

---

## 🚀 Getting Started

### 1. Understanding (30 minutes)
```bash
# Read the quick reference
cat QUICK_REFERENCE.md
```

### 2. Deep Dive (2 hours)
```bash
# Study the training data guide
cat TRAINING_DATA_GUIDE.md
```

### 3. Implementation (Start developing)
```bash
# Use code examples as templates
cat CODE_EXAMPLES.md
```

### 4. Execution
```bash
# Follow the phased plan in EXPLORATION_SUMMARY.md
# Phase 1: Behavioral cloning
# Phase 2: Evaluation
# Phase 3: RL fine-tuning (optional)
# Phase 4: Packaging & submission
```

---

## 📊 Data Overview

### Episode Structure
```
{
  task_id, variant_id, task_type,
  object_nominal: {shape, family, mass, surface, fragility},
  environment_context: {surface, layout, friction, noise},
  
  steps: [0..127] {
    low_dim_state: [14 floats],
    contact_summary: {coverage, slip_risk, damage_risk},
    tactile_heatmap_7x4: [7×4 matrix],
    action: {primitive, finger, force, direction}
  },
  
  metrics: {score01, success, pose_score, ...}
}
```

### Observation Space
- **low_dim_state** (14): Normalized pose, velocity, progress
- **contact_summary** (4): Coverage, slip risk, damage risk
- **tactile_heatmap** (7×4): Per-finger contact forces
- **stage_context**: Planning information

### Action Space
```python
{
    'primitive': str,           # 10 options: brace, push, drag, pivot, ...
    'finger': str,              # 7 options: thumb, index, ..., palm, wrist
    'force': float,             # [0.0, 1.0]
    'direction': [float, float] # 2D unit vector
}
```

---

## 🔧 Architecture Recommendations

### Minimal Model
```python
class Policy(nn.Module):
    # Input: low_dim (14) + tactile (7×4)
    encoder_low = nn.Linear(14, 128)
    encoder_tactile = nn.Conv1d(7, 32, 2)
    fusion = nn.Linear(160, 256)
    
    # Outputs
    primitive_head = nn.Linear(256, 10)
    finger_head = nn.Linear(256, 7)
    force_head = nn.Linear(256, 1)  # sigmoid
    direction_head = nn.Linear(256, 2)
```

### Training
- Epochs: 10-20
- Batch size: 32-64
- Learning rate: 1e-3 (Adam)
- Loss: CE(primitive) + CE(finger) + 0.5×MSE(force) + 0.5×cosine(direction)

---

## ⚠️ Critical Constraints

1. **Per-step timeout: 2 seconds** ← Keep model lightweight!
2. **Model limit: 12 GB** ← Compress carefully
3. **Success rate: 9.6%** ← Don't over-fit to weak data
4. **No vectorization** → Use act_batch() for efficiency

---

## 📋 Implementation Checklist

- [ ] Read QUICK_REFERENCE.md (30 min)
- [ ] Study TRAINING_DATA_GUIDE.md (2 hours)
- [ ] Load and inspect 1,200 episodes
- [ ] Extract ~154K (observation, action) pairs
- [ ] Build multi-head fusion model
- [ ] Implement training loop
- [ ] Train BC model (2-3 epochs to validate)
- [ ] Evaluate on valid_tasks.jsonl
- [ ] (Optional) Fine-tune with RL
- [ ] Create agent.py wrapper
- [ ] Package and test submission
- [ ] Submit!

---

## 📖 Related Files

**Data**:
- `demonstrations/weak_train_rollouts.jsonl` - Training data (97 MB)
- `demonstrations/manifest.json` - Metadata
- `tasks/valid_tasks.jsonl` - Validation tasks (360)
- `tasks/test_tasks.jsonl` - Test tasks (360)

**Code**:
- `simulator/gymnasium_env.py` - Gym wrapper (222 lines)
- `tools/check_format.py` - Validate submission
- `tools/run_public_eval.py` - Evaluate agent

**Documentation**:
- `agent_api.md` - Submission API specification
- `GETTING_STARTED.md` - Setup guide
- `RUNTIME_ENVIRONMENT.md` - Official runtime contract
- `requirements-runtime.txt` - Required packages

---

## 🎯 Key Insights

1. **Weak by design** - 9.6% success is intentional; data is a warm-start not a solution
2. **Multi-modal** - Observations combine low-dim state + tactile + contact info
3. **Multi-headed** - Actions require separate branches for discrete and continuous components
4. **Speed-critical** - 2 seconds per step is tight; prioritize inference speed
5. **Batch-friendly** - RTX 4090 can handle 64-rollout batches if optimized

---

## 📞 When You Get Stuck

| Problem | Solution |
|---------|----------|
| Don't understand episode structure | → Read EXPLORATION_SUMMARY.md "Episode Life Cycle" |
| Need code example | → Check CODE_EXAMPLES.md for similar use case |
| Inference too slow | → Reduce model size, use smaller backbone |
| Model too large | → Use quantization, knowledge distillation, pruning |
| Success rate stuck | → Try RL fine-tuning, check reward computation |
| Submission format error | → Use tools/check_format.py to validate |

---

## 📊 Documentation Stats

- Total lines: 1,770+
- TRAINING_DATA_GUIDE.md: 507 lines (12 KB)
- CODE_EXAMPLES.md: 578 lines (17 KB)
- QUICK_REFERENCE.md: 201 lines (6 KB)
- EXPLORATION_SUMMARY.md: 484 lines (17 KB)

---

## ✅ Exploration Status

- ✓ Demonstration data structure fully mapped
- ✓ Observation space analyzed with statistics
- ✓ Action space documented with ranges
- ✓ Gymnasium wrapper API documented
- ✓ Runtime requirements specified
- ✓ Architecture recommendations provided
- ✓ Training strategy outlined
- ✓ Code examples implemented
- ✓ FAQ addressed
- ✓ Submission checklist created

**Ready to build your policy!**

---

*Exploration Date: May 26, 2026*  
*Created by: Data exploration task*  
*Status: Complete and documented*
