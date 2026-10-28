"""Prototype: use per-step delta_events from probe observations to score profiles."""
import json, sys
from collections import Counter
sys.path.insert(0, '.')
from sim2 import simulate, EVENT_KEYS, DEFAULT_PROFILE, DIRS
from predict_v3 import candidate_profiles

def find_pos(grid, ch):
    for y, r in enumerate(grid):
        x = r.find(ch)
        if x >= 0: return (x,y)
    return None

def simulate_step_by_step(initial_grid, actions, profile):
    """Run simulator, return list of (events_at_each_step, agent_pos)."""
    # We need to instrument sim2 - or just call simulate with action prefixes
    results = []
    for i in range(len(actions)+1):
        partial_actions = actions[:i]
        r = simulate(initial_grid, partial_actions, profile)
        results.append({
            'agent_pos': r['agent_pos'],
            'events': dict(r['events']),
            'first_step': dict(r['first_step']),
        })
    return results

def score_profile_per_step(d, p):
    """Score profile based on per-step delta_events match."""
    score = 0
    n = 0
    for ep in d['context_episodes']:
        # Reconstruct probe's per-step events from delta_events
        probe_events_at = []
        for o in ep['observations']:
            probe_events_at.append(set(o['sensor'].get('delta_events', [])))
        # Simulate with profile, get per-step events
        sim_results = simulate_step_by_step(d['initial_full_grid'], ep['actions'], p)
        # Compare events at each step
        for step in range(len(ep['actions'])):
            # Sim event at step k = first_step[event] == k for any event
            sim_evs_at_step = set()
            sim_after = sim_results[step+1]['first_step']
            sim_before = sim_results[step]['first_step']
            for k in EVENT_KEYS:
                a, b = sim_before[k], sim_after[k]
                if a is None and b is not None and b == step:
                    sim_evs_at_step.add(k)
            probe_evs_at_step = probe_events_at[step+1] if step+1 < len(probe_events_at) else set()
            # Match score
            if sim_evs_at_step == probe_evs_at_step:
                score += 1
            n += 1
    return score, n

# Test: do different profiles give different per-step scores?
import time
t0 = time.time()
n_profiles_tested = 16
n_ctx_tested = 0
profile_diff = Counter()  # for each context, count profiles with distinct scores
with open('valid.jsonl') as f:
    for line in f:
        d = json.loads(line)
        scores = []
        for p in candidate_profiles():
            s, n = score_profile_per_step(d, p)
            scores.append((s, n))
        # Distinct count
        distinct = len(set(s[0] for s in scores))
        profile_diff[distinct] += 1
        n_ctx_tested += 1
        if n_ctx_tested >= 10: break

print(f'Tested {n_ctx_tested} contexts in {time.time()-t0:.1f}s')
print(f'Distinct score counts: {profile_diff}')
