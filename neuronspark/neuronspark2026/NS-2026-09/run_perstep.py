"""Build full predict using per-step picker. Compare full score."""
import json, sys, time, pickle, os
from collections import Counter
from functools import cmp_to_key
sys.path.insert(0, '.')
from sim2 import simulate, EVENT_KEYS, DEFAULT_PROFILE
import predict_v3 as pv

def per_step_score(d, p):
    total = 0; correct = 0
    for ep in d['context_episodes']:
        probe_events_at_step = []
        for o in ep['observations']:
            probe_events_at_step.append(set(o['sensor'].get('delta_events', [])))
        actions = ep['actions']
        r = simulate(d['initial_full_grid'], actions, p)
        sim_events_at_step = {step: set() for step in range(len(actions))}
        for k in EVENT_KEYS:
            fs = r['first_step'][k]
            if fs is not None and fs in sim_events_at_step:
                sim_events_at_step[fs].add(k)
        for step in range(len(actions)):
            probe_at = probe_events_at_step[step+1] if step+1 < len(probe_events_at_step) else set()
            sim_at = sim_events_at_step[step]
            if probe_at == sim_at: correct += 1
            total += 1
    return correct

def best_profile_per_step(d):
    """Use per_step as primary, aggregate as tiebreaker (numeric, not enforced)."""
    best_ps = -1; best_p = None
    for p in pv.candidate_profiles():
        ps = per_step_score(d, p)
        if ps > best_ps:
            best_ps = ps; best_p = p
    return best_p

# Patch best_profile_for to use per-step
original_best_profile_for = pv.best_profile_for
def best_profile_for_perstep(d):
    p = best_profile_per_step(d)
    return p, 0

pv.best_profile_for = best_profile_for_perstep

# Run
results = []
with open('valid.jsonl') as f:
    for line in f:
        d = json.loads(line)
        profile, _ = pv.best_profile_for(d)
        priors = pv.context_priors(d)
        for q in d['queries']:
            results.append(pv.predict_query(q, profile, priors, d['context_episodes']))
with open('results_perstep.json','w') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
import subprocess
out = subprocess.run(['python','score.py','results_perstep.json','valid.jsonl'], capture_output=True, text=True).stdout
print(out)
