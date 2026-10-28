"""Use aggregate picker for everything but per-step for agent_pos specifically."""
import json, sys, time, pickle, os, subprocess
from collections import Counter
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

# Cache per-step profile per context
ps_cache = {}
def best_per_step_profile(d):
    cid = id(d)
    if cid in ps_cache: return ps_cache[cid]
    best_ps = -1; best_p = None
    for p in pv.candidate_profiles():
        ps = per_step_score(d, p)
        if ps > best_ps:
            best_ps = ps; best_p = p
    ps_cache[cid] = best_p
    return best_p

# Override predict_query to use per-step profile for agent_pos
orig_predict_query = pv.predict_query
def predict_query_perstep_pos(q, profile, priors, eps):
    # Get original prediction
    pred = orig_predict_query(q, profile, priors, eps)
    # Get per-step profile (must be passed in priors? no, use closure on d)
    # Actually we need d here. Hack: store d as a global
    d = current_d
    ps_profile = best_per_step_profile(d)
    if ps_profile != profile:
        # Re-simulate with per-step profile
        r_ps = simulate(q['initial_full_grid'], q['future_actions'], ps_profile)
        # Override A position
        pred_a = r_ps['agent_pos']
        # Update final_grid: remove old A, place new A
        grid = [list(row) for row in pred['final_grid']]
        # Remove old A
        for y in range(12):
            for x in range(12):
                if grid[y][x] == 'A':
                    grid[y][x] = '.'
        # Place new A
        if pred_a is not None and 0 <= pred_a[0] < 12 and 0 <= pred_a[1] < 12:
            ax, ay = pred_a
            if grid[ay][ax] in '.GP^v<>IH':
                grid[ay][ax] = 'A'
        pred['final_grid'] = [''.join(r) for r in grid]
    return pred

pv.predict_query = predict_query_perstep_pos

results = []
current_d = None
with open('valid.jsonl') as f:
    for line in f:
        d = json.loads(line)
        current_d = d
        profile, _ = pv.best_profile_for(d)
        priors = pv.context_priors(d)
        for q in d['queries']:
            results.append(pv.predict_query(q, profile, priors, d['context_episodes']))
with open('results_hybrid.json','w') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
out = subprocess.run(['python','score.py','results_hybrid.json','valid.jsonl'], capture_output=True, text=True).stdout
print(out)
