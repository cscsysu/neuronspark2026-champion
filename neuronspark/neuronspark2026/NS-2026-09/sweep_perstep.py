"""Per-step as primary, with various tie-breakers and combinations."""
import json, sys, time
from collections import Counter
sys.path.insert(0, '.')
from sim2 import simulate, EVENT_KEYS, DEFAULT_PROFILE, DIRS
from predict_v3 import candidate_profiles, score_probe

def find_pos(grid, ch):
    for y, r in enumerate(grid):
        x = r.find(ch)
        if x >= 0: return (x,y)
    return None

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
            if probe_at == sim_at:
                correct += 1
            total += 1
    return correct

def aggregate_score(d, p):
    s = 0
    for ep in d['context_episodes']:
        r = simulate(d['initial_full_grid'], ep['actions'], p)
        s += score_probe(r, ep)
    return s

# Strategy: combined score = w_agg * aggregate + w_ps * per_step
# Sweep weights
n_total_dict = {}

import predict_v3
results_per_strategy = {}
for w_ps in [0, 0.1, 0.5, 1, 2, 5]:
    for w_agg in [0.5, 1, 2]:
        n_correct = 0
        n_total = 0
        with open('valid.jsonl') as f:
            for line in f:
                d = json.loads(line)
                # Score each profile
                scores = []
                for p in candidate_profiles():
                    agg = aggregate_score(d, p)
                    ps = per_step_score(d, p)
                    score = w_agg * agg + w_ps * ps
                    scores.append((score, p))
                scores.sort(key=lambda x: -x[0])
                top_p = scores[0][1]
                for q in d['queries']:
                    real_a = find_pos(q['label']['final_grid'], 'A')
                    n_total += 1
                    r = simulate(q['initial_full_grid'], q['future_actions'], top_p)
                    if r['agent_pos'] == real_a: n_correct += 1
        print(f'w_agg={w_agg}, w_ps={w_ps}: agent_pos={n_correct}/{n_total} = {n_correct/n_total:.4f}')
