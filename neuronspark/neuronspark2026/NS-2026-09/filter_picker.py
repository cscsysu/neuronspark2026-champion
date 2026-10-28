"""Hybrid picker: filter to top-half by per-step, then aggregate score among them."""
import json, sys, time, subprocess
from collections import Counter
sys.path.insert(0, '.')
from sim2 import simulate, EVENT_KEYS, DEFAULT_PROFILE
import predict_v3 as pv

def per_step_score(d, p):
    total = 0; correct = 0
    for ep in d['context_episodes']:
        probe_events_at_step = []
        for o in ep.get('observations', []):
            probe_events_at_step.append(set(o.get('sensor', {}).get('delta_events', [])))
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

def aggregate_score(d, p):
    s = 0
    for ep in d['context_episodes']:
        r = simulate(d['initial_full_grid'], ep['actions'], p)
        s += pv.score_probe(r, ep)
    return s

def find_pos(grid, ch):
    for y, r in enumerate(grid):
        x = r.find(ch)
        if x >= 0: return (x,y)
    return None

# Test: filter to top per-step, then pick best aggregate
def best_filter_picker(d, top_n_ps=8):
    ps_scores = [(per_step_score(d, p), p) for p in pv.candidate_profiles()]
    ps_scores.sort(key=lambda x: -x[0])
    candidates = [p for _, p in ps_scores[:top_n_ps]]
    best_p = None; best_agg = -1
    for p in candidates:
        agg = aggregate_score(d, p)
        if agg > best_agg:
            best_agg = agg; best_p = p
    return best_p

# Test multiple top_n
for top_n in [4, 6, 8, 10, 12, 16]:
    n_correct = 0; n_total = 0
    with open('valid.jsonl') as f:
        for line in f:
            d = json.loads(line)
            top_p = best_filter_picker(d, top_n)
            for q in d['queries']:
                real_a = find_pos(q['label']['final_grid'], 'A')
                n_total += 1
                r = simulate(q['initial_full_grid'], q['future_actions'], top_p)
                if r['agent_pos'] == real_a: n_correct += 1
    print(f'top_n={top_n}: agent_pos={n_correct}/{n_total} = {n_correct/n_total:.4f}')
