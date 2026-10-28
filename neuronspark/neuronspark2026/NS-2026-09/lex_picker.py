"""Per-step primary + aggregate tiebreaker, vs lexicographic ranking."""
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

# Strategy: lexicographic - per_step first, then aggregate
def best_profile_lex(d):
    scores = []
    for p in candidate_profiles():
        agg = aggregate_score(d, p)
        ps = per_step_score(d, p)
        scores.append((ps, agg, p))
    scores.sort(key=lambda x: (-x[0], -x[1]))
    return scores[0][2]

def best_profile_agg_first(d):
    scores = []
    for p in candidate_profiles():
        agg = aggregate_score(d, p)
        ps = per_step_score(d, p)
        scores.append((agg, ps, p))
    scores.sort(key=lambda x: (-x[0], -x[1]))
    return scores[0][2]

n_total = 0; n_lex = 0; n_agg_lex = 0
with open('valid.jsonl') as f:
    for line in f:
        d = json.loads(line)
        p_lex = best_profile_lex(d)
        p_agg = best_profile_agg_first(d)
        for q in d['queries']:
            real_a = find_pos(q['label']['final_grid'], 'A')
            n_total += 1
            r1 = simulate(q['initial_full_grid'], q['future_actions'], p_lex)
            if r1['agent_pos'] == real_a: n_lex += 1
            r2 = simulate(q['initial_full_grid'], q['future_actions'], p_agg)
            if r2['agent_pos'] == real_a: n_agg_lex += 1
print(f'Per-step then agg: {n_lex}/{n_total} = {n_lex/n_total:.4f}')
print(f'Agg then per-step: {n_agg_lex}/{n_total} = {n_agg_lex/n_total:.4f}')
