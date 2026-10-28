"""Better per-step scoring with weighted matches."""
import json, sys, time, subprocess
from collections import Counter
sys.path.insert(0, '.')
from sim2 import simulate, EVENT_KEYS, DEFAULT_PROFILE
import predict_v3 as pv

def per_step_score_v2(d, p, weight_correct=1, weight_wrong=0):
    """Score: +weight_correct for matching events, -weight_wrong for mismatches."""
    score = 0
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
            # Per-event scoring instead of all-or-nothing
            for ev in EVENT_KEYS:
                if (ev in probe_at) == (ev in sim_at):
                    score += weight_correct
                else:
                    score -= weight_wrong
    return score

def find_pos(grid, ch):
    for y, r in enumerate(grid):
        x = r.find(ch)
        if x >= 0: return (x,y)
    return None

best_score_a = 0
best_setting = None
for w_correct, w_wrong in [(1, 0), (1, 0.5), (1, 1), (1, 2), (2, 1)]:
    n_correct = 0; n_total = 0
    with open('valid.jsonl') as f:
        for line in f:
            d = json.loads(line)
            top_p = max(pv.candidate_profiles(), key=lambda p: per_step_score_v2(d, p, w_correct, w_wrong))
            for q in d['queries']:
                real_a = find_pos(q['label']['final_grid'], 'A')
                n_total += 1
                r = simulate(q['initial_full_grid'], q['future_actions'], top_p)
                if r['agent_pos'] == real_a: n_correct += 1
    print(f'wc={w_correct}, ww={w_wrong}: agent_pos={n_correct}/{n_total} = {n_correct/n_total:.4f}')
    if n_correct > best_score_a:
        best_score_a = n_correct
        best_setting = (w_correct, w_wrong)
print(f'Best agent_pos setting: {best_setting} = {best_score_a}/320')
