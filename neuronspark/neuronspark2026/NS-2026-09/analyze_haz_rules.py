"""Test hazard-event rules: combine sim + probe count + path-near-H."""
import json, sys
from collections import Counter
sys.path.insert(0, '.')
from sim2 import simulate, EVENT_KEYS, DEFAULT_PROFILE, DIRS
from predict_v3 import best_profile_for, context_priors, find_pos, find_all

with open('results_v25.json') as f:
    preds = {p['id']: p for p in json.load(f)}

# Test rules:
# Rule A: current (sim_fired or pec >= 5)
# Rule B: A or (path_near_H and pec >= 3)
# Rule C: A or (path_near_H and pec >= 2)

def get_path(q):
    """Naive path ignoring obstacles."""
    a = find_pos(q['initial_full_grid'], 'A')
    if a is None: return []
    cx, cy = a; path = [(cx, cy)]
    for ac in q['future_actions']:
        if ac in DIRS:
            dx, dy = DIRS[ac]
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < 12 and 0 <= ny < 12:
                cx, cy = nx, ny
        path.append((cx, cy))
    return path

def adj(p1, p2):
    return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1]) <= 1

def near_H(q, near=False):
    """Check if any H is on path (or near path)."""
    path = get_path(q)
    h_positions = []
    for y, r in enumerate(q['initial_full_grid']):
        for x, c in enumerate(r):
            if c == 'H': h_positions.append((x,y))
    if near:
        return any(adj(p, h) for p in path for h in h_positions)
    else:
        return any(p in h_positions for p in path)

def evaluate_rule(rule_name, rule_fn):
    n_correct = 0; n_total = 0
    n_fp = 0; n_fn = 0
    with open('valid.jsonl') as f:
        for line in f:
            d = json.loads(line)
            n_haz = sum(1 for ep in d['context_episodes'] if ep['observed_final_events']['hazard'])
            for q in d['queries']:
                lbl = q['label']
                pred = preds.get(q['query_id'])
                if pred is None: continue
                # Compute new hazard prediction
                new_haz = rule_fn(q, d, pred, n_haz)
                if new_haz == lbl['events']['hazard']:
                    n_correct += 1
                else:
                    if new_haz: n_fp += 1
                    else: n_fn += 1
                n_total += 1
    print(f'{rule_name}: {n_correct}/{n_total} ({n_correct/n_total:.4f}), FN={n_fn}, FP={n_fp}')

# Current rule (baseline)
evaluate_rule('Current rule', lambda q, d, pred, nh: pred['events']['hazard'])

# Rule: + path_in_H AND nh>=2
def rule_b(q, d, pred, nh):
    return pred['events']['hazard'] or (near_H(q) and nh >= 3)
evaluate_rule('A or (path_in_H AND nh>=3)', rule_b)

def rule_c(q, d, pred, nh):
    return pred['events']['hazard'] or (near_H(q, near=True) and nh >= 3)
evaluate_rule('A or (path_near_H AND nh>=3)', rule_c)

def rule_d(q, d, pred, nh):
    return pred['events']['hazard'] or (near_H(q) and nh >= 4)
evaluate_rule('A or (path_in_H AND nh>=4)', rule_d)

# also evaluate nh>=4 with no path constraint (i.e., full event prior)
def rule_e(q, d, pred, nh):
    return pred['events']['hazard'] or nh >= 4
evaluate_rule('A or nh>=4', rule_e)

def rule_f(q, d, pred, nh):
    return pred['events']['hazard'] or nh >= 3
evaluate_rule('A or nh>=3', rule_f)
