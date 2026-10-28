"""Oracle: for each query, what fraction of profiles give correct agent_pos?
And how often does ANY profile give correct agent_pos?
"""
import json, sys
from collections import Counter
sys.path.insert(0, '.')
from sim2 import simulate, EVENT_KEYS, DEFAULT_PROFILE
from predict_v3 import candidate_profiles, best_profile_for

def find_pos(grid, ch):
    for y, r in enumerate(grid):
        x = r.find(ch)
        if x >= 0: return (x,y)
    return None

n_total = 0
n_oracle = 0  # any profile correct
n_picked = 0  # picked profile correct
n_oracle_no_picker = 0  # oracle correct but picker wrong

with open('valid.jsonl') as f:
    for line in f:
        d = json.loads(line)
        picked_p, _ = best_profile_for(d)
        for q in d['queries']:
            real_a = find_pos(q['label']['final_grid'], 'A')
            n_total += 1
            # Oracle: any profile?
            any_correct = False
            picked_correct = False
            for p in candidate_profiles():
                r = simulate(q['initial_full_grid'], q['future_actions'], p)
                if r['agent_pos'] == real_a:
                    any_correct = True
                if p == picked_p:
                    picked_correct = (r['agent_pos'] == real_a)
            if any_correct: n_oracle += 1
            if picked_correct: n_picked += 1
            if any_correct and not picked_correct: n_oracle_no_picker += 1

print(f'Total queries: {n_total}')
print(f'Picked profile correct: {n_picked}/{n_total} = {n_picked/n_total:.4f}')
print(f'Oracle any-profile correct: {n_oracle}/{n_total} = {n_oracle/n_total:.4f}')
print(f'Oracle better than picker: {n_oracle_no_picker}/{n_total}')
