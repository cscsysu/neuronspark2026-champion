"""Inspect the 35 cases where some profile gives correct A but picker fails."""
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

# For each query where picker fails but oracle succeeds, what's the right knob?
right_knob_diffs = Counter()
n = 0
with open('valid.jsonl') as f:
    for line in f:
        d = json.loads(line)
        picked_p, _ = best_profile_for(d)
        for q in d['queries']:
            real_a = find_pos(q['label']['final_grid'], 'A')
            r = simulate(q['initial_full_grid'], q['future_actions'], picked_p)
            picked_correct = (r['agent_pos'] == real_a)
            if picked_correct: continue
            # Find any profile that's correct
            for p in candidate_profiles():
                rr = simulate(q['initial_full_grid'], q['future_actions'], p)
                if rr['agent_pos'] == real_a:
                    n += 1
                    diffs = []
                    for k in ['ice_mode','portal_pair','box_walk','orb_blocks']:
                        if p[k] != picked_p[k]:
                            diffs.append((k, picked_p[k], p[k]))
                    right_knob_diffs[tuple(sorted([d[0] for d in diffs]))] += 1
                    break

print(f'Total picker-fail-oracle-success: {n}')
print('Knob differences pattern:')
for k, v in right_knob_diffs.most_common():
    print(f'  {k}: {v}')
