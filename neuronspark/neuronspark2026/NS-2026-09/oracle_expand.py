"""Check if hazard_kills/goal_kills knobs improve oracle."""
import json, itertools, sys
from collections import Counter
sys.path.insert(0, '.')
from sim2 import simulate, EVENT_KEYS, DEFAULT_PROFILE
from predict_v3 import score_probe

def find_pos(grid, ch):
    for y, r in enumerate(grid):
        x = r.find(ch)
        if x >= 0: return (x,y)
    return None

KNOBS_BASE = {
    'ice_mode': ['classic', 'slide_one'],
    'portal_pair': [True, False],
    'box_walk': [False, True],
    'orb_blocks': [False, True],
}

KNOBS_KILLS = {
    **KNOBS_BASE,
    'hazard_kills': [False, True],
    'goal_kills': [False, True],
}

KNOBS_DOOR = {
    **KNOBS_BASE,
    'door_no_block': [False, True],
}

KNOBS_ALL = {
    **KNOBS_BASE,
    'hazard_kills': [False, True],
    'goal_kills': [False, True],
    'door_no_block': [False, True],
}

def candidate_profiles_K(KNOBS):
    keys = list(KNOBS.keys())
    for vals in itertools.product(*[KNOBS[k] for k in keys]):
        p = {**DEFAULT_PROFILE}
        for k, v in zip(KNOBS, vals): p[k] = v
        yield p

# Compare oracle ceilings
for name, KNOBS in [('base16', KNOBS_BASE), ('kills64', KNOBS_KILLS), ('door32', KNOBS_DOOR), ('all128', KNOBS_ALL)]:
    n_total = 0; n_oracle_a = 0; n_oracle_evset = 0; n_oracle_grid = 0
    n_profiles = sum(1 for _ in candidate_profiles_K(KNOBS))
    with open('valid.jsonl') as f:
        for line in f:
            d = json.loads(line)
            for q in d['queries']:
                lbl = q['label']
                real_a = find_pos(lbl['final_grid'], 'A')
                n_total += 1
                any_a = False; any_evset = False; any_grid = False
                lbl_evset = frozenset(k for k in EVENT_KEYS if lbl['events'][k])
                for p in candidate_profiles_K(KNOBS):
                    r = simulate(q['initial_full_grid'], q['future_actions'], p)
                    if r['agent_pos'] == real_a: any_a = True
                    pred_evset = frozenset(k for k in EVENT_KEYS if r['events'][k])
                    if pred_evset == lbl_evset: any_evset = True
                if any_a: n_oracle_a += 1
                if any_evset: n_oracle_evset += 1
    print(f'{name} ({n_profiles} profiles): oracle_a={n_oracle_a}/{n_total}, oracle_evset={n_oracle_evset}/{n_total}')
