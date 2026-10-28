"""Test expanding KNOBS with door/arrow modes."""
import json, itertools, sys
from collections import Counter
sys.path.insert(0, '.')
from sim2 import simulate, EVENT_KEYS, DEFAULT_PROFILE
from predict_v3 import score_probe, predict_query, context_priors

def find_pos(grid, ch):
    for y, r in enumerate(grid):
        x = r.find(ch)
        if x >= 0: return (x,y)
    return None

# Original 16 profiles
KNOBS_BASE = {
    'ice_mode': ['classic', 'slide_one'],
    'portal_pair': [True, False],
    'box_walk': [False, True],
    'orb_blocks': [False, True],
}

# Expanded with door_no_block
KNOBS_EXP = {
    **KNOBS_BASE,
    'door_no_block': [False, True],
}

KNOBS_EXP2 = {
    **KNOBS_BASE,
    'arrow_mode': ['pass', 'redirect'],
}

def candidate_profiles(KNOBS):
    keys = list(KNOBS.keys())
    for vals in itertools.product(*[KNOBS[k] for k in keys]):
        p = {**DEFAULT_PROFILE}
        for k, v in zip(KNOBS, vals): p[k] = v
        yield p

def best_profile_for_eps(d, KNOBS):
    best_p = None; best_s = -1
    for p in candidate_profiles(KNOBS):
        s = 0.0
        for ep in d['context_episodes']:
            r = simulate(d['initial_full_grid'], ep['actions'], p)
            s += score_probe(r, ep)
        if s > best_s:
            best_s = s; best_p = p
    return best_p

# Compare oracle agent_pos under each KNOB set
for name, KNOBS in [('base', KNOBS_BASE), ('door', KNOBS_EXP), ('arrow', KNOBS_EXP2)]:
    n = sum(1 for _ in candidate_profiles(KNOBS))
    n_oracle = 0; n_picked = 0; total = 0
    with open('valid.jsonl') as f:
        for line in f:
            d = json.loads(line)
            picked_p = best_profile_for_eps(d, KNOBS)
            for q in d['queries']:
                real_a = find_pos(q['label']['final_grid'], 'A')
                total += 1
                any_correct = False
                picked_correct = False
                for p in candidate_profiles(KNOBS):
                    r = simulate(q['initial_full_grid'], q['future_actions'], p)
                    if r['agent_pos'] == real_a: any_correct = True
                    if p == picked_p:
                        picked_correct = (r['agent_pos'] == real_a)
                if any_correct: n_oracle += 1
                if picked_correct: n_picked += 1
    print(f'{name} ({n} profiles): oracle={n_oracle}/{total}, picked={n_picked}/{total}')
