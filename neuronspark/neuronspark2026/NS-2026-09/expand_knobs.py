"""Try larger profile search to handle 'held-out compositions'."""
import json, itertools, sys, subprocess
from collections import Counter
sys.path.insert(0, '.')
from sim2 import simulate, EVENT_KEYS, DEFAULT_PROFILE
import predict_v3 as pv

def find_pos(grid, ch):
    for y, r in enumerate(grid):
        x = r.find(ch)
        if x >= 0: return (x,y)
    return None

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

# Try expanded knob spaces and check oracle agent_pos + per-step picker accuracy
KNOB_SPACES = {
    '16_base': {
        'ice_mode': ['classic', 'slide_one'],
        'portal_pair': [True, False],
        'box_walk': [False, True],
        'orb_blocks': [False, True],
    },
    '32_arrow': {
        'ice_mode': ['classic', 'slide_one'],
        'portal_pair': [True, False],
        'box_walk': [False, True],
        'orb_blocks': [False, True],
        'arrow_mode': ['pass', 'redirect'],
    },
    '64_arrow_kills': {
        'ice_mode': ['classic', 'slide_one'],
        'portal_pair': [True, False],
        'box_walk': [False, True],
        'orb_blocks': [False, True],
        'arrow_mode': ['pass', 'redirect'],
        'hazard_kills': [False, True],
    },
}

def candidate_profiles_K(KNOBS):
    keys = list(KNOBS.keys())
    for vals in itertools.product(*[KNOBS[k] for k in keys]):
        p = {**DEFAULT_PROFILE}
        for k, v in zip(KNOBS, vals): p[k] = v
        yield p

for name, KNOBS in KNOB_SPACES.items():
    n_total = 0; n_oracle = 0; n_picked = 0; n_picked_ps = 0
    n_profiles = sum(1 for _ in candidate_profiles_K(KNOBS))
    with open('valid.jsonl') as f:
        for line in f:
            d = json.loads(line)
            # Picker: per-step
            best_ps = -1; best_p = None
            best_agg = -1; agg_p = None
            for p in candidate_profiles_K(KNOBS):
                ps = per_step_score(d, p)
                if ps > best_ps:
                    best_ps = ps; best_p = p
                agg = sum(pv.score_probe(simulate(d['initial_full_grid'], ep['actions'], p), ep) for ep in d['context_episodes'])
                if agg > best_agg:
                    best_agg = agg; agg_p = p
            for q in d['queries']:
                real_a = find_pos(q['label']['final_grid'], 'A')
                n_total += 1
                # Oracle
                for p in candidate_profiles_K(KNOBS):
                    rr = simulate(q['initial_full_grid'], q['future_actions'], p)
                    if rr['agent_pos'] == real_a:
                        n_oracle += 1; break
                r1 = simulate(q['initial_full_grid'], q['future_actions'], agg_p)
                if r1['agent_pos'] == real_a: n_picked += 1
                r2 = simulate(q['initial_full_grid'], q['future_actions'], best_p)
                if r2['agent_pos'] == real_a: n_picked_ps += 1
    print(f'{name} ({n_profiles}): oracle={n_oracle}/{n_total}, agg_picked={n_picked}/{n_total}, ps_picked={n_picked_ps}/{n_total}')
