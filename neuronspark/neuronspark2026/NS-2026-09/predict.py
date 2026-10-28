"""Generate results.json for NS-2026-09 by:
1) For each test context, search profile space using probe trajectories (system ID).
2) Use best profile to simulate each query and emit predictions.
"""
import json, itertools, sys, time
from sim import simulate, EVENT_KEYS, DEFAULT_PROFILE

KNOBS = {
    'hazard_step_on': [True, False],
    'ice_slide': [True, False],
    'portal_pair': [True, False],
    'door_blocks_no_key': [True, False],
    'push_box': [True, False],
    'push_orb': [True, False],
}

def candidate_profiles():
    keys = list(KNOBS.keys())
    for vals in itertools.product(*[KNOBS[k] for k in keys]):
        p = {**DEFAULT_PROFILE}
        for k, v in zip(keys, vals): p[k] = v
        yield p

def find_pos(grid, ch):
    for y, r in enumerate(grid):
        x = r.find(ch)
        if x >= 0: return (x,y)
    return None

def score_probe(sim_r, ep):
    s = 0.0
    real_a = find_pos(ep['observed_final_full_grid'], 'A')
    if real_a == sim_r['agent_pos']: s += 3.0
    if all(sim_r['events'][k] == ep['observed_final_events'][k] for k in EVENT_KEYS): s += 3.0
    # partial event credit
    s += sum(1 for k in EVENT_KEYS if sim_r['events'][k] == ep['observed_final_events'][k]) * 0.3
    if sim_r['terminal'] == ep['observed_final_terminal']: s += 2.0
    if sim_r['event_order'] == ep['observed_final_event_order']: s += 1.0
    same = tot = 0
    for r1, r2 in zip(sim_r['final_grid'], ep['observed_final_full_grid']):
        for c1, c2 in zip(r1, r2):
            tot += 1
            if c1 == c2: same += 1
    s += (same/tot) * 1.5
    # timeline
    for k in EVENT_KEYS:
        if ep['observed_final_event_timeline'][k] != 'never':
            if sim_r['event_timeline'][k] == ep['observed_final_event_timeline'][k]:
                s += 0.5
    return s

def best_profile_for(d):
    best_p = None; best_s = -1
    for p in candidate_profiles():
        s = 0.0
        for ep in d['context_episodes']:
            r = simulate(d['initial_full_grid'], ep['actions'], p)
            s += score_probe(r, ep)
        if s > best_s:
            best_s = s; best_p = p
    return best_p, best_s

def predict_query(q, profile):
    r = simulate(q['initial_full_grid'], q['future_actions'], profile)
    return {
        'id': q['query_id'],
        'final_grid': r['final_grid'],
        'events': r['events'],
        'event_timeline': r['event_timeline'],
        'event_order': r['event_order'],
        'terminal': r['terminal'],
    }

def main():
    inp = sys.argv[1] if len(sys.argv) > 1 else 'test.jsonl'
    out = sys.argv[2] if len(sys.argv) > 2 else 'results.json'
    results = []
    t0 = time.time()
    n_ctx = 0
    with open(inp) as f:
        for line in f:
            d = json.loads(line)
            profile, _ = best_profile_for(d)
            for q in d['queries']:
                results.append(predict_query(q, profile))
            n_ctx += 1
            if n_ctx % 20 == 0:
                print(f'{n_ctx} ctx done ({time.time()-t0:.1f}s)')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f'wrote {len(results)} predictions to {out}')

if __name__ == '__main__':
    main()
