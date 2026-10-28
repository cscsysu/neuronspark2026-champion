"""Predict v2: combine simulator output with probe-derived priors.

Strategy:
- Run simulator with profile search (selecting best config matching probes).
- For each query, output simulator's grid/terminal as-is.
- For events: use both simulator and probe-event-count to decide. Specifically:
    if probe_count >= 4 -> trust probe prior strongly (event=True usually).
    if probe_count == 0 -> only set True if simulator says True.
    in between -> use simulator's output.
- For terminal: if simulator says timeout, prefer dominant probe terminal; else trust sim.
- For event_order: keep simulator's order; if it has 'none' first but probe orders all start
  with X, fill with X first.
"""
import json, itertools, sys, time
from collections import Counter
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
    s += sum(1 for k in EVENT_KEYS if sim_r['events'][k] == ep['observed_final_events'][k]) * 0.3
    if sim_r['terminal'] == ep['observed_final_terminal']: s += 2.0
    if sim_r['event_order'] == ep['observed_final_event_order']: s += 1.0
    same = tot = 0
    for r1, r2 in zip(sim_r['final_grid'], ep['observed_final_full_grid']):
        for c1, c2 in zip(r1, r2):
            tot += 1
            if c1 == c2: same += 1
    s += (same/tot) * 1.5
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

# Probe-based event prior thresholds (from train statistics)
EVENT_PRIOR_HIGH = {  # probe_count >= threshold => predict True
    'goal_reached': 5,
    'hazard': 5,
    'key_collected': 5,
    'portal_used': 5,
    'collision': 1,    # 83%+ collision when any probe has collision
    'box_on_goal': 99, # never override (rare)
}

EVENT_PRIOR_NONE = {  # probe_count == 0 => predict False
    'goal_reached': True,
    'hazard': True,
    'key_collected': True,
    'portal_used': True,
    'collision': False,  # collision can happen even if probe didn't show
    'box_on_goal': True,
}

def context_priors(d):
    eps = d['context_episodes']
    n = len(eps) or 1
    probe_event_counts = {k: sum(1 for ep in eps if ep['observed_final_events'][k]) for k in EVENT_KEYS}
    probe_terminals = Counter(ep['observed_final_terminal'] for ep in eps)
    # most-common probe order first event
    first_events = [ep['observed_final_event_order'][0] for ep in eps if ep['observed_final_event_order']]
    probe_first_event_mode = Counter(first_events).most_common(1)[0][0] if first_events else 'collision'
    return {
        'probe_event_counts': probe_event_counts,
        'probe_terminals': probe_terminals,
        'probe_first_event_mode': probe_first_event_mode,
        'n_probes': n,
    }

def merge_events(sim_events, sim_timeline, sim_order, priors, horizon):
    """Override events using probe priors when confident."""
    pc = priors['probe_event_counts']
    out_events = dict(sim_events)
    out_timeline = dict(sim_timeline)
    for k in EVENT_KEYS:
        # if all probes triggered, very likely query also triggers
        if pc[k] >= EVENT_PRIOR_HIGH[k] and not sim_events[k]:
            out_events[k] = True
            if out_timeline[k] == 'never':
                # default: use most common bucket among probes (early most likely)
                out_timeline[k] = 'early'
        # if no probe triggered AND simulator predicts True, trust simulator (already True)
        # if no probe triggered AND simulator predicts False, leave as False
    # event_order: rebuild based on out_events + out_timeline ordering
    # Order events that fired by their bucket (early < mid < late) and break ties by sim_order
    order_pri = {'early':0,'mid':1,'late':2,'never':9}
    fired = [k for k in EVENT_KEYS if out_events[k]]
    if fired:
        # try to keep simulator's relative order, but ensure events that were turned True by prior
        # are inserted by their (forced) timeline bucket.
        # default: sort fired by timeline-bucket then by EVENT_KEYS order
        fired.sort(key=lambda k: (order_pri.get(out_timeline[k], 9), EVENT_KEYS.index(k)))
        # but if simulator already had a non-trivial order, prefer it where consistent
        sim_fired_order = [e for e in sim_order if e != 'none' and e in fired]
        # any that aren't in sim's order, append
        final = list(sim_fired_order)
        for k in fired:
            if k not in final:
                final.append(k)
        new_order = list(final[:3])
    else:
        new_order = []
    while len(new_order) < 3:
        new_order.append('none')
    return out_events, out_timeline, new_order

def merge_terminal(sim_terminal, priors, sim_events):
    """Use probe terminal mode when simulator is uncertain."""
    if sim_terminal == 'goal' or sim_events.get('goal_reached'):
        return 'goal'
    if sim_terminal == 'hazard' or sim_events.get('hazard'):
        return 'hazard'
    pt = priors['probe_terminals']
    # fallback: if probes predominantly hazard/goal, prefer it
    most = pt.most_common(1)[0][0] if pt else 'blocked'
    if sim_terminal == 'timeout':
        return 'blocked'  # better default
    return sim_terminal

def predict_query(q, profile, priors):
    r = simulate(q['initial_full_grid'], q['future_actions'], profile)
    horizon = q['query_horizon']
    events, timeline, order = merge_events(r['events'], r['event_timeline'], r['event_order'], priors, horizon)
    # If we forced an event True, also force terminal accordingly
    terminal = r['terminal']
    if events.get('goal_reached') and not r['events']['goal_reached']:
        # Forced goal
        terminal = 'goal'
    if events.get('hazard') and not r['events']['hazard'] and terminal == 'blocked':
        terminal = 'hazard'
    terminal = merge_terminal(terminal, priors, events)
    return {
        'id': q['query_id'],
        'final_grid': r['final_grid'],
        'events': events,
        'event_timeline': timeline,
        'event_order': order,
        'terminal': terminal,
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
            priors = context_priors(d)
            for q in d['queries']:
                results.append(predict_query(q, profile, priors))
            n_ctx += 1
            if n_ctx % 40 == 0:
                print(f'{n_ctx} ctx done ({time.time()-t0:.1f}s)')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f'wrote {len(results)} predictions to {out}')

if __name__ == '__main__':
    main()
