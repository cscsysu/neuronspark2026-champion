"""Predict v3: hybrid strategy.

Key insight: For dynamic entities B/O/K, the most-common probe end position is highly
predictive (97% for orbs, 75% for boxes, 73% for keys; given consistency, 99/88/85% match).

Strategy:
1) Run simulator with profile search (for agent position).
2) Build final_grid from initial_grid:
   - Replace B/O/K with the most-common-probe-end positions of B/O/K
   - Place agent (A) where simulator predicts
3) For events: keep predict_v2 logic (probe priors).
4) Special case: if agent's predicted final tile coincides with goal G or hazard H,
   set the events accordingly.

Output schema unchanged from predict_v2.
"""
import json, itertools, sys, time, pickle, os
from collections import Counter
from sim2 import simulate, EVENT_KEYS, DEFAULT_PROFILE
from functools import cmp_to_key

# Try to load ML order classifier (optional - falls back to rule-based)
ORDER_CLF = None
try:
    if os.path.exists('order_clf.pkl'):
        with open('order_clf.pkl', 'rb') as f:
            ORDER_CLF = pickle.load(f)
except Exception:
    ORDER_CLF = None

# Try to load timeline classifiers (3-way: early/mid/late given event_pred=True)
TL_CLFS = None
try:
    if os.path.exists('tl_clfs2.pkl'):
        with open('tl_clfs2.pkl', 'rb') as f:
            TL_CLFS = pickle.load(f)
except Exception:
    TL_CLFS = None
TL_LABELS = ['early','mid','late']

# Terminal classifier
TERM_CLF = None
try:
    if os.path.exists('term_clf.pkl'):
        with open('term_clf.pkl', 'rb') as f:
            TERM_CLF = pickle.load(f)
except Exception:
    TERM_CLF = None
TERM_LABELS = ['blocked','timeout','goal','hazard']

# Event ML classifiers (mostly to override goal_reached and collision)
EV_CLFS = None
try:
    if os.path.exists('ev_clfs.pkl'):
        with open('ev_clfs.pkl', 'rb') as f:
            EV_CLFS = pickle.load(f)
except Exception:
    EV_CLFS = None
EV_USE_ML = {'goal_reached': True, 'collision': True}  # only events where ML wins

def build_ev_features(eps, k, priors, r):
    pec, pt, tlm, entity_pos, probe_tl_avg, probe_avg_rank = priors
    h = max(1, r['horizon'])
    f = [
        int(r['events'][k]),
        r['first_step'][k]/h if r['first_step'][k] is not None else -1,
        pec[k], h, len(eps),
    ]
    for kk in EVENT_KEYS:
        f.append(int(r['events'][kk]))
        f.append(pec[kk])
    for t in ['blocked','timeout','goal','hazard']:
        f.append(pt.get(t, 0))
    return f

def build_term_features(eps, k_actions, k_init, priors, r):
    pec, pt, tlm, entity_pos, probe_tl_avg, probe_avg_rank = priors
    h = max(1, r['horizon'])
    f = [h, int(r['last_blocked'])]
    for t in TERM_LABELS:
        f.append(1 if r['terminal']==t else 0)
        f.append(pt.get(t, 0))
    for kk in EVENT_KEYS:
        f.append(int(r['events'][kk]))
        f.append(pec[kk])
        if r['first_step'][kk] is not None:
            f.append(r['first_step'][kk]/h)
        else:
            f.append(-1)
    return f

def build_tl_features(d_eps, q_actions, q_init, k, priors, r):
    pec, pt, tlm, entity_pos, probe_tl_avg, probe_avg_rank = priors
    h = max(1, r['horizon'])
    bucket_cnt = Counter()
    for ep in d_eps:
        bucket_cnt[ep['observed_final_event_timeline'][k]] += 1
    f = [
        int(r['events'][k]),
        r['first_step'][k]/h if r['first_step'][k] is not None else -1,
        pec[k], h,
        bucket_cnt['early'], bucket_cnt['mid'], bucket_cnt['late'],
        probe_tl_avg.get(k, 0.5),
    ]
    for kk in EVENT_KEYS:
        f.append(int(r['events'][kk]))
        f.append(pec[kk])
        if r['first_step'][kk] is not None:
            f.append(r['first_step'][kk]/h)
        else:
            f.append(-1)
    return f

KNOBS = {
    'ice_mode': ['classic', 'slide_one'],
    'portal_pair': [True, False],
    'box_walk': [False, True],
    'orb_blocks': [False, True],
}

def candidate_profiles():
    keys = list(KNOBS.keys())
    for vals in itertools.product(*[KNOBS[k] for k in keys]):
        p = {**DEFAULT_PROFILE}
        for k, v in zip(KNOBS, vals): p[k] = v
        yield p

def find_pos(grid, ch):
    for y, r in enumerate(grid):
        x = r.find(ch)
        if x >= 0: return (x,y)
    return None

def find_all(grid, ch):
    out = []
    for y, r in enumerate(grid):
        for x, c in enumerate(r):
            if c == ch: out.append((x,y))
    return out

def score_probe(sim_r, ep):
    s = 0.0
    real_a = find_pos(ep['observed_final_full_grid'], 'A')
    if real_a == sim_r['agent_pos']: s += 1.0
    if all(sim_r['events'][k] == ep['observed_final_events'][k] for k in EVENT_KEYS): s += 1.0
    s += sum(1 for k in EVENT_KEYS if sim_r['events'][k] == ep['observed_final_events'][k]) * 0.3
    if sim_r['terminal'] == ep['observed_final_terminal']: s += 1.0
    if sim_r['event_order'] == ep['observed_final_event_order']: s += 1.0
    same = tot = 0
    for r1, r2 in zip(sim_r['final_grid'], ep['observed_final_full_grid']):
        for c1, c2 in zip(r1, r2):
            tot += 1
            if c1 == c2: same += 1
    s += (same/tot) * 1.0
    for k in EVENT_KEYS:
        if ep['observed_final_event_timeline'][k] != 'never':
            if sim_r['event_timeline'][k] == ep['observed_final_event_timeline'][k]:
                s += 1.5
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

THRESH = {  # Tuned for sim2: predict True if sim says True OR probe_count >= T
    'goal_reached': 5,
    'collision': 6,
    'hazard': 5,
    'box_on_goal': 3,
    'key_collected': 6,
    'portal_used': 6,
}

def context_priors(d, q=None):
    eps = d['context_episodes']
    pec = {k: sum(1 for ep in eps if ep['observed_final_events'][k]) for k in EVENT_KEYS}
    pt = Counter(ep['observed_final_terminal'] for ep in eps)
    tlm = {}
    for k in EVENT_KEYS:
        tls = [ep['observed_final_event_timeline'][k] for ep in eps if ep['observed_final_event_timeline'][k] != 'never']
        tlm[k] = Counter(tls).most_common(1)[0][0] if tls else 'early'
    # Most common final positions for dynamic entities across all probes
    entity_pos = {}
    for ent in 'BO':
        positions = []
        for ep in eps:
            positions.append(tuple(find_all(ep['observed_final_full_grid'], ent)))
        entity_pos[ent] = Counter(positions).most_common(1)[0][0]
    # Special K: gather positions when key NOT collected (uncollected position)
    K_when_not_collected = [tuple(find_all(ep['observed_final_full_grid'], 'K'))
                             for ep in eps if not ep['observed_final_events']['key_collected']]
    if K_when_not_collected:
        entity_pos['K'] = Counter(K_when_not_collected).most_common(1)[0][0]
    else:
        all_K = [tuple(find_all(ep['observed_final_full_grid'], 'K')) for ep in eps]
        entity_pos['K'] = Counter(all_K).most_common(1)[0][0]
    # Probe-derived avg timeline-bucket score (for forced event step estimation)
    bucket_map = {'early': 0.166, 'mid': 0.5, 'late': 0.833}
    probe_tl_avg = {}
    for k in EVENT_KEYS:
        scores = [bucket_map[ep['observed_final_event_timeline'][k]]
                  for ep in eps if ep['observed_final_event_timeline'][k] != 'never']
        probe_tl_avg[k] = sum(scores)/len(scores) if scores else 0.5
    # Probe-derived avg rank for event ordering
    probe_avg_rank = {}
    for k in EVENT_KEYS:
        ranks = []
        for ep in eps:
            for i, e in enumerate(ep['observed_final_event_order']):
                if e == k: ranks.append(i); break
        probe_avg_rank[k] = sum(ranks)/len(ranks) if ranks else 999.0
    return pec, pt, tlm, entity_pos, probe_tl_avg, probe_avg_rank

def best_match_probe(eps, q_actions):
    """Find probe with longest common action prefix to query (tiebreak: longest LCS)."""
    best = None; best_score = -1
    for ep in eps:
        # prefix match length
        n = 0
        for a, b in zip(q_actions, ep['actions']):
            if a == b: n += 1
            else: break
        score = n * 100 + sum(1 for x, y in zip(q_actions, ep['actions']) if x == y)
        if score > best_score:
            best_score = score; best = ep
    return best

def context_entity_pos_for_query(eps, q, predicted_key_collected=None):
    """Per-query entity positions. For K, special rule based on key_collected event."""
    out = {}
    # B: modal across all probes
    all_B = [tuple(find_all(ep['observed_final_full_grid'], 'B')) for ep in eps]
    out['B'] = Counter(all_B).most_common(1)[0][0]
    # O: modal
    all_O = [tuple(find_all(ep['observed_final_full_grid'], 'O')) for ep in eps]
    out['O'] = Counter(all_O).most_common(1)[0][0]
    # K: position when not collected
    K_not_collected = [tuple(find_all(ep['observed_final_full_grid'], 'K'))
                       for ep in eps if not ep['observed_final_events']['key_collected']]
    if K_not_collected:
        out['K_uncollected'] = Counter(K_not_collected).most_common(1)[0][0]
    else:
        all_K = [tuple(find_all(ep['observed_final_full_grid'], 'K')) for ep in eps]
        out['K_uncollected'] = Counter(all_K).most_common(1)[0][0]
    return out

def build_final_grid(initial_grid, agent_end_pos, entity_pos, events, sim_grid=None):
    """Build final grid. Agent placed first, then entities. If A & entity collide,
    the entity (B/O/K) wins for those cells.

    sim_grid: if provided, prefer simulator's B position (sim is more accurate for B than probe modal).
    """
    grid = [list(r) for r in initial_grid]
    goal_pos = None
    for y in range(12):
        for x in range(12):
            if grid[y][x] == 'G':
                goal_pos = (x, y)
    init_B = []
    for y in range(12):
        for x in range(12):
            if grid[y][x] == 'B':
                init_B.append((x, y))
    # Strip dynamic
    for y in range(12):
        for x in range(12):
            if grid[y][x] in 'ABOK':
                grid[y][x] = '.'
    # Place A first
    if agent_end_pos is not None:
        ax, ay = agent_end_pos
        if 0 <= ax < 12 and 0 <= ay < 12 and grid[ay][ax] != '#':
            grid[ay][ax] = 'A'
    # Place B: prefer sim's B position (more accurate than probe modal for box)
    if events.get('box_on_goal') and goal_pos and len(init_B) == 1:
        gx, gy = goal_pos
        if grid[gy][gx] != '#':
            grid[gy][gx] = 'B'
    elif sim_grid is not None:
        # Use sim's B positions
        sim_B = [(x, y) for y, row in enumerate(sim_grid) for x, c in enumerate(row) if c == 'B']
        for (x, y) in sim_B:
            if grid[y][x] != '#':
                grid[y][x] = 'B'
    else:
        for (x, y) in entity_pos.get('B', ()):
            if grid[y][x] != '#':
                grid[y][x] = 'B'
    for (x, y) in entity_pos.get('O', ()):
        if grid[y][x] != '#':
            grid[y][x] = 'O'
    if not events.get('key_collected'):
        for (x, y) in entity_pos.get('K_uncollected', ()):
            if grid[y][x] != '#':
                grid[y][x] = 'K'
    # Re-place A: if A was overwritten by an entity, find an empty cell near agent_end_pos
    has_A = any('A' in r for r in (''.join(row) for row in grid))
    if not has_A and agent_end_pos is not None:
        ax, ay = agent_end_pos
        if 0 <= ax < 12 and 0 <= ay < 12 and grid[ay][ax] in '.GP^v<>IH':
            grid[ay][ax] = 'A'
    return [''.join(r) for r in grid]

def per_step_score(d, p):
    """Score profile using per-step delta_events from probe observations."""
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

def best_per_step_profile(d):
    """Pick the profile with best per-step events match (better for agent_pos)."""
    best_ps = -1; best_p = None
    for p in candidate_profiles():
        ps = per_step_score(d, p)
        if ps > best_ps:
            best_ps = ps; best_p = p
    return best_p

def predict_query(q, profile, priors, eps, d=None):
    pec, pt, tlm, entity_pos, probe_tl_avg, probe_avg_rank = priors
    # Per-query entity positions
    entity_pos_q = context_entity_pos_for_query(eps, q)
    r = simulate(q['initial_full_grid'], q['future_actions'], profile)
    # Events with priors + optional ML for goal_reached/collision
    events = {}
    for k in EVENT_KEYS:
        events[k] = r['events'][k] or pec[k] >= THRESH[k]
        if EV_CLFS is not None and EV_USE_ML.get(k) and k in EV_CLFS and EV_CLFS[k][0] == 'ml':
            f = build_ev_features(eps, k, priors, r)
            events[k] = bool(EV_CLFS[k][1].predict([f])[0])
    timeline = dict(r['event_timeline'])
    # Compute per-event probe bucket counter (for unanimity rule)
    probe_buckets = {k: Counter() for k in EVENT_KEYS}
    for ep in eps:
        for k in EVENT_KEYS:
            tl = ep['observed_final_event_timeline'][k]
            if tl != 'never':
                probe_buckets[k][tl] += 1
    for k in EVENT_KEYS:
        if not events[k]:
            timeline[k] = 'never'
        else:
            # ML if available
            if TL_CLFS is not None and k in TL_CLFS and TL_CLFS[k][0] == 'ml':
                f = build_tl_features(eps, q['future_actions'], q['initial_full_grid'], k, priors, r)
                pred_idx = TL_CLFS[k][1].predict([f])[0]
                timeline[k] = TL_LABELS[pred_idx]
            elif r['events'][k]:
                # use sim's bucket (already in timeline)
                # Hazard-specific override: if all 5 probes have non-never hazard
                # AND they all share same bucket, trust the probe modal (analysis: 31/31 vs 25/31).
                if k == 'hazard':
                    bc = probe_buckets[k]
                    if sum(bc.values()) == 5 and len(bc) == 1:
                        timeline[k] = list(bc.keys())[0]
            else:
                # forced via prior, use probe modal
                timeline[k] = tlm[k]
    # Order: ML pairwise classifier if available, else step-based heuristic
    fired = [k for k in EVENT_KEYS if events[k]]
    h = max(1, r['horizon'])
    # Build per-event step features for ML / rule
    feats = {}
    for k in fired:
        if r['first_step'][k] is not None:
            step = r['first_step'][k] / h
        else:
            step = probe_tl_avg.get(k, 0.5)
        feats[k] = (step, probe_avg_rank.get(k, 999.0), int(r['events'][k]), pec[k])

    # Cache pairwise probe ordering counts
    if ORDER_CLF is not None and len(fired) >= 2 and ORDER_CLF.n_features_in_ == 15:
        # New v2 model with extra features
        def pair_pre_count(ka, kb):
            n_a_before = 0; n_both = 0
            for ep in eps:
                ord_list = [e for e in ep['observed_final_event_order'] if e in (ka, kb)]
                if ka in ord_list and kb in ord_list:
                    n_both += 1
                    if ord_list.index(ka) < ord_list.index(kb): n_a_before += 1
            return n_a_before / max(1, n_both), n_both
        def cmp_ml(a, b):
            fa = feats[a]; fb = feats[b]
            p_a_b, n_both = pair_pre_count(a, b)
            x = [[fa[0]-fb[0], fa[1]-fb[1], fa[2]-fb[2], fa[3]-fb[3],
                  fa[0], fb[0], fa[1], fb[1],
                  p_a_b, n_both,
                  fa[2], fb[2], fa[3], fb[3],
                  EVENT_KEYS.index(a) - EVENT_KEYS.index(b)]]
            p = ORDER_CLF.predict_proba(x)[0][1]
            if p > 0.5: return -1
            elif p < 0.5: return 1
            return EVENT_KEYS.index(a) - EVENT_KEYS.index(b)
        fired.sort(key=cmp_to_key(cmp_ml))
    elif ORDER_CLF is not None and len(fired) >= 2:
        def cmp_ml(a, b):
            fa = feats[a]; fb = feats[b]
            x = [[fa[0]-fb[0], fa[1]-fb[1], fa[2]-fb[2], fa[3]-fb[3],
                  fa[0], fb[0], fa[1], fb[1]]]
            p = ORDER_CLF.predict_proba(x)[0][1]
            if p > 0.5: return -1
            elif p < 0.5: return 1
            return EVENT_KEYS.index(a) - EVENT_KEYS.index(b)
        fired.sort(key=cmp_to_key(cmp_ml))
    else:
        def order_key(k):
            return (feats[k][0], feats[k][1], EVENT_KEYS.index(k))
        fired.sort(key=order_key)
    order = list(fired[:3])
    while len(order) < 3: order.append('none')
    # Terminal: ML classifier if available, else event-set match + rule
    if TERM_CLF is not None:
        f = build_term_features(eps, q['future_actions'], q['initial_full_grid'], priors, r)
        terminal = TERM_LABELS[TERM_CLF.predict([f])[0]]
    else:
        ev_set = frozenset(k for k in EVENT_KEYS if events[k])
        matching = [ep['observed_final_terminal'] for ep in eps
                    if frozenset(kk for kk in EVENT_KEYS if ep['observed_final_events'][kk]) == ev_set]
        if matching:
            terminal = Counter(matching).most_common(1)[0][0]
        else:
            terminal = r['terminal']
            if events['goal_reached']: terminal = 'goal'
            elif events['hazard'] and r['terminal'] != 'goal': terminal = 'hazard'
            elif terminal == 'timeout': terminal = 'blocked'
    # Build final grid using per-query probe entity positions + sim's B
    # Override agent_pos with per-step picker if d is provided and per-step picks differently
    final_agent_pos = r['agent_pos']
    if d is not None:
        ps_profile = best_per_step_profile(d)
        if ps_profile != profile:
            r_ps = simulate(q['initial_full_grid'], q['future_actions'], ps_profile)
            final_agent_pos = r_ps['agent_pos']
    final_grid = build_final_grid(q['initial_full_grid'], final_agent_pos, entity_pos_q, events, sim_grid=r['final_grid'])
    return {
        'id': q['query_id'],
        'final_grid': final_grid,
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
                results.append(predict_query(q, profile, priors, d['context_episodes'], d=d))
            n_ctx += 1
            if n_ctx % 40 == 0:
                print(f'{n_ctx} ctx done ({time.time()-t0:.1f}s)')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f'wrote {len(results)} predictions to {out}')

if __name__ == '__main__':
    main()
