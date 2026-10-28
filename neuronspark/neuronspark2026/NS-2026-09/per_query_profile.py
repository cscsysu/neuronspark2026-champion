"""Per-query profile selection: among tied profiles, pick by simulating query
and comparing to most action-similar probe."""
import json, sys
from collections import Counter
sys.path.insert(0, '.')
from sim2 import simulate, EVENT_KEYS, DEFAULT_PROFILE
from predict_v3 import candidate_profiles, score_probe, predict_query, context_priors, find_pos

def per_query_profile(d, q, top_k=3):
    """Pick best profile for this specific query."""
    # Compute scores per profile on probes
    profile_scores = []
    for p in candidate_profiles():
        s = 0.0
        for ep in d['context_episodes']:
            r = simulate(d['initial_full_grid'], ep['actions'], p)
            s += score_probe(r, ep)
        profile_scores.append((s, p))
    profile_scores.sort(key=lambda x: -x[0])

    # Among top scoring, find one whose query result is most consistent
    # Strategy: use top-1
    return profile_scores[0][1]

# Test multiple strategies
import predict_v3 as pv

# Strategy 1: standard (current)
def strategy_curr(d, q):
    return pv.best_profile_for(d)[0]

# Strategy 2: per-query (just top-1 same as curr)
def strategy_top1(d, q):
    profile, _ = pv.best_profile_for(d)
    return profile

# Strategy 3: per-query top-K with query path matching
def strategy_action_aware(d, q):
    """Among top-K profiles, pick one that simulates closest match
    to most-action-prefix-similar probe."""
    profile_scores = []
    for p in candidate_profiles():
        s = 0.0
        for ep in d['context_episodes']:
            r = simulate(d['initial_full_grid'], ep['actions'], p)
            s += score_probe(r, ep)
        profile_scores.append((s, p))
    profile_scores.sort(key=lambda x: -x[0])
    max_s = profile_scores[0][0]
    tied = [(s, p) for s, p in profile_scores if abs(s - max_s) < 1e-6]
    if len(tied) == 1:
        return tied[0][1]
    # Find probe with longest action prefix
    q_actions = q['future_actions']
    best_ep = None; best_pre = -1
    for ep in d['context_episodes']:
        k = 0
        for a, b in zip(q_actions, ep['actions']):
            if a == b: k += 1
            else: break
        if k > best_pre:
            best_pre = k; best_ep = ep
    if best_pre == 0:
        return tied[0][1]
    # Among tied profiles, pick one with best fit on best_ep
    best_p = tied[0][1]; best_fit = -1
    for s, p in tied:
        rr = simulate(d['initial_full_grid'], best_ep['actions'], p)
        f = score_probe(rr, best_ep)
        if f > best_fit:
            best_fit = f; best_p = p
    return best_p

# Eval each
for name, strat in [('current', strategy_curr), ('action-aware', strategy_action_aware)]:
    n_total = 0; n_a_correct = 0
    with open('valid.jsonl') as f:
        for line in f:
            d = json.loads(line)
            for q in d['queries']:
                profile = strat(d, q)
                r = simulate(q['initial_full_grid'], q['future_actions'], profile)
                real_a = find_pos(q['label']['final_grid'], 'A')
                n_total += 1
                if r['agent_pos'] == real_a: n_a_correct += 1
    print(f'{name}: agent_pos={n_a_correct}/{n_total} = {n_a_correct/n_total:.4f}')
