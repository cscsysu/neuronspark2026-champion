"""Try top-K profile ensemble for agent_pos."""
import json
from collections import Counter
import sys
sys.path.insert(0, '.')
from sim2 import simulate, EVENT_KEYS, DEFAULT_PROFILE
from predict_v3 import score_probe, candidate_profiles, context_priors, predict_query, find_pos

def best_K_profiles(d, K=3):
    scores = []
    for p in candidate_profiles():
        s = 0.0
        for ep in d['context_episodes']:
            r = simulate(d['initial_full_grid'], ep['actions'], p)
            s += score_probe(r, ep)
        scores.append((s, p))
    scores.sort(key=lambda x: -x[0])
    return scores[:K]

def predict_query_ensemble(q, top_profiles, priors, eps, K=3):
    """Run top-K profiles, vote on agent_pos with weighted votes."""
    pos_votes = Counter()
    grid_votes = Counter()
    sim_results = []
    for s, p in top_profiles:
        r = simulate(q['initial_full_grid'], q['future_actions'], p)
        sim_results.append((s, p, r))
        if r['agent_pos']:
            pos_votes[r['agent_pos']] += s
        for y, row in enumerate(r['final_grid']):
            for x, ch in enumerate(row):
                if ch == 'B':
                    grid_votes[('B', x, y)] += s
    # use top-1 for everything except override agent_pos with vote
    base_score, base_p, base_r = sim_results[0]
    if pos_votes:
        best_pos = pos_votes.most_common(1)[0][0]
        base_r['agent_pos'] = best_pos
    # use base profile + base sim for prediction
    return predict_query(q, base_p, priors, eps)

# Score on valid using ensemble
import time
results = []
t0 = time.time()
with open('valid.jsonl') as f:
    for line in f:
        d = json.loads(line)
        top_profiles = best_K_profiles(d, K=3)
        priors = context_priors(d)
        for q in d['queries']:
            results.append(predict_query_ensemble(q, top_profiles, priors, d['context_episodes']))
with open('results_ensemble3.json','w') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f'Done in {time.time()-t0:.1f}s')
