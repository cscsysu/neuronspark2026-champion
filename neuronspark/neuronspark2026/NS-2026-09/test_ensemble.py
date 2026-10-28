"""Top-K profile ensemble: use majority vote on agent_pos."""
import json, sys
from collections import Counter
sys.path.insert(0, '.')
from sim2 import simulate, EVENT_KEYS, DEFAULT_PROFILE
from predict_v3 import candidate_profiles, score_probe, predict_query, context_priors

def find_pos(grid, ch):
    for y, r in enumerate(grid):
        x = r.find(ch)
        if x >= 0: return (x,y)
    return None

def best_K_profiles(d, K):
    scores = []
    for p in candidate_profiles():
        s = 0.0
        for ep in d['context_episodes']:
            r = simulate(d['initial_full_grid'], ep['actions'], p)
            s += score_probe(r, ep)
        scores.append((s, p))
    scores.sort(key=lambda x: -x[0], reverse=False)
    scores.sort(key=lambda x: -x[0])
    return scores[:K]

# Test K=1, 2, 3, 5 with weighted vote on agent_pos
for K in [1, 2, 3, 5]:
    n_picked = 0; total = 0
    n_top1 = 0
    with open('valid.jsonl') as f:
        for line in f:
            d = json.loads(line)
            top_profiles = best_K_profiles(d, K)
            for q in d['queries']:
                real_a = find_pos(q['label']['final_grid'], 'A')
                total += 1
                # Top-1
                top1_p = top_profiles[0][1]
                r = simulate(q['initial_full_grid'], q['future_actions'], top1_p)
                if r['agent_pos'] == real_a: n_top1 += 1
                # Top-K weighted vote
                pos_votes = Counter()
                for s, p in top_profiles:
                    rr = simulate(q['initial_full_grid'], q['future_actions'], p)
                    if rr['agent_pos'] is not None:
                        pos_votes[rr['agent_pos']] += s
                if pos_votes:
                    best_pos = pos_votes.most_common(1)[0][0]
                    if best_pos == real_a: n_picked += 1
    print(f'K={K}: top1={n_top1}/{total}, vote={n_picked}/{total}')
