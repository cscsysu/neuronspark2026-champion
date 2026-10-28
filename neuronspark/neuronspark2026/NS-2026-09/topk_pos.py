"""Try: among top-K profiles, weighted vote on agent_pos."""
import json, sys
from collections import Counter
sys.path.insert(0, '.')
from sim2 import simulate, EVENT_KEYS, DEFAULT_PROFILE
from predict_v3 import candidate_profiles, score_probe

def find_pos(grid, ch):
    for y, r in enumerate(grid):
        x = r.find(ch)
        if x >= 0: return (x,y)
    return None

# For different K and weight strategies
def evaluate(K_top, weight_fn):
    n_total = 0; n_correct = 0
    with open('valid.jsonl') as f:
        for line in f:
            d = json.loads(line)
            scores = []
            for p in candidate_profiles():
                s = 0.0
                for ep in d['context_episodes']:
                    r = simulate(d['initial_full_grid'], ep['actions'], p)
                    s += score_probe(r, ep)
                scores.append((s, p))
            scores.sort(key=lambda x: -x[0])
            top_k = scores[:K_top]
            for q in d['queries']:
                real_a = find_pos(q['label']['final_grid'], 'A')
                n_total += 1
                pos_votes = Counter()
                top_pos = None
                for i, (s, p) in enumerate(top_k):
                    rr = simulate(q['initial_full_grid'], q['future_actions'], p)
                    if rr['agent_pos'] is not None:
                        w = weight_fn(s, top_k[0][0], i)
                        pos_votes[rr['agent_pos']] += w
                        if i == 0: top_pos = rr['agent_pos']
                if pos_votes:
                    best_pos = pos_votes.most_common(1)[0][0]
                    if best_pos == real_a: n_correct += 1
    return n_correct / n_total

# Test variants
for K in [1, 2, 3, 5, 8]:
    for wfn_name, wfn in [
        ('uniform', lambda s, top, i: 1.0),
        ('score_lin', lambda s, top, i: s),
        ('score_diff', lambda s, top, i: 2 ** (s - top)),  # exponential by score gap
        ('rank_inv', lambda s, top, i: 1.0/(i+1)),
    ]:
        acc = evaluate(K, wfn)
        print(f'K={K}, w={wfn_name}: {acc:.4f}')
