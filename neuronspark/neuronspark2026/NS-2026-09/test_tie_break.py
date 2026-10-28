"""Try per-query tie-breaking: among tied profiles, prefer one whose
sim of query is most similar to probe behavior."""
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

def sim_score(d, p):
    s = 0.0
    for ep in d['context_episodes']:
        r = simulate(d['initial_full_grid'], ep['actions'], p)
        s += score_probe(r, ep)
    return s

# Per-query: tie-break by best fit on closest-prefix probe
n_picked = 0
n_top1 = 0
n_total = 0
n_oracle_breakable = 0  # tied cases where some tied profile is correct

with open('valid.jsonl') as f:
    for line in f:
        d = json.loads(line)
        scores = []
        for p in candidate_profiles():
            s = sim_score(d, p)
            scores.append((s, p))
        scores.sort(key=lambda x: -x[0])
        max_s = scores[0][0]
        tied = [(s, p) for s, p in scores if abs(s - max_s) < 1e-6]
        for q in d['queries']:
            real_a = find_pos(q['label']['final_grid'], 'A')
            n_total += 1
            # Top-1
            top1_p = scores[0][1]
            r = simulate(q['initial_full_grid'], q['future_actions'], top1_p)
            if r['agent_pos'] == real_a: n_top1 += 1
            # Tie-break: among tied, find profile that gives most "consistent" sim
            if len(tied) > 1:
                # Find best probe match by action prefix
                best_pre = -1; best_ep = None
                for ep in d['context_episodes']:
                    k = 0
                    for a, b in zip(q['future_actions'], ep['actions']):
                        if a == b: k += 1
                        else: break
                    if k > best_pre:
                        best_pre = k; best_ep = ep
                # For each tied profile, simulate the closest probe and compare to actual probe end
                best_score = -1; best_p = tied[0][1]
                for s, p in tied:
                    rr = simulate(d['initial_full_grid'], best_ep['actions'], p)
                    fit = score_probe(rr, best_ep)
                    if fit > best_score:
                        best_score = fit; best_p = p
                rr = simulate(q['initial_full_grid'], q['future_actions'], best_p)
                if rr['agent_pos'] == real_a: n_picked += 1
            else:
                if r['agent_pos'] == real_a: n_picked += 1

print(f'Top-1 agent_pos: {n_top1}/{n_total} = {n_top1/n_total:.4f}')
print(f'Tie-break by closest-probe agent_pos: {n_picked}/{n_total} = {n_picked/n_total:.4f}')
