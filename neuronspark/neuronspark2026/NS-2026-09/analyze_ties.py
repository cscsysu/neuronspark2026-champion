"""Check tie patterns in profile selection."""
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

n_ties = 0
n_total_ctx = 0
ties_helped = 0  # tie broken to wrong but had right tied
ties_picked_wrong = 0

with open('valid.jsonl') as f:
    for line in f:
        d = json.loads(line)
        n_total_ctx += 1
        scores = []
        for p in candidate_profiles():
            s = 0.0
            for ep in d['context_episodes']:
                r = simulate(d['initial_full_grid'], ep['actions'], p)
                s += score_probe(r, ep)
            scores.append((s, p))
        scores.sort(key=lambda x: -x[0])
        max_s = scores[0][0]
        tied = [p for s, p in scores if abs(s - max_s) < 1e-6]
        if len(tied) > 1:
            n_ties += 1
            for q in d['queries']:
                real_a = find_pos(q['label']['final_grid'], 'A')
                # any tied profile gives correct?
                tied_correct = []
                for p in tied:
                    r = simulate(q['initial_full_grid'], q['future_actions'], p)
                    tied_correct.append(r['agent_pos'] == real_a)
                if any(tied_correct) and not all(tied_correct):
                    # tie matters
                    if not tied_correct[0]:  # we pick first one which is wrong
                        ties_picked_wrong += 1

print(f'Total contexts: {n_total_ctx}, tied ones: {n_ties}')
print(f'Among ties, picked wrong but tied profile would have been right: {ties_picked_wrong}')
