"""For tied profiles, what's the pattern of right vs wrong?"""
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

# For each tied query case, which knob values were correct?
right_knobs = Counter()
wrong_knobs = Counter()

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
        max_s = scores[0][0]
        tied = [p for s, p in scores if abs(s - max_s) < 1e-6]
        if len(tied) <= 1:
            continue
        for q in d['queries']:
            real_a = find_pos(q['label']['final_grid'], 'A')
            for p in tied:
                r = simulate(q['initial_full_grid'], q['future_actions'], p)
                correct = (r['agent_pos'] == real_a)
                key = (p['ice_mode'], p['portal_pair'], p['box_walk'], p['orb_blocks'])
                if correct: right_knobs[key] += 1
                else: wrong_knobs[key] += 1

print('Right (correct agent_pos in tied) by knob combo:')
for k, v in right_knobs.most_common():
    print(f'  {k}: right={v}, wrong={wrong_knobs[k]}, ratio={v/(v+wrong_knobs[k]):.2f}')
