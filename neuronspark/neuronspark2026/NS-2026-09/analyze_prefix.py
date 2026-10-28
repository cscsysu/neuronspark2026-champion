"""For each query, find the action-prefix-matching probe.
Use that probe's events at matching prefix to anchor the query's events.
"""
import json, sys
from collections import Counter
sys.path.insert(0, '.')
from sim2 import simulate, EVENT_KEYS, DEFAULT_PROFILE
from predict_v3 import best_profile_for, context_priors

def find_pos(grid, ch):
    for y, r in enumerate(grid):
        x = r.find(ch)
        if x >= 0: return (x,y)
    return None

# Stats: for queries, what's the avg action-prefix overlap with closest probe?
prefixes = Counter()
n_with_full_probe = 0
n = 0
with open('valid.jsonl') as f:
    for line in f:
        d = json.loads(line)
        for q in d['queries']:
            n += 1
            best_prefix = -1
            for ep in d['context_episodes']:
                k = 0
                for a, b in zip(q['future_actions'], ep['actions']):
                    if a == b: k += 1
                    else: break
                if k > best_prefix: best_prefix = k
            prefixes[best_prefix] += 1
            if best_prefix == len(q['future_actions']):
                n_with_full_probe += 1
print(f'Total queries: {n}')
print(f'Action prefix lengths: {sorted(prefixes.items())}')
print(f'Queries with FULL action match (some probe = query): {n_with_full_probe}')
