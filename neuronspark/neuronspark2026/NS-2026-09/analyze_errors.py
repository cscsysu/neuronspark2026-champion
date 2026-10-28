"""Analyze where current predictions are losing points on valid."""
import json
from collections import Counter, defaultdict

EVENT_KEYS = ["goal_reached", "collision", "hazard", "box_on_goal", "key_collected", "portal_used"]
DYN = set('ABOK')
ENTITY_CHARS = ['A','B','O','K']

def find_pos(grid, ch):
    for y, r in enumerate(grid):
        x = r.find(ch)
        if x >= 0: return (x,y)
    return None

with open('results_v18.json') as f:
    preds = {p['id']: p for p in json.load(f)}

# Stats
ev_misses = Counter()  # which event flipped wrong
ev_fp = Counter()  # predicted True, actual False
ev_fn = Counter()  # predicted False, actual True
tl_misses = Counter()
term_confusion = Counter()
ord_misses = 0
ent_misses = Counter()  # which entity wrong
n = 0

with open('valid.jsonl') as f:
    for line in f:
        d = json.loads(line)
        for q in d['queries']:
            n += 1
            lbl = q['label']
            pred = preds.get(q['query_id'])
            if pred is None: continue

            for k in EVENT_KEYS:
                if pred['events'][k] != lbl['events'][k]:
                    ev_misses[k] += 1
                    if pred['events'][k] and not lbl['events'][k]:
                        ev_fp[k] += 1
                    else:
                        ev_fn[k] += 1
            # timeline
            for k in EVENT_KEYS:
                if lbl['event_timeline'][k] != 'never':
                    if pred['event_timeline'][k] != lbl['event_timeline'][k]:
                        tl_misses[k] += 1
            # term
            if pred['terminal'] != lbl['terminal']:
                term_confusion[(pred['terminal'], lbl['terminal'])] += 1
            # order
            if pred['event_order'] != lbl['event_order']:
                ord_misses += 1
            # entity
            for ch in ENTITY_CHARS:
                pp = find_pos(pred['final_grid'], ch)
                lp = find_pos(lbl['final_grid'], ch)
                if pp != lp:
                    ent_misses[ch] += 1

print(f'queries: {n}')
print('event misses (total):')
for k, v in ev_misses.most_common():
    print(f'  {k}: {v} (FP={ev_fp[k]}, FN={ev_fn[k]})')
print('timeline misses (only when label is non-never):')
for k, v in tl_misses.most_common():
    print(f'  {k}: {v}')
print(f'order misses: {ord_misses} / {n}')
print(f'entity misses:')
for k, v in ent_misses.most_common():
    print(f'  {k}: {v}')
print(f'terminal confusion (pred -> actual):')
for k, v in term_confusion.most_common():
    print(f'  {k[0]} -> {k[1]}: {v}')
