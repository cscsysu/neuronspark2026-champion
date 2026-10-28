"""Analyze terminal misses - which path is the prediction taking?"""
import json
from collections import Counter

EVENT_KEYS = ["goal_reached", "collision", "hazard", "box_on_goal", "key_collected", "portal_used"]

with open('results_v19.json') as f:
    preds = {p['id']: p for p in json.load(f)}

# Look at blocked->timeout, blocked->hazard cases
miss_details = []
n = 0
with open('valid.jsonl') as f:
    for line in f:
        d = json.loads(line)
        for q in d['queries']:
            n += 1
            lbl = q['label']
            pred = preds.get(q['query_id'])
            if pred is None: continue
            if pred['terminal'] != lbl['terminal']:
                # what would event-set match give us?
                pred_evset = frozenset(k for k in EVENT_KEYS if pred['events'][k])
                lbl_evset = frozenset(k for k in EVENT_KEYS if lbl['events'][k])
                # event match
                ev_correct = (pred_evset == lbl_evset)
                # context probes with matching event vector to predicted
                matching_pred = [ep['observed_final_terminal'] for ep in d['context_episodes']
                                 if frozenset(kk for kk in EVENT_KEYS if ep['observed_final_events'][kk]) == pred_evset]
                miss_details.append({
                    'pred_term': pred['terminal'],
                    'lbl_term': lbl['terminal'],
                    'pred_ev_correct': ev_correct,
                    'matching_pred_terms': Counter(matching_pred).most_common(),
                })

print(f'total queries: {n}, miss count: {len(miss_details)}')
# Group by (pred, label)
groups = Counter()
ev_correct_count = Counter()
matching_summary = Counter()
for m in miss_details:
    key = (m['pred_term'], m['lbl_term'])
    groups[key] += 1
    if m['pred_ev_correct']:
        ev_correct_count[key] += 1
print('Misses by (pred -> label):')
for k, v in groups.most_common():
    print(f'  {k}: {v} (ev_correct: {ev_correct_count[k]})')

# Show details for blocked->timeout, blocked->hazard
print('\nblocked->timeout details:')
for m in miss_details:
    if m['pred_term']=='blocked' and m['lbl_term']=='timeout':
        print(f"  ev_correct={m['pred_ev_correct']}, matching: {m['matching_pred_terms']}")
print('\nblocked->hazard details:')
for m in miss_details[:20]:
    if m['pred_term']=='blocked' and m['lbl_term']=='hazard':
        print(f"  ev_correct={m['pred_ev_correct']}, matching: {m['matching_pred_terms']}")
