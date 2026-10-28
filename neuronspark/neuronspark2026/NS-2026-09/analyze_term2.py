"""Check terminal consistency with predicted goal_reached event."""
import json, sys
from collections import Counter
sys.path.insert(0, '.')

with open('results_v25.json') as f:
    preds = {p['id']: p for p in json.load(f)}

# Among queries where pred goal_reached=True, what's the predicted terminal vs label?
goal_pred_yes = Counter()
goal_pred_no = Counter()
n = 0
n_potential_fix_blocked_goal = 0
for_label = Counter()
with open('valid.jsonl') as f:
    for line in f:
        d = json.loads(line)
        for q in d['queries']:
            lbl = q['label']
            pred = preds.get(q['query_id'])
            if pred is None: continue
            n += 1
            if pred['events']['goal_reached']:
                goal_pred_yes[(pred['terminal'], lbl['terminal'])] += 1
            else:
                goal_pred_no[(pred['terminal'], lbl['terminal'])] += 1

print('When pred goal_reached=True, (pred_term, label_term):')
for k, v in goal_pred_yes.most_common():
    print(f'  {k}: {v}')

print('\nWhen pred goal_reached=False, (pred_term, label_term):')
for k, v in goal_pred_no.most_common():
    print(f'  {k}: {v}')

# What if we force terminal='goal' when goal_reached=True?
fix_correct = 0; fix_wrong = 0
for k, v in goal_pred_yes.items():
    pt, lt = k
    if pt != 'goal':
        if lt == 'goal':
            fix_correct += v
        else:
            fix_wrong += v
print(f'\nForce goal when goal_reached=True: fix={fix_correct}, break={fix_wrong}')

# What if we force terminal='hazard' when hazard=True and goal_reached=False?
hazard_pred = Counter()
for q_id, pred in preds.items():
    pass  # already in main loop
