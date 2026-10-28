"""Try to improve timeline by using probe consistency."""
import json
from collections import Counter

EVENT_KEYS = ["goal_reached", "collision", "hazard", "box_on_goal", "key_collected", "portal_used"]

with open('results_v18.json') as f:
    preds = {p['id']: p for p in json.load(f)}

# For each query: examine how often the probe modal (when available) is correct vs sim's bucket
mod_correct = Counter()  # probe modal matches label
sim_correct = Counter()  # current sim bucket matches label
agree = Counter()  # both predict same
ctx_for_query = {}
sim_buckets = {}
n = 0
total = Counter()

with open('valid.jsonl') as f:
    for line in f:
        d = json.loads(line)
        # Compute probe modal timeline per event
        probe_tl = {k: Counter() for k in EVENT_KEYS}
        for ep in d['context_episodes']:
            for k in EVENT_KEYS:
                tl = ep['observed_final_event_timeline'][k]
                if tl != 'never':
                    probe_tl[k][tl] += 1
        for q in d['queries']:
            lbl = q['label']
            pred = preds.get(q['query_id'])
            if pred is None: continue
            for k in EVENT_KEYS:
                if lbl['event_timeline'][k] != 'never':
                    total[k] += 1
                    cur = pred['event_timeline'][k]
                    modal = probe_tl[k].most_common(1)[0][0] if probe_tl[k] else 'never'
                    if cur == lbl['event_timeline'][k]:
                        sim_correct[k] += 1
                    if modal == lbl['event_timeline'][k]:
                        mod_correct[k] += 1
                    if cur == modal:
                        agree[k] += 1

print('event | sim_correct | modal_correct | agree | total')
for k in EVENT_KEYS:
    print(f'  {k}: {sim_correct[k]} / {mod_correct[k]} / {agree[k]} / {total[k]}')
