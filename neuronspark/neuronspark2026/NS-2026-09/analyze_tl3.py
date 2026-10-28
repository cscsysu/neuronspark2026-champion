"""Detailed: when sim fires hazard but bucket is wrong, what helps?"""
import json
from collections import Counter

EVENT_KEYS = ["goal_reached", "collision", "hazard", "box_on_goal", "key_collected", "portal_used"]

with open('results_v18.json') as f:
    preds = {p['id']: p for p in json.load(f)}

# For each event, when sim's bucket disagrees with label and probes have a modal,
# how often does probe modal match?
totals = {}  # k -> (sim_correct, modal_correct, both_correct, both_wrong, total)

with open('valid.jsonl') as f:
    for line in f:
        d = json.loads(line)
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
                if lbl['event_timeline'][k] != 'never' and pred['event_timeline'][k] != 'never':
                    if not probe_tl[k]: continue
                    modal, count = probe_tl[k].most_common(1)[0]
                    cur = pred['event_timeline'][k]
                    target = lbl['event_timeline'][k]
                    sc = (cur == target)
                    mc = (modal == target)
                    key = (k, count)
                    if key not in totals:
                        totals[key] = [0, 0, 0]  # sim_right, modal_right, total
                    totals[key][2] += 1
                    if sc: totals[key][0] += 1
                    if mc: totals[key][1] += 1

print('event, count_in_probes, sim_correct, modal_correct, total')
for k in EVENT_KEYS:
    for cnt in range(1, 6):
        if (k, cnt) in totals:
            sc, mc, tot = totals[(k, cnt)]
            print(f'  {k}, count={cnt}: sim={sc}/{tot}, modal={mc}/{tot}')
