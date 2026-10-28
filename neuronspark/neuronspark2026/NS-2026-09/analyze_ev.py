"""For hazard FN cases, how many probes had hazard?"""
import json
from collections import Counter
from sim2 import simulate, EVENT_KEYS, DEFAULT_PROFILE
from predict_v3 import best_profile_for, context_priors

with open('results_v21.json') as f:
    preds = {p['id']: p for p in json.load(f)}

# Hazard FN: predicted no hazard, actual hazard
# Hazard FP: predicted hazard, actual no hazard
fn_probe_counts = []
fp_probe_counts = []

with open('valid.jsonl') as f:
    for line in f:
        d = json.loads(line)
        n_probes_with_hazard = sum(1 for ep in d['context_episodes'] if ep['observed_final_events']['hazard'])
        for q in d['queries']:
            lbl = q['label']
            pred = preds.get(q['query_id'])
            if pred is None: continue
            if not pred['events']['hazard'] and lbl['events']['hazard']:
                fn_probe_counts.append(n_probes_with_hazard)
            if pred['events']['hazard'] and not lbl['events']['hazard']:
                fp_probe_counts.append(n_probes_with_hazard)

print(f'Hazard FN (n={len(fn_probe_counts)}): probe count distribution: {Counter(fn_probe_counts)}')
print(f'Hazard FP (n={len(fp_probe_counts)}): probe count distribution: {Counter(fp_probe_counts)}')

# Same for other events
for event in EVENT_KEYS:
    fn = []
    fp = []
    with open('valid.jsonl') as f:
        for line in f:
            d = json.loads(line)
            n_probes_event = sum(1 for ep in d['context_episodes'] if ep['observed_final_events'][event])
            for q in d['queries']:
                lbl = q['label']
                pred = preds.get(q['query_id'])
                if pred is None: continue
                if not pred['events'][event] and lbl['events'][event]:
                    fn.append(n_probes_event)
                if pred['events'][event] and not lbl['events'][event]:
                    fp.append(n_probes_event)
    print(f'\n{event}:')
    print(f'  FN (n={len(fn)}): probe count: {Counter(fn).most_common()}')
    print(f'  FP (n={len(fp)}): probe count: {Counter(fp).most_common()}')
