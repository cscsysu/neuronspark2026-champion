"""Test: if 0 probes have an event, force prediction to False."""
import json, sys
from collections import Counter
sys.path.insert(0, '.')
from sim2 import simulate, EVENT_KEYS, DEFAULT_PROFILE
from predict_v3 import best_profile_for, context_priors

# For each event: when 0 probes have it, what's our pred and label?
n = 0
sim_fired_when_no_probes = Counter()  # by event: (pred, label)
total_when_no_probes = Counter()

with open('valid.jsonl') as f:
    for line in f:
        d = json.loads(line)
        profile, _ = best_profile_for(d)
        for q in d['queries']:
            r = simulate(q['initial_full_grid'], q['future_actions'], profile)
            for k in EVENT_KEYS:
                pec_k = sum(1 for ep in d['context_episodes'] if ep['observed_final_events'][k])
                if pec_k == 0:  # 0 probes have it
                    total_when_no_probes[k] += 1
                    sim_says = r['events'][k]
                    label_says = q['label']['events'][k]
                    sim_fired_when_no_probes[(k, sim_says, label_says)] += 1

print('When 0 probes have event, (sim_pred, label):')
for k in EVENT_KEYS:
    print(f'  {k} (n={total_when_no_probes[k]}):')
    for (kk, s, l), c in sorted(sim_fired_when_no_probes.items()):
        if kk == k:
            print(f'    sim={s}, label={l}: {c}')

# How many cases would forcing False to "0 probes" break?
# Force False = any case where label=True is broken
n_break = 0
n_helped = 0
for (k, s, l), c in sim_fired_when_no_probes.items():
    if l == False:  # force to False is correct
        if s == True:  # we predicted True (wrong), now correct
            n_helped += c
    else:  # label True, force False is wrong
        if s == True:  # we predicted True (correct), now wrong
            n_break += c
print(f'\nForce False when 0 probes: helps={n_helped}, breaks={n_break}')
