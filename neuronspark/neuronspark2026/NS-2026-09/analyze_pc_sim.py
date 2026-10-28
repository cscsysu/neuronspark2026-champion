"""Analysis: cases where probe count is high but sim says no."""
import json, sys
from collections import Counter
sys.path.insert(0, '.')
from sim2 import simulate, EVENT_KEYS, DEFAULT_PROFILE
from predict_v3 import best_profile_for

# For each event, breakdown by probe count
data = {k: Counter() for k in EVENT_KEYS}

with open('valid.jsonl') as f:
    for line in f:
        d = json.loads(line)
        profile, _ = best_profile_for(d)
        for q in d['queries']:
            r = simulate(q['initial_full_grid'], q['future_actions'], profile)
            for k in EVENT_KEYS:
                pec_k = sum(1 for ep in d['context_episodes'] if ep['observed_final_events'][k])
                sim_says = r['events'][k]
                label_says = q['label']['events'][k]
                data[k][(pec_k, sim_says, label_says)] += 1

for k in EVENT_KEYS:
    print(f'\n{k}: (probe_count, sim, label) -> count')
    for (pc, s, l), c in sorted(data[k].items()):
        print(f'  ({pc}, sim={s}, label={l}): {c}')
