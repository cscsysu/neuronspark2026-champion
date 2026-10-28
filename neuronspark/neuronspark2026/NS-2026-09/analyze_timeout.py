"""Look at sim's last_blocked state for blocked->timeout misses."""
import json
from collections import Counter
from sim2 import simulate, EVENT_KEYS, DEFAULT_PROFILE
from predict_v3 import best_profile_for, context_priors

with open('results_v19.json') as f:
    preds = {p['id']: p for p in json.load(f)}

# Look at blocked->timeout cases: examine if sim shows last_blocked=False & no events fire
n_blocktime = 0
last_blocked_stats = Counter()
events_stats = Counter()
with open('valid.jsonl') as f:
    for line in f:
        d = json.loads(line)
        profile, _ = best_profile_for(d)
        for q in d['queries']:
            lbl = q['label']
            pred = preds.get(q['query_id'])
            if pred is None: continue
            if pred['terminal']=='blocked' and lbl['terminal']=='timeout':
                r = simulate(q['initial_full_grid'], q['future_actions'], profile)
                events_fired = [k for k in EVENT_KEYS if r['events'][k]]
                n_blocktime += 1
                last_blocked_stats[r['last_blocked']] += 1
                events_stats[tuple(events_fired)] += 1
                print(f"  q={q['query_id']}: last_blocked={r['last_blocked']}, sim_term={r['terminal']}, events_fired={events_fired}")

# Compare overall: when sim has last_blocked=False AND no events, which terminal?
all_no_event_no_block = Counter()
with open('valid.jsonl') as f:
    for line in f:
        d = json.loads(line)
        profile, _ = best_profile_for(d)
        for q in d['queries']:
            r = simulate(q['initial_full_grid'], q['future_actions'], profile)
            events_fired = sum(1 for k in EVENT_KEYS if r['events'][k])
            if not r['last_blocked'] and events_fired == 0:
                lbl_term = q['label']['terminal']
                all_no_event_no_block[lbl_term] += 1
print(f'\nWhen sim has last_blocked=False AND no events fire ({sum(all_no_event_no_block.values())} cases), label terminal:')
print(all_no_event_no_block.most_common())
