"""When does probe consensus differ from sim, and which is right?"""
import json
from collections import Counter

EVENT_KEYS = ["goal_reached", "collision", "hazard", "box_on_goal", "key_collected", "portal_used"]

with open('results_v18.json') as f:
    preds = {p['id']: p for p in json.load(f)}

# When probes agree strongly (4+/5 same bucket) and sim disagrees, who wins?
# Test threshold 3, 4, 5 (out of probes that have non-never)
for THRESH_CONSENSUS in [3, 4, 5]:
    print(f'\n=== Consensus threshold = {THRESH_CONSENSUS} (events fired in probe with that bucket) ===')
    flips = Counter()  # (correct, total) where consensus disagrees with sim
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
                    if lbl['event_timeline'][k] != 'never':
                        # Modal bucket and its count
                        if not probe_tl[k]: continue
                        modal, count = probe_tl[k].most_common(1)[0]
                        if count < THRESH_CONSENSUS: continue
                        cur = pred['event_timeline'][k]
                        if cur != modal:
                            # disagree case: record who's right
                            if modal == lbl['event_timeline'][k]:
                                flips[(k, 'modal_right')] += 1
                            elif cur == lbl['event_timeline'][k]:
                                flips[(k, 'sim_right')] += 1
                            else:
                                flips[(k, 'both_wrong')] += 1
    for k in EVENT_KEYS:
        m = flips[(k,'modal_right')]; s = flips[(k,'sim_right')]; b = flips[(k,'both_wrong')]
        if m+s+b > 0:
            print(f'  {k}: modal_right={m}, sim_right={s}, both_wrong={b}')
