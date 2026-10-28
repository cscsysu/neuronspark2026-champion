"""Check if agent's predicted path passes through H tile, helping hazard FN detection."""
import json, sys
from collections import Counter
sys.path.insert(0, '.')
from sim2 import simulate, EVENT_KEYS, DEFAULT_PROFILE, DIRS
from predict_v3 import best_profile_for, context_priors

def find_pos(grid, ch):
    for y, r in enumerate(grid):
        x = r.find(ch)
        if x >= 0: return (x,y)
    return None

with open('results_v22.json') as f:
    preds = {p['id']: p for p in json.load(f)}

# For hazard FN cases, check if H is adjacent to or in path
fn_path_h = 0
fn_total = 0
fp_path_h = 0
fp_total = 0
all_path_h_label = Counter()  # label hazard = T/F when H in path
with open('valid.jsonl') as f:
    for line in f:
        d = json.loads(line)
        h_positions = []
        for y, r in enumerate(d['initial_full_grid']):
            for x, c in enumerate(r):
                if c == 'H': h_positions.append((x,y))
        for q in d['queries']:
            lbl = q['label']
            pred = preds.get(q['query_id'])
            if pred is None: continue
            # trace agent path through actions assuming free movement
            start_a = find_pos(q['initial_full_grid'], 'A')
            if start_a is None: continue
            cx, cy = start_a
            path = [(cx, cy)]
            for a in q['future_actions']:
                if a in DIRS:
                    dx, dy = DIRS[a]
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < 12 and 0 <= ny < 12:
                        cx, cy = nx, ny
                path.append((cx, cy))
            # Check if any H position is in or adjacent to path
            h_in_path = any((hx, hy) in path for hx, hy in h_positions)
            # Aggregate
            all_path_h_label[(h_in_path, lbl['events']['hazard'])] += 1
            # FN: predicted no hazard, actual hazard
            if not pred['events']['hazard'] and lbl['events']['hazard']:
                fn_total += 1
                if h_in_path: fn_path_h += 1
            # FP: predicted hazard, no actual
            if pred['events']['hazard'] and not lbl['events']['hazard']:
                fp_total += 1
                if h_in_path: fp_path_h += 1

print(f'Hazard FN ({fn_total}): {fn_path_h} have H in naive path')
print(f'Hazard FP ({fp_total}): {fp_path_h} have H in naive path')
print(f'Overall (h_in_path, hazard_label): {all_path_h_label}')

# What if we add a rule: predict hazard if H in path AND probe_count >= 2?
print('\nRule analysis: hazard = sim OR pec>=5 OR (h_in_path AND pec>=N):')
for haz_thresh in [1, 2, 3]:
    new_correct = 0
    n = 0
    n_fixed_fn = 0
    n_added_fp = 0
    with open('valid.jsonl') as f:
        for line in f:
            d = json.loads(line)
            h_positions = []
            for y, r in enumerate(d['initial_full_grid']):
                for x, c in enumerate(r):
                    if c == 'H': h_positions.append((x,y))
            n_haz = sum(1 for ep in d['context_episodes'] if ep['observed_final_events']['hazard'])
            for q in d['queries']:
                lbl = q['label']
                pred = preds.get(q['query_id'])
                if pred is None: continue
                start_a = find_pos(q['initial_full_grid'], 'A')
                if start_a is None: continue
                cx, cy = start_a
                path = [(cx, cy)]
                for a in q['future_actions']:
                    if a in DIRS:
                        dx, dy = DIRS[a]
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < 12 and 0 <= ny < 12:
                            cx, cy = nx, ny
                    path.append((cx, cy))
                h_in_path = any((hx, hy) in path for hx, hy in h_positions)
                # New rule
                new_haz = pred['events']['hazard'] or (h_in_path and n_haz >= haz_thresh)
                old_haz = pred['events']['hazard']
                if old_haz != new_haz:
                    if new_haz == lbl['events']['hazard']:
                        n_fixed_fn += 1
                    else:
                        n_added_fp += 1
                if new_haz == lbl['events']['hazard']:
                    new_correct += 1
                n += 1
    print(f'  thresh={haz_thresh}: fixed FN={n_fixed_fn}, added FP={n_added_fp}, net={n_fixed_fn-n_added_fp}, accuracy={new_correct/n:.4f}')
