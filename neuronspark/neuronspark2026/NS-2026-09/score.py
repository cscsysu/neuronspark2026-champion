"""Score a results.json file against valid.jsonl labels."""
import json, sys
from collections import Counter

EVENT_KEYS = ["goal_reached", "collision", "hazard", "box_on_goal", "key_collected", "portal_used"]
DYN = set('ABOK')
ENTITY_CHARS = ['A','B','O','K']

def find_pos(grid, ch):
    for y, r in enumerate(grid):
        x = r.find(ch)
        if x >= 0: return (x,y)
    return None

def main():
    res_path = sys.argv[1]
    label_path = sys.argv[2] if len(sys.argv) > 2 else 'valid.jsonl'
    with open(res_path) as f:
        preds = {p['id']: p for p in json.load(f)}
    sums = Counter()
    n = 0
    with open(label_path) as f:
        for line in f:
            d = json.loads(line)
            for q in d['queries']:
                lbl = q['label']
                pred = preds.get(q['query_id'])
                if pred is None:
                    n += 1; continue
                # dyn grid
                dt = dc = 0
                for y in range(12):
                    for x in range(12):
                        pc = pred['final_grid'][y][x]
                        lc = lbl['final_grid'][y][x]
                        if pc in DYN or lc in DYN:
                            dt += 1
                            if pc == lc: dc += 1
                sums['dyn'] += (dc/dt) if dt else 1.0
                # entity
                et = ec = 0
                for ch in ENTITY_CHARS:
                    pp = find_pos(pred['final_grid'], ch)
                    lp = find_pos(lbl['final_grid'], ch)
                    if pp is None and lp is None: continue
                    et += 1
                    if pp == lp: ec += 1
                sums['ent'] += (ec/et) if et else 1.0
                sums['ev'] += 1.0 if all(pred['events'][k] == lbl['events'][k] for k in EVENT_KEYS) else 0.0
                tt = tc = 0
                for k in EVENT_KEYS:
                    if lbl['event_timeline'][k] != 'never':
                        tt += 1
                        if pred['event_timeline'][k] == lbl['event_timeline'][k]: tc += 1
                sums['tl'] += (tc/tt) if tt else 1.0
                sums['ord'] += 1.0 if pred['event_order'] == lbl['event_order'] else 0.0
                sums['term'] += 1.0 if pred['terminal'] == lbl['terminal'] else 0.0
                n += 1
    a = {k: sums[k]/n for k in sums}
    score = (100 + 225*a['dyn'] + 225*a['ent'] + 250*a['ev']
             + 250*a['tl'] + 250*a['ord'] + 200*a['term'])
    print(f'queries: {n}')
    for k, v in a.items():
        print(f'  {k}: {v:.4f}')
    print(f'score: {score:.1f} / 1500')

if __name__ == '__main__':
    main()
