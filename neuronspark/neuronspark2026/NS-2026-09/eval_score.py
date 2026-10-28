"""Evaluate simulator on train QUERIES (labeled). Computes score using the official 1500-pt formula."""
import json
from collections import Counter
from sim import simulate, EVENT_KEYS

def find_dyn_pos(grid, ch):
    for y, r in enumerate(grid):
        x = r.find(ch)
        if x >= 0: return (x,y)
    return None

DYNAMIC_CHARS = set('ABOK')
ENTITY_CHARS = ['A','B','O','K']

def score_one(pred, label):
    # dynamic grid: positions where either pred or label has dyn char
    dyn_total = 0
    dyn_correct = 0
    for y in range(12):
        for x in range(12):
            pc = pred['final_grid'][y][x]
            lc = label['final_grid'][y][x]
            if pc in DYNAMIC_CHARS or lc in DYNAMIC_CHARS:
                dyn_total += 1
                if pc == lc:
                    dyn_correct += 1
    dyn_acc = (dyn_correct / dyn_total) if dyn_total > 0 else 1.0
    # entity accuracy: positions of A/B/O/K
    e_total = 0; e_correct = 0
    for ch in ENTITY_CHARS:
        pp = find_dyn_pos(pred['final_grid'], ch)
        lp = find_dyn_pos(label['final_grid'], ch)
        if pp is None and lp is None:
            continue
        e_total += 1
        if pp == lp:
            e_correct += 1
    ent_acc = (e_correct / e_total) if e_total > 0 else 1.0
    # event vector
    ev_acc = 1.0 if all(pred['events'][k] == label['events'][k] for k in EVENT_KEYS) else 0.0
    # event timeline (only on non-never events in label)
    tl_total = 0; tl_correct = 0
    for k in EVENT_KEYS:
        if label['event_timeline'][k] != 'never':
            tl_total += 1
            if pred['event_timeline'][k] == label['event_timeline'][k]:
                tl_correct += 1
    tl_acc = (tl_correct / tl_total) if tl_total > 0 else 1.0
    # event order
    ord_acc = 1.0 if pred['event_order'] == label['event_order'] else 0.0
    # terminal
    term_acc = 1.0 if pred['terminal'] == label['terminal'] else 0.0
    return {
        'dyn': dyn_acc, 'ent': ent_acc, 'ev': ev_acc,
        'tl': tl_acc, 'ord': ord_acc, 'term': term_acc,
    }

def main(path):
    sums = Counter()
    n = 0
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            for q in d['queries']:
                n += 1
                r = simulate(q['initial_full_grid'], q['future_actions'])
                s = score_one(r, q['label'])
                for k, v in s.items():
                    sums[k] += v
    avg = {k: sums[k] / n for k in sums}
    score = (100
             + 225 * avg['dyn']
             + 225 * avg['ent']
             + 250 * avg['ev']
             + 250 * avg['tl']
             + 250 * avg['ord']
             + 200 * avg['term'])
    print(f'Queries: {n}')
    for k, v in avg.items():
        print(f'  {k}: {v:.4f}')
    print(f'Estimated score (out of 1500): {score:.1f}')

if __name__ == '__main__':
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else 'valid.jsonl')
