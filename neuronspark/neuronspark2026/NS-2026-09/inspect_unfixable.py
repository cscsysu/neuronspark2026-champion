"""Look at one specific case where no profile gives correct A position."""
import json, sys
sys.path.insert(0, '.')
from sim2 import simulate, EVENT_KEYS, DEFAULT_PROFILE
from predict_v3 import candidate_profiles

def find_pos(grid, ch):
    for y, r in enumerate(grid):
        x = r.find(ch)
        if x >= 0: return (x,y)
    return None

with open('valid.jsonl') as f:
    for line in f:
        d = json.loads(line)
        for q in d['queries']:
            real_a = find_pos(q['label']['final_grid'], 'A')
            any_correct = False
            best_p = None
            min_dist = 999
            for p in candidate_profiles():
                r = simulate(q['initial_full_grid'], q['future_actions'], p)
                if r['agent_pos'] == real_a:
                    any_correct = True
                    break
                if r['agent_pos']:
                    dist = abs(r['agent_pos'][0] - real_a[0]) + abs(r['agent_pos'][1] - real_a[1])
                    if dist < min_dist:
                        min_dist = dist
                        best_p = p
            if not any_correct:
                # Print this case
                print(f'\nQuery {q["query_id"]}:')
                print(f'  Initial agent: {find_pos(q["initial_full_grid"], "A")}')
                print(f'  Real final A: {real_a}')
                print(f'  Future actions: {"".join(a[0] for a in q["future_actions"])}')
                r = simulate(q['initial_full_grid'], q['future_actions'], best_p)
                print(f'  Best sim A: {r["agent_pos"]}, dist={min_dist}')
                print(f'  Best profile: {best_p}')
                # Show grids
                print('  Initial grid:')
                for row in q['initial_full_grid']:
                    print('   ', row)
                print('  Label final grid:')
                for row in q['label']['final_grid']:
                    print('   ', row)
                # Print events from label
                print(f'  Label events: {[k for k in EVENT_KEYS if q["label"]["events"][k]]}')
                print(f'  Sim events: {[k for k in EVENT_KEYS if r["events"][k]]}')
                # First non-fixable case
                break
        else:
            continue
        break
