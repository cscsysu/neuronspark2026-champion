"""For 44 cases where no profile gives correct agent_pos, what's special?"""
import json, sys
from collections import Counter
sys.path.insert(0, '.')
from sim2 import simulate, EVENT_KEYS, DEFAULT_PROFILE
from predict_v3 import candidate_profiles, best_profile_for, find_all

def find_pos(grid, ch):
    for y, r in enumerate(grid):
        x = r.find(ch)
        if x >= 0: return (x,y)
    return None

n = 0
no_profile_works = []
with open('valid.jsonl') as f:
    for line in f:
        d = json.loads(line)
        for q in d['queries']:
            real_a = find_pos(q['label']['final_grid'], 'A')
            n += 1
            any_correct = False
            for p in candidate_profiles():
                r = simulate(q['initial_full_grid'], q['future_actions'], p)
                if r['agent_pos'] == real_a:
                    any_correct = True; break
            if not any_correct:
                # Detect tiles in initial grid
                tiles = Counter()
                for row in q['initial_full_grid']:
                    for c in row:
                        tiles[c] += 1
                init_pos = find_pos(q['initial_full_grid'], 'A')
                no_profile_works.append({
                    'qid': q['query_id'],
                    'tiles': tiles,
                    'init_a': init_pos,
                    'real_a': real_a,
                    'horizon': len(q['future_actions']),
                })

print(f'Cases without correct profile: {len(no_profile_works)}/{n}')
# Tile distribution
total_tiles = Counter()
for c in no_profile_works:
    for t, count in c['tiles'].items():
        if t in 'IPHGKBOD^v<>':
            total_tiles[t] += 1
print('Special tiles present (count of cases):')
for t, c in total_tiles.most_common():
    print(f'  {t}: {c}/{len(no_profile_works)}')

# Compare to full distribution
full = Counter()
n_total = 0
with open('valid.jsonl') as f:
    for line in f:
        d = json.loads(line)
        for q in d['queries']:
            n_total += 1
            tiles = Counter()
            for row in q['initial_full_grid']:
                for c in row:
                    tiles[c] += 1
            for t in 'IPHGKBOD^v<>':
                if tiles[t] > 0: full[t] += 1
print(f'\nFull (n={n_total}):')
for t in 'IPHGKBOD^v<>':
    print(f'  {t}: {full[t]}/{n_total} = {full[t]/n_total:.3f}')

# Ratio
print(f'\nRatio (failing rate / overall rate):')
for t in 'IPHGKBOD^v<>':
    fail_rate = total_tiles[t]/len(no_profile_works) if no_profile_works else 0
    overall_rate = full[t]/n_total if n_total else 0
    if overall_rate > 0:
        print(f'  {t}: fail_rate={fail_rate:.3f}, overall_rate={overall_rate:.3f}, ratio={fail_rate/overall_rate:.2f}')
