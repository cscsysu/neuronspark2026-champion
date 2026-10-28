"""Evaluate simulator on train probe rollouts and labeled queries to gauge accuracy."""
import json
from collections import Counter
from sim import simulate, EVENT_KEYS

def grid_diff(a, b):
    same = 0; total = 0
    for r1, r2 in zip(a, b):
        for c1, c2 in zip(r1, r2):
            total += 1
            if c1 == c2: same += 1
    return same, total

def find_dyn_pos(grid, ch):
    for y, r in enumerate(grid):
        x = r.find(ch)
        if x >= 0: return (x,y)
    return None

stats = Counter()
n = 0
n_full_match = 0
sum_grid_acc = 0.0
sum_event_match = 0
sum_terminal_match = 0
sum_timeline_match = 0
sum_order_match = 0
agent_pos_match = 0

with open('train.jsonl') as f:
    for li, line in enumerate(f):
        d = json.loads(line)
        for ep in d['context_episodes']:
            n += 1
            r = simulate(d['initial_full_grid'], ep['actions'])
            same, total = grid_diff(r['final_grid'], ep['observed_final_full_grid'])
            sum_grid_acc += same/total
            if r['final_grid'] == ep['observed_final_full_grid']:
                n_full_match += 1
            ev_match = all(r['events'][k] == ep['observed_final_events'][k] for k in EVENT_KEYS)
            if ev_match: sum_event_match += 1
            tm_match = all(r['event_timeline'][k] == ep['observed_final_event_timeline'][k] for k in EVENT_KEYS)
            if tm_match: sum_timeline_match += 1
            ord_match = (r['event_order'] == ep['observed_final_event_order'])
            if ord_match: sum_order_match += 1
            if r['terminal'] == ep['observed_final_terminal']: sum_terminal_match += 1
            real_a = find_dyn_pos(ep['observed_final_full_grid'], 'A')
            sim_a = find_dyn_pos(r['final_grid'], 'A')
            if real_a == sim_a: agent_pos_match += 1

print(f'Probes evaluated: {n}')
print(f'Full grid match: {n_full_match}/{n} = {n_full_match/n:.3f}')
print(f'Avg grid char acc: {sum_grid_acc/n:.4f}')
print(f'Agent pos match: {agent_pos_match/n:.3f}')
print(f'Event vector match: {sum_event_match/n:.3f}')
print(f'Event timeline match: {sum_timeline_match/n:.3f}')
print(f'Event order match: {sum_order_match/n:.3f}')
print(f'Terminal match: {sum_terminal_match/n:.3f}')
