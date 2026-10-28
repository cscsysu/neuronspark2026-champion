"""For each query, find the most similar probe by action prefix.
If the probe's end position is reasonable (not blocked), use it as a candidate.
"""
import json
from collections import Counter
import sys
sys.path.insert(0, '.')
from sim2 import simulate, EVENT_KEYS, DEFAULT_PROFILE, DIRS
from predict_v3 import best_profile_for, context_priors

def find_pos(grid, ch):
    for y, r in enumerate(grid):
        x = r.find(ch)
        if x >= 0: return (x,y)
    return None

# Compare different agent_pos prediction strategies:
# 1) Sim agent_pos
# 2) Closest probe by action prefix - use that probe's end position
# 3) Hybrid

n_correct_sim = 0
n_correct_prefix = 0
n_total = 0
n_disagree = 0
n_correct_when_disagree_sim = 0
n_correct_when_disagree_prefix = 0

with open('valid.jsonl') as f:
    for line in f:
        d = json.loads(line)
        profile, _ = best_profile_for(d)
        for q in d['queries']:
            r = simulate(q['initial_full_grid'], q['future_actions'], profile)
            sim_pos = r['agent_pos']
            # Find longest action-prefix match
            q_actions = q['future_actions']
            best_prefix = -1
            best_ep = None
            for ep in d['context_episodes']:
                k = 0
                for a, b in zip(q_actions, ep['actions']):
                    if a == b: k += 1
                    else: break
                if k > best_prefix:
                    best_prefix = k
                    best_ep = ep
            # If prefix is full (q's actions are prefix of ep), end pos at step len(q_actions)
            # But we don't have intermediate positions in probe data.
            # Probe only gives us final position. So this is only useful if prefix == len(actions) AND len(probe actions) == len(q actions). And different actions.
            # Actually probe's final position gives us info IF the first `prefix` actions match perfectly.
            # If prefix == len(q_actions), and probe has more actions, then probe.final differs.
            # So probe position is not directly usable.

            real_a = find_pos(q['label']['final_grid'], 'A')
            n_total += 1
            if sim_pos == real_a: n_correct_sim += 1
            # Probe end pos
            probe_pos = find_pos(best_ep['observed_final_full_grid'], 'A')
            if probe_pos == real_a: n_correct_prefix += 1
            if sim_pos != probe_pos:
                n_disagree += 1
                if sim_pos == real_a: n_correct_when_disagree_sim += 1
                if probe_pos == real_a: n_correct_when_disagree_prefix += 1

print(f'Sim correct: {n_correct_sim}/{n_total} = {n_correct_sim/n_total:.4f}')
print(f'Prefix-probe correct: {n_correct_prefix}/{n_total} = {n_correct_prefix/n_total:.4f}')
print(f'When disagree ({n_disagree}): sim={n_correct_when_disagree_sim}, probe={n_correct_when_disagree_prefix}')
