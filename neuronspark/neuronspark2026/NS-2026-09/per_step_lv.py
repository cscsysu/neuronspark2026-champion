"""Use local_view from probe observations to verify sim's intermediate states."""
import json, sys, time
from collections import Counter
sys.path.insert(0, '.')
from sim2 import simulate, EVENT_KEYS, DEFAULT_PROFILE, DIRS
from predict_v3 import candidate_profiles, score_probe

def find_pos(grid, ch):
    for y, r in enumerate(grid):
        x = r.find(ch)
        if x >= 0: return (x,y)
    return None

def get_local_view_at_step(initial_grid, actions, profile, step):
    """Run sim for `step` actions, return 5x5 local view around agent."""
    r = simulate(initial_grid, actions[:step], profile)
    pos = r['agent_pos']
    if pos is None: return None
    cx, cy = pos
    grid = r['final_grid']
    # Build 5x5 centered on agent
    lv = [['?' for _ in range(5)] for _ in range(5)]
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            x, y = cx + dx, cy + dy
            lx, ly = dx + 2, dy + 2
            if 0 <= x < 12 and 0 <= y < 12:
                lv[ly][lx] = grid[y][x]
            # else stays '?'
    return [''.join(r) for r in lv]

def local_view_match(sim_lv, probe_lv):
    """Count how many cells match between sim and probe local views.
    Probe_lv may have ? for unobserved."""
    if sim_lv is None: return 0
    matches = 0
    for sr, pr in zip(sim_lv, probe_lv):
        for sc, pc in zip(sr, pr):
            if pc == '?': continue
            if sc == pc: matches += 1
    return matches

def per_step_full_score(d, p):
    """Score profile P using per-step events AND local_view matches."""
    correct_events = 0; total_events = 0
    lv_matches = 0; lv_total = 0
    for ep in d['context_episodes']:
        probe_events_at_step = []
        probe_lv_at_step = []
        for o in ep['observations']:
            probe_events_at_step.append(set(o['sensor'].get('delta_events', [])))
            if o.get('observation_type') == 'sparse_view' and 'local_view' in o:
                probe_lv_at_step.append(o['local_view'])
            else:
                probe_lv_at_step.append(None)
        actions = ep['actions']
        # Sim per-step events
        r = simulate(d['initial_full_grid'], actions, p)
        sim_events_at_step = {step: set() for step in range(len(actions))}
        for k in EVENT_KEYS:
            fs = r['first_step'][k]
            if fs is not None and fs in sim_events_at_step:
                sim_events_at_step[fs].add(k)
        for step in range(len(actions)):
            probe_at = probe_events_at_step[step+1] if step+1 < len(probe_events_at_step) else set()
            sim_at = sim_events_at_step[step]
            if probe_at == sim_at: correct_events += 1
            total_events += 1
            # Local view at step+1 (after action)
            if step+1 < len(probe_lv_at_step) and probe_lv_at_step[step+1] is not None:
                sim_lv = get_local_view_at_step(d['initial_full_grid'], actions, p, step+1)
                m = local_view_match(sim_lv, probe_lv_at_step[step+1])
                lv_matches += m
                lv_total += 25  # 5x5 cells
    return correct_events, total_events, lv_matches, lv_total

# Test: how well does this distinguish profiles?
import time
t0 = time.time()
n_total = 0; n_correct = 0
with open('valid.jsonl') as f:
    n_ctx = 0
    for line in f:
        d = json.loads(line)
        n_ctx += 1
        scores = []
        for p in candidate_profiles():
            ce, te, lm, lt = per_step_full_score(d, p)
            # Combined score
            score = ce / max(1, te) + lm / max(1, lt)
            scores.append((score, p))
        scores.sort(key=lambda x: -x[0])
        top_p = scores[0][1]
        for q in d['queries']:
            real_a = find_pos(q['label']['final_grid'], 'A')
            n_total += 1
            r = simulate(q['initial_full_grid'], q['future_actions'], top_p)
            if r['agent_pos'] == real_a: n_correct += 1
        if n_ctx % 30 == 0: print(f'  {n_ctx} ctx done in {time.time()-t0:.1f}s')
print(f'Per-step + local_view picker: agent_pos={n_correct}/{n_total} = {n_correct/n_total:.4f}')
print(f'Time: {time.time()-t0:.1f}s')
