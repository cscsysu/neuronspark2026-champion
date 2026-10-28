"""Add per-step events scoring as part of score_probe weight."""
import json, sys, time, subprocess
from collections import Counter
sys.path.insert(0, '.')
from sim2 import simulate, EVENT_KEYS, DEFAULT_PROFILE
import predict_v3 as pv

def find_pos(grid, ch):
    for y, r in enumerate(grid):
        x = r.find(ch)
        if x >= 0: return (x,y)
    return None

def per_step_score_for_ep(ep, sim_r, actions):
    """Per-step events match score for one probe."""
    probe_events_at_step = []
    for o in ep['observations']:
        probe_events_at_step.append(set(o['sensor'].get('delta_events', [])))
    sim_events_at_step = {step: set() for step in range(len(actions))}
    for k in EVENT_KEYS:
        fs = sim_r['first_step'][k]
        if fs is not None and fs in sim_events_at_step:
            sim_events_at_step[fs].add(k)
    matches = 0
    for step in range(len(actions)):
        probe_at = probe_events_at_step[step+1] if step+1 < len(probe_events_at_step) else set()
        sim_at = sim_events_at_step[step]
        if probe_at == sim_at: matches += 1
    return matches / max(1, len(actions))

# Modify score_probe to include per-step
W_PS = 0  # weight for per-step
def score_probe_v2(sim_r, ep, actions=None):
    s = 0.0
    real_a = find_pos(ep['observed_final_full_grid'], 'A')
    if real_a == sim_r['agent_pos']: s += 1.0
    if all(sim_r['events'][k] == ep['observed_final_events'][k] for k in EVENT_KEYS): s += 1.0
    s += sum(1 for k in EVENT_KEYS if sim_r['events'][k] == ep['observed_final_events'][k]) * 0.3
    if sim_r['terminal'] == ep['observed_final_terminal']: s += 1.0
    if sim_r['event_order'] == ep['observed_final_event_order']: s += 1.0
    same = tot = 0
    for r1, r2 in zip(sim_r['final_grid'], ep['observed_final_full_grid']):
        for c1, c2 in zip(r1, r2):
            tot += 1
            if c1 == c2: same += 1
    s += (same/tot) * 1.0
    for k in EVENT_KEYS:
        if ep['observed_final_event_timeline'][k] != 'never':
            if sim_r['event_timeline'][k] == ep['observed_final_event_timeline'][k]:
                s += 1.5
    if actions is not None:
        ps_score = per_step_score_for_ep(ep, sim_r, actions)
        s += W_PS * ps_score
    return s

# Patch best_profile_for to pass actions
def patched_best(d):
    best_p = None; best_s = -1
    for p in pv.candidate_profiles():
        s = 0.0
        for ep in d['context_episodes']:
            r = simulate(d['initial_full_grid'], ep['actions'], p)
            s += score_probe_v2(r, ep, actions=ep['actions'])
        if s > best_s:
            best_s = s; best_p = p
    return best_p, best_s

pv.best_profile_for = patched_best

best_score = 0; best_w = 0
for w_ps in [0, 0.5, 1, 2, 3, 5, 10]:
    W_PS = w_ps
    # Re-patch
    def make_patched(w):
        def patched(d):
            best_p = None; best_s = -1
            for p in pv.candidate_profiles():
                s = 0.0
                for ep in d['context_episodes']:
                    r = simulate(d['initial_full_grid'], ep['actions'], p)
                    s += score_probe_v2(r, ep, actions=ep['actions']) * 1.0  # keep base
                    # add per-step
                    ps_score = per_step_score_for_ep(ep, r, ep['actions'])
                    s += w * ps_score
                if s > best_s:
                    best_s = s; best_p = p
            return best_p, best_s
        return patched
    pv.best_profile_for = make_patched(w_ps)
    results = []
    with open('valid.jsonl') as f:
        for line in f:
            d = json.loads(line)
            profile, _ = pv.best_profile_for(d)
            priors = pv.context_priors(d)
            for q in d['queries']:
                results.append(pv.predict_query(q, profile, priors, d['context_episodes']))
    with open('temp_r.json','w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    out = subprocess.run(['python','score.py','temp_r.json','valid.jsonl'], capture_output=True, text=True).stdout
    score = float(out.strip().split()[-3])
    print(f'w_ps={w_ps}: {score}')
    if score > best_score:
        best_score = score; best_w = w_ps
print(f'Best w_ps={best_w}: {best_score}')
