"""Combine per-step with aggregate score, find best mix."""
import json, sys, time, pickle, os, subprocess
from collections import Counter
sys.path.insert(0, '.')
from sim2 import simulate, EVENT_KEYS, DEFAULT_PROFILE
import predict_v3 as pv

def per_step_score(d, p):
    total = 0; correct = 0
    for ep in d['context_episodes']:
        probe_events_at_step = []
        for o in ep['observations']:
            probe_events_at_step.append(set(o['sensor'].get('delta_events', [])))
        actions = ep['actions']
        r = simulate(d['initial_full_grid'], actions, p)
        sim_events_at_step = {step: set() for step in range(len(actions))}
        for k in EVENT_KEYS:
            fs = r['first_step'][k]
            if fs is not None and fs in sim_events_at_step:
                sim_events_at_step[fs].add(k)
        for step in range(len(actions)):
            probe_at = probe_events_at_step[step+1] if step+1 < len(probe_events_at_step) else set()
            sim_at = sim_events_at_step[step]
            if probe_at == sim_at: correct += 1
            total += 1
    return correct

orig_best_profile_for = pv.best_profile_for
W_PS = 0
def patched_best(d):
    best_p = None; best_s = -1
    for p in pv.candidate_profiles():
        s = 0.0
        for ep in d['context_episodes']:
            r = simulate(d['initial_full_grid'], ep['actions'], p)
            s += pv.score_probe(r, ep)
        s += W_PS * per_step_score(d, p)
        if s > best_s:
            best_s = s; best_p = p
    return best_p, best_s

pv.best_profile_for = patched_best

best_score = 0
best_w = 0
for w_ps in [0, 0.1, 0.5, 1, 2, 5, 10, 20]:
    W_PS = w_ps
    # Patch pv's W_PS
    def patched_best_w(d):
        best_p = None; best_s = -1
        for p in pv.candidate_profiles():
            s = 0.0
            for ep in d['context_episodes']:
                r = simulate(d['initial_full_grid'], ep['actions'], p)
                s += pv.score_probe(r, ep)
            s += w_ps * per_step_score(d, p)
            if s > best_s:
                best_s = s; best_p = p
        return best_p, best_s
    pv.best_profile_for = patched_best_w
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
