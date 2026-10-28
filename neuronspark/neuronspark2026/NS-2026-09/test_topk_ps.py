"""Inverse hybrid: top-K by aggregate, then pick best per-step."""
import json, sys, time, subprocess
from collections import Counter
sys.path.insert(0, '.')
from sim2 import simulate, EVENT_KEYS, DEFAULT_PROFILE
import predict_v3 as pv

def per_step_score(d, p):
    total = 0; correct = 0
    for ep in d['context_episodes']:
        probe_events_at_step = []
        for o in ep.get('observations', []):
            probe_events_at_step.append(set(o.get('sensor', {}).get('delta_events', [])))
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

def find_pos(grid, ch):
    for y, r in enumerate(grid):
        x = r.find(ch)
        if x >= 0: return (x,y)
    return None

# Approach: among top-K aggregate, pick one with best per-step for agent_pos override
def picker_v2(d, K_top=3):
    agg_scores = []
    for p in pv.candidate_profiles():
        s = 0
        for ep in d['context_episodes']:
            r = simulate(d['initial_full_grid'], ep['actions'], p)
            s += pv.score_probe(r, ep)
        agg_scores.append((s, p))
    agg_scores.sort(key=lambda x: -x[0])
    # Top-1 for predictions
    top1 = agg_scores[0][1]
    # Among top-K, find best per-step for agent_pos
    top_k_profiles = [p for _, p in agg_scores[:K_top]]
    best_ps = -1; best_ps_p = top1
    for p in top_k_profiles:
        ps = per_step_score(d, p)
        if ps > best_ps:
            best_ps = ps; best_ps_p = p
    return top1, best_ps_p

# Override predict
orig_predict = pv.predict_query
def predict_v3_ps(q, profile, priors, eps, d=None):
    pred = orig_predict(q, profile, priors, eps)
    if d is not None:
        # Run picker, get profile for pos
        for K in [4]:
            _, ps_profile = picker_v2(d, K)
            if ps_profile != profile:
                r_ps = simulate(q['initial_full_grid'], q['future_actions'], ps_profile)
                # Override agent A in final_grid
                grid = [list(r) for r in pred['final_grid']]
                # Remove old A
                for y in range(12):
                    for x in range(12):
                        if grid[y][x] == 'A':
                            grid[y][x] = '.'
                pa = r_ps['agent_pos']
                if pa is not None and 0 <= pa[0] < 12 and 0 <= pa[1] < 12:
                    if grid[pa[1]][pa[0]] in '.GP^v<>IH':
                        grid[pa[1]][pa[0]] = 'A'
                pred['final_grid'] = [''.join(r) for r in grid]
    return pred

# Test K=2,3,4,5,8,16
for K in [2, 3, 4, 5, 8, 16]:
    pv.predict_query = orig_predict
    def predict_K(q, profile, priors, eps, d=None):
        pred = orig_predict(q, profile, priors, eps)
        if d is not None:
            _, ps_profile = picker_v2(d, K)
            if ps_profile != profile:
                r_ps = simulate(q['initial_full_grid'], q['future_actions'], ps_profile)
                grid = [list(r) for r in pred['final_grid']]
                for y in range(12):
                    for x in range(12):
                        if grid[y][x] == 'A': grid[y][x] = '.'
                pa = r_ps['agent_pos']
                if pa is not None and 0 <= pa[0] < 12 and 0 <= pa[1] < 12:
                    if grid[pa[1]][pa[0]] in '.GP^v<>IH':
                        grid[pa[1]][pa[0]] = 'A'
                pred['final_grid'] = [''.join(r) for r in grid]
        return pred
    pv.predict_query = predict_K
    results = []
    with open('valid.jsonl') as f:
        for line in f:
            d = json.loads(line)
            profile, _ = pv.best_profile_for(d)
            priors = pv.context_priors(d)
            for q in d['queries']:
                results.append(pv.predict_query(q, profile, priors, d['context_episodes'], d=d))
    with open('temp_r.json','w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    out = subprocess.run(['python','score.py','temp_r.json','valid.jsonl'], capture_output=True, text=True).stdout
    score = float(out.strip().split()[-3])
    print(f'K={K}: {score}')
