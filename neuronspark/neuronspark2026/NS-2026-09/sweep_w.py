"""Smarter grid search around (1,1,1,1,1.0,1.0). Also save current scoreboard."""
import json, subprocess, sys
sys.path.insert(0, '.')
import predict_v3
from sim2 import EVENT_KEYS

def make_sp(wa, wevset, wevk, wt, wo, wgrid, wtl):
    def sp(sim_r, ep):
        from predict_v3 import find_pos
        s = 0.0
        real_a = find_pos(ep['observed_final_full_grid'], 'A')
        if real_a == sim_r['agent_pos']: s += wa
        if all(sim_r['events'][k] == ep['observed_final_events'][k] for k in EVENT_KEYS): s += wevset
        s += sum(1 for k in EVENT_KEYS if sim_r['events'][k] == ep['observed_final_events'][k]) * wevk
        if sim_r['terminal'] == ep['observed_final_terminal']: s += wt
        if sim_r['event_order'] == ep['observed_final_event_order']: s += wo
        same = tot = 0
        for r1, r2 in zip(sim_r['final_grid'], ep['observed_final_full_grid']):
            for c1, c2 in zip(r1, r2):
                tot += 1
                if c1 == c2: same += 1
        s += (same/tot) * wgrid
        for k in EVENT_KEYS:
            if ep['observed_final_event_timeline'][k] != 'never':
                if sim_r['event_timeline'][k] == ep['observed_final_event_timeline'][k]:
                    s += wtl
        return s
    return sp

def eval_w(w):
    predict_v3.score_probe = make_sp(*w)
    results = []
    with open('valid.jsonl') as f:
        for line in f:
            d = json.loads(line)
            profile, _ = predict_v3.best_profile_for(d)
            priors = predict_v3.context_priors(d)
            for q in d['queries']:
                results.append(predict_v3.predict_query(q, profile, priors, d['context_episodes']))
    with open('temp_r.json','w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    out = subprocess.run(['python','score.py','temp_r.json','valid.jsonl'], capture_output=True, text=True).stdout
    return float(out.strip().split()[-3])

# Current best: (1, 1, 0.3, 1, 1, 1.0, 1.0) = 1320.7
best_w = (1, 1, 0.3, 1, 1, 1.0, 1.0)
best_score = eval_w(best_w)
print(f'Baseline: {best_w}: {best_score}')

# Coordinate-wise search around best
import sys
sys.stdout.flush()

for round_n in range(2):
    for i, vs in enumerate([
        [0.5, 1, 1.5, 2],   # wa
        [0.5, 1, 1.5, 2],   # wevset
        [0.1, 0.3, 0.5],    # wevk
        [0.5, 1, 1.5],      # wt
        [0.5, 1, 1.5, 2],   # wo
        [0.5, 1, 1.5, 2, 3],# wgrid
        [0.5, 1, 1.5, 2],   # wtl
    ]):
        for v in vs:
            new_w = list(best_w)
            new_w[i] = v
            new_w = tuple(new_w)
            if new_w == best_w: continue
            s = eval_w(new_w)
            if s > best_score:
                best_score = s
                best_w = new_w
                print(f'NEW: {best_w}: {best_score}')
                sys.stdout.flush()
print(f'Final: {best_w}: {best_score}')
