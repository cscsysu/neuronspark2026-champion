"""Per-event per-step picker."""
import json, sys, time, subprocess
from collections import Counter
sys.path.insert(0, '.')
from sim2 import simulate, EVENT_KEYS, DEFAULT_PROFILE
import predict_v3 as pv

def per_step_event_score(d, p, target_event):
    """Score: how well does profile match probe events for a specific event."""
    correct = 0; total = 0
    for ep in d['context_episodes']:
        probe_events_at_step = []
        for o in ep.get('observations', []):
            probe_events_at_step.append(set(o.get('sensor', {}).get('delta_events', [])))
        actions = ep['actions']
        r = simulate(d['initial_full_grid'], actions, p)
        sim_first = r['first_step'][target_event]
        # Find probe's first_step for target event
        probe_first = None
        for step, evs in enumerate(probe_events_at_step):
            if target_event in evs:
                probe_first = step - 1  # delta at step+1 means action step is step
                break
        # Match: both fire at same step (or both don't fire)
        if sim_first == probe_first:
            correct += 1
        total += 1
    return correct

def find_pos(grid, ch):
    for y, r in enumerate(grid):
        x = r.find(ch)
        if x >= 0: return (x,y)
    return None

# Get per-event preferred profile
def get_event_profile(d, ev):
    best_s = -1; best_p = None
    for p in pv.candidate_profiles():
        s = per_step_event_score(d, p, ev)
        if s > best_s:
            best_s = s; best_p = p
    return best_p

# For each query, for each event, use per-event profile
import predict_v3
orig_predict = predict_v3.predict_query
def predict_per_event(q, profile, priors, eps, d=None):
    pred = orig_predict(q, profile, priors, eps, d=d)
    if d is None: return pred
    # For each event, override with per-event profile
    for ev in EVENT_KEYS:
        pe_profile = get_event_profile(d, ev)
        if pe_profile != profile:
            r_pe = simulate(q['initial_full_grid'], q['future_actions'], pe_profile)
            # Replace event prediction
            sim_pe = r_pe['events'][ev]
            pred['events'][ev] = sim_pe
            # Update timeline if needed
            # ... skip for now
    return pred

predict_v3.predict_query = predict_per_event

# Run
results = []
with open('valid.jsonl') as f:
    for line in f:
        d = json.loads(line)
        profile, _ = predict_v3.best_profile_for(d)
        priors = predict_v3.context_priors(d)
        for q in d['queries']:
            results.append(predict_v3.predict_query(q, profile, priors, d['context_episodes'], d=d))
with open('results_per_ev.json','w') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
out = subprocess.run(['python','score.py','results_per_ev.json','valid.jsonl'], capture_output=True, text=True).stdout
print(out)
