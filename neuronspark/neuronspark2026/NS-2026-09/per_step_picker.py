"""Better profile scoring: use per-step delta_events AND per-step agent position from local_view."""
import json, sys, time
from collections import Counter
sys.path.insert(0, '.')
from sim2 import simulate, EVENT_KEYS, DEFAULT_PROFILE, DIRS
from predict_v3 import candidate_profiles

def find_pos(grid, ch):
    for y, r in enumerate(grid):
        x = r.find(ch)
        if x >= 0: return (x,y)
    return None

def get_agent_pos_from_local(local_view, prev_pos, action):
    """Local_view is 5x5 centered on agent."""
    # If local_view is a 5x5 grid, agent is at (2, 2) in local coords
    # But we already know which cell the agent moved to. Just look at sim's prediction.
    return None  # not implementing for now

def reconstruct_probe_trajectory(d, ep):
    """Get agent position at each step using simulate (oracle: with no profile constraints)."""
    # If we sim with default and get matching events, we have the trajectory
    pass

def per_step_score(d, p):
    """Score profile P using per-step probe events."""
    total = 0; correct = 0
    for ep in d['context_episodes']:
        # Probe per-step events from observations
        probe_events_at_step = []  # delta_events at each step
        for o in ep['observations']:
            probe_events_at_step.append(set(o['sensor'].get('delta_events', [])))
        # Sim by running incrementally
        actions = ep['actions']
        # Build sim's first_step for this profile
        r = simulate(d['initial_full_grid'], actions, p)
        # Sim's events fired at step = first_step[k]
        sim_events_at_step = {step: set() for step in range(len(actions))}
        for k in EVENT_KEYS:
            fs = r['first_step'][k]
            if fs is not None:
                if fs in sim_events_at_step:
                    sim_events_at_step[fs].add(k)
        # Compare per step
        for step in range(len(actions)):
            probe_at = probe_events_at_step[step+1] if step+1 < len(probe_events_at_step) else set()
            sim_at = sim_events_at_step[step]
            # Match score: 1 if matches, weighted by signal
            if probe_at == sim_at:
                correct += 1
            total += 1
    return correct, total

# Compare: per-step vs aggregate scoring
def aggregate_score(d, p):
    from predict_v3 import score_probe
    s = 0
    for ep in d['context_episodes']:
        r = simulate(d['initial_full_grid'], ep['actions'], p)
        s += score_probe(r, ep)
    return s

# For each context, compare profile rankings
n_total = 0
n_picker_correct = 0  # current picker
n_per_step_correct = 0  # per-step picker
n_combined_correct = 0  # per-step + aggregate

n_ctx = 0
t0 = time.time()
with open('valid.jsonl') as f:
    for line in f:
        d = json.loads(line)
        n_ctx += 1
        # Score each profile
        agg_scores = []
        ps_scores = []
        for p in candidate_profiles():
            agg = aggregate_score(d, p)
            ps_c, ps_t = per_step_score(d, p)
            agg_scores.append((agg, p))
            ps_scores.append((ps_c, p))
        # Pick top
        agg_scores.sort(key=lambda x: -x[0])
        ps_scores.sort(key=lambda x: -x[0])
        for q in d['queries']:
            real_a = find_pos(q['label']['final_grid'], 'A')
            n_total += 1
            # Agg picker
            r1 = simulate(q['initial_full_grid'], q['future_actions'], agg_scores[0][1])
            if r1['agent_pos'] == real_a: n_picker_correct += 1
            # Per-step picker
            r2 = simulate(q['initial_full_grid'], q['future_actions'], ps_scores[0][1])
            if r2['agent_pos'] == real_a: n_per_step_correct += 1
            # Combined: agg first, ps as tie break
            agg_top = agg_scores[0][0]
            tied = [p for s, p in agg_scores if abs(s - agg_top) < 1e-6]
            if len(tied) > 1:
                # Use per-step
                best_p = max(tied, key=lambda p: per_step_score(d, p)[0])
                r3 = simulate(q['initial_full_grid'], q['future_actions'], best_p)
                if r3['agent_pos'] == real_a: n_combined_correct += 1
            else:
                if r1['agent_pos'] == real_a: n_combined_correct += 1
print(f'Aggregate picker: {n_picker_correct}/{n_total} = {n_picker_correct/n_total:.4f}')
print(f'Per-step picker: {n_per_step_correct}/{n_total} = {n_per_step_correct/n_total:.4f}')
print(f'Combined picker: {n_combined_correct}/{n_total} = {n_combined_correct/n_total:.4f}')
print(f'Time: {time.time()-t0:.1f}s')
