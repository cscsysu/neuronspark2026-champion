"""Try training a per-query profile picker.
Goal: predict which knob configuration is correct given context features.
"""
import json, pickle, sys
from collections import Counter
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
sys.path.insert(0, '.')
from sim2 import simulate, EVENT_KEYS, DEFAULT_PROFILE
from predict_v3 import candidate_profiles, score_probe, context_priors, find_pos, find_all

# For each query, learn which knob settings were correct
# Features: probe-derived signals about the dynamics

def context_feats(d):
    """Compute features describing the dynamics from probes."""
    eps = d['context_episodes']
    feats = []
    # For each tile type, count appearances in initial grid
    init = d['initial_full_grid']
    tile_counts = Counter()
    for r in init:
        for c in r:
            tile_counts[c] += 1
    for t in 'IPHGKBOD^v<>#.':
        feats.append(tile_counts.get(t, 0))
    # Probe-derived: avg n_events fired
    avg_ev = sum(sum(ep['observed_final_events'].values()) for ep in eps) / len(eps)
    feats.append(avg_ev)
    # n_probes with each event
    for k in EVENT_KEYS:
        feats.append(sum(1 for ep in eps if ep['observed_final_events'][k]))
    # avg horizon
    feats.append(sum(len(ep['actions']) for ep in eps) / len(eps))
    # avg events per probe
    feats.append(avg_ev)
    return feats

# Build training data: each context, the right knob settings (4 binary classifications)
# We define "correct knob" via oracle: which profile gives best fit on PROBES (trying actual labels) - but we don't have query labels for context_episodes.
# Instead: for each context, find profile that maximizes correct agent_pos across queries.

# Actually let me do this with valid data (we have labels for valid queries):
all_data = []
print('Build train data: which profile is best per context (using train query labels)?')
with open('train.jsonl') as f:
    for line in f:
        d = json.loads(line)
        feats = context_feats(d)
        # Per-context best profile: which one matches most query labels?
        best_p = None; best_ac = -1
        for p in candidate_profiles():
            ac = 0
            for q in d['queries']:
                r = simulate(q['initial_full_grid'], q['future_actions'], p)
                real_a = find_pos(q['label']['final_grid'], 'A')
                if r['agent_pos'] == real_a: ac += 1
            if ac > best_ac:
                best_ac = ac; best_p = p
        all_data.append({'feats': feats, 'best_p': best_p})

print(f'Train contexts: {len(all_data)}')
# Train a separate classifier for each knob
KNOBS = ['ice_mode', 'portal_pair', 'box_walk', 'orb_blocks']
X = np.array([d['feats'] for d in all_data])
clfs = {}
for k in KNOBS:
    y = np.array([d['best_p'][k] for d in all_data])
    clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=0)
    clf.fit(X, y)
    clfs[k] = clf
    print(f'{k}: train_acc={clf.score(X, y):.4f}')

# Validate: pick profile per context using ML
# Compare to score_probe-based picker
n_total = 0
n_pickedscore = 0
n_pickedML = 0
with open('valid.jsonl') as f:
    for line in f:
        d = json.loads(line)
        feats = context_feats(d)
        # ML pick
        pick_ml = {**DEFAULT_PROFILE}
        for k in KNOBS:
            v = clfs[k].predict([feats])[0]
            pick_ml[k] = v
        # Score-based pick
        best_p = None; best_s = -1
        for p in candidate_profiles():
            s = 0.0
            for ep in d['context_episodes']:
                r = simulate(d['initial_full_grid'], ep['actions'], p)
                s += score_probe(r, ep)
            if s > best_s:
                best_s = s; best_p = p
        for q in d['queries']:
            real_a = find_pos(q['label']['final_grid'], 'A')
            n_total += 1
            r = simulate(q['initial_full_grid'], q['future_actions'], best_p)
            if r['agent_pos'] == real_a: n_pickedscore += 1
            r2 = simulate(q['initial_full_grid'], q['future_actions'], pick_ml)
            if r2['agent_pos'] == real_a: n_pickedML += 1
print(f'\nValid: score-picker {n_pickedscore}/{n_total}, ML-picker {n_pickedML}/{n_total}')
