"""Train timeline classifiers (3-way: early/mid/late) on train.jsonl only."""
import json, pickle, sys
from collections import Counter
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
sys.path.insert(0, '.')
from sim2 import simulate, EVENT_KEYS, DEFAULT_PROFILE
from predict_v3 import best_profile_for, context_priors, build_tl_features

TL_LABELS = ['early','mid','late']

train_data = []
print('Building train features...')
with open('train.jsonl') as f:
    for line in f:
        d = json.loads(line)
        profile, _ = best_profile_for(d)
        priors = context_priors(d)
        for q in d['queries']:
            r = simulate(q['initial_full_grid'], q['future_actions'], profile)
            for k in EVENT_KEYS:
                if q['label']['event_timeline'][k] == 'never': continue
                if not q['label']['events'][k]: continue
                f = build_tl_features(d['context_episodes'], q['future_actions'], q['initial_full_grid'], k, priors, r)
                train_data.append({
                    'feat': f,
                    'event': k,
                    'label': TL_LABELS.index(q['label']['event_timeline'][k]),
                    'sim_fired': bool(f[0]),
                })

print('Building valid features...')
valid_data = []
with open('valid.jsonl') as f:
    for line in f:
        d = json.loads(line)
        profile, _ = best_profile_for(d)
        priors = context_priors(d)
        for q in d['queries']:
            r = simulate(q['initial_full_grid'], q['future_actions'], profile)
            for k in EVENT_KEYS:
                if q['label']['event_timeline'][k] == 'never': continue
                if not q['label']['events'][k]: continue
                f = build_tl_features(d['context_episodes'], q['future_actions'], q['initial_full_grid'], k, priors, r)
                valid_data.append({
                    'feat': f,
                    'event': k,
                    'label': TL_LABELS.index(q['label']['event_timeline'][k]),
                    'sim_fired': bool(f[0]),
                    'sim_step': f[1],
                })

# Per-event ML vs sim's bucket
clfs = {}
print('\nPer-event TIMELINE accuracy on valid (only sim_fired cases):')
print(f'{"event":<15} {"sim":<8} {"ml":<8} {"better":<6}')
for k in EVENT_KEYS:
    Xt = np.array([d['feat'] for d in train_data if d['event'] == k and d['sim_fired']])
    yt = np.array([d['label'] for d in train_data if d['event'] == k and d['sim_fired']])
    Xv = np.array([d['feat'] for d in valid_data if d['event'] == k and d['sim_fired']])
    yv = np.array([d['label'] for d in valid_data if d['event'] == k and d['sim_fired']])
    if len(Xv) == 0: continue
    # Sim bucket: based on f[1] (sim_step / horizon)
    def bucket(s):
        if s < 0: return 0
        if s < 1/3: return 0
        if s < 2/3: return 1
        return 2
    sim_pred = np.array([bucket(s) for s in Xv[:, 1]])
    sim_acc = (sim_pred == yv).mean()
    if len(set(yt)) >= 2 and len(Xt) >= 50:
        clf = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=0)
        clf.fit(Xt, yt)
        ml_pred = clf.predict(Xv)
        ml_acc = (ml_pred == yv).mean()
        better = '+' if ml_acc > sim_acc else '-'
        clfs[k] = ('ml', clf) if ml_acc > sim_acc else ('rule', None)
    else:
        ml_acc = sim_acc
        better = '='
        clfs[k] = ('rule', None)
    print(f'{k:<15} {sim_acc:.4f}  {ml_acc:.4f}  {better} ({len(Xv)} valid samples)')

# Save
with open('tl_clfs2.pkl', 'wb') as f:
    pickle.dump(clfs, f)
print('saved tl_clfs2.pkl')
