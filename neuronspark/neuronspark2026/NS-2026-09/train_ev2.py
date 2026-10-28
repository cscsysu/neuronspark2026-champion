"""Train event classifiers on train only. See if adding to predict_v3 helps."""
import json, pickle, sys
from collections import Counter
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
sys.path.insert(0, '.')
from sim2 import simulate, EVENT_KEYS, DEFAULT_PROFILE
from predict_v3 import best_profile_for, context_priors, build_ev_features

# Train on train.jsonl
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
                f = build_ev_features(d['context_episodes'], k, priors, r)
                train_data.append({
                    'feat': f,
                    'event': k,
                    'label': int(q['label']['events'][k]),
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
                f = build_ev_features(d['context_episodes'], k, priors, r)
                valid_data.append({
                    'feat': f,
                    'event': k,
                    'label': int(q['label']['events'][k]),
                    'sim_fired': bool(f[0]),
                })

# Train per-event classifier
clfs = {}
print('\nPer-event accuracy:')
print(f'{"event":<15} {"rule":<8} {"ml":<8} {"better":<6}')
for k in EVENT_KEYS:
    Xt = np.array([d['feat'] for d in train_data if d['event'] == k])
    yt = np.array([d['label'] for d in train_data if d['event'] == k])
    Xv = np.array([d['feat'] for d in valid_data if d['event'] == k])
    yv = np.array([d['label'] for d in valid_data if d['event'] == k])
    # Rule: feat[0]=sim_fired, feat[2]=pec
    THRESH = {'goal_reached': 5, 'collision': 6, 'hazard': 5,
              'box_on_goal': 3, 'key_collected': 6, 'portal_used': 6}
    rule_pred = (Xv[:, 0].astype(bool)) | (Xv[:, 2] >= THRESH[k])
    rule_acc = (rule_pred == yv.astype(bool)).mean()
    if len(set(yt)) >= 2:
        clf = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=0)
        clf.fit(Xt, yt)
        ml_pred = clf.predict(Xv)
        ml_acc = (ml_pred == yv).mean()
        better = '+' if ml_acc > rule_acc else '-'
        clfs[k] = ('ml', clf)
    else:
        ml_acc = rule_acc
        better = '='
        clfs[k] = ('rule', None)
    print(f'{k:<15} {rule_acc:.4f}  {ml_acc:.4f}  {better}')

# Save
with open('ev_clfs_v2.pkl', 'wb') as f:
    pickle.dump(clfs, f)
