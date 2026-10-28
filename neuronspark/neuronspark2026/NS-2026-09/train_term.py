"""Train term classifier on train.jsonl."""
import json, pickle, sys
from collections import Counter
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
sys.path.insert(0, '.')
from sim2 import simulate, EVENT_KEYS, DEFAULT_PROFILE
from predict_v3 import best_profile_for, context_priors, build_term_features

TERM_LABELS = ['blocked','timeout','goal','hazard']

train_data = []
print('Building train features...')
with open('train.jsonl') as f:
    for line in f:
        d = json.loads(line)
        profile, _ = best_profile_for(d)
        priors = context_priors(d)
        for q in d['queries']:
            r = simulate(q['initial_full_grid'], q['future_actions'], profile)
            f = build_term_features(d['context_episodes'], q['future_actions'], q['initial_full_grid'], priors, r)
            train_data.append({'feat': f, 'label': TERM_LABELS.index(q['label']['terminal'])})

print('Building valid features...')
valid_data = []
with open('valid.jsonl') as f:
    for line in f:
        d = json.loads(line)
        profile, _ = best_profile_for(d)
        priors = context_priors(d)
        for q in d['queries']:
            r = simulate(q['initial_full_grid'], q['future_actions'], profile)
            f = build_term_features(d['context_episodes'], q['future_actions'], q['initial_full_grid'], priors, r)
            valid_data.append({'feat': f, 'label': TERM_LABELS.index(q['label']['terminal'])})

X = np.array([d['feat'] for d in train_data])
y = np.array([d['label'] for d in train_data])
Xv = np.array([d['feat'] for d in valid_data])
yv = np.array([d['label'] for d in valid_data])
print(f'Train: {X.shape}, valid: {Xv.shape}')

clf = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=0)
clf.fit(X, y)
acc = clf.score(Xv, yv)
print(f'TERM ML valid accuracy: {acc}')
print(f'Current rule term accuracy from results_v24: 0.9156')

with open('term_clf.pkl', 'wb') as f:
    pickle.dump(clf, f)
