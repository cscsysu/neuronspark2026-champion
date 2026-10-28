"""Train hazard event classifier on train.jsonl only."""
import json, pickle, sys
from collections import Counter
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
sys.path.insert(0, '.')
from sim2 import simulate, EVENT_KEYS, DEFAULT_PROFILE
from predict_v3 import best_profile_for, context_priors, build_ev_features

X = []
y = []
with open('train.jsonl') as f:
    for line in f:
        d = json.loads(line)
        profile, _ = best_profile_for(d)
        priors = context_priors(d)
        for q in d['queries']:
            r = simulate(q['initial_full_grid'], q['future_actions'], profile)
            f = build_ev_features(d['context_episodes'], 'hazard', priors, r)
            X.append(f)
            y.append(int(q['label']['events']['hazard']))
X = np.array(X); y = np.array(y)
print(f'Train: {len(X)} samples, positive: {sum(y)}')
clf = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=0)
clf.fit(X, y)
print('train acc:', clf.score(X, y))
print('feature_importances:', clf.feature_importances_)

# Validate on valid
X_v = []; y_v = []
with open('valid.jsonl') as f:
    for line in f:
        d = json.loads(line)
        profile, _ = best_profile_for(d)
        priors = context_priors(d)
        for q in d['queries']:
            r = simulate(q['initial_full_grid'], q['future_actions'], profile)
            f = build_ev_features(d['context_episodes'], 'hazard', priors, r)
            X_v.append(f)
            y_v.append(int(q['label']['events']['hazard']))
X_v = np.array(X_v); y_v = np.array(y_v)
print(f'Valid: {len(X_v)}, accuracy: {clf.score(X_v, y_v):.4f}')
# Compare to current rule
from predict_v3 import THRESH
correct_rule = 0
for f, y_true in zip(X_v, y_v):
    # f[0] = sim event fired, f[2] = pec for hazard
    pred = bool(f[0]) or f[2] >= THRESH['hazard']
    if pred == bool(y_true):
        correct_rule += 1
print(f'Current rule accuracy: {correct_rule/len(y_v):.4f}')

# Save model
with open('haz_clf_v2.pkl','wb') as f:
    pickle.dump(clf, f)
