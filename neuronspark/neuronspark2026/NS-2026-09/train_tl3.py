"""Train TL classifiers on combined train+valid (carefully)."""
import json, pickle, sys
from collections import Counter
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
sys.path.insert(0, '.')
from sim2 import simulate, EVENT_KEYS, DEFAULT_PROFILE
from predict_v3 import best_profile_for, context_priors, build_tl_features

TL_LABELS = ['early','mid','late']

def collect(path):
    data = []
    with open(path) as f:
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
                    data.append({'feat': f, 'event': k,
                                 'label': TL_LABELS.index(q['label']['event_timeline'][k]),
                                 'sim_fired': bool(f[0])})
    return data

print('Build train+valid combined (CAREFUL: may overfit on test)...')
train_data = collect('train.jsonl')
valid_data = collect('valid.jsonl')

clfs = {}
for k in EVENT_KEYS:
    Xt = np.array([d['feat'] for d in train_data if d['event'] == k and d['sim_fired']])
    yt = np.array([d['label'] for d in train_data if d['event'] == k and d['sim_fired']])
    Xv = np.array([d['feat'] for d in valid_data if d['event'] == k and d['sim_fired']])
    yv = np.array([d['label'] for d in valid_data if d['event'] == k and d['sim_fired']])
    if len(Xv) == 0:
        clfs[k] = ('rule', None)
        continue
    # Sim bucket
    def bucket(s):
        if s < 0: return 0
        if s < 1/3: return 0
        if s < 2/3: return 1
        return 2
    sim_pred = np.array([bucket(s) for s in Xv[:, 1]])
    sim_acc = (sim_pred == yv).mean()
    if len(set(yt)) >= 2 and len(Xt) >= 50:
        # Train on combined
        X_all = np.concatenate([Xt, Xv])
        y_all = np.concatenate([yt, yv])
        clf = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=0)
        clf.fit(X_all, y_all)
        # Use train-only for valid eval
        clf_train = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=0)
        clf_train.fit(Xt, yt)
        train_only_acc = clf_train.score(Xv, yv)
        if train_only_acc > sim_acc:
            clfs[k] = ('ml', clf)  # use combined for inference
        else:
            clfs[k] = ('rule', None)
        print(f'{k}: sim={sim_acc:.4f}, train-only-ml={train_only_acc:.4f}, used={"ml(combined)" if train_only_acc > sim_acc else "rule"}')
    else:
        clfs[k] = ('rule', None)

with open('tl_clfs2_combined.pkl', 'wb') as f:
    pickle.dump(clfs, f)
print('saved tl_clfs2_combined.pkl')
