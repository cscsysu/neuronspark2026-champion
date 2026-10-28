"""5-fold CV for EV ML."""
import json, pickle, sys, random
from collections import Counter
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
sys.path.insert(0, '.')
from sim2 import simulate, EVENT_KEYS, DEFAULT_PROFILE
from predict_v3 import best_profile_for, context_priors, build_ev_features

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
                    f = build_ev_features(d['context_episodes'], k, priors, r)
                    data.append({'feat': f, 'event': k, 'label': int(q['label']['events'][k])})
    return data

train_data = collect('train.jsonl')
valid_data = collect('valid.jsonl')

random.seed(0)
random.shuffle(valid_data)

print(f'5-fold CV: train-only vs train+v_train (each event)')
for k in EVENT_KEYS:
    items = [d for d in valid_data if d['event'] == k]
    if len(items) < 10: continue
    folds = 5
    fold_size = len(items) // folds
    train_only_accs = []
    train_v_accs = []
    rule_accs = []
    THRESH = {'goal_reached': 5, 'collision': 6, 'hazard': 5, 'box_on_goal': 3, 'key_collected': 6, 'portal_used': 6}
    for fi in range(folds):
        v_test = items[fi*fold_size:(fi+1)*fold_size]
        v_train = items[:fi*fold_size] + items[(fi+1)*fold_size:]
        Xt = np.array([d['feat'] for d in train_data if d['event'] == k])
        yt = np.array([d['label'] for d in train_data if d['event'] == k])
        if len(set(yt)) < 2 or len(Xt) < 50: continue
        Xv_train = np.array([d['feat'] for d in v_train])
        yv_train = np.array([d['label'] for d in v_train])
        Xv_test = np.array([d['feat'] for d in v_test])
        yv_test = np.array([d['label'] for d in v_test])
        # Rule
        rule_pred = (Xv_test[:, 0].astype(bool)) | (Xv_test[:, 2] >= THRESH[k])
        rule_accs.append((rule_pred == yv_test.astype(bool)).mean())
        # Train-only
        clf_t = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=0)
        clf_t.fit(Xt, yt)
        train_only_accs.append((clf_t.predict(Xv_test) == yv_test).mean())
        # Train + valid_train
        Xtv = np.concatenate([Xt, Xv_train])
        ytv = np.concatenate([yt, yv_train])
        clf_tv = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=0)
        clf_tv.fit(Xtv, ytv)
        train_v_accs.append((clf_tv.predict(Xv_test) == yv_test).mean())
    print(f'  {k:<15} rule={np.mean(rule_accs):.4f}, train-only={np.mean(train_only_accs):.4f}, train+v={np.mean(train_v_accs):.4f}')
