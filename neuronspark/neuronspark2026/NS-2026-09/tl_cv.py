"""Cross-val TL ML: train on train+half_valid, eval other half."""
import json, pickle, sys, random
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

train_data = collect('train.jsonl')
valid_data = collect('valid.jsonl')

random.seed(0)
random.shuffle(valid_data)
half = len(valid_data) // 2
v1 = valid_data[:half]; v2 = valid_data[half:]

print(f'{"event":<15} {"sim":<8} {"train_only":<10} {"train+v1":<10} (eval on v2)')
for k in EVENT_KEYS:
    Xt = np.array([d['feat'] for d in train_data if d['event'] == k and d['sim_fired']])
    yt = np.array([d['label'] for d in train_data if d['event'] == k and d['sim_fired']])
    Xv1 = np.array([d['feat'] for d in v1 if d['event'] == k and d['sim_fired']])
    yv1 = np.array([d['label'] for d in v1 if d['event'] == k and d['sim_fired']])
    Xv2 = np.array([d['feat'] for d in v2 if d['event'] == k and d['sim_fired']])
    yv2 = np.array([d['label'] for d in v2 if d['event'] == k and d['sim_fired']])
    if len(Xv2) == 0: continue
    def bucket(s):
        if s < 1/3: return 0
        if s < 2/3: return 1
        return 2
    sim_pred = np.array([bucket(s) for s in Xv2[:, 1]])
    sim_acc = (sim_pred == yv2).mean()

    if len(set(yt)) >= 2 and len(Xt) >= 50:
        # Train-only
        clf_t = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=0)
        clf_t.fit(Xt, yt)
        ml_t = (clf_t.predict(Xv2) == yv2).mean()
        # Train + v1
        Xtv = np.concatenate([Xt, Xv1])
        ytv = np.concatenate([yt, yv1])
        clf_tv = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=0)
        clf_tv.fit(Xtv, ytv)
        ml_tv = (clf_tv.predict(Xv2) == yv2).mean()
        print(f'  {k:<15} {sim_acc:.4f}  {ml_t:.4f}    {ml_tv:.4f}     ({len(Xv2)} samples)')
