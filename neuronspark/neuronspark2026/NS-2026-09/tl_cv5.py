"""Conservative combined TL ML: smaller model, more regularization."""
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

# Cross-validate on valid: split valid in 2, eval each model on held-out half
random.seed(0)
random.shuffle(valid_data)

def cross_val_score(model_factory, k):
    """5-fold CV on valid_data only for event k."""
    items = [d for d in valid_data if d['event'] == k and d['sim_fired']]
    if len(items) < 10: return None, None
    folds = 5
    fold_size = len(items) // folds
    train_only_accs = []
    train_v_accs = []
    for fi in range(folds):
        v_test = items[fi*fold_size:(fi+1)*fold_size]
        v_train = items[:fi*fold_size] + items[(fi+1)*fold_size:]
        Xt = np.array([d['feat'] for d in train_data if d['event'] == k and d['sim_fired']])
        yt = np.array([d['label'] for d in train_data if d['event'] == k and d['sim_fired']])
        if len(set(yt)) < 2 or len(Xt) < 50: continue
        Xv_train = np.array([d['feat'] for d in v_train])
        yv_train = np.array([d['label'] for d in v_train])
        Xv_test = np.array([d['feat'] for d in v_test])
        yv_test = np.array([d['label'] for d in v_test])
        # Train-only
        clf_t = model_factory()
        clf_t.fit(Xt, yt)
        train_only_accs.append((clf_t.predict(Xv_test) == yv_test).mean())
        # Train + valid_train
        Xtv = np.concatenate([Xt, Xv_train])
        ytv = np.concatenate([yt, yv_train])
        clf_tv = model_factory()
        clf_tv.fit(Xtv, ytv)
        train_v_accs.append((clf_tv.predict(Xv_test) == yv_test).mean())
    return np.mean(train_only_accs), np.mean(train_v_accs)

print(f'5-fold CV: train-only vs train+v_train (each event)')
for k in EVENT_KEYS:
    factory = lambda: GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=0)
    t, tv = cross_val_score(factory, k)
    if t is not None:
        diff = tv - t
        print(f'  {k:<15} train-only={t:.4f}, train+v={tv:.4f}, diff={diff:+.4f}')
