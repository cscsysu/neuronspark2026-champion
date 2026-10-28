"""Train order classifier on combined train+valid."""
import json, pickle, sys
from collections import Counter
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
sys.path.insert(0, '.')
from sim2 import simulate, EVENT_KEYS, DEFAULT_PROFILE
from predict_v3 import best_profile_for, context_priors

def build_pair_features(d, q, r, k_a, k_b, eps, priors):
    pec, pt, tlm, entity_pos, probe_tl_avg, probe_avg_rank = priors
    h = max(1, r['horizon'])
    fa_step = r['first_step'][k_a]/h if r['first_step'][k_a] is not None else probe_tl_avg.get(k_a, 0.5)
    fb_step = r['first_step'][k_b]/h if r['first_step'][k_b] is not None else probe_tl_avg.get(k_b, 0.5)
    rk_a = probe_avg_rank.get(k_a, 999)
    rk_b = probe_avg_rank.get(k_b, 999)
    n_a_before_b = 0; n_both = 0
    for ep in eps:
        order_list = [e for e in ep['observed_final_event_order'] if e in (k_a, k_b)]
        if k_a in order_list and k_b in order_list:
            n_both += 1
            if order_list.index(k_a) < order_list.index(k_b): n_a_before_b += 1
    p_a_before = n_a_before_b / max(1, n_both)
    fired_a = int(r['events'][k_a]); fired_b = int(r['events'][k_b])
    pec_a = pec[k_a]; pec_b = pec[k_b]
    return [
        fa_step - fb_step, rk_a - rk_b, fired_a - fired_b, pec_a - pec_b,
        fa_step, fb_step, rk_a, rk_b,
        p_a_before, n_both,
        fired_a, fired_b, pec_a, pec_b,
        EVENT_KEYS.index(k_a) - EVENT_KEYS.index(k_b),
    ]

def collect(path):
    X, y = [], []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            profile, _ = best_profile_for(d)
            priors = context_priors(d)
            for q in d['queries']:
                lbl = q['label']
                lbl_order = [e for e in lbl['event_order'] if e != 'none']
                if len(lbl_order) < 2: continue
                r = simulate(q['initial_full_grid'], q['future_actions'], profile)
                eps = d['context_episodes']
                for i, ka in enumerate(lbl_order):
                    for kb in lbl_order[i+1:]:
                        f1 = build_pair_features(d, q, r, ka, kb, eps, priors)
                        X.append(f1); y.append(1)
                        f2 = build_pair_features(d, q, r, kb, ka, eps, priors)
                        X.append(f2); y.append(0)
    return X, y

print('Train:'); Xt, yt = collect('train.jsonl')
print('Valid:'); Xv, yv = collect('valid.jsonl')
Xt, yt = np.array(Xt), np.array(yt)
Xv, yv = np.array(Xv), np.array(yv)
X_all = np.concatenate([Xt, Xv])
y_all = np.concatenate([yt, yv])
clf = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=0)
clf.fit(X_all, y_all)
print(f'combined acc: {clf.score(X_all, y_all):.4f}')
print(f'valid acc: {clf.score(Xv, yv):.4f}')
with open('order_clf_combined.pkl', 'wb') as f:
    pickle.dump(clf, f)
