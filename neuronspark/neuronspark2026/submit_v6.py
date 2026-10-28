"""
NS-2026-04 v6: 突破630天花板
关键改进:
1. 更低学习率(0.01) + 更多树(5000) = 更好的拟合
2. Huber loss (对异常值鲁棒)
3. Dart boosting (减少过拟合)
4. 多种loss函数集成
5. 全量8500训练, 2076维特征(验证最好的特征集)
"""
import csv, json, math, os, re, warnings, zipfile
from collections import Counter
import numpy as np
from scipy import sparse
import lightgbm as lgb
warnings.filterwarnings('ignore')

# Load data
print("Loading data...")
with open("train.csv") as f: tr=list(csv.DictReader(f))
y=np.array([float(r['target']) for r in tr])
smiles_tr=[r['smiles'] for r in tr]
with open("test.csv") as f: te=list(csv.DictReader(f))
ids=[r['id'] for r in te]; smiles_te=[r['smiles'] for r in te]

# Features
h_tr=sparse.load_npz("features/hashed_smiles_train.npz").toarray()
h_te=sparse.load_npz("features/hashed_smiles_test.npz").toarray()

TOKEN_RE=re.compile(r"Cl|Br|Si|Na|Li|Mg|Ca|Al|@@?|\[[^\]]+\]|[A-Za-z]|\d|\(|\)|=|#|-|\+|\\|/|\.|:")
ATOM_RE=re.compile(r"Cl|Br|Si|Na|Li|Mg|Ca|Al|[BCNOPSFIKH]|[bcnops]")
def desc(slist):
    feats=[]
    for smi in slist:
        c=Counter(ATOM_RE.findall(str(smi)));t=TOKEN_RE.findall(str(smi))
        ar=sum(c.get(x,0) for x in "cnops");ca=c.get("C",0)+c.get("c",0)
        ni=c.get("N",0)+c.get("n",0);ox=c.get("O",0)+c.get("o",0)
        su=c.get("S",0)+c.get("s",0);he=sum(v for k,v in c.items() if k not in{"C","c","H"})
        ha=c.get("Cl",0)+c.get("Br",0)+c.get("F",0)+c.get("I",0);ta=sum(c.values())
        nr=sum(ch.isdigit() for ch in smi);nb=smi.count("(");nd=smi.count("=")
        nt=smi.count("#");nc=smi.count("+")+smi.count("-");nch=smi.count("@")
        sl=len(smi);nt2=len(t);mw=ca*12+ni*14+ox*16+su*32+ha*35+ta
        feats.append([sl,nt2,ta,ca,ni,ox,su,he,ar,ha,nr,nb,nd,nt,nc,nch,
            ca/max(ta,1),he/max(ta,1),ar/max(ta,1),mw,math.log1p(sl),
            math.log1p(ta),math.log1p(mw),ca*nr,ar*nd,he*nb,ta*nr,sl*ha])
    return np.array(feats,dtype=np.float32)

Xtr=np.hstack([h_tr,desc(smiles_tr)]); Xte=np.hstack([h_te,desc(smiles_te)])
print(f"Features: {Xtr.shape[1]}")

# Strategy: multiple loss functions + learning rates + boosting types
configs = [
    # Config A: low lr + many trees (deeper convergence)
    {"objective": "regression", "learning_rate": 0.01, "n_estimators": 5000,
     "num_leaves": 63, "min_child_samples": 10, "subsample": 0.8,
     "colsample_bytree": 0.6, "reg_alpha": 0.1, "reg_lambda": 1.0},
    # Config B: huber loss (robust to outliers)
    {"objective": "huber", "learning_rate": 0.01, "n_estimators": 5000,
     "num_leaves": 63, "min_child_samples": 10, "subsample": 0.8,
     "colsample_bytree": 0.6, "reg_alpha": 0.1, "reg_lambda": 1.0},
    # Config C: original good config (0.03 lr, 2000 trees)
    {"objective": "regression", "learning_rate": 0.03, "n_estimators": 2000,
     "num_leaves": 63, "min_child_samples": 10, "subsample": 0.8,
     "colsample_bytree": 0.6, "reg_alpha": 0.1, "reg_lambda": 1.0},
    # Config D: dart boosting (prevents overfitting)
    {"objective": "regression", "boosting_type": "dart", "learning_rate": 0.05,
     "n_estimators": 500, "num_leaves": 63, "min_child_samples": 10,
     "subsample": 0.8, "colsample_bytree": 0.6, "reg_alpha": 0.1, "reg_lambda": 1.0,
     "drop_rate": 0.1, "max_drop": 50},
    # Config E: very low lr + very many trees
    {"objective": "regression", "learning_rate": 0.005, "n_estimators": 8000,
     "num_leaves": 31, "min_child_samples": 20, "subsample": 0.7,
     "colsample_bytree": 0.5, "reg_alpha": 0.5, "reg_lambda": 2.0},
    # Config F: MAE loss (L1, different optimization landscape)
    {"objective": "regression_l1", "learning_rate": 0.01, "n_estimators": 5000,
     "num_leaves": 63, "min_child_samples": 10, "subsample": 0.8,
     "colsample_bytree": 0.6, "reg_alpha": 0.1, "reg_lambda": 1.0},
]

N_SEEDS = 5
test_preds = np.zeros(len(ids))
n_total = 0

for cfg_idx, cfg in enumerate(configs):
    cfg_preds = np.zeros(len(ids))
    boosting = cfg.pop("boosting_type", "gbdt")
    for seed in range(N_SEEDS):
        m = lgb.LGBMRegressor(boosting_type=boosting, verbose=-1,
                              random_state=42+seed*77+cfg_idx*999, **cfg)
        m.fit(Xtr, y)
        cfg_preds += m.predict(Xte)
        n_total += 1
    cfg_preds /= N_SEEDS
    test_preds += cfg_preds
    print(f"  Config {cfg_idx+1}/{len(configs)} done: mean={cfg_preds.mean():.4f}")

test_preds /= len(configs)
print(f"\nFinal ensemble ({n_total} models): mean={test_preds.mean():.4f}, std={test_preds.std():.4f}")

# Save
os.makedirs("submission_v6", exist_ok=True)
with open("submission_v6/results.csv", "w", newline="") as f:
    w=csv.writer(f); w.writerow(["id","prediction"])
    for tid,p in zip(ids,test_preds): w.writerow([tid,f"{p:.6f}"])

import subprocess
r=subprocess.run(["python3","tools/check_format.py","submission_v6/results.csv","--test-csv","test.csv"],
                 capture_output=True,text=True)
print(f"Format: {r.stdout.strip()}")

with zipfile.ZipFile("NS-2026-04-answer-v6.zip","w",zipfile.ZIP_DEFLATED) as z:
    z.write("submission_v6/results.csv","results.csv")
print("Done! NS-2026-04-answer-v6.zip")
