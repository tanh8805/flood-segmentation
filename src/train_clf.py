"""Bo phan loai khung hinh: CLIP dong bang + logistic regression. Chia train/val theo nhom (nguon/chuoi/cum anh giong nhau).
Dung: python train_clf.py [--emb b32] ; ra models/clf_<emb>.npz"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT

d = np.load(ROOT / "tmp/train_set.npz", allow_pickle=True)
X, y, hard, group, src = d["X"], d["y"], d["hard"], d["group"], d["src"]
rng = np.random.default_rng(1)
ug = np.array(sorted(set(group)))
val_g = set(rng.choice(ug, size=int(0.2 * len(ug)), replace=False))
isval = np.array([g in val_g for g in group])
tr, va = ~isval, isval


def fit(C, mask):
    mu, sd = X[mask].mean(0), X[mask].std(0) + 1e-6
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000)
    clf.fit((X[mask] - mu) / sd, y[mask])
    return clf, mu, sd


def report(clf, mu, sd, m, thr=0.5):
    p = clf.predict_proba((X[m] - mu) / sd)[:, 1]
    pred = p >= thr
    pr, rc, f1, _ = precision_recall_fscore_support(y[m], pred, average="binary", zero_division=0)
    hm = hard[m] == 1
    fp_hard = float(pred[hm].mean()) if hm.any() else float("nan")
    fp_all = float(pred[y[m] == 0].mean())
    return {"acc": round(float((pred == y[m]).mean()), 4), "prec": round(pr, 4), "rec": round(rc, 4), "f1": round(f1, 4),
            "fp_rate_all_neg": round(fp_all, 4), "fp_rate_hard_neg": round(fp_hard, 4), "n": int(m.sum()), "n_hard": int(hm.sum())}


print("train", int(tr.sum()), "val", int(va.sum()), "val pos", int(y[va].sum()), "val hard", int(hard[va].sum()))
best = None
for C in (0.003, 0.01, 0.03, 0.1, 0.3, 1):
    clf, mu, sd = fit(C, tr)
    r = report(clf, mu, sd, va)
    print(C, r)
    if best is None or r["f1"] > best[0]:
        best = (r["f1"], C, r)
_, C, r = best
clf, mu, sd = fit(C, np.ones(len(y), bool))      # model cuoi: fit tren toan bo
(ROOT / "models").mkdir(exist_ok=True)
np.savez(ROOT / "models/clf_b32.npz", w=clf.coef_[0], b=clf.intercept_[0], mu=mu, sd=sd, C=C)
(ROOT / "models/clf_b32_val.json").write_text(json.dumps({"C": C, "val": r}, indent=1))
print("chon C", C, r)
