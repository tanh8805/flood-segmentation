"""v5: doc clip + nhan + fold; khung huan luyen tu clip (3 khung/doan) ; do chinh xac theo doan."""
import csv
import json
from pathlib import Path

import numpy as np

from common import ROOT, clips

FRAMES_PER_SEG = 3


def folds():
    f = json.loads((ROOT / "folds.json").read_text())
    return {int(k): v for k, v in f.items()}


def clip_fold():
    return {c: k for k, v in folds().items() for c in v}


def labels():
    lab = {}
    for r in csv.DictReader((ROOT / "review" / "labels.csv").open()):
        lab[(r["clip"], int(r["seg"]))] = r["label"]
    return lab


def load_clip(name):
    rows = [json.loads(x) for x in (ROOT / "cache" / f"{name}.jsonl").read_text().splitlines()]
    return {
        "emb": np.load(ROOT / "cache" / f"{name}.clip_b32.npy"),
        "ratio": np.array([r["v3_ratio"] for r in rows]), "scene": np.array([r["scene"] for r in rows]),
        "seg": np.array([r["seg"] for r in rows]), "lum": np.array([r["lum"] for r in rows]),
        "v1_level": np.array([r["v1_level"] for r in rows]),
    }


def clip_frames(name, d, lab):
    """Chi so khung huan luyen: 3 khung/doan (1/6, 3/6, 5/6 doan), nhan cua doan, bo bo_qua. Tra (idx, y)."""
    idx, y = [], []
    for s in sorted(set(d["seg"])):
        l = lab.get((name, int(s)))
        if l not in ("ngap", "khong_ngap"):
            continue
        ks = np.where(d["seg"] == s)[0]
        pick = [ks[int(round(f * (len(ks) - 1)))] for f in (1 / 6, 3 / 6, 5 / 6)] if len(ks) > 1 else list(ks)
        idx += pick
        y += [int(l == "ngap")] * len(pick)
    return np.array(idx, int), np.array(y, int)


def fit_lr(X, y, C=0.003, w=None):
    from sklearn.linear_model import LogisticRegression
    mu, sd = X.mean(0), X.std(0) + 1e-6
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000)
    clf.fit((X - mu) / sd, y, sample_weight=w)
    return {"w": clf.coef_[0], "b": float(clf.intercept_[0]), "mu": mu, "sd": sd}


def predict(m, emb):
    z = ((emb - m["mu"]) / m["sd"]) @ m["w"] + m["b"]
    return 1 / (1 + np.exp(-z))
