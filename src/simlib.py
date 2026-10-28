"""Doc cache + nhan, tinh p cua bo phan loai, va danh gia theo doan 2 s (nhan doan = da so khung)."""
import csv
import json
from pathlib import Path

import numpy as np

from common import ROOT, SEG, splits


def load_labels():
    lab = {}
    for r in csv.DictReader((ROOT / "review" / "labels.csv").open()):
        lab[(r["clip"], int(r["seg"]))] = r["label"]
    return lab


def load_clip_cache(clip):
    rows = [json.loads(x) for x in (ROOT / "cache" / f"{clip}.jsonl").read_text().splitlines()]
    t = np.array([r["t"] for r in rows])
    return rows, t


def clf_p(emb, model):
    z = ((emb - model["mu"]) / model["sd"]) @ model["w"] + model["b"]
    return 1 / (1 + np.exp(-z))


def load_model(path):
    d = np.load(path)
    return {k: d[k] for k in ("w", "b", "mu", "sd")}


def seg_index(t):
    return np.floor(t / SEG).astype(int)
