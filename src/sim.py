"""Mo phong luat tren cache DEV (khong chay lai model). Tieu chi chon (DINH TRUOC, khong doi sau khi xem so):
  F1 cao nhat voi recall >= 0.90 va ti le bao nham (FPR) <= 0.10; chenh F1 <= 0.01 thi chon luat don gian hon
  (thu tu: clf < water < combo), roi bao thu hon (tong nguong cao hon). Bao so leave-one-clip-out (LOCO).
Dung: python sim.py <model.npz> [out.json]"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import ROOT, splits
from floodapp.rules import make_rule, run, segment_states
from simlib import clf_p, load_clip_cache, load_labels, load_model

INF_FPS = 5.0
COMPLEX = {"clf": 0, "water": 1, "combo": 2}
GRID = []
for W in (1, 2, 3):
    for on, off in itertools.product((0.5, 0.6, 0.7, 0.8, 0.9), (0.3, 0.4, 0.5, 0.6, 0.7)):
        if off <= on:
            GRID.append(("clf", {"W": W, "on": on, "off": off}))
    for on, off in itertools.product((0.15, 0.25, 0.35, 0.5, 0.7), (0.1, 0.2, 0.3, 0.4, 0.5)):
        if off <= on:
            GRID.append(("water", {"W": W, "on": on, "off": off}))
    for a, b, c in itertools.product((0.7, 0.8, 0.9), (0.2, 0.3, 0.4, 0.5, 0.6), (0.1, 0.2, 0.3, 0.5)):
        GRID.append(("combo", {"W": W, "a": a, "b": b, "c": c}))


def load_data(model, part="dev"):
    lab, sp, data = load_labels(), splits(), {}
    for clip, pt in sorted(sp.items()):
        if pt != part:
            continue
        rows, t = load_clip_cache(clip)
        emb = np.load(ROOT / f"cache/{clip}.clip_b32.npy")
        data[clip] = {"p": clf_p(emb, model), "ratio": np.array([r["v3_ratio"] for r in rows]), "scene": np.array([r["scene"] for r in rows]),
                      "seg": np.array([r["seg"] for r in rows]), "lab": {s: lab[(clip, s)] for s in set(r["seg"] for r in rows) if lab.get((clip, s)) in ("ngap", "khong_ngap")}}
    return data


def predict(data, name, kw):
    return {c: segment_states(run(make_rule({"name": name, **kw}), d["p"], d["ratio"], d["scene"]), d["seg"]) for c, d in data.items()}


def counts(data, pred, clips):
    tp = fp = tn = fn = 0
    for c in clips:
        for s, l in data[c]["lab"].items():
            y, p = l == "ngap", pred[c][s]
            tp += y and p; fn += y and not p; fp += (not y) and p; tn += (not y) and not p
    return tp, fp, tn, fn


def metrics(tp, fp, tn, fn):
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"prec": round(prec, 3), "rec": round(rec, 3), "f1": round(f1, 3), "fpr": round(fp / (fp + tn) if fp + tn else 0.0, 3), "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def choose(cands):
    """cands: [(name, kw, metrics)] -> luat theo tieu chi dinh truoc; None neu khong luat nao dat."""
    ok = [c for c in cands if c[2]["rec"] >= 0.90 and c[2]["fpr"] <= 0.10]
    if not ok:
        return None
    top = max(c[2]["f1"] for c in ok)
    near = [c for c in ok if top - c[2]["f1"] <= 0.01]
    return min(near, key=lambda c: (COMPLEX[c[0]], -(sum(v for k, v in c[1].items() if k != "W")), ))


def main():
    model = load_model(sys.argv[1])
    data = load_data(model)
    clips = sorted(data)
    preds = [predict(data, n, kw) for n, kw in GRID]
    res = []
    for (n, kw), pr in zip(GRID, preds):
        res.append((n, kw, metrics(*counts(data, pr, clips))))
    pick = choose(res)
    print("so to hop", len(GRID), "| so doan DEV", sum(len(d["lab"]) for d in data.values()))
    best3 = {n: max((r for r in res if r[0] == n), key=lambda r: r[2]["f1"]) for n in COMPLEX}
    for n, r in best3.items():
        print("tot nhat", n, r[1], r[2])
    print("CHON:", pick[:2] if pick else None, pick[2] if pick else "khong luat nao dat recall>=.90 va FPR<=.10")
    # leave-one-clip-out: chon luat tren cac clip con lai, danh gia tren clip bi giu
    tot = [0, 0, 0, 0]
    for held in clips:
        rest = [c for c in clips if c != held]
        cands = [(n, kw, metrics(*counts(data, pr, rest))) for (n, kw), pr in zip(GRID, preds)]
        ch = choose(cands) or max(cands, key=lambda c: c[2]["f1"])
        i = [(n, kw) for n, kw in GRID].index((ch[0], ch[1]))
        c4 = counts(data, preds[i], [held])
        tot = [a + b for a, b in zip(tot, c4)]
        print("  LOCO", held, ch[0], ch[1], metrics(*c4))
    print("LOCO tong", metrics(*tot))
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(json.dumps({"choose": pick[:2] if pick else None, "all": [[n, kw, m] for n, kw, m in res]}))


if __name__ == "__main__":
    main()
