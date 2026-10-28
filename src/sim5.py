"""v5: chon luat tren du doan out-of-fold (cache/<clip>.oof_p.npy). Luoi 138 to hop:
  clf-only: W in {1,2,3} s, bat in {.5..,.9}, tat in {.3..,.6} (tat <= bat)
  combo (nhanh nuoc chi bat khi p >= b): bat neu p>=a hoac (p>=b va ratio>=c), a in {.7,.8,.9}, b in {.4,.5,.6}, c in {.2,.3,.5}
Tieu chi (dinh truoc): recall >= 0.90; bao nham (FPR) thap nhat; trong nhom FPR chenh <= 0.01 so voi thap nhat lay F1 cao nhat;
  trong nhom F1 chenh <= 0.01 lay luat don gian hon (clf < combo), roi bao thu hon (nguong cao hon).
Dung: python sim5.py [--write]"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from baselines import baseline_pred
from common import ROOT, clips
from floodapp import rules as R
from v5lib import labels, load_clip

GRID = []
for W in (1, 2, 3):
    for on, off in itertools.product((0.5, 0.6, 0.7, 0.8, 0.9), (0.3, 0.4, 0.5, 0.6)):
        if off <= on:
            GRID.append({"name": "clf", "W": W, "on": on, "off": off})
    for a, b, c in itertools.product((0.7, 0.8, 0.9), (0.4, 0.5, 0.6), (0.2, 0.3, 0.5)):
        GRID.append({"name": "combo", "W": W, "a": a, "b": b, "c": c})


def load(pfile="oof_p"):
    lab = labels()
    data = {}
    for c, meta in clips().items():
        d = load_clip(c)
        d["p"] = np.load(ROOT / "cache" / f"{c}.{pfile}.npy")
        d["lab"] = {s: lab[(c, s)] for s in set(d["seg"]) if lab.get((c, s)) in ("ngap", "khong_ngap")}
        d["tags"] = meta.get("tags", [])
        data[c] = d
    return data


def predict(data, cfg):
    return {c: R.segment_states(R.run(R.make_rule(cfg), d["p"], d["ratio"], d["scene"]), d["seg"]) for c, d in data.items()}


def counts(data, pred, sel):
    """sel: ham (clip, seg, label) -> bool de chon doan tinh."""
    tp = fp = tn = fn = 0
    for c, d in data.items():
        for s, l in d["lab"].items():
            if not sel(c, s, l):
                continue
            y, p = l == "ngap", pred[c][s]
            tp += y and p; fn += y and not p; fp += (not y) and p; tn += (not y) and not p
    return tp, fp, tn, fn


def metrics(tp, fp, tn, fn):
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"prec": round(prec, 3), "rec": round(rec, 3), "f1": round(f1, 3), "fpr": round(fp / (fp + tn) if fp + tn else 0.0, 3), "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def choose(res):
    ok = [r for r in res if r[1]["rec"] >= 0.90]
    if not ok:
        return None
    fmin = min(r[1]["fpr"] for r in ok)
    tier = [r for r in ok if r[1]["fpr"] <= fmin + 0.01]
    top = max(r[1]["f1"] for r in tier)
    near = [r for r in tier if top - r[1]["f1"] <= 0.01]
    thr = lambda cfg: sum(v for k, v in cfg.items() if k not in ("name", "W"))
    return min(near, key=lambda r: (r[0]["name"] != "clf", -thr(r[0])))


def night_pos(data, c, s):
    return float(data[c]["lum"][data[c]["seg"] == s].mean()) < 70


def main():
    data = load()
    allsel = lambda c, s, l: True
    res = [(cfg, metrics(*counts(data, predict(data, cfg), allsel))) for cfg in GRID]
    pick = choose(res)
    print("to hop", len(GRID), "| doan", sum(len(d["lab"]) for d in data.values()))
    for n in ("clf", "combo"):
        best = max((r for r in res if r[0]["name"] == n), key=lambda r: (r[1]["rec"] >= 0.9, -r[1]["fpr"], r[1]["f1"]))
        print("tot nhat", n, best[0], best[1])
    print("CHON:", pick[0] if pick else None, pick[1] if pick else "khong luat nao dat recall >= 0.90")
    if not pick:
        return
    cfg = pick[0]
    out = {"rule": cfg, "grid": len(GRID)}
    nightwet = lambda c, s, l: "wet" in data[c]["tags"] and "night" in data[c]["tags"]
    nightpos = lambda c, s, l: l == "ngap" and night_pos(data, c, s)
    from sim_v4 import v4_pred
    DEFAULT = {"name": "clf", "W": 3, "on": 0.5, "off": 0.5}
    systems = {"moi": predict(data, cfg), "mac_dinh": predict(data, DEFAULT), "v1": {c: baseline_pred(c, "v1") for c in data}, "v3": {c: baseline_pred(c, "v3") for c in data}, "v4": v4_pred(data)}
    for name, pr in systems.items():
        out[name] = {"tong": metrics(*counts(data, pr, allsel)),
                     "dem_uot_khong_ngap": metrics(*counts(data, pr, nightwet)),
                     "dem_ngap": metrics(*counts(data, pr, nightpos)),
                     "clip": {c: metrics(*counts(data, pr, lambda cc, s, l, c=c: cc == c)) for c in data if data[c]["lab"]}}
        t = out[name]["tong"]
        print(f"{name:5s} tong {t}\n      dem uot (khong ngap) {out[name]['dem_uot_khong_ngap']}\n      dem ngap {out[name]['dem_ngap']}")
    for c in data:
        if data[c]["lab"]:
            print(f"  {c:14s}", {n: (out[n]["clip"][c]["fp"] + out[n]["clip"][c]["fn"]) for n in systems}, "sai/doan", len(data[c]["lab"]))
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/v5_oof.json").write_text(json.dumps(out, indent=1))
    if "--write" in sys.argv:
        (ROOT / "models/rule_v5.json").write_text(json.dumps(cfg))
        print("da ghi models/rule_v5.json")


if __name__ == "__main__":
    main()
