"""Dung tap train cho bo phan loai tu tmp/pool2.jsonl (+ tmp/pool_bdd*.jsonl neu co).
Quy tac: ti le nuoc tren mat dat >= 0.3 -> ngap; 0.02..0.3 -> bo (mo ho); < 0.02 -> am. Nguon co mask nguoi gan: deepflood, sazara, pixabay, fredf, ade.
Ra: tmp/train_set5.npz (X, y, group, src, hard) + tmp/train_set5.jsonl"""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT

LEAK_MAX = 0.93
rows = [json.loads(l) for l in (ROOT / "tmp/pool2.jsonl").read_text().splitlines()]
emb = np.load(ROOT / "tmp/pool_emb.npy")
if (ROOT / "tmp/pool_bdd2.jsonl").exists():
    rows += [json.loads(l) for l in (ROOT / "tmp/pool_bdd2.jsonl").read_text().splitlines()]
    emb = np.concatenate([emb, np.load(ROOT / "tmp/pool_bdd_emb.npy")])
bdd_meta = json.loads((ROOT / "data/bdd/meta.json").read_text()) if (ROOT / "data/bdd/meta.json").exists() else {}
HUMAN = {"deepflood", "sazara", "pixabay", "fredf", "ade"}
DROP = {"up0", "houston", "old"}
rng = np.random.default_rng(0)
# v5: anh dem khong mua -> chi giu 300 anh "dem uot phan chieu" nhat theo CLIP zero-shot (prompt night_wet)
night = [(r["zs"]["night_wet"], r["id"]) for r in rows if r["src"] == "bdd" and bdd_meta.get(r["id"].split(":")[1], {}).get("kind") == "night"]
night_keep = {i for _, i in sorted(night, reverse=True)[:300]}

keep, y, hard, grp = [], [], [], []
for i, r in enumerate(rows):
    s = r["src"]
    if s in DROP or r["leak"] >= LEAK_MAX:
        continue
    label, h = None, 0
    if s in HUMAN:
        if r["ratio"] >= 0.3:
            label = 1
        elif r["ratio"] < 0.02:
            label = 0
            h = int(s == "fredf")
    elif s == "cocoval":
        continue                      # chon rieng ben duoi
    elif s in ("fredd", "coco"):
        if r["segwater"] < 0.005:
            label, h = 0, int(s == "fredd")
    elif s == "bdd":
        m = bdd_meta.get(r["id"].split(":")[1], {})
        if m.get("kind") == "night" and r["id"] not in night_keep:
            continue
        if r["segwater"] < 0.01 and r["ground"] >= 0.08:
            label, h = 0, int(m.get("kind") in ("rain", "night"))
    if label is None:
        continue
    g = s
    m = re.match(r"fred[fd]:(?:fred[fd]_)?([A-Za-z\-]+_\d{8})", r["id"])
    if m:
        g = "fred_" + m.group(1)
    keep.append(i); y.append(label); hard.append(h); grp.append(g)

# COCO val: am tinh thuong (duong kho), toi da 1000 anh co mat dat >= 0.1 va khong co nuoc
coco = [i for i, r in enumerate(rows) if r["src"] == "cocoval" and r["ground"] >= 0.1 and r["segwater"] < 0.005 and r["leak"] < LEAK_MAX]
pick = rng.choice(coco, size=min(1000, len(coco)), replace=False)
for i in pick:
    keep.append(int(i)); y.append(0); hard.append(0); grp.append("cocoval")

# nhom cho split: moi anh Deepflood/Sazara/pixabay/ade/coco/bdd la mot nhom rieng theo cum CLIP (cos >= 0.9)
X = emb[keep]
y = np.array(y); hard = np.array(hard); grp = np.array(grp, dtype=object)
src = np.array([rows[i]["src"] for i in keep], dtype=object)
cl = -np.ones(len(keep), int); n = 0
for i in range(len(keep)):
    if cl[i] >= 0:
        continue
    sims = X @ X[i]
    sel = np.where((sims >= 0.9) & (cl < 0))[0]
    cl[sel] = n; n += 1
grp = np.array([g if g.startswith("fred_") else f"{g}#{c}" for g, c in zip(grp, cl)], dtype=object)
np.savez(ROOT / "tmp/train_set5.npz", X=X, y=y, hard=hard, group=grp, src=src)
(ROOT / "tmp/train_set5.jsonl").write_text("\n".join(json.dumps({"id": rows[i]["id"], "path": rows[i]["path"], "y": int(a), "hard": int(b), "src": rows[i]["src"], "group": g})
                                                    for i, a, b, g in zip(keep, y, hard, grp)))
import collections
print("n", len(y), "pos", int(y.sum()), "neg", int((1 - y).sum()), "hard", int(hard.sum()), "groups", len(set(grp)))
print({k: (int(((src == k) & (y == 1)).sum()), int(((src == k) & (y == 0)).sum())) for k in sorted(set(src))})
