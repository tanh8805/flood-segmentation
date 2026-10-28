"""Grouped 5-fold theo clip: moi fold huan luyen CLIP+LR tren anh + khung clip cua 4 fold con lai, du doan clip fold con lai.
Ra cache/<clip>.oof_p.npy (p tren moi lan suy luan) va models/clf_b32_v5_fold<k>.npz. Dung: python oof.py [C]"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT
from v5lib import clip_fold, clip_frames, fit_lr, labels, load_clip, predict

C = float(sys.argv[1]) if len(sys.argv) > 1 else 0.003
ts = np.load(ROOT / "tmp/train_set5.npz", allow_pickle=True)
Xi, yi = ts["X"], ts["y"]
lab = labels()
cf = clip_fold()
data = {c: load_clip(c) for c in cf}
fr = {c: clip_frames(c, data[c], lab) for c in cf}
(ROOT / "models").mkdir(exist_ok=True)
for k in sorted(set(cf.values())):
    tr = [c for c in cf if cf[c] != k]
    X = np.concatenate([Xi] + [data[c]["emb"][fr[c][0]] for c in tr])
    y = np.concatenate([yi] + [fr[c][1] for c in tr])
    m = fit_lr(X, y, C)
    np.savez(ROOT / f"models/clf_b32_v5_fold{k}.npz", **m)
    for c in cf:
        if cf[c] == k:
            np.save(ROOT / f"cache/{c}.oof_p.npy", predict(m, data[c]["emb"]))
    print("fold", k, "train", len(y), "pos", int(y.sum()), "test clips", [c for c in cf if cf[c] == k], flush=True)
