"""Train bo phan loai cuoi: anh (train_set5) + 3 khung moi doan cua moi clip da duyet. Ra floodapp/weights/clf_b32.npz + rule.json."""
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT
from v5lib import clip_fold, clip_frames, fit_lr, labels, load_clip, predict

C = 0.003
ts = np.load(ROOT / "tmp/train_set5.npz", allow_pickle=True)
lab = labels()
cf = clip_fold()
data = {c: load_clip(c) for c in cf}
fr = {c: clip_frames(c, data[c], lab) for c in cf}
X = np.concatenate([ts["X"]] + [data[c]["emb"][fr[c][0]] for c in cf])
y = np.concatenate([ts["y"]] + [fr[c][1] for c in cf])
m = fit_lr(X, y, C)
np.savez(ROOT / "models/clf_b32_v5.npz", **m)
shutil.copy(ROOT / "models/clf_b32_v5.npz", ROOT / "floodapp/weights/clf_b32.npz")
shutil.copy(ROOT / "models/rule_v5.json", ROOT / "floodapp/weights/rule.json")
for c in cf:
    np.save(ROOT / f"cache/{c}.final_p.npy", predict(m, data[c]["emb"]))
print("train", len(y), "pos", int(y.sum()))
