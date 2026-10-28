"""v4 tren cung doan: bo phan loai v4 (chi anh, models/clf_v4.npz) + luat v4 (combo W2 a.9 b.2 c.3)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import ROOT
from floodapp import rules as R
from simlib import clf_p, load_model

V4_RULE = {"name": "combo", "W": 2, "a": 0.9, "b": 0.2, "c": 0.3}


def v4_pred(data):
    m = load_model(ROOT / "models/clf_v4.npz")
    out = {}
    for c, d in data.items():
        p = clf_p(d["emb"], m)
        out[c] = R.segment_states(R.run(R.make_rule(V4_RULE), p, d["ratio"], d["scene"]), d["seg"])
    return out
