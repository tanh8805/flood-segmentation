"""Cham MOT LAN tren TEST voi luat da dong bang (floodapp/weights/rule.json): v1 / v3 / ban moi. Khong chinh gi sau khi xem."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines import baseline_pred
from common import ROOT
from floodapp import rules as R
from sim import counts, load_data, metrics
from simlib import load_model

model = load_model(ROOT / "floodapp/weights/clf_b32.npz")
data = load_data(model, "test")
clips = sorted(data)
rule_cfg = ROOT / "floodapp/weights/rule.json"
preds = {"v1": {c: baseline_pred(c, "v1") for c in clips}, "v3": {c: baseline_pred(c, "v3") for c in clips}}
preds["moi"] = {c: R.segment_states(R.run(R.load_rule(rule_cfg), d["p"], d["ratio"], d["scene"]), d["seg"]) for c, d in data.items()}
out = {"rule": json.loads(rule_cfg.read_text()), "n_segments": sum(len(d["lab"]) for d in data.values())}
for name, pr in preds.items():
    out[name] = {"tong": metrics(*counts(data, pr, clips)), "clip": {c: metrics(*counts(data, pr, [c])) for c in clips}}
errs = []
for c, d in data.items():
    for s, l in sorted(d["lab"].items()):
        if preds["moi"][c][s] != (l == "ngap"):
            m = d["seg"] == s
            errs.append({"clip": c, "seg": s, "t": s * 2, "label": l, "p": round(float(d["p"][m].mean()), 2), "ratio": round(float(d["ratio"][m].mean()), 2)})
out["loi_moi"] = errs
out["loi_v1"] = [(c, s) for c, d in data.items() for s, l in d["lab"].items() if preds["v1"][c][s] != (l == "ngap")]
out["loi_v3"] = [(c, s) for c, d in data.items() for s, l in d["lab"].items() if preds["v3"][c][s] != (l == "ngap")]
Path(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results/test_eval.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
for name in ("v1", "v3", "moi"):
    print(name, out[name]["tong"])
    for c in clips:
        print("   ", c, {k: v for k, v in out[name]["clip"][c].items() if k in ("tp", "fp", "tn", "fn", "f1")})
print("loi moi:", errs)
