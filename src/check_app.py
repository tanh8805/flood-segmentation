"""Kiem tra: app that (floodapp.analyzer) vs mo phong tren cache cho cac clip chi dinh -> ti le doan khop + do lech p. Do thoi gian."""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, clip_path
from floodapp import analyzer, rules as R
from simlib import clf_p, load_clip_cache, load_model

rule_cfg = ROOT / "floodapp/weights/rule.json"
model = load_model(ROOT / "floodapp/weights/clf_b32.npz")
tot = hit = 0
for clip in sys.argv[1:]:
    t0 = time.time()
    res = analyzer.process_video(clip_path(clip), None)
    dt = time.time() - t0
    rows, _ = load_clip_cache(clip)
    p = clf_p(np.load(ROOT / f"cache/{clip}.clip_b32.npy"), model)
    st = R.run(R.load_rule(rule_cfg), p, [r["v3_ratio"] for r in rows], [r["scene"] for r in rows])
    sim = R.segment_states(st, [r["seg"] for r in rows])
    app = {i: s["flooded"] for i, s in enumerate(res["segments"])}
    n = len(res["frames"])
    dp = np.abs(np.array([f["p"] for f in res["frames"]]) - p).max()
    dr = np.abs(np.array([f["water_ratio"] for f in res["frames"]]) - np.array([r["v3_ratio"] for r in rows])).max()
    match = sum(app[i] == sim[i] for i in sim)
    tot += len(sim); hit += match
    print(f"{clip}: doan app {len(app)} sim {len(sim)} khop {match}/{len(sim)} | frames {n} == cache {len(rows)} | max|dp| {dp:.4f} max|dratio| {dr:.4f} | {dt:.1f}s = {1000 * dt / n:.0f} ms/lan suy luan | device {analyzer.get_classifier().dev}", flush=True)
print("TONG khop", hit, "/", tot)
