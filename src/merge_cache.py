"""Gop tin hieu v1 / v3 / CLIP thanh cache/<clip>.jsonl (mot dong moi khung suy luan, ~5/giay)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT

raw = ROOT / "tmp" / "raw"
for clip in sys.argv[1:]:
    v1 = json.loads((raw / f"{clip}.v1.json").read_text())["frames"]
    v3 = [json.loads(x) for x in (raw / f"{clip}.v3.sig.jsonl").read_text().splitlines()]
    emb = np.load(ROOT / "cache" / f"{clip}.clip_b32.npy")
    n = len(v3)
    seg_len = json.loads((raw / f"{clip}.v3.sig.meta.json").read_text())["seg_len"]
    assert len(v1) == n == len(emb), (clip, len(v1), n, len(emb))
    with (ROOT / "cache" / f"{clip}.jsonl").open("w") as f:
        for k in range(n):
            a, b = v1[k], v3[k]
            assert abs(a["time_s"] - b["t"]) < 0.02, (clip, k, a["time_s"], b["t"])
            f.write(json.dumps({
                "k": k, "idx": b["idx"], "seg": b["idx"] // seg_len, "t": b["t"], "scene": b["scene"], "lum": b["lum"],
                "v1_level": a["level"], "v1_flood_ratio": a["flood_ratio"], "v1_scene": a["scene"],
                "v3_ratio": b["water_ratio"], "v3_ground": b["ground_ratio"], "v3_frac": b["water_frac"], "v3_base": b["base"],
            }) + "\n")
    print(clip, n)
