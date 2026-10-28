"""Du doan theo doan cua v1 (muc >= 1 = ngap, nhu README v3) va v3 (flooded) tu ket qua da luu trong tmp/raw."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT

RAW = ROOT / "tmp" / "raw"


def baseline_pred(clip, ver):
    segs = json.loads((RAW / f"{clip}.{ver}.json").read_text())["segments"]
    meta = json.loads((RAW / f"{clip}.v3.sig.meta.json").read_text())
    out = {}
    for i, s in enumerate(segs):
        out[i] = bool(s["level"] >= 1) if ver == "v1" else bool(s["flooded"])
    return out
