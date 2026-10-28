"""Chay code cu (v1 = main, v3 = nhanh v3) tren moi clip, luu tin hieu tho vao tmp/raw/<clip>.<ver>.json.
Dung: python cache_old.py v1|v3 clip [clip ...]"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT, clip_path

ver = sys.argv[1]
sys.path.insert(0, str(ROOT / "tmp" / ("v1_src" if ver == "v1" else "v3_src") / "backend"))
import analyzer as A  # noqa: E402

raw = ROOT / "tmp" / "raw"
for clip in sys.argv[2:]:
    out = raw / f"{clip}.{ver}.json"
    if out.exists():
        continue
    src = clip_path(clip)
    if ver == "v1":
        res = A.process_video(src, raw / "_v1.mp4")
    else:
        res = A.process_video(src, None, signals_path=raw / f"{clip}.v3.sig.jsonl")
    out.write_text(json.dumps({"frames": res["frames"], "segments": res["segments"],
                               "flooded_seconds": res.get("flooded_seconds")}))
    print(ver, clip, len(res["frames"]), flush=True)
