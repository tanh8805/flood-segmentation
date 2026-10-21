"""Chạy model MỘT lần trên mọi clip, lưu tín hiệu từng lần suy luận -> work/v3/cache/<clip>.jsonl (+ .meta.json).
Dùng:  work/.venv/bin/python -I work/v3/scripts/cache_signals.py [clip ...]   (bỏ qua clip đã có cache)"""
import sys
from pathlib import Path

ROOT = Path("/home/tuananh/Documents/flood_segmentation")
sys.path.insert(0, str(ROOT / "flood-segmentation" / "backend"))
import analyzer  # noqa: E402

from clips import CLIPS  # noqa: E402

OUT = ROOT / "work/v3/cache"
OUT.mkdir(parents=True, exist_ok=True)
for name in sys.argv[1:] or CLIPS:
    dst = OUT / f"{name}.jsonl"
    if dst.with_suffix(".meta.json").exists():
        continue
    r = analyzer.process_video(CLIPS[name], None, signals_path=dst)
    print(name, r["duration_s"], r["frames_analyzed"], flush=True)
print("DONE", flush=True)
