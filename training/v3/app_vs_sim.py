"""Chạy app thật (process_video, có ghi video) trên clip rồi so từng đoạn với mô phỏng từ cache. Dùng: app_vs_sim.py clip [clip ...]"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from clips import CLIPS, ROOT  # noqa: E402
import sim_binary as S  # noqa: E402
import analyzer  # noqa: E402

P = S.P
cfg = (P.FLOOD_ON, P.FLOOD_OFF, P.FLOOD_HOLD_S, P.FLOOD_SKIP_DARK)
print("cấu hình app:", cfg)
for clip in sys.argv[1:]:
    out = ROOT / "work/tmp" / f"{clip}_result.mp4"
    r = analyzer.process_video(CLIPS[clip], out)
    app = {round(s["start_s"] / 2) * 2: s["flooded"] for s in r["segments"]}
    sim = S.predict(clip, cfg, {clip: S.load_cache(clip)})
    diff = {t: (app.get(t), sim.get(t)) for t in set(app) | set(sim) if app.get(t) != sim.get(t)}
    print(clip, "đoạn:", len(app), "khớp:", len(app) - len(diff), "lệch:", diff, "|", r["summary"], "|", out.stat().st_size // 1024, "KB")
