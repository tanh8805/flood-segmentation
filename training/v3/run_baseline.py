"""Chạy một phiên bản app cũ (v1 = main, v2) trên mọi clip -> work/v3/baselines/<ver>.json: {clip: [{t, level, evidence}]}.
Dùng: run_baseline.py <ver> <backend_dir>   (worktree của nhánh tương ứng)"""
import json
import os
import sys
from pathlib import Path

ver, backend = sys.argv[1], sys.argv[2]
sys.path.insert(0, str(Path(__file__).parent))
from clips import CLIPS, ROOT  # noqa: E402

sys.path.insert(0, backend)
os.environ["MAX_SECONDS"] = "600"
import analyzer as A  # noqa: E402

out = ROOT / "work/v3/baselines" / f"{ver}.json"
out.parent.mkdir(exist_ok=True)
res = json.loads(out.read_text()) if out.exists() else {}
tmp = ROOT / "work/tmp/baseline_out.mp4"
for name, path in CLIPS.items():
    if name in res:
        continue
    r = A.process_video(str(path), str(tmp))
    res[name] = [dict(t=s["start_s"], level=s["level"], evidence=s.get("evidence")) for s in r["segments"]]
    out.write_text(json.dumps(res, ensure_ascii=False))
    print(ver, name, len(res[name]), flush=True)
print("DONE", flush=True)
