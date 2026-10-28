import json
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = Path.home() / "Videos" / "flood"
SEG = 2.0


def clips():
    """Danh ba clip v5: {ten: {path, cls, src}} (clips.json). ngap_4 (gara trong nhan) va ngap_8 (trung ngap_6) khong co trong do."""
    return json.loads((ROOT / "clips.json").read_text())


def clip_path(name):
    return Path(clips()[name]["path"])


def splits():
    s = json.loads((ROOT / "splits.json").read_text())
    out = {}
    for part in ("dev", "test"):
        for cls, names in s[part].items():
            for n in names:
                out[n] = part
    return out


def video_info(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    if n <= 0:
        return None
    return {"fps": fps, "frames": n, "duration": n / fps}


def n_segments(duration):
    full = int(duration // SEG)
    return full + (1 if duration - full * SEG >= 1.0 else 0)


def grab_at(path, times):
    """Doc tuan tu, tra ve {t: frame BGR} gan nhat voi cac moc thoi gian (tang dan)."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    targets = sorted(times)
    out, i, idx = {}, 0, 0
    while i < len(targets):
        ok = cap.grab()
        if not ok:
            break
        if idx >= int(round(targets[i] * fps)):
            ok, fr = cap.retrieve()
            if ok:
                out[targets[i]] = fr
            i += 1
        idx += 1
    cap.release()
    return out
