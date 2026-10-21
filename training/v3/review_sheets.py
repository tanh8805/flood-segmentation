"""Bộ test mới: 1 contact sheet/clip (khung giữa mỗi đoạn 2 s, ô đánh số t_s) + review/labels.csv mặc định theo thư mục.
Cột label: 1 ngập, 0 không ngập, x bỏ qua (logo/đen/chuyển cảnh/trái nhóm). Bỏ qua clip đã có sheet."""
import csv
import math
from pathlib import Path

import cv2
import numpy as np

V = Path.home() / "Videos" / "flood_test_new"
OUT = Path(__file__).resolve().parents[1] / "review"
OUT.mkdir(exist_ok=True)
rows = []
for group, default in (("ngap", 1), ("khong_ngap", 0)):
    for p in sorted((V / group).glob("*.mp4")):
        cap = cv2.VideoCapture(str(p))
        fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        dur = n / fps
        segs = math.ceil(dur / 2)
        thumbs = []
        for k in range(segs):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(min(k * 2 + 1, dur - 0.1) * fps))
            ok, f = cap.read()
            f = cv2.resize(f, (128, round(128 * f.shape[0] / f.shape[1]))) if ok else np.zeros((228, 128, 3), np.uint8)
            cv2.putText(f, str(k * 2), (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            thumbs.append(f)
            rows.append((p.stem, k * 2, default))
        cols = 10
        th = max(t.shape[0] for t in thumbs)
        sheet = np.zeros((math.ceil(len(thumbs) / cols) * th, cols * 128, 3), np.uint8)
        for i, t in enumerate(thumbs):
            r, c = divmod(i, cols)
            sheet[r * th:r * th + t.shape[0], c * 128:(c + 1) * 128] = t
        cv2.imwrite(str(OUT / f"{p.stem}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
        print(p.stem, f"{dur:.1f}s", segs, "ô", sheet.shape[:2])
if not (OUT / "labels.csv").exists():
    with open(OUT / "labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["clip", "t_s", "label", "ghi_chu"])
        w.writerows([(c, t, d, "") for c, t, d in rows])
