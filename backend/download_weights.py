"""Tải các weights công khai vào backend/weights/ (yolo11n-pose, yolo11n của Ultralytics) và nạp trước SegFormer vào cache.
water.pt và depth_cls.pt là model tự train: nếu quá lớn để commit thì đặt tay vào backend/weights/ (xem README)."""
import sys
from pathlib import Path

from ultralytics.utils.downloads import attempt_download_asset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline as P

W = Path(__file__).parent / "weights"
W.mkdir(exist_ok=True)
for name in ("yolo11n-pose.pt", "yolo11n.pt"):
    if not (W / name).exists():
        attempt_download_asset(str(W / name))
    print(name, "ok" if (W / name).exists() else "LỖI")
try:
    from transformers import SegformerForSemanticSegmentation
    SegformerForSemanticSegmentation.from_pretrained(P.GROUND_MODEL)
    print(P.GROUND_MODEL, "ok")
except Exception as e:
    print("SegFormer lỗi:", type(e).__name__, file=sys.stderr)
for name in ("water.pt", "depth_cls.pt"):
    print(name, "có" if (W / name).exists() else "THIẾU (tự train, xem README)")
