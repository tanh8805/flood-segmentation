"""Nạp trước SegFormer (mặt đất, veto nước) vào cache của Hugging Face.
water.pt là model tự train: đặt tay vào backend/weights/ (xem README)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline as P

try:
    from transformers import SegformerForSemanticSegmentation
    SegformerForSemanticSegmentation.from_pretrained(P.GROUND_MODEL)
    print(P.GROUND_MODEL, "ok")
except Exception as e:
    print("SegFormer lỗi:", type(e).__name__, file=sys.stderr)
print("water.pt", "có" if Path(P.WATER_MODEL_PATH).exists() else "THIẾU (tự train, xem README)")
