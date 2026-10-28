"""Tải trước CLIP ViT-B-32 (open_clip) và SegFormer-B2 (mặt đất, veto nước) vào cache của thư viện.
water.pt và clf_b32.npz nằm sẵn trong floodapp/weights/."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from floodapp import classifier, pipeline as P

try:
    import open_clip
    open_clip.create_model_and_transforms(classifier.CLIP_NAME, pretrained=classifier.CLIP_PRETRAINED)
    print(classifier.CLIP_NAME, "ok")
except Exception as e:
    print("CLIP lỗi:", type(e).__name__, e, file=sys.stderr)
try:
    from transformers import SegformerForSemanticSegmentation
    SegformerForSemanticSegmentation.from_pretrained(P.GROUND_MODEL)
    print(P.GROUND_MODEL, "ok")
except Exception as e:
    print("SegFormer lỗi:", type(e).__name__, file=sys.stderr)
for f in (Path(P.WATER_MODEL_PATH), Path(classifier.CLF_PATH)):
    print(f.name, "có" if f.exists() else "THIẾU")
