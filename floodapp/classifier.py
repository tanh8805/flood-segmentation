"""Bộ phân loại khung hình: CLIP ViT-B-32 đóng băng + hồi quy logistic. Khung BGR nguyên bản -> p(ngập) trong [0, 1]."""
import os
from pathlib import Path

import cv2
import numpy as np

WEIGHTS = Path(__file__).parent / "weights"
CLF_PATH = os.getenv("CLF_PATH", str(WEIGHTS / "clf_b32.npz"))
CLIP_NAME, CLIP_PRETRAINED = "ViT-B-32-quickgelu", "openai"
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)


class FloodClassifier:
    def __init__(self, device=None):
        import open_clip
        import torch
        self.torch = torch
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model, _, _ = open_clip.create_model_and_transforms(CLIP_NAME, pretrained=CLIP_PRETRAINED)
        self.model = model.eval().to(self.dev)
        if self.dev == "cuda":
            self.model = self.model.half()
        d = np.load(CLF_PATH)
        self.w, self.b, self.mu, self.sd = d["w"], float(d["b"]), d["mu"], d["sd"]

    def embed(self, frames_bgr):
        """(N,H,W,3) hoặc list khung BGR -> embedding CLIP L2-chuẩn hoá. Ép về 224x224 (không cắt) bằng INTER_AREA."""
        x = np.stack([cv2.resize(f, (224, 224), interpolation=cv2.INTER_AREA) for f in frames_bgr])[..., ::-1].astype(np.float32) / 255.0
        t = self.torch.from_numpy(((x - MEAN) / STD).transpose(0, 3, 1, 2)).to(self.dev)
        with self.torch.no_grad():
            f = self.model.encode_image(t.half() if self.dev == "cuda" else t).float()
            return self.torch.nn.functional.normalize(f, dim=-1).cpu().numpy()

    def predict(self, frame_bgr):
        z = ((self.embed([frame_bgr])[0] - self.mu) / self.sd) @ self.w + self.b
        return float(1 / (1 + np.exp(-z)))
