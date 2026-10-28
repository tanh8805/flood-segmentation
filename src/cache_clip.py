"""Khung 224x224 (squash, khong crop) + embedding CLIP ViT-B-32 tai cac khung suy luan (5 khung/giay).
Luu tmp/raw/<clip>.f224.npy va cache/<clip>.clip_b32.npy."""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT, clip_path
from embed import load_clip, embed_frames

INFER_FPS = 5.0
model, _ = load_clip("ViT-B-32")
for clip in sys.argv[1:]:
    out = ROOT / "cache" / f"{clip}.clip_b32.npy"
    if out.exists():
        continue
    cap = cv2.VideoCapture(str(clip_path(clip)))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps > 240:
        fps = 25.0
    every = max(1, round(fps / INFER_FPS))
    frames, idx = [], 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if idx % every == 0:
            frames.append(cv2.resize(fr, (224, 224), interpolation=cv2.INTER_AREA))
        idx += 1
    cap.release()
    arr = np.stack(frames)
    np.save(ROOT / "tmp" / "raw" / f"{clip}.f224.npy", arr)
    np.save(out, embed_frames(model, arr))
    print("clip", clip, arr.shape, flush=True)
