"""CLIP dong bang: khung BGR 224x224 (squash) -> embedding L2-chuan hoa."""
import numpy as np
import open_clip
import torch

MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)


def load_clip(name="ViT-B-32", pretrained="openai"):
    name = name + "-quickgelu" if pretrained == "openai" and not name.endswith("quickgelu") else name
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, _ = open_clip.create_model_and_transforms(name, pretrained=pretrained)
    model = model.eval().to(dev)
    if dev == "cuda":
        model = model.half()
    model.dev = dev
    return model, dev


@torch.no_grad()
def embed_frames(model, frames_bgr, batch=64):
    """frames_bgr: (N,224,224,3) uint8 BGR."""
    out = []
    for i in range(0, len(frames_bgr), batch):
        x = frames_bgr[i:i + batch][..., ::-1].astype(np.float32) / 255.0
        x = torch.from_numpy(((x - MEAN) / STD).transpose(0, 3, 1, 2)).to(model.dev)
        x = x.half() if model.dev == "cuda" else x
        f = model.encode_image(x).float()
        out.append(torch.nn.functional.normalize(f, dim=-1).cpu().numpy())
    return np.concatenate(out)
