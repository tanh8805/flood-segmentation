"""Quet ro ri (do giong CLIP giua anh train va khung clip DEV/TEST) + khai thac am tinh kho tu COCO bang CLIP zero-shot."""
import json
import sys
from pathlib import Path

import numpy as np
import open_clip
import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT
from embed import load_clip

SUF = "_bdd" if len(sys.argv) > 1 and sys.argv[1] == "bdd" else ""
rows = [json.loads(l) for l in (ROOT / f"tmp/pool{SUF}.jsonl").read_text().splitlines()]
emb = np.load(ROOT / f"tmp/pool{SUF}_emb.npy")
assert len(rows) == len(emb)
clips = sorted(p.name.split(".")[0] for p in (ROOT / "cache").glob("*.clip_b32.npy"))
cemb = {c: np.load(ROOT / f"cache/{c}.clip_b32.npy") for c in clips}
allc = np.concatenate(list(cemb.values()))
who = np.concatenate([[c] * len(e) for c, e in cemb.items()])
sim = emb @ allc.T
mx, arg = sim.max(1), sim.argmax(1)
for r, m, a in zip(rows, mx, arg):
    r["leak"] = round(float(m), 4)
    r["leak_clip"] = str(who[a])

model, dev = load_clip("ViT-B-32")
tok = open_clip.get_tokenizer("ViT-B-32-quickgelu")
prompts = {
    "night_wet": "a wet street at night with light reflections on the road",
    "rain": "a street in heavy rain",
    "shadow": "a road with tree shadows on the asphalt",
    "wet": "a wet road after rain",
    "dry": "a dry road in daylight",
}
with torch.no_grad():
    t = model.encode_text(tok(list(prompts.values())).to(dev)).float()
    t = torch.nn.functional.normalize(t, dim=-1).cpu().numpy()
sc = emb @ t.T
for r, s in zip(rows, sc):
    r["zs"] = {k: round(float(v), 4) for k, v in zip(prompts, s)}
(ROOT / f"tmp/pool{SUF}2.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
import collections
bysrc = collections.defaultdict(list)
for r in rows:
    bysrc[r["src"]].append(r["leak"])
for s, v in bysrc.items():
    v = np.array(v)
    print(s, "leak>0.95:", int((v > 0.95).sum()), ">0.92:", int((v > 0.92).sum()), ">0.90:", int((v > 0.90).sum()), "n", len(v))
