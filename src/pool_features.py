"""Dac trung cho ung vien train: SegFormer (mat dat/veto/nuoc), ti le nuoc tren mat dat tu mask nguoi gan, CLIP B-32.
Ra: tmp/pool.jsonl + tmp/pool_emb.npy"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import SegformerForSemanticSegmentation

sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT
from embed import embed_frames, load_clip

OLD = Path.home() / "Documents/flood_segmentation/work"
GROUND = ("road", "sidewalk", "path", "earth", "floor")
VETO = ("sky", "building", "house", "tree", "wall", "ceiling", "signboard")
WATERISH = ("water", "sea", "river", "lake", "swimming pool", "waterfall", "fountain")
HUMAN = {"deepflood", "sazara", "pixabay", "fredf", "ade"}   # nguon co mask nuoc nguoi gan trong work/dataset


SUF = "_bdd" if len(sys.argv) > 1 and sys.argv[1] == "bdd" else ""


def candidates():
    out = []
    if len(sys.argv) > 1 and sys.argv[1] == "bdd":
        for p in sorted((ROOT / "data/bdd").glob("*.jpg")):
            out.append({"id": f"bdd:{p.stem}", "path": str(p), "src": "bdd", "label": None})
        return out
    for split in ("train", "val", "test_flood"):
        for p in sorted((OLD / "dataset/images" / split).iterdir()):
            src = p.name.split("_")[0]
            if src in ("up", "sturm"):
                continue
            lab = OLD / "dataset/labels" / split / (p.stem + ".txt")
            out.append({"id": f"{src}:{p.stem}", "path": str(p), "src": src, "label": str(lab) if lab.exists() else None})
    for p in sorted((ROOT / "data/coco_val/val2017").iterdir()):
        out.append({"id": f"cocoval:{p.stem}", "path": str(p), "src": "cocoval", "label": None})
    return out


def human_water(label, h, w):
    m = np.zeros((h, w), np.uint8)
    if not label:
        return m
    for ln in Path(label).read_text().splitlines():
        v = ln.split()
        if not v or v[0] != "0":
            continue
        pts = (np.array(v[1:], np.float32).reshape(-1, 2) * [w, h]).astype(np.int32)
        if len(pts) >= 3:
            cv2.fillPoly(m, [pts], 1)
    return m.astype(bool)


def main():
    dev = "cuda"
    seg = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512").eval().half().to(dev)
    names = {int(i): str(n).split(",")[0].strip().lower() for i, n in seg.config.id2label.items()}
    ids = lambda group: torch.tensor([i for i, n in names.items() if n in group], device=dev)
    g_ids, v_ids, w_ids = ids(GROUND), ids(VETO), ids(WATERISH)
    clip, _ = load_clip("ViT-B-32")
    rows, embs, chunks = [], [], []
    cands = candidates()
    for k, c in enumerate(cands):
        img = cv2.imread(c["path"])
        if img is None:
            continue
        h, w = img.shape[:2]
        sh = max(32, round(h * 512 / w / 32) * 32)
        x = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (512, sh), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        t = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(dev).half()
        with torch.no_grad():
            lab = seg(pixel_values=t).logits.float().argmax(1)[0]
            lab = torch.nn.functional.interpolate(lab[None, None].float(), size=(sh, 512), mode="nearest")[0, 0].long()
            ground = torch.isin(lab, g_ids).cpu().numpy()
            veto = torch.isin(lab, v_ids).cpu().numpy()
            sw = torch.isin(lab, w_ids).cpu().numpy()
        hw = human_water(c["label"], sh, 512) if c["src"] in HUMAN else np.zeros((sh, 512), bool)
        hw &= ~veto
        gpx = int((ground | hw).sum())
        c.update(h=h, w=w, ground=round(float(ground.mean()), 4), veto=round(float(veto.mean()), 4),
                 segwater=round(float(sw.mean()), 4), water=round(float(hw.mean()), 4),
                 ratio=round(int(hw.sum()) / gpx, 4) if gpx else 0.0, gpx=round(gpx / (sh * 512), 4))
        rows.append(c)
        embs.append(cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA))
        if len(embs) == 256 or k == len(cands) - 1:
            chunks.append(embed_frames(clip, np.stack(embs)))
            embs = []
        if k % 500 == 0:
            print(k, len(cands), flush=True)
    np.save(ROOT / f"tmp/pool{SUF}_emb.npy", np.concatenate(chunks))
    (ROOT / f"tmp/pool{SUF}.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    print("done", len(rows))


main()
