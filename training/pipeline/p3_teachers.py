"""Phase 3: run teachers on every candidate once and cache raw outputs (resumable, sequential, fp16).
Per id: data/img/<id>.jpg (canonical image, max side 1280), cache/<id>.npz (fn mask, segformer label map, gt, car masks), cache/<id>.json.
Decisions (union / filters / levels) are made later in p4 from this cache, so rules can be tuned without re-running models."""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import cv2, numpy as np, torch
from PIL import Image
import teachers as T

R = T.ROOT
os.makedirs(f"{R}/data/img", exist_ok=True); os.makedirs(f"{R}/cache", exist_ok=True)
cands = json.load(open(f"{R}/manifest/candidates.json"))
only = set(sys.argv[1:])
if only: cands = [c for c in cands if c["source"] in only]
todo = [c for c in cands if not os.path.exists(f"{R}/cache/{c['id']}.json")]
print("candidates", len(cands), "todo", len(todo), flush=True)
fn, sf, cars = T.FloodNet(), T.SegFormer(), T.Cars()
MAXS = 1280


def prep(c):
    im = Image.open(c["path"]).convert("RGB")
    w, h = im.size; s = min(1.0, MAXS / max(w, h))
    if s < 1: im = im.resize((round(w * s), round(h * s)), Image.LANCZOS)
    rgb = np.array(im); H, W = rgb.shape[:2]
    gt = None
    if c.get("gt"):
        g = np.array(Image.open(c["gt"])); g = g if g.ndim == 2 else g[..., 0]
        gt = cv2.resize(g.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        if c.get("gt_type") == "flood01": gt = (gt > 0).astype(np.uint8)
    return rgb, gt


def seg_cars(bgr, conf, classes):
    s = cars.seg.predict(bgr, conf=conf, classes=classes, imgsz=960, half=True, retina_masks=True, verbose=False, device=0)[0]
    H, W = bgr.shape[:2]
    if s.masks is None or len(s.boxes) == 0: return np.zeros((0, 4)), np.zeros(0), np.zeros((0, H, W), bool)
    ms = s.masks.data.cpu().numpy().astype(bool)
    if ms.shape[1:] != (H, W): ms = np.stack([cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool) for m in ms])
    return s.boxes.xyxy.cpu().numpy(), s.boxes.conf.cpu().numpy(), ms


t0 = time.time()
for i, c in enumerate(todo):
    try:
        rgb, gt = prep(c)
        H, W = rgb.shape[:2]; bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{R}/data/img/{c['id']}.jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        fnm = (fn.prob(rgb) > 0.5)
        lab = sf.labels(rgb)
        sturm = c["source"] == "sturm"
        sb, sc, sm = seg_cars(bgr, 0.10 if sturm else 0.25, [cars.car_id, 7] if sturm else [cars.car_id])
        lv = []
        if not sturm:
            r = cars.level.predict(bgr, conf=0.25, half=True, verbose=False, device=0)[0]
            lv = [dict(box=b.tolist(), cls=int(k), conf=float(p)) for b, k, p in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy())]
        np.savez_compressed(f"{R}/cache/{c['id']}.npz", fn=fnm, sf=lab, car_masks=sm, **({"gt": gt} if gt is not None else {}))
        json.dump(dict(id=c["id"], hw=[H, W], level_dets=lv, seg_boxes=sb.tolist(), seg_conf=sc.tolist()), open(f"{R}/cache/{c['id']}.json", "w"))
    except Exception as e:
        print("ERR", c["id"], repr(e)[:200], flush=True)
    if i % 100 == 0: print(f"{i}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
print("P3 DONE", flush=True)
