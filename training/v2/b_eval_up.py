import glob, os, sys, json, numpy as np
from ultralytics import YOLO
sys.path.insert(0, "/home/tuananh/Documents/flood_segmentation/flood-segmentation/backend")
import pipeline as P
W = "/home/tuananh/Documents/flood_segmentation/work/v2"
m = YOLO(f"{W}/runs/depth_cls/weights/best.pt"); out = {}
rows = []
for c in range(4):
    for f in glob.glob(f"{W}/depth_cls/test/d{c}/*.jpg"):
        rows.append((c, os.path.basename(f), f))
res = m.predict([r[2] for r in rows], verbose=False, device=0, imgsz=224)
order = sorted(range(4), key=lambda i: int(m.names[i][-1]))
probs = np.array([r.probs.data.cpu().numpy()[order] for r in res])
y = np.array([r[0] for r in rows]); name = [r[1] for r in rows]
def stats(pred, sel):
    p, t = pred[sel], y[sel]
    return dict(n=int(sel.sum()), exact=round(float((p == t).mean()), 3), within1=round(float((abs(p - t) <= 1).mean()), 3), under=round(float((p < t).mean()), 3), over=round(float((p > t).mean()), 3))
before = probs.argmax(1); after = np.array([P.pick_depth(p) for p in probs])
for tag, pre in (("all", ""), ("sturm", "car-"), ("own", "own_")):
    sel = np.array([n.startswith(pre) for n in name])
    out[tag] = dict(before=stats(before, sel), after=stats(after, sel))
json.dump(out, open(f"{W}/depth_cls_eval_up.json", "w"), indent=1); print(json.dumps(out))
