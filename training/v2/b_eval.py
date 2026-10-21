import glob, os, numpy as np, json
from ultralytics import YOLO
W = "/home/tuananh/Documents/flood_segmentation/work/v2"
m = YOLO(f"{W}/runs/depth_cls/weights/best.pt"); out = {}
for tag, pre in (("all", ""), ("sturm", "car-"), ("own", "own_")):
    C = np.zeros((4, 4), int)
    for c in range(4):
        fs = [f for f in glob.glob(f"{W}/depth_cls/test/d{c}/*.jpg") if os.path.basename(f).startswith(pre)]
        for f, r in zip(fs, m.predict(fs, verbose=False, device=0, imgsz=224) if fs else []): C[c, int(r.names[r.probs.top1][-1])] += 1
    out[tag] = dict(n=int(C.sum()), acc=round(float(np.trace(C) / max(1, C.sum())), 3), within1=round(float(sum(C[i, j] for i in range(4) for j in range(4) if abs(i - j) <= 1) / max(1, C.sum())), 3), confusion=C.tolist())
json.dump(out, open(f"{W}/depth_cls_eval.json", "w"), indent=1); print(json.dumps(out))
