"""Cut candidate crops (motorcycle/bicycle/bus/truck) from Deepflood+Sazara images and VN test videos. contact = water in bottom 25% of box."""
import json, os, glob, csv, cv2, numpy as np, random
from ultralytics import YOLO
W = "/home/tuananh/Documents/flood_segmentation/work"; OUT = f"{W}/v2/depth_cls_raw"; os.makedirs(f"{OUT}/crops", exist_ok=True)
random.seed(0)
det = YOLO(f"{W}/../flood-segmentation/backend/weights/yolo11n.pt"); old = YOLO(f"{W}/../flood-segmentation/backend/weights/best.pt")
NAMES = {1: "bicycle", 3: "motorcycle", 5: "bus", 7: "truck"}
rows = []
def handle(img, water, src, key):
    H, Wd = img.shape[:2]
    r = det.predict(img, conf=0.4, imgsz=640, classes=list(NAMES), verbose=False, device=0)[0]
    for k, (b, c, cf) in enumerate(zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int), r.boxes.conf.cpu().numpy())):
        x1, y1, x2, y2 = (int(v) for v in b)
        if y2 - y1 < 40 or x2 - x1 < 20: continue
        low = water[y1 + int(.75 * (y2 - y1)):y2 + 1, x1:x2]
        contact = bool(low.size and low.mean() >= 0.03)
        px, py = int(.1 * (x2 - x1)), int(.1 * (y2 - y1))
        cr = img[max(0, y1 - py):min(H, y2 + py), max(0, x1 - px):min(Wd, x2 + px)]
        name = f"{key}_{k}.jpg"; cv2.imwrite(f"{OUT}/crops/{name}", cr)
        rows.append(dict(crop=name, src=src, key=key, cls=NAMES[c], contact=int(contact), conf=round(float(cf), 2)))
cands = json.load(open(f"{W}/manifest/candidates.json"))
for c in cands:
    if c["source"] not in ("deepflood", "sazara"): continue
    img = cv2.imread(c["path"]); gt = cv2.imread(c["gt"], 0)
    if img is None or gt is None: continue
    if gt.shape != img.shape[:2]: gt = cv2.resize(gt, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    handle(img, gt > 0, c["source"], c["id"])
info = json.load(open(f"{W}/v2/vn_sheets/clips.json"))
vids = [(k, v["path"]) for k, v in info.items() if k not in ("clip3", "clip4")] + [(f"up{i}", p) for i, p in enumerate(sorted(glob.glob(os.path.expanduser("~/Downloads/PTHTTM/uploads/*.mp4"))))]
for k, p in vids:
    cap = cv2.VideoCapture(p); fps = cap.get(cv2.CAP_PROP_FPS) or 25; n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for t in np.arange(0, n / fps, 1.0):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps)); ok, f = cap.read()
        if not ok: continue
        s = 960 / f.shape[1]; f = cv2.resize(f, (960, int(f.shape[0] * s)))
        r = old.predict(f, conf=0.4, imgsz=640, retina_masks=True, verbose=False, device=0)[0]
        wm = np.zeros(f.shape[:2], bool)
        if r.masks is not None:
            for m, cc in zip(r.masks.data.cpu().numpy() > 0.5, r.boxes.cls.cpu().numpy().astype(int)):
                if cc == 0: wm |= m
        handle(f, wm, "video", f"{k}_{int(t)}")
with open(f"{OUT}/index.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, rows[0].keys()); w.writeheader(); w.writerows(rows)
import collections
print(len(rows), collections.Counter((r["src"], r["cls"], r["contact"]) for r in rows).most_common(20))
