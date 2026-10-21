import sys, glob, os, json, cv2, numpy as np
sys.path.insert(0, "/home/tuananh/Documents/flood_segmentation/flood-segmentation/backend")
import pipeline as P
DS = "/home/tuananh/Documents/flood_segmentation/work/v2/water_ds"; DARK = 70
pipe = P.FloodPipeline(); res = []
def lum(p): return float(cv2.cvtColor(cv2.resize(cv2.imread(p), (160, 90)), cv2.COLOR_BGR2GRAY).mean())
stats = {}
for sp in ("train", "val", "test"):
    L = [lum(p) for p in glob.glob(f"{DS}/images/{sp}/*.jpg")]
    stats[sp] = dict(n=len(L), dark=int(sum(l < DARK for l in L)), dark_frac=round(sum(l < DARK for l in L) / len(L), 3), p10=round(float(np.percentile(L, 10)), 1))
print("độ sáng (dark = TB xám <%d):" % DARK, json.dumps(stats))
for p in sorted(glob.glob(f"{DS}/images/test/*.jpg")):
    img = cv2.imread(p); H, W = img.shape[:2]; gt = np.zeros((H, W), np.uint8)
    for l in open(f"{DS}/labels/test/{os.path.basename(p)[:-4]}.txt").read().splitlines():
        v = l.split()
        if len(v) >= 7: cv2.fillPoly(gt, [(np.array(v[1:], float).reshape(-1, 2) * [W, H]).astype(np.int32)], 1)
    gt = gt.astype(bool)
    r = pipe.water.predict(img, conf=0.25, imgsz=640, retina_masks=True, verbose=False, device=0)[0]
    pr = np.zeros((H, W), bool)
    if r.masks is not None:
        for m in r.masks.data.cpu().numpy().astype(bool):
            if m.shape != (H, W): m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
            pr |= m
    _, veto = pipe.scene_masks(img); pv = pr & ~veto
    iou = lambda a: float((a & gt).sum() / max(1, (a | gt).sum()))
    res.append(dict(id=os.path.basename(p)[:-4], src=os.path.basename(p).split("_")[0], lum=lum(p), before=iou(pr), after=iou(pv), removed=float((pr & veto).sum() / max(1, pr.sum())), gt_in_veto=float((gt & veto).sum() / max(1, gt.sum()))))
import pandas as pd
d = pd.DataFrame(res); d.to_csv("/home/tuananh/Documents/flood_segmentation/work/v2/veto_iou.csv", index=False)
print("tất cả", len(d), d[["before", "after"]].mean().round(4).to_dict())
print(d.groupby("src")[["before", "after"]].mean().round(4).to_dict())
dk = d[d.lum < DARK]; print("ảnh tối", len(dk), dk[["before", "after"]].mean().round(4).to_dict() if len(dk) else "-")
print("ảnh sáng", len(d) - len(dk), d[d.lum >= DARK][["before", "after"]].mean().round(4).to_dict())
print("ảnh giảm >0.05:", int(((d.before - d.after) > 0.05).sum()), "tăng >0.05:", int(((d.after - d.before) > 0.05).sum()), "| GT nằm trong vùng veto TB:", round(d.gt_in_veto.mean(), 4))
