"""Dataset format check with ultralytics (no training): load data.yaml, count images/labels per split, report polygon errors."""
import os, sys, glob, collections, yaml, numpy as np
D = os.path.join(os.environ.get("FLOOD_WORK", os.path.abspath("work")), "dataset")
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import check_det_dataset
cfg = yaml.safe_load(open(f"{D}/data.yaml")); cfg["path"] = D; tmpy = os.path.join(os.path.dirname(D), "data_abs.yaml"); yaml.safe_dump(cfg, open(tmpy, "w"))
names = cfg["names"]; errs = []
tot = {}
for sp in ("train", "val", "test_flood", "test_car"):
    imgs = sorted(glob.glob(f"{D}/images/{sp}/*.jpg")); labs = sorted(glob.glob(f"{D}/labels/{sp}/*.txt"))
    ib = {os.path.basename(p)[:-4] for p in imgs}; lb = {os.path.basename(p)[:-4] for p in labs}
    cnt = collections.Counter(); empty = 0
    for p in labs:
        L = [l.split() for l in open(p).read().splitlines() if l.strip()]
        empty += not L
        for l in L:
            v = np.array(l[1:], float)
            if len(v) < 6 or len(v) % 2: errs.append((p, "bad point count")); continue
            if v.min() < 0 or v.max() > 1: errs.append((p, "coords out of [0,1]"))
            xy = v.reshape(-1, 2); a = 0.5 * abs(np.dot(xy[:, 0], np.roll(xy[:, 1], 1)) - np.dot(xy[:, 1], np.roll(xy[:, 0], 1)))
            if a <= 1e-7: errs.append((p, "zero-area polygon"))
            if int(l[0]) not in names: errs.append((p, "bad class")); continue
            cnt[names[int(l[0])]] += 1
    print(f"{sp}: images={len(imgs)} labels={len(labs)} empty_labels={empty} img_without_label={len(ib - lb)} label_without_img={len(lb - ib)} instances={dict(cnt)}")
    tot[sp] = len(imgs)
    ds = YOLODataset(img_path=f"{D}/images/{sp}", data=dict(names=names, nc=len(names), channels=3), task="segment", imgsz=640, augment=False)
    nseg = sum(len(x["segments"]) for x in ds.labels)
    print(f"   ultralytics: loaded {len(ds.labels)} images, {nseg} segments")
print("check_det_dataset:", {k: (v if not isinstance(v, dict) else "dict") for k, v in check_det_dataset(tmpy).items() if k in ("nc", "train", "val", "test")})
print("polygon errors:", len(errs), errs[:10])
