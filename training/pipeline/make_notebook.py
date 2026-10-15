"""Generate work/train_kaggle.ipynb (+ kaggle upload folders). The eval cell source is also exec'd locally as a smoke test."""
import json, os, ast

R = os.environ.get("FLOOD_WORK", os.path.abspath("work"))
USER = os.environ.get("KAGGLE_USER", "your-kaggle-username")
DS = "flood-street-seg-v1"

SETUP = r'''
import os, glob, shutil, yaml, json, subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.177"], check=False)
import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
SRC = os.path.dirname(glob.glob("/kaggle/input/**/data.yaml", recursive=True)[0]); print("dataset at", SRC)
DST = "/tmp/data"   # writable copy outside /kaggle/working so it is not part of the notebook output
if not os.path.exists(DST): shutil.copytree(SRC, DST, ignore=shutil.ignore_patterns("*.cache"))
for n in ("data", "data_test_flood", "data_test_car"):
    c = yaml.safe_load(open(f"{DST}/{n}.yaml")); c["path"] = DST; yaml.safe_dump(c, open(f"{DST}/{n}.yaml", "w"))
for sp in ("train", "val", "test_flood", "test_car"): print(sp, len(os.listdir(f"{DST}/images/{sp}")), "images")
'''

TRAIN = r'''
from ultralytics import YOLO
RUN = "/kaggle/working/runs"; NAME = "flood_street"
last = f"{RUN}/{NAME}/weights/last.pt"
if os.path.exists(last):                      # resume an interrupted run (e.g. when runs/ was attached as input and copied back)
    YOLO(last).train(resume=True)
else:
    YOLO("yolo11n-seg.pt").train(data=f"{DST}/data.yaml", epochs=150, patience=40, imgsz=640, batch=16, optimizer="auto",
                                 device=0, plots=True, project=RUN, name=NAME, exist_ok=True, workers=2, cache=True, seed=0)
'''

EVAL = r'''
import numpy as np, cv2, glob, os, json, pandas as pd
from ultralytics import YOLO

NAMES = ["flood", "road", "car_dry", "car_shallow", "car_deep"]
OLD2NEW = {0: 0, 1: 1}   # old best.pt classes are flood, road, wet_surface (no car classes): compare flood/road only

def read_label(p, H, W):
    out = []
    for l in open(p).read().splitlines():
        v = l.split()
        if len(v) >= 7: out.append((int(v[0]), (np.array(v[1:], float).reshape(-1, 2) * [W, H]).astype(np.int32)))
    return out

def iou(a, b):
    u = (a | b).sum(); return float((a & b).sum() / u) if u else np.nan

def seg_masks(model, img, cls_ids, mapping=None):
    r = model.predict(img, conf=0.25, imgsz=640, retina_masks=True, verbose=False, device=0)[0]
    H, W = img.shape[:2]; out = {}
    if r.masks is None: return r, out
    for m, c in zip(r.masks.data.cpu().numpy().astype(bool), r.boxes.cls.cpu().numpy().astype(int)):
        c = mapping.get(c, -1) if mapping else c
        if c in cls_ids:
            if m.shape != (H, W): m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
            out[c] = out.get(c, np.zeros((H, W), bool)) | m
    return r, out

def flood_road_iou(model, DST, mapping=None):
    """pixel IoU of predicted flood / road union vs labels on test_flood. flood GT = human masks; road GT = SegFormer pseudo-label (not human)."""
    rows = []
    for p in sorted(glob.glob(f"{DST}/images/test_flood/*.jpg")):
        img = cv2.imread(p); H, W = img.shape[:2]
        gt = {0: np.zeros((H, W), np.uint8), 1: np.zeros((H, W), np.uint8)}
        for c, poly in read_label(f"{DST}/labels/test_flood/{os.path.basename(p)[:-4]}.txt", H, W):
            if c in gt: cv2.fillPoly(gt[c], [poly], 1)
        _, pr = seg_masks(model, img, (0, 1), mapping)
        row = dict(id=os.path.basename(p)[:-4], src=os.path.basename(p).split("_")[0])
        row["flood_iou"] = iou(pr.get(0, np.zeros((H, W), bool)), gt[0].astype(bool))
        if gt[1].any(): row["road_iou"] = iou(pr.get(1, np.zeros((H, W), bool)), gt[1].astype(bool))
        rows.append(row)
    return pd.DataFrame(rows)

def car_level_eval(model, DST, mapping=None):
    """match predicted cars to GT cars (box IoU>=0.5); returns recall, level accuracy on matched, 3x3 confusion (rows GT: dry/shallow/deep)"""
    conf = np.zeros((3, 3), int); n_gt = 0; miss = 0
    for p in sorted(glob.glob(f"{DST}/images/test_car/*.jpg")):
        img = cv2.imread(p); H, W = img.shape[:2]
        gts = [(c - 2, cv2.boundingRect(poly)) for c, poly in read_label(f"{DST}/labels/test_car/{os.path.basename(p)[:-4]}.txt", H, W) if c >= 2]
        r = model.predict(img, conf=0.25, imgsz=640, verbose=False, device=0)[0]
        pb = [((mapping.get(int(c), -1) if mapping else int(c)) - 2, b) for c, b in zip(r.boxes.cls.cpu().numpy(), r.boxes.xyxy.cpu().numpy())]
        pb = [(c, b) for c, b in pb if c >= 0]
        for g, (x, y, w, h) in gts:
            n_gt += 1; best, bi = 0, None
            for c, b in pb:
                ix = max(0, min(x + w, b[2]) - max(x, b[0])); iy = max(0, min(y + h, b[3]) - max(y, b[1])); i = ix * iy
                u = w * h + (b[2] - b[0]) * (b[3] - b[1]) - i; s = i / u if u else 0
                if s > best: best, bi = s, c
            if best >= 0.5: conf[g, bi] += 1
            else: miss += 1
    m = conf.sum()
    return dict(n_gt=n_gt, recall=m / max(1, n_gt), level_acc=float(np.trace(conf) / max(1, m)), confusion_rows_gt_dry_shallow_deep=conf.tolist())

def ultra_val(model, DST, yaml_name):
    try:
        r = model.val(data=f"{DST}/{yaml_name}.yaml", imgsz=640, batch=16, plots=False, verbose=False, device=0)
        return {NAMES[int(i)]: dict(mask_ap50=float(r.seg.class_result(k)[2]), box_ap50=float(r.box.class_result(k)[2])) for k, i in enumerate(r.ap_class_index)}
    except Exception as e:
        return dict(error=repr(e)[:300])

def run_eval(new_path, old_path, DST, out_dir):
    os.makedirs(out_dir, exist_ok=True); res = {}
    for tag, path, mp in (("new", new_path, None), ("old", old_path, OLD2NEW)):
        m = YOLO(path); d = flood_road_iou(m, DST, mp)
        d.to_csv(f"{out_dir}/test_flood_iou_{tag}.csv", index=False)
        res[tag] = dict(flood_iou_mean=float(d.flood_iou.mean()), flood_iou_median=float(d.flood_iou.median()),
                        flood_iou_by_source={k: float(v) for k, v in d.groupby("src").flood_iou.mean().items()},
                        road_iou_vs_pseudo_mean=float(d.road_iou.mean()), car_level=(car_level_eval(m, DST, mp) if tag == "new" else "n/a (old model has no car classes)"))
    res["new"]["ultralytics_val_test_flood"] = ultra_val(YOLO(new_path), DST, "data_test_flood")
    res["new"]["ultralytics_val_test_car"] = ultra_val(YOLO(new_path), DST, "data_test_car")
    json.dump(res, open(f"{out_dir}/eval_summary.json", "w"), indent=1)
    return res
'''

RUNEVAL = r'''
import shutil
best = f"{RUN}/{NAME}/weights/best.pt"
res = run_eval(best, f"{DST}/old_model/best.pt", DST, "/kaggle/working/eval")
print(json.dumps(res, indent=1))
# copy deliverables next to the notebook output
W = "/kaggle/working"
shutil.copy(best, f"{W}/best.pt"); shutil.copy(f"{RUN}/{NAME}/results.csv", f"{W}/results.csv")
os.makedirs(f"{W}/plots", exist_ok=True)
for p in glob.glob(f"{RUN}/{NAME}/*.png") + glob.glob(f"{RUN}/{NAME}/*.jpg"): shutil.copy(p, f"{W}/plots/")
print(sorted(os.listdir(W)))
'''

def code(s): return dict(cell_type="code", metadata={}, source=s.strip("\n").splitlines(True), outputs=[], execution_count=None)
def md(s): return dict(cell_type="markdown", metadata={}, source=s.strip("\n").splitlines(True))

cells = [
    md("# flood-street-seg-v1: train YOLO11n-seg (flood / road / car_dry / car_shallow / car_deep)\n\nDataset is a pseudo-labelled set (teachers: V-FloodNet, SegFormer-b5, yolo11x-seg, best_car_YOLOv11; human masks from Deepflood, Sazara, FRED, flooding_pixabay; human car levels from STURM). "
       "Contains FRED (CC BY-NC-SA 4.0) -> keep private, non-commercial."),
    code(SETUP), code(TRAIN), code(EVAL), code(RUNEVAL),
]
nb = dict(cells=cells, metadata=dict(kernelspec=dict(display_name="Python 3", language="python", name="python3"), language_info=dict(name="python")), nbformat=4, nbformat_minor=5)
if __name__ == "__main__":
    for c in cells:
        if c["cell_type"] == "code": ast.parse("".join(c["source"]))
    json.dump(nb, open(f"{R}/train_kaggle.ipynb", "w"), indent=1)
    os.makedirs(f"{R}/kaggle_kernel", exist_ok=True)
    json.dump(nb, open(f"{R}/kaggle_kernel/train_kaggle.ipynb", "w"), indent=1)
    json.dump(dict(id=f"{USER}/{DS}-train", title=f"{DS}-train", code_file="train_kaggle.ipynb", language="python", kernel_type="notebook",
                   is_private=True, enable_gpu=True, enable_internet=True, dataset_sources=[f"{USER}/{DS}"], competition_sources=[], kernel_sources=[]),
              open(f"{R}/kaggle_kernel/kernel-metadata.json", "w"), indent=1)
    print("notebook ok")
