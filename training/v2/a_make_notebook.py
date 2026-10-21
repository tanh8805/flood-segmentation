"""Build Kaggle upload folder + kernel for the water-seg v2 training (private)."""
import json, os, ast
V2 = "/home/tuananh/Documents/flood_segmentation/work/v2"
USER, DS, KN = "tuananhanubis", "flood-water-seg-v2", "flood-water-seg-v2-train"

SETUP = r'''
import os, glob, shutil, yaml, json, subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.177"], check=False)
import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())
SRC = os.path.dirname(glob.glob("/kaggle/input/**/data.yaml", recursive=True)[0]); print("dataset at", SRC)
DST = "/tmp/data"
if not os.path.exists(DST): shutil.copytree(SRC, DST, ignore=shutil.ignore_patterns("*.cache"))
c = yaml.safe_load(open(f"{DST}/data.yaml")); c["path"] = DST; yaml.safe_dump(c, open(f"{DST}/data.yaml", "w"))
for sp in ("train", "val", "test"): print(sp, len(os.listdir(f"{DST}/images/{sp}")), "images")
'''
TRAIN = r'''
from ultralytics import YOLO
RUN = "/kaggle/working/runs"; NAME = "water"
last = f"{RUN}/{NAME}/weights/last.pt"
if os.path.exists(last):
    YOLO(last).train(resume=True)
else:
    YOLO("yolo11s-seg.pt").train(data=f"{DST}/data.yaml", epochs=100, patience=30, imgsz=640, batch=16, optimizer="auto",
                                 device=0, plots=True, project=RUN, name=NAME, exist_ok=True, workers=2, cache=True, seed=0)
'''
EVAL = r'''
import numpy as np, cv2, pandas as pd
best = f"{RUN}/{NAME}/weights/best.pt"; model = YOLO(best); rows = []
for p in sorted(glob.glob(f"{DST}/images/test/*.jpg")):
    img = cv2.imread(p); H, W = img.shape[:2]; gt = np.zeros((H, W), np.uint8)
    for l in open(f"{DST}/labels/test/{os.path.basename(p)[:-4]}.txt").read().splitlines():
        v = l.split()
        if len(v) >= 7: cv2.fillPoly(gt, [(np.array(v[1:], float).reshape(-1, 2) * [W, H]).astype(np.int32)], 1)
    r = model.predict(img, conf=0.25, imgsz=640, retina_masks=True, verbose=False, device=0)[0]
    pr = np.zeros((H, W), bool)
    if r.masks is not None:
        for m in r.masks.data.cpu().numpy().astype(bool):
            if m.shape != (H, W): m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
            pr |= m
    u = (pr | gt.astype(bool)).sum()
    rows.append(dict(id=os.path.basename(p)[:-4], src=os.path.basename(p).split("_")[0], iou=float((pr & gt.astype(bool)).sum() / u) if u else np.nan))
d = pd.DataFrame(rows); os.makedirs("/kaggle/working/eval", exist_ok=True); d.to_csv("/kaggle/working/eval/test_flood_iou.csv", index=False)
res = dict(iou_mean=float(d.iou.mean()), iou_median=float(d.iou.median()), by_source={k: float(v) for k, v in d.groupby("src").iou.mean().items()})
json.dump(res, open("/kaggle/working/eval/eval_summary.json", "w"), indent=1); print(json.dumps(res, indent=1))
shutil.copy(best, "/kaggle/working/best.pt"); shutil.copy(f"{RUN}/{NAME}/results.csv", "/kaggle/working/results.csv")
'''
def code(s): return dict(cell_type="code", metadata={}, source=s.strip("\n").splitlines(True), outputs=[], execution_count=None)
def md(s): return dict(cell_type="markdown", metadata={}, source=s.strip("\n").splitlines(True))
cells = [md("# flood-water-seg-v2: YOLO11s-seg, one class `water`\n\nTrained on human-annotated masks only (Deepflood, Sazara, flooding_pixabay, FRED, ADE20K water) plus ~15% negatives."),
         code(SETUP), code(TRAIN), code(EVAL)]
for c in cells:
    if c["cell_type"] == "code": ast.parse("".join(c["source"]))
nb = dict(cells=cells, metadata=dict(kernelspec=dict(display_name="Python 3", language="python", name="python3"), language_info=dict(name="python")), nbformat=4, nbformat_minor=5)
os.makedirs(f"{V2}/kaggle_kernel", exist_ok=True)
json.dump(nb, open(f"{V2}/kaggle_kernel/train_water.ipynb", "w"), indent=1)
json.dump(dict(id=f"{USER}/{KN}", title=KN, code_file="train_water.ipynb", language="python", kernel_type="notebook", is_private=True,
               enable_gpu=True, enable_internet=True, dataset_sources=[f"{USER}/{DS}"], competition_sources=[], kernel_sources=[]), open(f"{V2}/kaggle_kernel/kernel-metadata.json", "w"), indent=1)
json.dump(dict(title=DS, id=f"{USER}/{DS}", licenses=[dict(name="CC-BY-NC-SA-4.0")], isPrivate=True), open(f"{V2}/water_ds/dataset-metadata.json", "w"), indent=1)
