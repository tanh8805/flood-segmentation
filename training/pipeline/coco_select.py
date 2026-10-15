"""COCO val2017 -> up to 500 dry-road images with road + car (by SegFormer), no water. Reads straight from the zip."""
import io, os, sys, zipfile, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image
import teachers as T

ZIP = f"{T.ROOT}/external/coco/val2017.zip"
OUT = f"{T.ROOT}/external/coco_sel"; os.makedirs(OUT, exist_ok=True)
CAND = f"{T.ROOT}/external/coco_candidates.json"
MAXN = 500
sf = T.SegFormer()
car_id = [i for i, n in sf.id2label.items() if n.split(",")[0].strip().lower() == "car"][0]
side_ids = [i for i, n in sf.id2label.items() if n.split(",")[0].strip().lower() in ("sidewalk", "building")]
z = zipfile.ZipFile(ZIP)
names = sorted(n for n in z.namelist() if n.endswith(".jpg"))
done = json.load(open(CAND)) if os.path.exists(CAND) else {}
for k, n in enumerate(names):
    if n in done: continue
    rgb = np.array(Image.open(io.BytesIO(z.read(n))).convert("RGB"))
    lab = sf.labels(rgb)
    done[n] = dict(road=float(np.isin(lab, sf.road_ids).mean()), car=float((lab == car_id).mean()),
                   water=float(np.isin(lab, sf.water_ids).mean()), side=float(np.isin(lab, side_ids).mean()))
    if k % 500 == 0:
        json.dump(done, open(CAND, "w")); print(k, len(names), flush=True)
json.dump(done, open(CAND, "w"))
ok = [(n, d) for n, d in done.items() if d["road"] >= 0.08 and d["car"] >= 0.01 and d["water"] < 0.01]
ok.sort(key=lambda t: -(min(t[1]["road"], 0.4) + min(t[1]["car"], 0.2)))
sel = ok[:MAXN]
for n, d in sel:
    open(f"{OUT}/{os.path.basename(n)}", "wb").write(z.read(n))
print("candidates passing filter:", len(ok), "selected:", len(sel))
