"""Phase 5: select final images (balancing / negatives / group split) and export a YOLO-seg dataset + review overlays.
Run after p4. Output: work/dataset/{images,labels}/{train,val,test_flood,test_car}, data.yaml, export_stats.json, review sheets."""
import json, os, random, shutil, sys, collections
import cv2, numpy as np

R = os.environ.get("FLOOD_WORK", os.path.abspath("work"))
D = f"{R}/dataset"
NAMES = ["flood", "road", "car_dry", "car_shallow", "car_deep"]
rng = random.Random(11)
cands = {c["id"]: c for c in json.load(open(f"{R}/manifest/candidates.json"))}
dec = json.load(open(f"{R}/final/decisions.json"))
excl = set(open(f"{R}/final/coco_exclude.txt").read().split()) if os.path.exists(f"{R}/final/coco_exclude.txt") else set()
PER_LEVEL, NEG_FRAC, VAL_FRAC = 160, 0.05, 0.15

keep = {i: d for i, d in dec.items() if d["status"] == "keep" and i not in excl}
test_flood = [i for i, d in keep.items() if d["split"] == "test_flood"]
test_car = [i for i, d in keep.items() if d["split"] == "test_car"]
pool = {i: d for i, d in keep.items() if d["split"] is None or d["split"] == "train"}

# STURM: <=800 crops balanced by original level
st = collections.defaultdict(list)
for i in pool:
    if cands[i]["source"] == "sturm": st[cands[i]["meta"]["level"]].append(i)
for L, ids in st.items():
    rng.shuffle(ids)
    for i in ids[PER_LEVEL:]: pool.pop(i)

# lake/sea/fountain negatives (empty flood), <=5% of final dataset
negs = [i for i, d in dec.items() if d["status"] == "neg_pool" and d["split"] is None and d["source"] in ("ade", "houston", "old")]
rng.shuffle(negs)
n_neg = int(NEG_FRAC / (1 - NEG_FRAC) * (len(pool) + len(test_flood) + len(test_car)))
neg_sel = negs[:n_neg]
for i in neg_sel: pool[i] = dec[i]

# group split 85/15 per source (whole groups)
split = {}
bysrc = collections.defaultdict(lambda: collections.defaultdict(list))
for i in pool: bysrc[cands[i]["source"]][cands[i]["group"]].append(i)
for src, gs in bysrc.items():
    order = list(gs); rng.shuffle(order); tot = sum(len(v) for v in gs.values()); got = 0
    for g in order:
        s = "val" if got < VAL_FRAC * tot else "train"
        if s == "val": got += len(gs[g])
        for i in gs[g]: split[i] = s
for i in test_flood: split[i] = "test_flood"
for i in test_car: split[i] = "test_car"

# ---- write ----
if os.path.exists(D): shutil.rmtree(D)
for s in ("train", "val", "test_flood", "test_car"):
    os.makedirs(f"{D}/images/{s}"); os.makedirs(f"{D}/labels/{s}")
stats = collections.defaultdict(lambda: collections.Counter())
inst_cls = collections.defaultdict(lambda: collections.Counter())
for i, s in split.items():
    lab = json.load(open(f"{R}/final/labels/{i}.json"))
    ins = [] if i in neg_sel else lab["instances"]
    lines = [f"{k['cls']} " + " ".join(f"{x:.5f} {y:.5f}" for x, y in k["poly"]) for k in ins]
    shutil.copy(f"{R}/data/img/{i}.jpg", f"{D}/images/{s}/{i}.jpg")
    open(f"{D}/labels/{s}/{i}.txt", "w").write("\n".join(lines) + ("\n" if lines else ""))
    src = "neg_lake_sea" if i in neg_sel else cands[i]["source"]
    stats[src][s] += 1
    for k in ins: inst_cls[s][NAMES[k["cls"]]] += 1
open(f"{D}/data.yaml", "w").write(f"path: {D}\ntrain: images/train\nval: images/val\ntest: images/test_flood\nnames:\n" + "".join(f"  {k}: {n}\n" for k, n in enumerate(NAMES)))
tot_inst = collections.Counter()
for s in inst_cls: tot_inst.update(inst_cls[s])
out = dict(images_per_source={k: dict(v) for k, v in stats.items()}, instances_per_split={k: dict(v) for k, v in inst_cls.items()}, instances_total=dict(tot_inst),
           n_images=len(split), sturm_levels_kept={L: min(len(v), PER_LEVEL) for L, v in st.items()}, neg_pool=len(negs), neg_used=len(neg_sel))
json.dump(out, open(f"{R}/final/export_stats.json", "w"), indent=1)
print(json.dumps(out, indent=1))

# ---- review overlays (100 random from train+val) ----
COL = [(255, 60, 60), (60, 120, 255), (60, 220, 60), (255, 170, 0), (220, 0, 200)]
def overlay(i, s):
    im = cv2.imread(f"{D}/images/{s}/{i}.jpg"); H, W = im.shape[:2]; o = im.copy()
    L = [l.split() for l in open(f"{D}/labels/{s}/{i}.txt").read().splitlines() if l.strip()]
    for l in sorted(L, key=lambda l: int(l[0])):
        k = int(l[0]); pts = (np.array(l[1:], float).reshape(-1, 2) * [W, H]).astype(np.int32)
        cv2.fillPoly(o, [pts], COL[k])
    im = cv2.addWeighted(o, 0.45, im, 0.55, 0)
    for l in L:
        k = int(l[0]); pts = (np.array(l[1:], float).reshape(-1, 2) * [W, H]).astype(np.int32)
        cv2.polylines(im, [pts], True, COL[k], 2)
    sc = 400 / W; im = cv2.resize(im, (400, max(1, int(H * sc))))
    im = im[:300]; pad = np.zeros((300, 400, 3), np.uint8); pad[:im.shape[0]] = im
    cv2.rectangle(pad, (0, 0), (400, 16), (0, 0, 0), -1); cv2.putText(pad, f"{i[-38:]}", (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 1)
    return pad
if "--review" in sys.argv:
    ids = [i for i, s in split.items() if s in ("train", "val")]; rng2 = random.Random(6); sel = rng2.sample(ids, min(100, len(ids)))
    os.makedirs(f"{R}/review", exist_ok=True)
    for f in os.listdir(f"{R}/review"): os.remove(f"{R}/review/{f}")
    for k in range(0, len(sel), 12):
        tiles = [overlay(i, split[i]) for i in sel[k:k + 12]]
        while len(tiles) < 12: tiles.append(np.zeros((300, 400, 3), np.uint8))
        rows = [np.concatenate(tiles[r * 3:(r + 1) * 3], 1) for r in range(4)]
        cv2.imwrite(f"{R}/review/review_{k // 12 + 1:02d}.jpg", np.concatenate(rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 88])
    json.dump(sel, open(f"{R}/review/ids.json", "w"))
    print("review sheets", (len(sel) + 11) // 12)
