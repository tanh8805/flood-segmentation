"""Phase 2: build the candidate manifest (what goes through the teachers), with dedupe, groups, TEST hold-outs.
Idempotent: re-run after FRED finishes downloading. Output: work/manifest/candidates.json
Entry: id, source, path, group, split ('train'|'test_flood'|'test_car'|None -> decided at export), gt (path/type) , meta."""
import glob, json, os, random, re, sys
import cv2, numpy as np

R = os.environ.get("FLOOD_WORK", os.path.abspath("work"))
X = f"{R}/external"
WD_A = os.environ.get("WATERDATASET_A", f"{R}/raw/WaterDataset")      # có train_images/{JPEGImages,Annotations}
WD_B = os.environ.get("WATERDATASET_B", f"{R}/raw/WaterDataset_B")  # bản bổ sung (test_videos/...)
OLD = os.environ.get("OLD_DATASET_IMAGES", f"{R}/raw/old_dataset/images")  # chỉ dùng ảnh, không dùng nhãn
UP = os.environ.get("UPLOAD_VIDEOS", f"{R}/raw/uploads")
os.makedirs(f"{R}/manifest", exist_ok=True); os.makedirs(f"{R}/data/frames", exist_ok=True)
rng = random.Random(2025)


def dhash(path):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None: return None
    s = cv2.resize(im, (9, 8), interpolation=cv2.INTER_AREA)
    return int("".join("1" if v else "0" for v in (s[:, 1:] > s[:, :-1]).flatten()), 2)


def ham(a, b): return bin(a ^ b).count("1")


def clusters(items, thr=6):
    """union-find on dhash hamming<=thr -> group name per item"""
    n = len(items); par = list(range(n))
    def f(i):
        while par[i] != i: par[i] = par[par[i]]; i = par[i]
        return i
    hs = [it["hash"] for it in items]
    for i in range(n):
        for j in range(i + 1, n):
            if hs[i] is not None and hs[j] is not None and ham(hs[i], hs[j]) <= thr: par[f(i)] = f(j)
    for i, it in enumerate(items): it["group"] = f"{it['source']}_c{f(i)}"


cands = []
def add(**kw): cands.append(kw)

# ---------- human-mask sources ----------
for p in sorted(glob.glob(f"{X}/ds_Deepflood/Deepflood/image/*.jpg")):
    add(id="deepflood_" + os.path.basename(p)[:-4], source="deepflood", path=p, gt=p.replace("/image/", "/mask/")[:-4] + ".png", gt_type="flood01")
for p in sorted(glob.glob(f"{X}/ds_Sazara/Sazara/image/*.jpg")):
    add(id="sazara_" + os.path.basename(p)[:-4], source="sazara", path=p, gt=p.replace("/image/", "/mask/").replace("image_", "label_")[:-4] + ".png", gt_type="flood01")
for p in sorted(glob.glob(f"{WD_A}/train_images/JPEGImages/flooding_pixabay/*.jpg")):
    g = p.replace("JPEGImages", "Annotations")[:-4] + ".png"
    add(id="pixabay_" + os.path.basename(p)[:-4], source="pixabay", path=p, gt=g, gt_type="flood01")
for p in sorted(glob.glob(f"{WD_A}/train_images/JPEGImages/ADE20K_water/*.jpg")):
    g = p.replace("JPEGImages", "Annotations")[:-4] + ".png"
    add(id="ade_" + os.path.basename(p)[:-4], source="ade", path=p, gt=g, gt_type="flood01")

# ---------- houston (<=50, spread in time, A+B) ----------
hp = sorted(glob.glob(f"{WD_A}/test_videos/houston_buffalo_*/*.jpg") + glob.glob(f"{WD_B}/test_videos/houston_buffalo_*/*.jpg"), key=os.path.basename)
sel = [hp[int(i * (len(hp) - 1) / 49)] for i in range(50)]
for p in sel: add(id="houston_" + os.path.basename(p)[:-4], source="houston", path=p, gt=None)

# ---------- old dataset (images only, never labels) ----------
for p in sorted(glob.glob(f"{OLD}/*/*.jpg")):
    add(id="old_" + os.path.basename(p)[:-4], source="old", path=p, gt=None)

# ---------- uploads: frames every ~12th, dedup, block groups ----------
for vi, vp in enumerate(sorted(glob.glob(f"{UP}/*.mp4"))):
    cap = cv2.VideoCapture(vp); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); step = max(1, n // 100)
    kept, k = [], 0
    for fi in range(0, n, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi); ok, fr = cap.read()
        if not ok: continue
        out = f"{R}/data/frames/up{vi}_{fi:05d}.jpg"
        if not os.path.exists(out): cv2.imwrite(out, fr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        h = dhash(out)
        if any(ham(h, kh) <= 4 for kh in kept): continue
        kept.append(h); add(id=f"up{vi}_{fi:05d}", source="uploads", path=out, gt=None, group=f"up{vi}_b{k // 10}"); k += 1

# ---------- FRED ----------
FRED = f"{X}/fred"
for kind in ("flooded", "dry"):
    for sd in sorted(glob.glob(f"{FRED}/{kind}/*")):
        seq = os.path.basename(sd)
        imgs = sorted(p for p in glob.glob(f"{sd}/img/*.png"))
        if kind == "flooded": imgs = [p for p in imgs if os.path.exists(p.replace("/img/", "/lab/"))]
        kept = []
        for p in imgs:
            h = dhash(p)
            if h is None or any(ham(h, kh) <= 5 for _, kh in kept): continue
            kept.append((p, h))
        cap_n = 60 if kind == "flooded" else 20
        if len(kept) > cap_n: kept = [kept[int(i * (len(kept) - 1) / (cap_n - 1))] for i in range(cap_n)]
        for p, h in kept:
            add(id=f"fred{kind[0]}_{seq}_{os.path.basename(p)[:-4]}", source=f"fred_{kind}", path=p,
                gt=p.replace("/img/", "/lab/") if kind == "flooded" else None, gt_type="fred", group=f"fred_{seq}")

# ---------- COCO ----------
for p in sorted(glob.glob(f"{X}/coco_sel/*.jpg")):
    add(id="coco_" + os.path.basename(p)[:-4], source="coco", path=p, gt=None)

# ---------- STURM: original split; train/val files whose (car id, level) also appears in test.txt are removed ----------
S = f"{X}/sturm"
sp = {n: [l.strip().replace("\\", "/") for l in open(f"{S}/{n}.txt") if l.strip()] for n in ("train", "val", "test")}
key = lambda l: re.search(r"(car\d*-\d+-Level\d)", l).group(1)
test_keys = {key(l) for l in sp["test"]}
by_key = {}
for n in ("train", "val", "test"):
    for l in sp[n]:
        if n != "test" and key(l) in test_keys: continue
        by_key.setdefault((n == "test", key(l)), []).append(l)
for (is_test, k), ls in sorted(by_key.items()):
    l = sorted(ls)[0]  # one crop per physical car
    lv = int(re.search(r"Level(\d)", l).group(1))
    add(id="sturm_" + os.path.basename(l)[:-4], source="sturm", path=f"{X}/ds_sturm/upscaled_images/{l}", gt=None, group="sturm_" + k,
        split="test_car" if is_test else "train", meta=dict(level=lv))

# ---------- hashes, cross-source dedupe, groups ----------
for c in cands:
    if "hash" not in c: c["hash"] = dhash(c["path"])
prio = ["deepflood", "sazara", "pixabay", "fred_flooded", "fred_dry", "sturm", "ade", "uploads", "houston", "coco", "old"]
seen, out, dup = [], [], []
for c in sorted(cands, key=lambda c: prio.index(c["source"])):
    h = c["hash"]
    if c["source"] not in ("sturm",) and h is not None and any(ham(h, s) <= 3 for s in seen):
        dup.append(c["id"]); continue
    if h is not None and c["source"] != "sturm": seen.append(h)
    out.append(c)
cands = out
# groups for sources without natural groups
for src in ("deepflood", "sazara", "pixabay", "ade", "houston", "coco", "old"):
    its = [c for c in cands if c["source"] == src]
    clusters(its, 6 if src != "houston" else 12)
for c in cands: c.setdefault("group", c["source"] + "_" + c["id"])

# ---------- TEST hold-outs: ~120 Deepflood + ~40 pixabay (whole groups) ----------
def hold(src, n):
    gs = {}
    for c in cands:
        if c["source"] == src: gs.setdefault(c["group"], []).append(c)
    order = list(gs); rng.shuffle(order); got = 0
    for g in order:
        if got >= n: break
        for c in gs[g]: c["split"] = "test_flood"
        got += len(gs[g])
    return got
print("held out test_flood: deepflood", hold("deepflood", 120), "pixabay", hold("pixabay", 40))
for c in cands: c.setdefault("split", None); c.pop("hash", None)
json.dump(cands, open(f"{R}/manifest/candidates.json", "w"))
import collections
print("duplicates dropped:", len(dup), collections.Counter(d.split("_")[0] for d in dup))
cnt = collections.Counter((c["source"], c["split"]) for c in cands)
for k, v in sorted(cnt.items(), key=lambda t: str(t)): print(k, v)
print("total", len(cands))
