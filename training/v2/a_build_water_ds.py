"""Water dataset v2: class 0 (flood) of human-masked sources only (+ ~15% negatives). test_flood = the 160 old images."""
import os, random, shutil, glob, json, collections
W = "/home/tuananh/Documents/flood_segmentation/work"
SRC, DST = f"{W}/dataset", f"{W}/v2/water_ds"
POS = ("deepflood", "sazara", "pixabay", "fredf", "ade")
NEG = ("fredd", "coco")
random.seed(0)

def water_lines(p):
    return [l for l in open(p).read().splitlines() if l.split()[:1] == ["0"]]

def put(split, name, lines):
    stem = os.path.splitext(name)[0]
    for sub in ("images", "labels"):
        os.makedirs(f"{DST}/{sub}/{split}", exist_ok=True)
    shutil.copy(f"{SRC}/images/{split if split != 'test' else 'test_flood'}/{name}", f"{DST}/images/{split}/{name}")
    open(f"{DST}/labels/{split}/{stem}.txt", "w").write("\n".join(lines) + ("\n" if lines else ""))

stats = collections.Counter()
for split in ("train", "val"):
    pos, neg = [], []
    for img in sorted(os.listdir(f"{SRC}/images/{split}")):
        pre = img.split("_")[0]
        lab = f"{SRC}/labels/{split}/{os.path.splitext(img)[0]}.txt"
        if pre in POS:
            ls = water_lines(lab)
            if ls: pos.append((img, ls))
        elif pre in NEG:
            neg.append(img)
    k = round(len(pos) * 0.15 / 0.85)
    random.shuffle(neg)
    for img, ls in pos: put(split, img, ls); stats[split, "pos"] += 1
    for img in neg[:k]: put(split, img, []); stats[split, "neg"] += 1
for img in sorted(os.listdir(f"{SRC}/images/test_flood")):
    ls = water_lines(f"{SRC}/labels/test_flood/{os.path.splitext(img)[0]}.txt")
    put("test", img, ls); stats["test", "pos" if ls else "empty"] += 1
open(f"{DST}/data.yaml", "w").write("path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: water\n")
print(dict(stats))
