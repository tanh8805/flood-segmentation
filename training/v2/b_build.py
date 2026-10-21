"""Build v2/depth_cls (ImageFolder d0..d3). Own crops labelled from contact sheets (labels below, -1 = skip); STURM mapping L0->0 L1->1 L2->2 L3,L4->3,
split by car id (test = car ids in STURM test.txt; no id appears in two splits). Own crops from VN videos go to test only (VN eval must stay unseen)."""
import csv, os, random, shutil, glob, collections, re
W = "/home/tuananh/Documents/flood_segmentation/work"; O = f"{W}/v2/depth_cls_raw"; D = f"{W}/v2/depth_cls"
random.seed(0)
S = """1 2 1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
"""
raw = {
 0: "2 1 -1 -1 1 1 1 1 1 1 2 2 1 1 2 2 -1 1 1 1",
 1: "1 2 1 2 1 2 2 1 2 2 0 2 1 1 1 1 1 1 1 1",
 2: "1 0 1 1 0 1 1 1 -1 -1 1 1 1 1 0 0 2 2 2 2",
 3: "1 0 1 -1 -1 -1 0 1 1 1 1 1 1 1 1 1 1 1 1 2",
 4: "2 2 2 2 2 2 2 2 2 1 2 2 2 2 2 2 2 -1 2 2",
 5: "1 2 1 2 2 1 1 1 2 1 2 1 2 1 2 1 2 2 -1 -1",
 6: "1 1 2 1 2 3 2 -1 1 2 1 2 1 1 1 2 1 2 2 2",
 7: "0 1 1 2 3 3 1 2 2 1 3 2 1 2 2 0 3 1 1 1",
 8: "0 1 2 3 3 2 2 -1 2 2 3 2 1 1 2 1 1 3 2 3",
 9: "1 2 1 2 2 1 2 2 2 3 2 3 1 2 2 2 2 2 1 1",
 10: "0 1 -1 2 2 2 1 1 2 1 2 0 1 1 1 1 1 2 2 1",
 11: "2 1 1 2 -1 1 0 2 1 1 2 1 2 1 3 2 0 1 2 3",
 12: "-1 2 1 2 -1 1 2 1 1 3 1 2 3 3 1 -1 1 2 3 2",
 13: "-1 -1 3 3 1 3 1 1 1 3 0 3 -1 1 2 3 1 1 2 1",
 14: "3 -1 2 -1 0 -1 -1 2 1 1 1 2 3 1 2 2 1 2 1 1",
 15: "1 2 2 1 -1 1 -1 1 -1 2 1 3 2 1 0 2 2 3 -1 2",
 16: "-1 1 -1 1 2 1 -1 1 1 3 3 -1 1 0 -1 -1 1 -1 2 2",
 17: "-1 2 1 1 2 2 1 2 1 1 -1 1 2 1 2 2 2 1 2 1",
 18: "-1 1 1 1 1 -1 -1 2 -1 1 2 1 1 -1 2 0 1 1 1 1",
 19: "-1 1 1 2 2 -1 1 2 1 -1 1 2 2 1 1 1 1 -1 1 1",
}
lab = {}
for s, line in raw.items():
    v = [int(x) for x in line.split()]; assert len(v) == 20, (s, len(v))
    for j, x in enumerate(v): lab[s * 20 + j] = x
sel = {int(r["id"]): r for r in csv.DictReader(open(f"{O}/selected.csv"))}
with open(f"{W}/v2/depth_cls_own_labels.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["id", "crop", "src", "key", "cls", "level"])
    for i, r in sel.items(): w.writerow([i, r["crop"], r["src"], r["key"], r["cls"], lab[i]])
if os.path.exists(D): shutil.rmtree(D)
def put(split, cls, src, name):
    os.makedirs(f"{D}/{split}/d{cls}", exist_ok=True); shutil.copy(src, f"{D}/{split}/d{cls}/{name}")
# own
keys = sorted({r["key"] for r in sel.values() if r["src"] != "video"}); random.shuffle(keys); test_keys = set(keys[:len(keys) // 5])
val_keys = set(keys[len(keys) // 5: len(keys) // 5 + len(keys) // 10])
for i, r in sel.items():
    if lab[i] < 0: continue
    sp = "test" if r["src"] == "video" or r["key"] in test_keys else "val" if r["key"] in val_keys else "train"
    put(sp, lab[i], f"{O}/crops/{r['crop']}", "own_" + r["crop"])
# STURM
R = f"{W}/external/ds_sturm/upscaled_images"; MAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3}
test_ids = {re.match(r"Level\d.car-(\d+)-", l.strip().replace("\\", "/")).group(1) if re.match(r"Level\d.car-(\d+)-", l.strip().replace("\\", "/")) else None for l in open(f"{W}/external/sturm/test.txt")}
by = collections.defaultdict(list)
for f in glob.glob(f"{R}/Level*/car-*.jpg"):
    m = re.match(r"car-(\d+)-Level(\d)_(\d+)", os.path.basename(f)); by[(m.group(1), int(m.group(2)))].append(f)
ids = sorted({k[0] for k in by}); rest = [i for i in ids if i not in test_ids]; random.shuffle(rest)
val_ids = set(rest[:len(rest) // 8]); cnt = collections.Counter()
for (cid, lv), fs in sorted(by.items()):
    f = sorted(fs)[0]; c = MAP[lv]
    sp = "test" if cid in test_ids else "val" if cid in val_ids else "train"
    cnt[sp, c] += 1
    if sp == "train" and cnt[sp, c] > 330: continue   # balance: cap train per class
    put(sp, c, f, os.path.basename(f))
print("test car ids", len(test_ids)); 
for sp in ("train", "val", "test"): print(sp, {c: len(os.listdir(f"{D}/{sp}/d{c}")) if os.path.exists(f"{D}/{sp}/d{c}") else 0 for c in range(4)})
