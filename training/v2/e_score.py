import sys, json, csv, numpy as np
lab = {}
for r in csv.DictReader(open("/home/tuananh/Documents/flood_segmentation/work/v2/vn_test_labels.csv")): lab[(r["clip"], int(r["t_s"]))] = int(r["level"])
for path in sys.argv[1:]:
    pred = json.load(open(path)); C = np.zeros((4, 4), int)
    for (c, t), g in lab.items():
        seg = [s for s in pred.get(c, []) if abs(s["t"] - t) < 0.6]
        if seg: C[g, seg[0]["level"]] += 1
    print(path.split("/")[-1], "n", C.sum(), "exact", round(np.trace(C) / C.sum(), 3), "within1", round(sum(C[i, j] for i in range(4) for j in range(4) if abs(i - j) <= 1) / C.sum(), 3), "confusion(rows=label)", C.tolist())
