import sys, json, csv, numpy as np
lab = {(r["clip"], int(r["t_s"])): int(r["level"]) for r in csv.DictReader(open("/home/tuananh/Documents/flood_segmentation/work/v2/vn_test_labels.csv"))}
def load(path):
    pred = json.load(open(path)); out = {}
    for (c, t), g in lab.items():
        seg = [s for s in pred.get(c, []) if abs(s["t"] - t) < 0.6]
        if seg: out[(c, t)] = (g, seg[0]["level"], seg[0].get("evidence"))
    return out
names = ["v1 (cũ)", "v2 cũ", "v2 sửa"]; paths = sys.argv[1:4]
print("| | n | đúng mức | lệch ≤1 | báo thiếu | báo dư |\n|---|---|---|---|---|---|")
for n, p in zip(names, paths):
    d = load(p); g = np.array([v[0] for v in d.values()]); q = np.array([v[1] for v in d.values()])
    print(f"| {n} | {len(g)} | {(g==q).mean():.3f} | {(abs(g-q)<=1).mean():.3f} | {(q<g).mean():.3f} | {(q>g).mean():.3f} |")
d = load(paths[2]); C = np.zeros((4, 4), int)
for g, q, _ in d.values(): C[g, q] += 1
print("ma trận v2 sửa (hàng=nhãn, cột=dự đoán):", C.tolist())
for (c, t), (g, q, e) in d.items():
    if g != q: print(f"{c} t={t}s nhãn {g} → {q}: {e}")
