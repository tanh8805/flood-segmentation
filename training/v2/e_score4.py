import sys, json, csv, numpy as np
lab = {(r["clip"], int(r["t_s"])): int(r["level"]) for r in csv.DictReader(open("/home/tuananh/Documents/flood_segmentation/work/v2/vn_test_labels.csv"))}
CL = __import__("os").environ.get("CLIPS", "clip0,clip1,clip2,clip5,clip7").split(",")   # clip6: video đã mất
def load(path):
    pred = json.load(open(path)); d = {}
    for (c, t), g in lab.items():
        if c not in CL: continue
        s = sorted([x for x in pred.get(c, []) if abs(x["t"] - t) < 1.0], key=lambda x: abs(x["t"] - t))
        if s: d[(c, t)] = (g, s[0]["level"], s[0].get("evidence"))
    return d
hdr = "| | n | đúng | ≤1 | thiếu | dư | " + " | ".join(f"{c} thiếu/dư" for c in CL) + " |"
print(hdr); print("|---" * (6 + len(CL)) + "|")
last = None
for a in sys.argv[1:]:
    n, p = a.split("="); d = load(p); last = (n, d)
    g = np.array([v[0] for v in d.values()]); q = np.array([v[1] for v in d.values()])
    pc = []
    for c in CL:
        x = [v for k, v in d.items() if k[0] == c]
        pc.append(f"{sum(b < a_ for a_, b, _ in x)}/{sum(b > a_ for a_, b, _ in x)} của {len(x)}")
    print(f"| {n} | {len(g)} | {(g==q).mean():.3f} | {(abs(g-q)<=1).mean():.3f} | {(q<g).mean():.3f} | {(q>g).mean():.3f} | " + " | ".join(pc) + " |")
for n, d in [last]:
    C = np.zeros((4, 4), int)
    for g, q, _ in d.values(): C[g, q] += 1
    print(n, "ma trận:", C.tolist())
