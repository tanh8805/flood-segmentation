import sys, os, time, csv, json
dev = sys.argv[1]
if dev == "cpu": os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, "/home/tuananh/Documents/flood_segmentation/flood-segmentation/backend")
import cv2, torch, numpy as np
import pipeline as P
pipe = P.FloodPipeline()
path = next(csv.DictReader(open("/home/tuananh/Documents/flood_segmentation/work/v2/vn_test_labels.csv")))["path"]
cap = cv2.VideoCapture(path); frames = []
cap.set(cv2.CAP_PROP_POS_FRAMES, 200)
for _ in range(30):
    ok, f = cap.read()
    if ok: frames.append(f)
for f in frames[:3]: pipe.infer(f)
if dev == "cuda": torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
n = 5 if dev == "cpu" else 20; ts = []
for f in frames[3:3 + n]:
    t = time.perf_counter(); pipe.infer(f)
    if dev == "cuda": torch.cuda.synchronize()
    ts.append((time.perf_counter() - t) * 1000)
print(json.dumps(dict(dev=dev, ms_mean=round(float(np.mean(ts)), 1), ms_median=round(float(np.median(ts)), 1), peak_vram_mb=round(torch.cuda.max_memory_allocated() / 2**20) if dev == "cuda" else None)))
