import sys, os, cv2, numpy as np
for kv in sys.argv[1:]: k, v = kv.split("=", 1); os.environ[k] = v
sys.path.insert(0, "/home/tuananh/Documents/flood_segmentation/flood-segmentation/backend")
import pipeline as P
pipe = P.FloodPipeline()
path = [l.split(",")[1] for l in open("/home/tuananh/Documents/flood_segmentation/work/v2/vn_test_labels.csv") if l.startswith("clip2,")][0]
cap = cv2.VideoCapture(path); fps = cap.get(5) or 25
for t in np.arange(2, 21, 1.0):
    cap.set(1, int(t * fps)); ok, f = cap.read()
    w, objs = pipe.infer(f)
    print(t, f"water={w.mean():.3f}", [(o["cls"], round(o["conf"], 2), int(o["box"][3] - o["box"][1]), o["level"]) for o in objs])
