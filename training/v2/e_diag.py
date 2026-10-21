import sys, cv2, numpy as np
sys.path.insert(0, "/home/tuananh/Documents/flood_segmentation/flood-segmentation/backend")
import pipeline as P
from ultralytics import YOLO
path = [l.split(",")[1] for l in open("/home/tuananh/Documents/flood_segmentation/work/v2/vn_test_labels.csv") if l.startswith("clip2,")][0]
det = YOLO(P.DETECT_MODEL_PATH); wat = YOLO(P.WATER_MODEL_PATH)
cap = cv2.VideoCapture(path); fps = cap.get(5) or 25; tiles = []
for t in range(2, 22, 2):
    cap.set(1, int(t * fps)); ok, f = cap.read()
    if not ok: continue
    h, w = f.shape[:2]
    r = det.predict(f, conf=0.05, imgsz=640, classes=list(P.COCO_VEHICLES) + [0], verbose=False, device=0)[0]
    img = f.copy(); info = []
    for b, c, k in zip(r.boxes.xyxy.cpu().numpy().astype(int), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)):
        col = (0, 255, 0) if c >= P.OBJ_CONF else (0, 0, 255)
        cv2.rectangle(img, tuple(b[:2]), tuple(b[2:]), col, max(2, w // 300)); info.append(f"{det.names[k]}{c:.2f} h{b[3]-b[1]}")
        cv2.putText(img, f"{det.names[k]} {c:.2f}", (b[0], max(15, b[1] - 4)), 0, w / 700, col, 2)
    m = P.WATER_CONF
    rw = wat.predict(f, conf=m, imgsz=640, verbose=False, device=0)[0]
    wf = 0.0
    if rw.masks is not None: wf = float(np.mean(np.any(cv2.resize((rw.masks.data.cpu().numpy() > .5).any(0).astype(np.uint8), (w, h)) > 0, axis=None))) 
    print(t, f"mean_lum={cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean():.0f}", "|", "; ".join(info) or "-")
    tiles.append(cv2.resize(img, (426, 240)))
while len(tiles) % 3: tiles.append(np.zeros_like(tiles[0]))
rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
cv2.imwrite("/home/tuananh/Documents/flood_segmentation/work/tmp/clip2_diag.jpg", np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
