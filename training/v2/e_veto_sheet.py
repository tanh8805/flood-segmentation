import sys, os, cv2, numpy as np
sys.path.insert(0, "/home/tuananh/Documents/flood_segmentation/flood-segmentation/backend")
import pipeline as P
pipe = P.FloodPipeline(); cap = cv2.VideoCapture(os.path.expanduser("~/Videos/flood_test/tuoitre_dem.mp4")); fps = cap.get(5)
tiles = []
for t in (70, 78, 83, 94):
    cap.set(1, int(t * fps)); ok, f = cap.read(); f = cv2.resize(f, (640, 360))
    w = pipe.water_mask(f); _, veto = pipe.scene_masks(f)
    row = []
    for m, tag in ((w, "truoc"), (w & ~veto, "sau")):
        o = f.copy(); o[m] = (0.5 * o[m] + 0.5 * np.array([255, 120, 0])).astype(np.uint8); cv2.putText(o, f"{t}s {tag} {m.mean():.2f}", (6, 22), 0, 0.7, (0, 255, 255), 2); row.append(o)
    print(t, "nước trên vùng cấm:", round(float((w & veto).sum() / max(1, w.sum())), 3))
    tiles.append(np.hstack(row))
cv2.imwrite("/home/tuananh/Documents/flood_segmentation/work/tmp/veto_sheet.jpg", cv2.resize(np.vstack(tiles), (1280, 1440 // 2 * 1)), [cv2.IMWRITE_JPEG_QUALITY, 85])
