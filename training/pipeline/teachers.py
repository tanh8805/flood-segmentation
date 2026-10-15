"""Teacher models used offline to pseudo-label the dataset (never shipped in the app)."""
import os
import warnings
warnings.filterwarnings("ignore")
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.environ.get("FLOOD_WORK", os.path.abspath("work"))
FLOODNET_PTH = os.environ.get("FLOODNET_PTH", f"{ROOT}/models/link_efficientb4_model.pth")
CAR_LEVEL_PT = os.environ.get("CAR_LEVEL_PT", f"{ROOT}/models/best_car_YOLOv11.pt")
CAR_SEG_PT = f"{ROOT}/models/yolo11x-seg.pt"
SEGFORMER = "nvidia/segformer-b5-finetuned-ade-640-640"
DEV = "cuda"

ROAD_NAMES = {"road"}
WATER_NAMES = {"water", "sea", "river", "lake"}


# ---------------- V-FloodNet (water) ----------------
class FloodNet:
    """LinkNet-EfficientNet-b4. The .pth is a full-model pickle; we additionally verify that its
    weights load with strict=True into smp.Linknet built exactly like repo train_image_seg.py."""

    def __init__(self):
        import segmentation_models_pytorch as smp
        pickled = torch.load(FLOODNET_PTH, map_location="cpu", weights_only=False)
        ref = smp.Linknet(encoder_name="efficientnet-b4", encoder_depth=5, encoder_weights=None,
                          in_channels=3, classes=1, activation="sigmoid")
        ref.load_state_dict(pickled.state_dict(), strict=True)  # raises on any mismatch
        self.model = ref.to(DEV).eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)

    @torch.no_grad()
    def prob(self, rgb):
        """Returns float32 water probability at original resolution (repo: resize to 416x416)."""
        h, w = rgb.shape[:2]
        x = torch.from_numpy(rgb).to(DEV).permute(2, 0, 1)[None].float() / 255.0
        x = F.interpolate(x, size=(416, 416), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        p = self.model(x)  # sigmoid already applied
        p = F.interpolate(p, size=(h, w), mode="bilinear", align_corners=False)
        return p[0, 0].float().cpu().numpy()

    def mask(self, rgb, postprocess=False):
        m = (self.prob(rgb) > 0.5).astype(np.uint8)
        if postprocess:  # repo's myutils.postprocessing_pred: keep the single largest component
            n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
            if n > 2:
                k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                m = (lab == k).astype(np.uint8)
        return m


# ---------------- SegFormer ADE20K ----------------
class SegFormer:
    def __init__(self):
        from transformers import AutoModelForSemanticSegmentation
        self.model = AutoModelForSemanticSegmentation.from_pretrained(SEGFORMER).to(DEV).half().eval()
        id2 = self.model.config.id2label
        norm = lambda s: s.split(",")[0].strip().lower()
        self.road_ids = [int(i) for i, n in id2.items() if norm(n) in ROAD_NAMES]
        self.water_ids = [int(i) for i, n in id2.items() if norm(n) in WATER_NAMES]
        assert self.road_ids and len(self.water_ids) == len(WATER_NAMES), (self.road_ids, self.water_ids)
        self.id2label = {int(i): n for i, n in id2.items()}
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)

    @torch.no_grad()
    def labels(self, rgb):
        """Full ADE20K argmax label map at original resolution (uint8)."""
        h, w = rgb.shape[:2]
        x = torch.from_numpy(rgb).to(DEV).permute(2, 0, 1)[None].float() / 255.0
        x = F.interpolate(x, size=(640, 640), mode="bilinear", align_corners=False)
        x = ((x - self.mean) / self.std).half()
        lg = self.model(pixel_values=x).logits  # 1,150,160,160
        lg = F.interpolate(lg.float(), size=(h, w), mode="bilinear", align_corners=False)
        return lg.argmax(1)[0].to(torch.uint8).cpu().numpy()

    def road_water(self, rgb):
        lab = self.labels(rgb)
        return np.isin(lab, self.road_ids).astype(np.uint8), np.isin(lab, self.water_ids).astype(np.uint8), lab


# ---------------- cars ----------------
def box_iou(a, b):
    ix1, iy1 = np.maximum(a[0], b[0]), np.maximum(a[1], b[1])
    ix2, iy2 = np.minimum(a[2], b[2]), np.minimum(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class Cars:
    def __init__(self, conf_level=0.5, conf_seg=0.25, iou_match=0.5, imgsz_seg=960):
        from ultralytics import YOLO
        self.level = YOLO(CAR_LEVEL_PT)
        self.seg = YOLO(CAR_SEG_PT)
        self.conf_level, self.conf_seg, self.iou_match, self.imgsz_seg = conf_level, conf_seg, iou_match, imgsz_seg
        self.car_id = [k for k, v in self.seg.names.items() if v == "car"][0]

    def run(self, bgr):
        """Returns list of dicts {level, conf, box, mask(bool HxW)} for level boxes matched to a
        yolo11x-seg car mask (IoU>=0.5), plus count of unmatched level boxes."""
        h, w = bgr.shape[:2]
        r = self.level.predict(bgr, conf=self.conf_level, half=True, verbose=False, device=0)[0]
        boxes = r.boxes.xyxy.cpu().numpy(); cls = r.boxes.cls.cpu().numpy().astype(int); cf = r.boxes.conf.cpu().numpy()
        if len(boxes) == 0:
            return [], 0
        s = self.seg.predict(bgr, conf=self.conf_seg, classes=[self.car_id], imgsz=self.imgsz_seg, half=True,
                             retina_masks=True, verbose=False, device=0)[0]
        if s.masks is None:
            return [], len(boxes)
        sb = s.boxes.xyxy.cpu().numpy()
        sm = s.masks.data.cpu().numpy().astype(bool)
        used, out, unmatched = set(), [], 0
        for i in np.argsort(-cf):
            ious = [box_iou(boxes[i], b) if j not in used else -1 for j, b in enumerate(sb)]
            j = int(np.argmax(ious))
            if ious[j] >= self.iou_match:
                used.add(j)
                m = sm[j]
                if m.shape != (h, w):
                    m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                out.append(dict(level=int(cls[i]), conf=float(cf[i]), box=boxes[i].tolist(), mask=m))
            else:
                unmatched += 1
        return out, unmatched
