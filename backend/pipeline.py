"""Cơ chế v2: model nước tự train + vật tham chiếu COCO (người, xe) + mặt đất SegFormer.

FloodPipeline nạp model một lần. Phần luật (độ sâu người, luật cảnh báo, tracker) là hàm thuần,
không cần model nên test được độc lập (tests/test_pipeline.py)."""
import os
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np

WEIGHTS = Path(__file__).parent / "weights"


def _flag(name):
    return os.getenv(name, "1").strip().lower() not in ("0", "false", "no", "off")


USE_POSE = _flag("USE_POSE")
USE_DETECT = _flag("USE_DETECT")
USE_DEPTH_CLS = _flag("USE_DEPTH_CLS")
USE_GROUND = _flag("USE_GROUND")

_default_water = WEIGHTS / "water.pt" if (WEIGHTS / "water.pt").exists() else WEIGHTS / "best.pt"  # best.pt: model cũ, dự phòng
WATER_MODEL_PATH = os.getenv("WATER_MODEL_PATH", str(_default_water))
POSE_MODEL_PATH = os.getenv("POSE_MODEL_PATH", str(WEIGHTS / "yolo11n-pose.pt"))
DETECT_MODEL_PATH = os.getenv("DETECT_MODEL_PATH", str(WEIGHTS / "yolo11n.pt"))
DEPTH_MODEL_PATH = os.getenv("DEPTH_MODEL_PATH", str(WEIGHTS / "depth_cls.pt"))
GROUND_MODEL = os.getenv("GROUND_MODEL", "nvidia/segformer-b2-finetuned-ade-512-512")

IMG_SIZE = int(os.getenv("IMG_SIZE", "640"))
WATER_CONF = float(os.getenv("WATER_CONF", "0.4"))
OBJ_CONF = float(os.getenv("OBJ_CONF", "0.35"))
KP_CONF = float(os.getenv("KP_CONF", "0.5"))                   # ngưỡng tin cậy của keypoint (mắt cá, gối, hông)
KP_HIGH = float(os.getenv("KP_HIGH", "0.7"))                   # mắt cá conf >= KP_HIGH: chân còn nhìn rõ, nước chưa quá mắt cá
DEPTH_UP_P = float(os.getenv("DEPTH_UP_P", "0.35"))            # depth_cls: xác suất bậc cao hơn >= ngưỡng này thì chọn bậc cao (thiên về an toàn)
WATER_MIN_FRAME_FRAC = float(os.getenv("WATER_MIN_FRAME_FRAC", "0.005"))  # nước < 0.5% khung hình: coi như không có
WATER_MIN_RATIO = float(os.getenv("WATER_MIN_RATIO", "0.05"))  # độ phủ nước/mặt đất tối thiểu để tính là "có nước"
COVER_WIDE = float(os.getenv("COVER_WIDE", "0.30"))        # độ phủ >= 30%: chỉ đổi evidence ("nước phủ rộng"), mức vẫn tối đa 1
GROUND_MIN_FRAC = float(os.getenv("GROUND_MIN_FRAC", "0.05"))  # mặt đất < 5% khung hình -> logic dự phòng
FLOOD_COVER_MIN_FRAC = float(os.getenv("FLOOD_COVER_MIN_FRAC", "0.10"))  # dự phòng: nước >= 10% khung hình -> mặt đường bị phủ kín
PERSON_MIN_H = int(os.getenv("PERSON_MIN_H", "60"))
VEHICLE_MIN_H = int(os.getenv("VEHICLE_MIN_H", "40"))
CONTACT_MIN_FRAC = float(os.getenv("CONTACT_MIN_FRAC", "0.03"))  # nước chiếm >= 3% phần dưới box xe mới đưa vào depth_cls
STABLE_STEPS = int(os.getenv("STABLE_STEPS", "2"))             # giữ bậc >= 2 lần suy luận liên tiếp
SHALLOW_F = float(os.getenv("SHALLOW_F", "0.25"))              # mép nước cao tới 25% ống chân: bậc 1
KNEE_F = float(os.getenv("KNEE_F", "0.9"))                     # tới 90% ống chân trở lên (gần gối): bậc 3

COCO_VEHICLES = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
TWO_WHEELERS = ("bicycle", "motorcycle")
VEHICLE_VN = {"bicycle": "Xe đạp", "motorcycle": "Xe máy", "car": "Ô tô", "bus": "Xe buýt", "truck": "Xe tải"}
VEHICLE_TAG = {"bicycle": "xe dap", "motorcycle": "xe may", "car": "oto", "bus": "bus", "truck": "xe tai"}  # nhãn OpenCV: không dấu
VEHICLE_DEPTH_TEXT = {1: "dưới nửa bánh", 2: "quá nửa bánh", 3: "hết bánh"}
PERSON_DEPTH_TEXT = {1: "tới mắt cá", 2: "tới giữa ống chân", 3: "tới gối"}
GROUND_LABELS = ("road", "sidewalk", "path", "earth", "floor")
NO_REF_EVIDENCE = "Không có vật tham chiếu, chỉ dựa vào độ phủ"
WIDE_WATER_EVIDENCE = "Nước phủ rộng, không có vật tham chiếu để ước lượng độ sâu"
NO_WATER_EVIDENCE = "Không có nước đáng kể"


def log(msg):
    print(f"[pipeline] {msg}", file=sys.stderr)


# ---------------- Luật hình học (hàm thuần) ----------------

def box_iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def center_inside(inner, outer):
    cx, cy = (inner[0] + inner[2]) / 2, (inner[1] + inner[3]) / 2
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def water_edge_y(water, box, x_range=None, min_y=None):
    """Mép nước = hàng trên cùng của vùng nước nối liền đáy box, trong cột x_range (mặc định cả box).
    Trả về None nếu không có nước chạm đáy box."""
    h, w = water.shape
    x1, y1, x2, y2 = (int(v) for v in box)
    if x_range is not None:
        x1, x2 = int(x_range[0]), int(x_range[1])
    x1, y1, x2, y2 = max(0, x1), max(0, y1 if min_y is None else int(min_y)), min(w, x2), min(h, y2)
    if x2 - x1 < 2 or y2 - y1 < 4:
        return None
    band = water[y1:y2, x1:x2]
    if not band.any():
        return None
    _, lab = cv2.connectedComponents(band.astype(np.uint8), connectivity=8)
    rows = max(3, band.shape[0] // 20)
    ids = np.unique(lab[-rows:])
    ids = ids[ids > 0]
    if not len(ids):
        return None
    reach = np.isin(lab, ids)
    return y1 + int(np.where(reach.any(axis=1))[0].min())


def person_depth(kxy, kconf, box, water, kp_conf=KP_CONF):
    """Bậc độ sâu 0..3 của một người, hoặc None nếu không đủ keypoint để neo (cần gối hoặc hông tin cậy).
    kxy (17,2), kconf (17,). COCO: hông 11,12; gối 13,14; mắt cá 15,16."""
    def pick(ids):
        ys = [kxy[i][1] for i in ids if kconf[i] >= kp_conf]
        return float(np.mean(ys)) if ys else None
    hip, knee, ankle = pick((11, 12)), pick((13, 14)), pick((15, 16))
    if knee is None and hip is None:
        return None
    if ankle is None:
        ankle = float(box[3])  # mắt cá bị nước/khung che: lấy đáy box
    if knee is None:
        knee = (hip + ankle) / 2
    shin = ankle - knee
    if shin < 4:
        return None
    ankle_c = max((kconf[i] for i in (15, 16)), default=0.0)
    knee_c = max((kconf[i] for i in (13, 14)), default=0.0)
    hip_c = max((kconf[i] for i in (11, 12)), default=0.0)
    # chỉ xét cột của chân (hông-gối-mắt cá) để nước phía sau người không bị tính là mép nước
    xs = [kxy[i][0] for i in range(11, 17) if kconf[i] >= kp_conf]
    bw = box[2] - box[0]
    xr = (max(box[0], min(xs) - 0.15 * bw), min(box[2], max(xs) + 0.15 * bw)) if xs else (box[0] + 0.25 * bw, box[2] - 0.25 * bw)
    edge = water_edge_y(water, box, xr, min_y=(hip if hip is not None else knee) - 0.05 * shin)
    if edge is None:
        lv = 0
    else:
        f = (ankle - edge) / shin
        lv = 1 if f < SHALLOW_F else 2 if f < KNEE_F else 3
    return combine_person_signals(lv, ankle_c, knee_c, hip_c, kp_conf)


def combine_person_signals(edge_lv, ankle_c, knee_c, hip_c, kp_conf=KP_CONF):
    """Kết hợp bậc theo mép nước với tín hiệu conf keypoint; mâu thuẫn thì lấy bậc thấp hơn, trừ khi cả hai cùng chỉ bậc 3.
    mắt cá conf cao -> <= 1 | mắt cá thấp + gối cao -> 1 (mép nước <= 1) hoặc 2 | mắt cá + gối thấp + hông cao -> 3."""
    if ankle_c >= KP_HIGH:
        kp = 1
    elif ankle_c < kp_conf and knee_c >= kp_conf:
        kp = 1 if edge_lv <= 1 else 2
    elif ankle_c < kp_conf and knee_c < kp_conf and hip_c >= kp_conf:
        kp = 3
    else:
        return edge_lv
    return edge_lv if edge_lv == kp else min(edge_lv, kp)


def pick_depth(probs, up_p=DEPTH_UP_P):
    """probs: xác suất theo bậc 0..3. Lấy bậc top-1, nhưng nếu có bậc cao hơn với xác suất >= up_p thì chọn bậc cao nhất trong số đó."""
    top = int(np.argmax(probs))
    up = [j for j in range(top + 1, len(probs)) if probs[j] >= up_p]
    return max(up) if up else top


def object_text(obj):
    lv = obj["level"]
    if obj["kind"] == "person":
        return f"Người lội nước {PERSON_DEPTH_TEXT[lv]}"
    return f"{VEHICLE_VN[obj['cls']]} ngập {VEHICLE_DEPTH_TEXT[lv]}"


def is_valid(obj):
    return obj["box"][3] - obj["box"][1] >= (PERSON_MIN_H if obj["kind"] == "person" else VEHICLE_MIN_H)


def frame_decision(water, ground, objects, h, w):
    """Luật cảnh báo cho một khung hình -> (bậc, evidence, thống kê)."""
    water_px = int(water.sum())
    frac = water_px / (h * w)
    ground_px = int((ground | water).sum()) if ground is not None else 0
    if ground is not None and ground_px >= GROUND_MIN_FRAC * h * w:
        ratio, base = water_px / ground_px, "ground"
    elif frac >= FLOOD_COVER_MIN_FRAC:
        ratio, base = 1.0, "covered"   # logic dự phòng: không thấy mặt đất mà nước nhiều -> đã phủ kín
    else:
        ratio, base = frac, "frame"
    valid = [o for o in objects if is_valid(o)] if frac >= WATER_MIN_FRAME_FRAC else []
    top = max(valid, key=lambda o: o["level"], default=None)
    obj_lv = top["level"] if top else 0
    has_water = frac >= WATER_MIN_FRAME_FRAC and ratio >= WATER_MIN_RATIO
    level = 0
    if has_water or obj_lv >= 1:
        level = 1
    if obj_lv >= 2:   # độ phủ một mình tối đa mức 1: mức 2, 3 chỉ đến từ vật tham chiếu
        level = 2
    if obj_lv >= 3:
        level = 3
    if level == 0:
        evidence = NO_WATER_EVIDENCE
    elif top and obj_lv == level:
        evidence = object_text(top)
    elif valid:
        evidence = f"Vật tham chiếu chỉ ngập nông, độ phủ nước {ratio:.0%}"
    elif ratio >= COVER_WIDE:
        evidence = WIDE_WATER_EVIDENCE
    else:
        evidence = NO_REF_EVIDENCE
    vehicles = [o["level"] for o in valid if o["kind"] == "vehicle"]
    return level, evidence, {
        "flood_ratio": round(ratio, 3), "ratio_base": base, "objects": len(valid),
        "max_object_level": obj_lv if valid else None, "max_car_level": max(vehicles, default=None),
        "level": level, "evidence": evidence,
    }


# ---------------- Làm mượt theo thời gian ----------------

class WaterSmoother:
    """Trung bình trượt mask nước + ngưỡng trễ (bật khi > on, chỉ tắt khi < off) để vùng không nhấp nháy."""

    def __init__(self, smooth=0.6, on=0.5, off=0.25):
        self.smooth, self.on, self.off = smooth, on, off
        self.cur = self.state = None

    def update(self, mask):
        cur = mask.astype(np.float32)
        if self.smooth > 0 and self.cur is not None:
            cur = (1 - self.smooth) * cur + self.smooth * self.cur
        self.cur = cur
        out = cur > self.on
        if self.smooth > 0 and self.state is not None:
            out |= self.state & (cur > self.off)
        self.state = out
        return out


class ObjectTracker:
    """Nối vật (người, xe) giữa các lần suy luận theo IoU box cùng loại. Bậc của vật = min của STABLE_STEPS
    lần suy luận gần nhất: bậc >= 2 phải giữ ít nhất 2 lần liên tiếp mới tính; bậc 1 tính ngay. Vật bị sót 1 lần vẫn giữ."""

    def __init__(self, min_iou=0.3, hold=1, steps=None):
        self.tracks, self.min_iou, self.hold = [], min_iou, hold
        self.steps = steps or STABLE_STEPS

    def update(self, objects):
        pairs = sorted(((box_iou(t["obj"]["box"], o["box"]), ti, oi)
                        for ti, t in enumerate(self.tracks) for oi, o in enumerate(objects)
                        if t["obj"]["kind"] == o["kind"]), key=lambda p: -p[0])
        t2o, o2t = {}, {}
        for iou, ti, oi in pairs:
            if iou < self.min_iou:
                break
            if ti not in t2o and oi not in o2t:
                t2o[ti], o2t[oi] = oi, ti
        tracks = []
        for ti, t in enumerate(self.tracks):
            if ti in t2o:
                t["obj"], t["missed"] = objects[t2o[ti]], 0
                t["levels"].append(t["obj"]["level"])
            else:
                t["missed"] += 1
            if t["missed"] <= self.hold:
                tracks.append(t)
        for oi, o in enumerate(objects):
            if oi not in o2t:
                tracks.append({"obj": o, "missed": 0, "levels": deque([o["level"]], maxlen=self.steps)})
        self.tracks = tracks
        out = []
        for t in tracks:
            lv = t["levels"]
            stable = min(lv) if len(lv) >= self.steps else min(lv[-1], 1)
            out.append({**t["obj"], "level": stable})
        return out


class GroundCache:
    """Mặt đất chạy ~1 lần/giây hoặc khi chuyển cảnh; giữa các lần dùng lại mask cũ."""

    def __init__(self, pipe, every):
        self.pipe, self.every, self.mask, self.n = pipe, max(1, every), None, 0

    def update(self, frame, cut=False):
        if self.pipe.ground is None:
            return None
        if cut:
            self.mask, self.n = None, 0
        if self.mask is None or self.n % self.every == 0:
            self.mask = self.pipe.ground_mask(frame)
        self.n += 1
        return self.mask


# ---------------- Nạp model và suy luận ----------------

class FloodPipeline:
    """Nạp model một lần. Thiếu weights của phần nào thì tắt phần đó (trừ model nước)."""

    def __init__(self):
        from ultralytics import YOLO
        import torch
        self.device = 0 if torch.cuda.is_available() else "cpu"
        if not Path(WATER_MODEL_PATH).exists():
            raise FileNotFoundError(f"Không thấy model nước: {WATER_MODEL_PATH} (chạy python download_weights.py hoặc set WATER_MODEL_PATH)")
        self.water = YOLO(WATER_MODEL_PATH)
        self.water_ids = {int(i) for i, n in self.water.names.items() if str(n).lower() in ("water", "flood")}
        if not self.water_ids:
            raise ValueError(f"Model nước không có class 'water' hoặc 'flood': {self.water.names}")
        self.pose = self._load(YOLO, USE_POSE, POSE_MODEL_PATH, "pose")
        self.detect = self._load(YOLO, USE_DETECT, DETECT_MODEL_PATH, "detect")
        self.depth = self._load(YOLO, USE_DEPTH_CLS, DEPTH_MODEL_PATH, "depth_cls")
        self.ground = self._load_ground() if USE_GROUND else None

    @staticmethod
    def _load(YOLO, on, path, name):
        if not on:
            return None
        if not Path(path).exists():
            log(f"thiếu weights {name}: {path} -> tắt")
            return None
        return YOLO(path)

    def _load_ground(self):
        try:
            import torch
            from transformers import SegformerForSemanticSegmentation
            model = SegformerForSemanticSegmentation.from_pretrained(GROUND_MODEL).eval()
            dev = "cuda" if self.device != "cpu" else "cpu"
            model = (model.half() if dev == "cuda" else model).to(dev)
            names = {int(i): str(n).split(",")[0].strip().lower() for i, n in model.config.id2label.items()}
            ids = [i for i, n in names.items() if n in GROUND_LABELS]
            self._ground_dev, self._ground_ids = dev, torch.tensor(ids, device=dev)
            return model
        except Exception as e:  # không tải được SegFormer: chạy tiếp bằng logic dự phòng
            log(f"không nạp được mặt đất ({type(e).__name__}) -> tắt")
            return None

    def status(self):
        return {"water": WATER_MODEL_PATH, "pose": self.pose is not None, "detect": self.detect is not None,
                "depth_cls": self.depth is not None, "ground": self.ground is not None}

    # --- từng phần ---
    def water_mask(self, frame):
        h, w = frame.shape[:2]
        r = self.water.predict(frame, conf=WATER_CONF, imgsz=IMG_SIZE, retina_masks=True, verbose=False, device=self.device)[0]
        out = np.zeros((h, w), bool)
        if r.masks is None or not len(r.boxes):
            return out
        for m, c in zip(r.masks.data.cpu().numpy() > 0.5, r.boxes.cls.cpu().numpy().astype(int)):
            if c in self.water_ids:
                if m.shape != (h, w):
                    m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                out |= m
        return out

    def ground_mask(self, frame):
        import torch
        h, w = frame.shape[:2]
        sh = max(32, round(h * 512 / w / 32) * 32)
        x = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (512, sh), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        t = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(self._ground_dev)
        with torch.no_grad():
            logits = self.ground(pixel_values=t.half() if self._ground_dev == "cuda" else t.float()).logits
            lab = torch.nn.functional.interpolate(logits.float(), size=(h, w), mode="bilinear", align_corners=False).argmax(1)[0]
            return torch.isin(lab, self._ground_ids).cpu().numpy()

    def persons(self, frame, water):
        if self.pose is None:
            return []
        r = self.pose.predict(frame, conf=OBJ_CONF, imgsz=IMG_SIZE, verbose=False, device=self.device)[0]
        if r.keypoints is None or not len(r.boxes):
            return []
        out = []
        for box, conf, kxy, kc in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                                      r.keypoints.xy.cpu().numpy(), r.keypoints.conf.cpu().numpy()):
            if box[3] - box[1] < PERSON_MIN_H:
                continue
            lv = person_depth(kxy, kc, box, water)
            if lv is not None:
                out.append({"kind": "person", "cls": "person", "box": box.astype(int), "conf": float(conf), "level": lv})
        return out

    def vehicles(self, frame, water):
        if self.detect is None:
            return [], []
        r = self.detect.predict(frame, conf=OBJ_CONF, imgsz=IMG_SIZE, classes=list(COCO_VEHICLES), verbose=False, device=self.device)[0]
        cand = []
        for box, conf, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)):
            cand.append({"kind": "vehicle", "cls": COCO_VEHICLES[c], "box": box.astype(int), "conf": float(conf), "level": 0})
        two = [v["box"] for v in cand if v["cls"] in TWO_WHEELERS]
        h, w = water.shape
        todo = []
        for v in cand:
            x1, y1, x2, y2 = v["box"]
            if y2 - y1 < VEHICLE_MIN_H:
                continue
            low = water[max(0, y1 + int(0.75 * (y2 - y1))):min(h, y2 + 1), max(0, x1):min(w, x2)]
            if low.size and low.mean() >= CONTACT_MIN_FRAC:  # đáy xe không chạm nước: bậc 0, khỏi chạy depth_cls
                v["level"] = 1
                todo.append(v)
        if todo and self.depth is not None:
            crops = []
            for v in todo:
                x1, y1, x2, y2 = v["box"]
                px, py = 0.2 * (x2 - x1) / 2, 0.2 * (y2 - y1) / 2   # nới box 20%
                crops.append(frame[max(0, int(y1 - py)):min(h, int(y2 + py)), max(0, int(x1 - px)):min(w, int(x2 + px))])
            for v, res in zip(todo, self.depth.predict(crops, verbose=False, device=self.device)):
                p = res.probs.data.cpu().numpy()
                order = sorted(range(len(p)), key=lambda i: int(str(res.names[i])[-1]))  # xác suất theo thứ tự bậc 0..3
                v["level"] = pick_depth(p[order])
        return cand, two

    def infer(self, frame):
        """Khung hình -> (mask nước thô, danh sách vật tham chiếu). Không có nước: bỏ qua pose/detect/cls."""
        water = self.water_mask(frame)
        if water.mean() < WATER_MIN_FRAME_FRAC:
            return water, []
        cars, two = self.vehicles(frame, water)
        people = [p for p in self.persons(frame, water) if not any(center_inside(p["box"], t) for t in two)]
        return water, people + [c for c in cars if c["level"] > 0 or is_valid(c)]
