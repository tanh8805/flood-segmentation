"""Model nước tự train (water.pt) + mặt đất SegFormer (mẫu số độ phủ nước, veto nước) -> vùng nước và water_ratio của một khung hình.

FloodPipeline nạp model một lần; water_signal và WaterSmoother là hàm/lớp thuần, test được không cần model."""
import os
import sys
from pathlib import Path

import cv2
import numpy as np

WEIGHTS = Path(__file__).parent / "weights"


def _flag(name, default="1"):
    return os.getenv(name, default).strip().lower() not in ("0", "false", "no", "off")


USE_GROUND = _flag("USE_GROUND")
USE_WATER_VETO = _flag("USE_WATER_VETO")   # xoá nước trên sky/building/house/tree/wall/ceiling/signboard (SegFormer)

WATER_MODEL_PATH = os.getenv("WATER_MODEL_PATH", str(WEIGHTS / "water.pt"))
GROUND_MODEL = os.getenv("GROUND_MODEL", "nvidia/segformer-b2-finetuned-ade-512-512")

IMG_SIZE = int(os.getenv("IMG_SIZE", "640"))
WATER_CONF = float(os.getenv("WATER_CONF", "0.4"))
WATER_MIN_FRAME_FRAC = float(os.getenv("WATER_MIN_FRAME_FRAC", "0.005"))  # nước < 0.5% khung hình: coi như không có
WATER_MIN_RATIO = float(os.getenv("WATER_MIN_RATIO", "0.05"))  # chỉ để viết căn cứ: dưới ngưỡng này là "không thấy nước"
GROUND_MIN_FRAC = float(os.getenv("GROUND_MIN_FRAC", "0.05"))  # mặt đất < 5% khung hình -> logic dự phòng
FLOOD_COVER_MIN_FRAC = float(os.getenv("FLOOD_COVER_MIN_FRAC", "0.10"))  # dự phòng: nước >= 10% khung hình -> mặt đường bị phủ kín

# Luật nhị phân (chọn bằng mô phỏng trên cache tín hiệu, xem README)

GROUND_LABELS = ("road", "sidewalk", "path", "earth", "floor")
VETO_LABELS = ("sky", "building", "house", "tree", "wall", "ceiling", "signboard")   # nước không thể nằm trên các lớp này


def log(msg):
    print(f"[pipeline] {msg}", file=sys.stderr)


# ---------------- Luật (hàm thuần) ----------------

def water_signal(water, ground, h, w):
    """Tín hiệu một lần suy luận: độ phủ nước/mặt đường (water_ratio) + số liệu phụ. Làm tròn ngay để app và mô phỏng dùng cùng số."""
    water_px = int(water.sum())
    frac = water_px / (h * w)
    ground_px = int((ground | water).sum()) if ground is not None else 0
    if ground is not None and ground_px >= GROUND_MIN_FRAC * h * w:
        ratio, base = water_px / ground_px, "ground"
    elif frac >= FLOOD_COVER_MIN_FRAC:
        ratio, base = 1.0, "covered"   # logic dự phòng: không thấy mặt đất mà nước nhiều -> đã phủ kín
    else:
        ratio, base = frac, "frame"
    if frac < WATER_MIN_FRAME_FRAC:
        ratio = 0.0
    return {"water_ratio": round(ratio, 4), "ground_ratio": round(ground_px / (h * w), 4),
            "water_frac": round(frac, 4), "base": base}


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


class GroundCache:
    """Mặt đất chạy ~1 lần/giây hoặc khi chuyển cảnh; giữa các lần dùng lại mask cũ."""

    def __init__(self, pipe, every):
        self.pipe, self.every, self.mask, self.veto, self.n = pipe, max(1, every), None, None, 0

    def update(self, frame, cut=False):
        if self.pipe.ground is None:
            return None
        if cut:
            self.mask, self.veto, self.n = None, None, 0
        if self.mask is None or self.n % self.every == 0:
            self.mask, self.veto = self.pipe.scene_masks(frame)
        self.n += 1
        return self.mask


# ---------------- Nạp model và suy luận ----------------

class FloodPipeline:
    """Nạp model nước một lần; SegFormer (mặt đất, veto) tuỳ chọn: không tải được thì chạy bằng logic dự phòng."""

    def __init__(self):
        from ultralytics import YOLO
        import torch
        self.device = 0 if torch.cuda.is_available() else "cpu"
        if not Path(WATER_MODEL_PATH).exists():
            raise FileNotFoundError(f"Không thấy model nước: {WATER_MODEL_PATH} (đặt water.pt vào backend/weights/ hoặc set WATER_MODEL_PATH, xem README)")
        self.water = YOLO(WATER_MODEL_PATH)
        self.water_ids = {int(i) for i, n in self.water.names.items() if str(n).lower() in ("water", "flood")}
        if not self.water_ids:
            raise ValueError(f"Model nước không có class 'water' hoặc 'flood': {self.water.names}")
        self.ground = self._load_ground() if USE_GROUND else None

    def _load_ground(self):
        try:
            import torch
            from transformers import SegformerForSemanticSegmentation
            model = SegformerForSemanticSegmentation.from_pretrained(GROUND_MODEL).eval()
            dev = "cuda" if self.device != "cpu" else "cpu"
            model = (model.half() if dev == "cuda" else model).to(dev)
            names = {int(i): str(n).split(",")[0].strip().lower() for i, n in model.config.id2label.items()}
            ids = [i for i, n in names.items() if n in GROUND_LABELS]
            veto = [i for i, n in names.items() if n in VETO_LABELS]
            self._ground_dev = dev
            self._ground_ids, self._veto_ids = torch.tensor(ids, device=dev), torch.tensor(veto, device=dev)
            return model
        except Exception as e:  # không tải được SegFormer: chạy tiếp bằng logic dự phòng
            log(f"không nạp được mặt đất ({type(e).__name__}) -> tắt")
            return None

    def status(self):
        return {"water": WATER_MODEL_PATH, "ground": self.ground is not None}

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

    def scene_masks(self, frame):
        """SegFormer -> (mặt đất, vùng cấm nước: sky/building/house/tree/wall/ceiling/signboard)."""
        import torch
        h, w = frame.shape[:2]
        sh = max(32, round(h * 512 / w / 32) * 32)
        x = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (512, sh), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        t = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(self._ground_dev)
        with torch.no_grad():
            logits = self.ground(pixel_values=t.half() if self._ground_dev == "cuda" else t.float()).logits
            lab = torch.nn.functional.interpolate(logits.float(), size=(h, w), mode="bilinear", align_corners=False).argmax(1)[0]
            return torch.isin(lab, self._ground_ids).cpu().numpy(), torch.isin(lab, self._veto_ids).cpu().numpy()

    def infer(self, frame, veto=None):
        """Khung hình -> mask nước thô (đã xoá nước trên vùng cấm)."""
        water = self.water_mask(frame)
        if veto is not None and USE_WATER_VETO:
            water &= ~veto
        return water
