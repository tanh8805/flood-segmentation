"""Cơ chế v3: chỉ phát hiện NGẬP / KHÔNG NGẬP. Model nước tự train + mặt đất SegFormer (mẫu số độ phủ, veto nước).

FloodPipeline nạp model một lần. Phần luật (tín hiệu nước, luật bật/tắt, đoạn 2 s) là hàm thuần,
không cần model nên test được độc lập (tests/test_pipeline.py) và mô phỏng được trên cache tín hiệu."""
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
FLOOD_ON = float(os.getenv("FLOOD_ON", "0.25"))        # độ phủ nước/mặt đường >= ngưỡng này liên tục FLOOD_HOLD_S giây -> NGẬP
FLOOD_OFF = float(os.getenv("FLOOD_OFF", "0.2"))      # đang NGẬP: độ phủ < ngưỡng này liên tục FLOOD_HOLD_S giây -> KHÔNG NGẬP
FLOOD_HOLD_S = float(os.getenv("FLOOD_HOLD_S", "1"))
FLOOD_SKIP_DARK = _flag("FLOOD_SKIP_DARK", "0")        # khung tối (logo, đen, chuyển cảnh): không cập nhật trạng thái
DARK_LUM = float(os.getenv("DARK_LUM", "40"))          # độ sáng xám trung bình < ngưỡng này là khung tối
FLOOD_RESET_ON_CUT = _flag("FLOOD_RESET_ON_CUT", "0")  # 1: đổi cảnh thì về KHÔNG NGẬP và đếm lại; 0: bỏ qua đổi cảnh (video ghép ảnh tĩnh vẫn đếm đủ H giây)
MIN_FRAME_FRAC = float(os.getenv("MIN_FRAME_FRAC", "0.5"))  # đoạn NGẬP khi >= 50% lần suy luận trong đoạn ở trạng thái NGẬP

GROUND_LABELS = ("road", "sidewalk", "path", "earth", "floor")
VETO_LABELS = ("sky", "building", "house", "tree", "wall", "ceiling", "signboard")   # nước không thể nằm trên các lớp này
NO_WATER_EVIDENCE = "Không thấy nước"


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


class FloodRule:
    """Trạng thái NGẬP với ngưỡng trễ và giữ H giây: KHÔNG NGẬP -> NGẬP khi ratio >= on liên tục hold giây;
    NGẬP -> KHÔNG NGẬP khi ratio < off liên tục hold giây. Khung tối (nếu bật) bị bỏ qua.
    reset_on_cut: đổi cảnh thì về KHÔNG NGẬP và đếm lại (mặc định tắt)."""

    def __init__(self, on=None, off=None, hold_s=None, skip_dark=None, dark_lum=None, reset_on_cut=None):
        self.on = FLOOD_ON if on is None else on
        self.off = FLOOD_OFF if off is None else off
        self.hold_s = FLOOD_HOLD_S if hold_s is None else hold_s
        self.skip_dark = FLOOD_SKIP_DARK if skip_dark is None else skip_dark
        self.dark_lum = DARK_LUM if dark_lum is None else dark_lum
        self.reset_on_cut = FLOOD_RESET_ON_CUT if reset_on_cut is None else reset_on_cut
        self.state, self.since, self.scene = False, None, None

    def update(self, t, scene, ratio, lum):
        if scene != self.scene:
            if self.reset_on_cut:
                self.state, self.since = False, None
            self.scene = scene
        if self.skip_dark and lum < self.dark_lum:
            return self.state
        against = ratio < self.off if self.state else ratio >= self.on
        if not against:
            self.since = None
        else:
            if self.since is None:
                self.since = t
            if t - self.since >= self.hold_s - 1e-9:
                self.state, self.since = not self.state, None
        return self.state


def evidence_text(flooded, ratio):
    if flooded:
        return f"Nước phủ {ratio:.0%} mặt đường"
    if ratio >= WATER_MIN_RATIO:
        return f"Nước ít ({ratio:.0%}), chưa ngập"
    return NO_WATER_EVIDENCE


def segment_info(steps):
    """Các lần suy luận (có `flooded`, `water_ratio`) của một đoạn -> (ngập?, độ phủ trung bình, căn cứ)."""
    flooded = sum(s["flooded"] for s in steps) / len(steps) >= MIN_FRAME_FRAC
    ratio = round(sum(s["water_ratio"] for s in steps) / len(steps), 3)
    return flooded, ratio, evidence_text(flooded, ratio)


def run_rule(steps, fps, rule=None):
    """Áp luật lên danh sách tín hiệu đã lưu (mỗi phần tử: idx, scene, water_ratio, lum), gắn `flooded`."""
    rule = rule or FloodRule()
    return [{**s, "flooded": rule.update(s["idx"] / fps, s["scene"], s["water_ratio"], s["lum"])} for s in steps]


def segments_from_steps(steps, total_frames, seg_len):
    """Chia các lần suy luận (đã có `flooded`) thành đoạn seg_len khung hình như analyzer: [{start_idx, end_idx, flooded, water_ratio, evidence}]."""
    out = []
    for start in range(0, total_frames, seg_len):
        end = min(start + seg_len, total_frames)
        cur = [s for s in steps if start <= s["idx"] < end]
        if not cur:
            cur = [max((s for s in steps if s["idx"] < end), key=lambda s: s["idx"])]
        flooded, ratio, ev = segment_info(cur)
        out.append({"start_idx": start, "end_idx": end, "flooded": flooded, "water_ratio": ratio, "evidence": ev})
    return out


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
