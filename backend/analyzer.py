"""Phân tích video camera đường phố -> video segmentation + mức cảnh báo ngập theo từng giai đoạn.

Cơ chế (xem pipeline.py): model nước tự train, người (yolo11n-pose) và xe (yolo11n) làm vật tham chiếu độ sâu,
SegFormer ADE20K cho mặt đất. Mức 0..3 theo luật: 0 không có nước; 1 có nước hoặc vật bậc 1;
2 chỉ khi có vật bậc 2; 3 chỉ khi có vật bậc 3 (độ phủ một mình tối đa mức 1).
Chạy thử không cần server:  python analyzer.py video.mp4 [video_ket_qua.mp4]
"""
import os
import sys
import json
import threading
import unicodedata
from collections import Counter, deque
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

import pipeline as P
INFER_FPS = float(os.getenv("INFER_FPS", "5"))        # số lần chạy model mỗi giây video; khung hình ở giữa dùng lại mask gần nhất
SEGMENT_SECONDS = float(os.getenv("SEGMENT_SECONDS", "2"))  # độ dài mỗi đoạn để tính mức cảnh báo
MIN_FRAME_FRAC = float(os.getenv("MIN_FRAME_FRAC", "0.5"))  # mức của đoạn = mức cao nhất mà >= 50% khung hình trong đoạn đạt
GROUND_EVERY_SECONDS = float(os.getenv("GROUND_EVERY_SECONDS", "1"))  # mặt đất (SegFormer) chạy ~1 lần/giây hoặc khi chuyển cảnh
LEVEL_HOLD_SECONDS = float(os.getenv("LEVEL_HOLD_SECONDS", "0"))  # > 0: mức chỉ được giảm khi đã thấp hơn liên tục bấy nhiêu giây; mặc định 0 để mức theo đúng thời điểm
SCENE_CUT = float(os.getenv("SCENE_CUT", "0.6"))  # khác biệt màu so với ~1 giây trước vượt ngưỡng này -> chuyển cảnh (1 = tắt)
MASK_SMOOTH = float(os.getenv("MASK_SMOOTH", "0.6"))  # trọng số lịch sử khi làm mượt mask nước giữa các lần chạy model (0 = tắt)
OUT_WIDTH = int(os.getenv("OUT_WIDTH", "960"))        # chiều rộng video kết quả (thu nhỏ cho nhẹ)
MAX_SECONDS = float(os.getenv("MAX_SECONDS", "600"))  # chỉ xử lý tối đa 10 phút đầu

# Mức cảnh báo: (mã, tên, màu BGR). Luật chọn mức nằm ở pipeline.frame_decision.
LEVELS = [
    (0, "An toàn", (70, 160, 40)),
    (1, "Chú ý", (0, 185, 235)),
    (2, "Cảnh báo", (0, 120, 245)),
    (3, "Nguy hiểm", (40, 30, 210)),
]
NO_REF_NOTE = "không có vật tham chiếu để ước lượng độ sâu"
# Màu vẽ (BGR)
CLASS_COLORS = {"water": (230, 120, 20), "person": (80, 200, 80), "vehicle": (0, 215, 255)}
LEVEL_COLORS = {0: (80, 200, 80), 1: (60, 220, 200), 2: (0, 150, 255), 3: (40, 40, 230)}

_pipe = None
_lock = threading.Lock()


def get_pipeline():
    global _pipe
    if _pipe is None:
        _pipe = P.FloodPipeline()
    return _pipe



def segment_level(stats):
    n = len(stats)
    return max(code for code, *_ in LEVELS if code == 0 or sum(s["level"] >= code for s in stats) / n >= MIN_FRAME_FRAC)


def held_level(raw, start_s, scene, prev_segments):
    """Nước không rút trong vài giây: mức tăng ngay, nhưng chỉ giảm khi các đoạn trong
    LEVEL_HOLD_SECONDS giây trước đều thấp hơn. Lấp các đoạn model sót nước (bị che, nhận nhầm).
    Chỉ giữ trong cùng một cảnh: sang cảnh khác (logo, trường quay...) thì không kéo mức cũ sang."""
    recent = [p["raw_level"] for p in prev_segments
              if p["scene"] == scene and p["end_s"] > start_s - LEVEL_HOLD_SECONDS]
    return max([raw] + recent)

def max_or_none(values):
    values = [v for v in values if v is not None]
    return max(values) if values else None


class SceneCutDetector:
    """Phát hiện chuyển cảnh (video tin tức cắt ghép) bằng histogram màu HSV: so khung hình hiện tại với
    khung hình ~1 giây trước trong cùng cảnh, để bắt được cả cắt thẳng lẫn chuyển mờ dần.
    Ngưỡng cao để người/xe đi ngang che camera hay camera rung không bị tính là chuyển cảnh."""

    def __init__(self, window):
        self.hists = deque(maxlen=max(1, window))

    def update(self, frame):
        hsv = cv2.cvtColor(cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [16, 4, 4], [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        cut = bool(self.hists) and cv2.compareHist(self.hists[0], hist, cv2.HISTCMP_BHATTACHARYYA) > SCENE_CUT
        if cut:
            self.hists.clear()
        self.hists.append(hist)
        return cut


# ---------------- Vẽ lên video ----------------

def _find_font():
    env = os.getenv("FONT_PATH", "")
    if env and Path(env).exists():
        return env
    # matplotlib (luôn được cài cùng ultralytics) có sẵn DejaVu Sans, hiển thị đủ dấu tiếng Việt
    try:
        import matplotlib
        p = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans-Bold.ttf"
        if p.exists():
            return str(p)
    except Exception:
        pass
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ]
    return next((c for c in candidates if Path(c).exists()), None)


def _font(size):
    if FONT_FILE:
        return ImageFont.truetype(FONT_FILE, size)
    try:
        return ImageFont.load_default(size)  # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


FONT_FILE = _find_font()


def vn_text(s):
    """Không có font tiếng Việt thì bỏ dấu để chữ không bị lỗi ô vuông."""
    if FONT_FILE:
        return s
    s = unicodedata.normalize("NFD", s.replace("đ", "d").replace("Đ", "D"))
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")


@lru_cache(maxsize=512)
def banner(width, level, line2):
    """Thanh cảnh báo phía trên video (PIL để hiển thị được tiếng Việt)."""
    scale = width / 960
    height = int(70 * scale)
    b, g, r = LEVELS[level][2]
    img = Image.new("RGB", (width, height), (r, g, b))
    draw = ImageDraw.Draw(img)
    big = _font(int(30 * scale))
    small = _font(int(18 * scale))
    draw.text((int(14 * scale), int(5 * scale)), vn_text(f"MỨC {level}: {LEVELS[level][1].upper()}"), font=big, fill=(255, 255, 255))
    draw.text((int(14 * scale), int(42 * scale)), vn_text(line2), font=small, fill=(255, 255, 255))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def build_overlay(water, objects, h, w):
    """Lớp màu nước + viền + khung người/xe (kèm bậc) của một lần chạy model."""
    ov = {"area_color": np.zeros((h, w, 3), np.uint8), "area": water.copy(), "contours": [], "labels": [], "boxes": []}
    ov["area_color"][water] = CLASS_COLORS["water"]
    cs, _ = cv2.findContours(water.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ov["contours"].append((cs, CLASS_COLORS["water"]))
    for o in objects:
        if not P.is_valid(o):
            continue
        col = LEVEL_COLORS[o["level"]]
        x1, y1, x2, y2 = (int(v) for v in o["box"])
        tag = "nguoi" if o["kind"] == "person" else P.VEHICLE_TAG[o["cls"]]
        ov["boxes"].append(((x1, y1, x2, y2), col))
        ov["labels"].append((f"{tag} B{o['level']}", (x1, y1), col))
    return ov


def draw_legend(img):
    h = img.shape[0]
    items = [("nuoc", CLASS_COLORS["water"])] + [(f"vat bac {k}", LEVEL_COLORS[k]) for k in range(4)]
    y0 = h - 10 - 20 * len(items)
    cv2.rectangle(img, (6, y0 - 6), (130, h - 6), (0, 0, 0), -1)
    for k, (name, col) in enumerate(items):
        y = y0 + 20 * k
        cv2.rectangle(img, (12, y + 2), (26, y + 14), col, -1)
        cv2.putText(img, name, (32, y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def render_frame(frame, ov_a, ov_b, t, level, stats, held=False):
    """Vẽ khung hình nằm giữa 2 lần chạy model: vùng nước hòa trộn dần từ ov_a sang ov_b (t = 0..1);
    khung vật, viền, nhãn lấy theo lần chạy gần hơn."""
    near = ov_a if t < 0.5 else ov_b
    if ov_a is ov_b or t == 0:
        out = frame.copy()
        m = ov_a["area"]
        out[m] = (out[m] * 0.5 + ov_a["area_color"][m] * 0.5).astype(np.uint8)
    else:
        wa = ov_a["area"].astype(np.float32) * (1 - t)
        wb = ov_b["area"].astype(np.float32) * t
        alpha = (0.5 * (wa + wb))[..., None]
        color = ov_a["area_color"] * wa[..., None] + ov_b["area_color"] * wb[..., None]
        out = (frame * (1 - alpha) + 0.5 * color).astype(np.uint8)
    for cs, col in near["contours"]:
        cv2.drawContours(out, cs, -1, col, 2)
    for (x1, y1, x2, y2), col in near["boxes"]:
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
    for text, (x, y), col in near["labels"]:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y = max(y, th + 4)
        cv2.rectangle(out, (x, y - th - 4), (x + tw + 4, y), col, -1)
        cv2.putText(out, text, (x + 2, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    draw_legend(out)
    line2 = f"Căn cứ: {stats['evidence']}"
    if held:
        line2 = f"Giữ mức do vừa ngập vài giây trước · hiện thấy: {stats['evidence'].lower()}"
    return np.vstack([banner(out.shape[1], level, line2), out])

def open_writer(path, w, h, fps):
    """Ghi H.264 để trình duyệt phát được (mp4v của OpenCV không phát trên web)."""
    writer = imageio_ffmpeg.write_frames(
        str(path), (w, h), fps=fps, codec="libx264", pix_fmt_in="bgr24", quality=None, macro_block_size=2,
        output_params=["-preset", "veryfast", "-crf", "23", "-movflags", "+faststart"])
    writer.send(None)
    return writer



# ---------------- Xử lý cả video ----------------

def process_video(video_path, out_path, progress=None):
    """Tạo video segmentation có thanh cảnh báo; trả về mức cảnh báo tổng + từng giai đoạn."""
    pipe = get_pipeline()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("Không đọc được video (định dạng không hỗ trợ hoặc file hỏng)")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps > 240:
        fps = 25.0
    src_w, src_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = min(OUT_WIDTH, src_w) // 2 * 2
    h = round(src_h * w / src_w) // 2 * 2
    max_frames = int(MAX_SECONDS * fps)
    total = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0), max_frames) or max_frames
    infer_every = max(1, round(fps / INFER_FPS))
    seg_len = max(infer_every, round(SEGMENT_SECONDS * fps))
    ground_every = max(1, round(GROUND_EVERY_SECONDS * fps / infer_every))

    frames, kf_index, segments = [], [], []   # số liệu từng lần chạy model + chỉ số khung hình tương ứng
    buf = []       # khung hình đã sẵn overlay của đoạn hiện tại (chờ biết mức của đoạn rồi mới ghi)
    pending = []   # khung hình từ lần chạy model trước tới trước lần chạy kế tiếp (chờ để hòa trộn mask)
    prev = None    # (overlay, stats) của lần chạy model trước
    smoother, tracker, ground = P.WaterSmoother(MASK_SMOOTH), P.ObjectTracker(), P.GroundCache(pipe, ground_every)
    memory = P.DepthMemory()
    scenes = SceneCutDetector(round(INFER_FPS))
    scene = 0
    writer = open_writer(out_path, w, h + banner(w, 0, "").shape[0], fps)
    idx = seg_start = 0

    def flush(end):
        nonlocal seg_start
        if not buf:
            return
        cur = [f for f, i in zip(frames, kf_index) if seg_start <= i < end]
        if not cur:
            cur = [max(((f, i) for f, i in zip(frames, kf_index) if i < end), key=lambda x: x[1])[0]]
        raw = segment_level(cur)
        seg_scene = cur[-1]["scene"]
        at = [s for s in cur if s["level"] == raw] or [s for s in cur if s["level"] >= raw]
        evidence = Counter(s["evidence"] for s in at).most_common(1)[0][0]
        no_ref = all(s["max_object_level"] is None for s in cur)
        level = held_level(raw, round(seg_start / fps, 2), seg_scene, segments)
        segments.append({
            "start_s": round(seg_start / fps, 2), "end_s": round(end / fps, 2),
            "level": level, "warning": LEVELS[level][1], "raw_level": raw, "scene": seg_scene, "evidence": evidence,
            "flood_ratio_max": max(s["flood_ratio"] for s in cur),
            "object_level_max": max_or_none(s["max_object_level"] for s in cur),
            "car_level_max": max_or_none(s["max_car_level"] for s in cur),
            "note": NO_REF_NOTE if no_ref else None,
        })
        for frame, ov_a, ov_b, t, st in buf:
            writer.send(np.ascontiguousarray(render_frame(frame, ov_a, ov_b, t, level, st, held=level > raw)))
        buf.clear()
        seg_start = end

    def emit(nxt):
        """Chuyển các khung hình đang chờ vào đoạn, hòa trộn từ lần chạy trước (prev) tới lần chạy mới (nxt)."""
        n = len(pending)
        for j, (i, frame) in enumerate(pending):
            if i and i % seg_len == 0:
                flush(i)
            t = j / n
            st = prev[1] if t < 0.5 else nxt[1]
            buf.append((frame, prev[0], nxt[0], t, st))
        pending.clear()

    with _lock:  # model không an toàn khi nhiều video dùng cùng lúc
        try:
            while idx < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
                if idx % infer_every == 0:
                    cut = scenes.update(frame)
                    if cut:
                        # cảnh mới: không hòa trộn mask từ cảnh cũ, bỏ lịch sử làm mượt mask/vật
                        if prev is not None:
                            emit(prev)
                        smoother, tracker = P.WaterSmoother(MASK_SMOOTH), P.ObjectTracker()
                        scene += 1
                    water_raw, objs = pipe.infer(frame)
                    water, objs = smoother.update(water_raw), tracker.update(objs)
                    gmask = ground.update(frame, cut) if water.any() else None
                    remembered = memory.update(idx / fps, scene, objs, float(water.mean())) if P.USE_DEPTH_MEMORY else []
                    _, _, st = P.frame_decision(water, gmask, objs + remembered, h, w)
                    stats = {"time_s": round(idx / fps, 2), "scene": scene, **st}
                    frames.append(stats)
                    kf_index.append(idx)
                    cur = (build_overlay(water, objs, h, w), stats)
                    if prev is not None:
                        emit(cur)
                    prev = cur
                pending.append((idx, frame))
                idx += 1
                if progress:
                    progress(min(idx / total, 0.99))
            if prev is not None:
                emit(prev)
                flush(idx)
        finally:
            cap.release()
            writer.close()
    if not frames:
        raise ValueError("Video không có khung hình nào")

    # Gộp các đoạn liền nhau cùng mức thành một giai đoạn
    phases = []
    for s in segments:
        if phases and phases[-1]["level"] == s["level"]:
            p = phases[-1]
            p["end_s"] = s["end_s"]
            p["flood_ratio_max"] = max(p["flood_ratio_max"], s["flood_ratio_max"])
            p["object_level_max"] = max_or_none([p["object_level_max"], s["object_level_max"]])
            p["car_level_max"] = max_or_none([p["car_level_max"], s["car_level_max"]])
            p["note"] = NO_REF_NOTE if p["object_level_max"] is None else None
        else:
            phases.append(dict(s))
    level = max(s["level"] for s in segments)
    return {
        "warning_level": level,
        "warning": LEVELS[level][1],
        "evidence": next(s["evidence"] for s in segments if s["level"] == level),
        "duration_s": round(idx / fps, 2),
        "frames_analyzed": len(frames),
        "seconds_per_level": {LEVELS[c][1]: round(sum(p["end_s"] - p["start_s"] for p in phases if p["level"] == c), 2)
                              for c in range(len(LEVELS))},
        "flood_ratio_max": max(f["flood_ratio"] for f in frames),
        "object_level_max": max_or_none(f["max_object_level"] for f in frames),
        "car_level_max": max_or_none(f["max_car_level"] for f in frames),
        "note": NO_REF_NOTE if all(f["max_object_level"] is None for f in frames) else None,
        "scene_cuts": [f["time_s"] for k, f in enumerate(frames) if k and f["scene"] != frames[k - 1]["scene"]],
        "components": pipe.status(),
        "phases": phases,
        "segments": segments,
        "frames": frames,
    }


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        sys.exit("Dùng: python analyzer.py video.mp4 [video_ket_qua.mp4]")
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) == 3 else src.with_name(src.stem + "_result.mp4")
    out = process_video(src, dst)
    out.pop("frames")
    out.pop("segments")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print("Video kết quả:", dst)
