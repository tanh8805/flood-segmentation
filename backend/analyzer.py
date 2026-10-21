"""Phân tích video camera đường phố -> video segmentation + NGẬP / KHÔNG NGẬP theo thời gian.

Cơ chế (xem pipeline.py): model nước tự train + SegFormer ADE20K cho mặt đất. Mỗi lần suy luận cho độ phủ nước/mặt đường;
luật nhị phân (ngưỡng trễ + giữ H giây) đổi độ phủ thành trạng thái NGẬP; mỗi đoạn 2 giây NGẬP khi >= 50% lần suy luận ngập.
Chạy thử không cần server:  python analyzer.py video.mp4 [video_ket_qua.mp4]
"""
import os
import sys
import json
import threading
import unicodedata
from collections import deque
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

import pipeline as P
INFER_FPS = float(os.getenv("INFER_FPS", "5"))        # số lần chạy model mỗi giây video; khung hình ở giữa dùng lại mask gần nhất
SEGMENT_SECONDS = float(os.getenv("SEGMENT_SECONDS", "2"))  # độ dài mỗi đoạn
GROUND_EVERY_SECONDS = float(os.getenv("GROUND_EVERY_SECONDS", "1"))  # mặt đất (SegFormer) chạy ~1 lần/giây hoặc khi chuyển cảnh
SCENE_CUT = float(os.getenv("SCENE_CUT", "0.6"))  # khác biệt màu so với ~1 giây trước vượt ngưỡng này -> chuyển cảnh (1 = tắt)
MASK_SMOOTH = float(os.getenv("MASK_SMOOTH", "0.6"))  # trọng số lịch sử khi làm mượt mask nước giữa các lần chạy model (0 = tắt)
OUT_WIDTH = int(os.getenv("OUT_WIDTH", "960"))        # chiều rộng video kết quả (thu nhỏ cho nhẹ)
MAX_SECONDS = float(os.getenv("MAX_SECONDS", "600"))  # chỉ xử lý tối đa 10 phút đầu

FLOOD_BGR = (40, 30, 210)    # nền banner NGẬP (đỏ)
SAFE_BGR = (70, 160, 40)     # nền banner KHÔNG NGẬP (xanh)
WATER_BGR = (230, 120, 20)

_pipe = None
_lock = threading.Lock()


def get_pipeline():
    global _pipe
    if _pipe is None:
        _pipe = P.FloodPipeline()
    return _pipe


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
def banner(width, flooded, ratio_pct, line2):
    """Thanh trạng thái phía trên video (PIL để hiển thị được tiếng Việt)."""
    scale = width / 960
    height = int(70 * scale)
    b, g, r = FLOOD_BGR if flooded else SAFE_BGR
    img = Image.new("RGB", (width, height), (r, g, b))
    draw = ImageDraw.Draw(img)
    title = "NGẬP" if flooded else "KHÔNG NGẬP"
    draw.text((int(14 * scale), int(5 * scale)), vn_text(f"{title} · nước {ratio_pct}%"), font=_font(int(30 * scale)), fill=(255, 255, 255))
    draw.text((int(14 * scale), int(42 * scale)), vn_text(line2), font=_font(int(18 * scale)), fill=(255, 255, 255))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def build_overlay(water, h, w):
    """Lớp màu nước + viền của một lần chạy model."""
    area_color = np.zeros((h, w, 3), np.uint8)
    area_color[water] = WATER_BGR
    cs, _ = cv2.findContours(water.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return {"area_color": area_color, "area": water.copy(), "contours": cs}


def render_frame(frame, ov_a, ov_b, t, seg):
    """Vẽ khung hình nằm giữa 2 lần chạy model: vùng nước hòa trộn dần từ ov_a sang ov_b (t = 0..1)."""
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
    cv2.drawContours(out, near["contours"], -1, WATER_BGR, 2)
    return np.vstack([banner(out.shape[1], seg["flooded"], round(seg["water_ratio"] * 100), f"Căn cứ: {seg['evidence']}"), out])


def open_writer(path, w, h, fps):
    """Ghi H.264 để trình duyệt phát được (mp4v của OpenCV không phát trên web)."""
    writer = imageio_ffmpeg.write_frames(
        str(path), (w, h), fps=fps, codec="libx264", pix_fmt_in="bgr24", quality=None, macro_block_size=2,
        output_params=["-preset", "veryfast", "-crf", "23", "-movflags", "+faststart"])
    writer.send(None)
    return writer


# ---------------- Xử lý cả video ----------------

def process_video(video_path, out_path=None, progress=None, signals_path=None):
    """Phân tích video; trả về kết quả chung + từng đoạn. out_path=None: không ghi video kết quả.
    signals_path: ghi mỗi lần suy luận một dòng JSON (tín hiệu thô, chưa qua luật) để mô phỏng ngưỡng mà không chạy lại model."""
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
    buf = []       # khung hình đã sẵn overlay của đoạn hiện tại (chờ biết kết quả của đoạn rồi mới ghi)
    pending = []   # khung hình từ lần chạy model trước tới trước lần chạy kế tiếp (chờ để hòa trộn mask)
    prev = None    # (overlay, stats) của lần chạy model trước
    smoother, ground, rule = P.WaterSmoother(MASK_SMOOTH), P.GroundCache(pipe, ground_every), P.FloodRule()
    scenes = SceneCutDetector(round(INFER_FPS))
    scene = 0
    writer = open_writer(out_path, w, h + banner(w, False, 0, "").shape[0], fps) if out_path else None
    sig_file = open(signals_path, "w", encoding="utf-8") if signals_path else None
    idx = seg_start = 0

    def flush(end):
        nonlocal seg_start
        if not buf:
            return
        cur = [f for f, i in zip(frames, kf_index) if seg_start <= i < end]
        if not cur:
            cur = [max(((f, i) for f, i in zip(frames, kf_index) if i < end), key=lambda x: x[1])[0]]
        flooded, ratio, evidence = P.segment_info(cur)
        seg = {"start_s": round(seg_start / fps, 2), "end_s": round(end / fps, 2), "flooded": flooded,
               "water_ratio": ratio, "evidence": evidence, "scene": cur[-1]["scene"]}
        segments.append(seg)
        if writer:
            for frame, ov_a, ov_b, t in buf:
                writer.send(np.ascontiguousarray(render_frame(frame, ov_a, ov_b, t, seg)))
        buf.clear()
        seg_start = end

    def emit(nxt):
        """Chuyển các khung hình đang chờ vào đoạn, hòa trộn từ lần chạy trước (prev) tới lần chạy mới (nxt)."""
        n = len(pending)
        for j, (i, frame) in enumerate(pending):
            if i and i % seg_len == 0:
                flush(i)
            buf.append((frame, prev[0], nxt[0], j / n))
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
                        # cảnh mới: không hòa trộn mask từ cảnh cũ, bỏ lịch sử làm mượt mask
                        if prev is not None:
                            emit(prev)
                        smoother = P.WaterSmoother(MASK_SMOOTH)
                        scene += 1
                    ground.update(frame, cut)
                    water = smoother.update(pipe.infer(frame, veto=ground.veto))
                    sig = P.water_signal(water, ground.mask, h, w)
                    lum = round(float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()), 1)
                    if sig_file:
                        sig_file.write(json.dumps({"idx": idx, "t": round(idx / fps, 3), "scene": scene, **sig, "lum": lum}) + "\n")
                    stats = {"time_s": round(idx / fps, 2), "idx": idx, "scene": scene, **sig, "lum": lum,
                             "flooded": rule.update(idx / fps, scene, sig["water_ratio"], lum)}
                    frames.append(stats)
                    kf_index.append(idx)
                    cur = (build_overlay(water, h, w) if writer else None, stats)
                    if prev is not None:
                        emit(cur)
                    prev = cur
                pending.append((idx, frame if writer else None))
                idx += 1
                if progress:
                    progress(min(idx / total, 0.99))
            if prev is not None:
                emit(prev)
                flush(idx)
        finally:
            cap.release()
            if writer:
                writer.close()
            if sig_file:
                sig_file.close()
    if not frames:
        raise ValueError("Video không có khung hình nào")
    if signals_path:
        Path(signals_path).with_suffix(".meta.json").write_text(json.dumps(
            {"fps": fps, "total_frames": idx, "seg_len": seg_len, "infer_every": infer_every, "w": w, "h": h}), encoding="utf-8")

    intervals = []   # các khoảng NGẬP liền nhau
    for s in segments:
        if s["flooded"]:
            if intervals and intervals[-1]["end_s"] == s["start_s"]:
                intervals[-1]["end_s"] = s["end_s"]
            else:
                intervals.append({"start_s": s["start_s"], "end_s": s["end_s"]})
    flooded_s = round(sum(i["end_s"] - i["start_s"] for i in intervals), 2)
    return {
        "flooded": bool(intervals),
        "flooded_seconds": flooded_s,
        "flood_intervals": intervals,
        "summary": (f"Có ngập: {flooded_s:g} giây, {len(intervals)} khoảng" if intervals else "Không ngập"),
        "duration_s": round(idx / fps, 2),
        "frames_analyzed": len(frames),
        "water_ratio_max": max(s["water_ratio"] for s in segments),
        "scene_cuts": [f["time_s"] for k, f in enumerate(frames) if k and f["scene"] != frames[k - 1]["scene"]],
        "components": pipe.status(),
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
