"""Mở video bằng OpenCV; không đọc được thì chuyển sang H.264 bằng ffmpeg của imageio-ffmpeg rồi đọc lại."""
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg


def _readable(path):
    cap = cv2.VideoCapture(str(path))
    ok = cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) > 0 and cap.grab()
    cap.release()
    return bool(ok)


def transcode_h264(src, dst):
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-v", "error", "-i", str(src), "-an", "-c:v", "libx264", "-preset", "veryfast",
           "-crf", "23", "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", str(dst)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not Path(dst).exists():
        raise ValueError("Không đọc được video (định dạng không hỗ trợ hoặc file hỏng)")
    return Path(dst)


def open_video(path):
    """Trả (cap, đường dẫn thực sự dùng, đường dẫn tạm cần xoá hoặc None)."""
    path = Path(path)
    if _readable(path):
        return cv2.VideoCapture(str(path)), path, None
    tmp = path.with_name(path.stem + "_h264.mp4")
    transcode_h264(path, tmp)
    if not _readable(tmp):
        tmp.unlink(missing_ok=True)
        raise ValueError("Không đọc được video (định dạng không hỗ trợ hoặc file hỏng)")
    return cv2.VideoCapture(str(tmp)), tmp, tmp
