"""Danh sách clip dùng cho cache/baseline v3 (tên -> đường dẫn)."""
from pathlib import Path

ROOT = Path("/home/tuananh/Documents/flood_segmentation")
H = Path.home()
V = H / "Videos" / "flood_test_new"
UP = H / "Downloads" / "PTHTTM" / "uploads"
CLIPS = {
    "clip0": UP / "0924(1).mp4",
    "clip1": UP / "76037d05016a.mp4",
    "clip2": next(UP.glob("FSave.com_Reels_10K*")),
    "clip5": next((V / "_used_v2").glob("YTSave*")),
    "tuoitre_dem": V / "_used_v2" / "tuoitre_dem.mp4",
    **{f"fred_dry_0{i}": ROOT / "work/v3/clips" / f"fred_dry_0{i}.mp4" for i in range(1, 7)},
    **{p.stem: p for p in sorted((V / "ngap").glob("*.mp4"))},
    **{p.stem: p for p in sorted((V / "khong_ngap").glob("*.mp4"))},
}
