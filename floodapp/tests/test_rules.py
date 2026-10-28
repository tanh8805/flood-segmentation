import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from floodapp import rules as R
from floodapp.video_io import _readable, open_video, transcode_h264


def run(rule, p, ratio=None, scene=None):
    n = len(p)
    return R.run(rule, p, ratio or [0.0] * n, scene or [0] * n).tolist()


def test_clf_hysteresis():
    r = R.ClfRule(W=0.2, on=0.8, off=0.3)          # W=0.2 s -> cửa sổ 1 lần suy luận
    assert run(r, [0.5, 0.9, 0.5, 0.4, 0.2, 0.5]) == [False, True, True, True, False, False]


def test_clf_window_average():
    r = R.ClfRule(W=0.4, on=0.8, off=0.3)          # 2 lần suy luận: trung bình 0.5, 0.7, 0.9 -> chỉ bật ở lần cuối
    assert run(r, [0.5, 0.5, 0.9, 0.9]) == [False, False, False, True]


def test_window_resets_on_scene_cut_but_state_kept():
    r = R.ClfRule(W=1.0, on=0.8, off=0.3)          # 5 lần suy luận
    out = run(r, [1.0] * 5 + [0.5, 0.5], scene=[0] * 5 + [1, 1])
    assert out[4] is True and out[5] is True       # cảnh mới: cửa sổ chỉ còn 0.5 >= off nên giữ NGẬP
    assert run(R.ClfRule(1.0, 0.8, 0.3), [0.0, 0.0, 0.9, 0.9, 0.9, 0.9, 0.9], scene=[0, 0, 1, 1, 1, 1, 1])[2] is True  # không lẫn p của cảnh cũ


def test_water_rule():
    r = R.WaterRule(W=0.2, on=0.25, off=0.2)
    assert run(r, [0] * 4, ratio=[0.1, 0.3, 0.22, 0.1]) == [False, True, True, False]


def test_combo_rule():
    r = R.ComboRule(W=0.2, a=0.9, b=0.2, c=0.3)
    assert run(r, [0.3] * 3, ratio=[0.1, 0.35, 0.35]) == [False, True, True]       # p thấp nhưng nước phủ nhiều
    assert run(R.ComboRule(0.2, 0.9, 0.2, 0.3), [0.95, 0.8, 0.1], ratio=[0, 0, 0]) == [True, True, False]  # giữ bằng ngưỡng thấp hơn
    assert run(R.ComboRule(0.2, 0.9, 0.2, 0.3), [0.5], ratio=[0.0]) == [False]


def test_segment_states_majority():
    st = [True, True, False, False, False, True]
    assert R.segment_states(st, [0, 0, 1, 1, 1, 2]) == {0: True, 1: False, 2: True}
    assert R.segment_states([True, False], [0, 0]) == {0: True}      # đúng 50% -> NGẬP


def test_evidence_text_vietnamese():
    assert R.evidence_text(True, 0.93, 0.85) == "Bộ phân loại 0.93, nước phủ 85% mặt đường"
    assert "chưa thấy vùng nước rõ" in R.evidence_text(False, 0.1, 0.01)


def test_load_rule_from_weights():
    rule = R.load_rule(Path(__file__).resolve().parents[1] / "weights" / "rule.json")
    assert isinstance(rule, tuple(R.RULES.values()))


def test_open_video_transcodes_unreadable(tmp_path):
    src = tmp_path / "a.avi"
    w = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    for i in range(20):
        w.write(np.full((48, 64, 3), i * 10, np.uint8))
    w.release()
    out = transcode_h264(src, tmp_path / "b.mp4")
    assert _readable(out)
    cap, used, tmp = open_video(src)
    assert cap.isOpened()
    cap.release()
