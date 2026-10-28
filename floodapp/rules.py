"""Luật quyết định NGẬP / KHÔNG NGẬP theo thời gian: hàm/lớp thuần, không cần model.
App chạy luật này theo từng lần suy luận; mô phỏng (src/sim.py) chạy đúng các lớp này trên cache tín hiệu.
Mỗi luật nhận p (bộ phân loại), ratio (độ phủ nước) và scene (id cảnh) của từng lần suy luận, trả trạng thái NGẬP."""
import json
from collections import deque
from pathlib import Path

import numpy as np

INFER_FPS = 5.0
MIN_FRAME_FRAC = 0.5      # đoạn NGẬP khi >= 50% lần suy luận trong đoạn ở trạng thái NGẬP
WATER_MIN_RATIO = 0.05    # chỉ để viết căn cứ: dưới ngưỡng này là "không thấy nước"


class _Windowed:
    """Trung bình trượt của p và ratio trên n lần suy luận gần nhất trong cùng một cảnh; đổi cảnh thì xoá cửa sổ (trạng thái ngập được giữ)."""

    def __init__(self, W, fps_inf=INFER_FPS):
        self.n = max(1, int(round(W * fps_inf)))
        self.p, self.r, self.scene, self.state = deque(maxlen=self.n), deque(maxlen=self.n), None, False

    def _push(self, p, ratio, scene):
        if scene != self.scene:
            self.p.clear()
            self.r.clear()
            self.scene = scene
        self.p.append(p)
        self.r.append(ratio)
        return sum(self.p) / len(self.p), sum(self.r) / len(self.r)


class ClfRule(_Windowed):
    """Chỉ bộ phân loại: p trung bình trượt W giây; bật khi >= on, giữ cho tới khi < off."""

    def __init__(self, W, on, off, fps_inf=INFER_FPS):
        super().__init__(W, fps_inf)
        self.on, self.off = on, off

    def update(self, p, ratio, scene):
        ps, _ = self._push(p, ratio, scene)
        self.state = ps >= (self.off if self.state else self.on)
        return self.state


class WaterRule(_Windowed):
    """Chỉ độ phủ nước (như v3): ratio trung bình trượt; bật khi >= on, giữ cho tới khi < off."""

    def __init__(self, W, on, off, fps_inf=INFER_FPS):
        super().__init__(W, fps_inf)
        self.on, self.off = on, off

    def update(self, p, ratio, scene):
        _, rs = self._push(p, ratio, scene)
        self.state = rs >= (self.off if self.state else self.on)
        return self.state


class ComboRule(_Windowed):
    """Kết hợp: ngập nếu p >= a, hoặc (p >= b và ratio >= c); giữ với ngưỡng thấp hơn (p: -margin, ratio: -0.1)."""

    def __init__(self, W, a, b, c, margin=0.15, fps_inf=INFER_FPS):
        super().__init__(W, fps_inf)
        self.a, self.b, self.c, self.margin = a, b, c, margin

    def update(self, p, ratio, scene):
        ps, rs = self._push(p, ratio, scene)
        m, mr = (self.margin, 0.1) if self.state else (0.0, 0.0)
        self.state = bool(ps >= self.a - m or (ps >= self.b - m and rs >= self.c - mr))
        return self.state


RULES = {"clf": ClfRule, "water": WaterRule, "combo": ComboRule}


def make_rule(cfg):
    cfg = dict(cfg)
    return RULES[cfg.pop("name")](**cfg)


def load_rule(path):
    return make_rule(json.loads(Path(path).read_text()))


def run(rule, p, ratio, scene):
    return np.array([rule.update(float(a), float(b), int(c)) for a, b, c in zip(p, ratio, scene)], bool)


def segment_states(states, seg, min_frac=MIN_FRAME_FRAC):
    """Nhãn đoạn = đa số lần suy luận ở trạng thái NGẬP. Trả {đoạn: bool}."""
    states, seg = np.asarray(states, float), np.asarray(seg)
    return {int(s): bool(states[seg == s].mean() >= min_frac) for s in np.unique(seg)}


def evidence_text(flooded, p, ratio):
    water = f"nước phủ {ratio:.0%} mặt đường" if ratio >= WATER_MIN_RATIO else "chưa thấy vùng nước rõ"
    return f"Bộ phân loại {p:.2f}, {water}" if flooded else f"Không ngập: bộ phân loại {p:.2f}, {water}"
