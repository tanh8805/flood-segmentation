"""Unit test luật nhị phân NGẬP / KHÔNG NGẬP, tín hiệu nước, đoạn 2 s, mặt đất, bật/tắt (không cần weights).
Chạy: cd backend && python -m unittest discover tests"""
import importlib
import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pipeline as P

H = W = 100


def mask(frac, shape=(H, W)):
    m = np.zeros(shape, bool)
    m.flat[:int(frac * m.size)] = True
    return m


def feed(rule, ratios, dt=0.2, scene=0, lum=100, t0=0.0):
    """Đưa chuỗi độ phủ vào luật (mỗi lần cách dt giây) -> danh sách trạng thái."""
    return [rule.update(round(t0 + i * dt, 3), scene, r, lum) for i, r in enumerate(ratios)]


class SignalTest(unittest.TestCase):
    def test_ratio_over_ground(self):
        water, ground = mask(0.2), np.zeros((H, W), bool)
        ground[40:80] = True                                  # 40% khung, không chồng nước
        s = P.water_signal(water, ground, H, W)
        self.assertEqual(s["base"], "ground")
        self.assertAlmostEqual(s["water_ratio"], 0.2 / 0.6, places=3)   # nước / (mặt đất ∪ nước)
        self.assertAlmostEqual(s["ground_ratio"], 0.6, places=3)

    def test_fallback_covered(self):
        s = P.water_signal(mask(0.3), None, H, W)             # không có mặt đất (SegFormer tắt), nước >= 10% khung
        self.assertEqual((s["base"], s["water_ratio"]), ("covered", 1.0))

    def test_fallback_frame(self):
        s = P.water_signal(mask(0.08), None, H, W)
        self.assertEqual((s["base"], s["water_ratio"]), ("frame", 0.08))

    def test_tiny_water_ignored(self):
        s = P.water_signal(mask(0.003), np.ones((H, W), bool), H, W)
        self.assertEqual(s["water_ratio"], 0.0)

    def test_rounded(self):
        s = P.water_signal(mask(1 / 3), np.ones((H, W), bool), H, W)
        self.assertEqual(s["water_ratio"], round(s["water_ratio"], 4))


class RuleTest(unittest.TestCase):
    def rule(self, **kw):
        return P.FloodRule(**{"on": 0.5, "off": 0.3, "hold_s": 1.0, "skip_dark": True, "dark_lum": 40, "reset_on_cut": False, **kw})

    def test_on_needs_hold(self):
        out = feed(self.rule(), [0.6] * 8)                    # 0.2 s/lần: đủ 1 s ở lần thứ 6
        self.assertEqual(out, [False] * 5 + [True] * 3)

    def test_short_spike_ignored(self):
        out = feed(self.rule(), [0.6, 0.6, 0.1, 0.6, 0.6, 0.1] + [0.1] * 4)
        self.assertFalse(any(out))

    def test_hysteresis_band_keeps_state(self):
        out = feed(self.rule(hold_s=0), [0.6, 0.4, 0.4, 0.4])   # 0.3 <= 0.4 < 0.5: giữ NGẬP
        self.assertEqual(out, [True] * 4)
        out = feed(self.rule(hold_s=0), [0.4, 0.4])             # chưa ngập thì vùng giữa không bật
        self.assertEqual(out, [False, False])

    def test_off_needs_hold(self):
        out = feed(self.rule(), [0.6] * 7 + [0.1] * 8)
        self.assertTrue(out[6] and out[7])                     # vừa tụt dưới off: chưa tắt ngay
        self.assertTrue(out[11])
        self.assertFalse(out[12])                              # tụt từ lần 7, đủ 1 s ở lần 12

    def test_dip_during_flood_does_not_turn_off(self):
        out = feed(self.rule(), [0.6] * 7 + [0.1, 0.1] + [0.6] * 3)
        self.assertTrue(all(out[6:]))

    def test_scene_cut_ignored_by_default(self):
        r = self.rule()
        out = [r.update(round(i * 0.2, 3), i // 2, 0.6, 100) for i in range(8)]   # đổi cảnh mỗi 0.4 s (ảnh tĩnh ghép)
        self.assertTrue(out[-1])                                # vẫn đủ 1 s liên tục -> NGẬP

    def test_scene_cut_resets_when_enabled(self):
        r = self.rule(hold_s=0, reset_on_cut=True)
        self.assertTrue(r.update(0, 0, 0.9, 100))
        self.assertFalse(r.update(0.2, 1, 0.4, 100))           # cảnh mới: về KHÔNG NGẬP (0.4 < on)
        self.assertTrue(r.update(0.4, 1, 0.9, 100))
        r = self.rule(reset_on_cut=True)
        feed(r, [0.6] * 4)
        self.assertFalse(any(feed(r, [0.6] * 4, scene=1, t0=0.8)))   # cảnh mới phải chờ lại từ đầu

    def test_dark_frames_skipped(self):
        r = self.rule(hold_s=0)
        self.assertTrue(r.update(0, 0, 0.9, 100))
        self.assertTrue(r.update(0.2, 0, 0.0, 10))             # khung tối: bị bỏ qua, không tắt
        r2 = self.rule(hold_s=0, skip_dark=False)
        r2.update(0, 0, 0.9, 100)
        self.assertFalse(r2.update(0.2, 0, 0.0, 10))           # tắt chế độ bỏ khung tối: khung tối vẫn tính

    def test_defaults_valid(self):
        self.assertLess(P.FLOOD_OFF, P.FLOOD_ON)
        self.assertGreaterEqual(P.FLOOD_HOLD_S, 0)


class SegmentTest(unittest.TestCase):
    def steps(self, pairs, per=10):
        """pairs: [(flooded, ratio)] mỗi phần tử lặp `per` lần suy luận (khung hình cách 5)."""
        out = []
        for flooded, ratio in pairs:
            out += [{"scene": 0, "water_ratio": ratio, "lum": 100, "flooded": flooded}] * per
        return [{**s, "idx": i * 5} for i, s in enumerate(out)]

    def test_evidence_text(self):
        self.assertEqual(P.evidence_text(True, 0.85), "Nước phủ 85% mặt đường")
        self.assertEqual(P.evidence_text(False, 0.12), "Nước ít (12%), chưa ngập")
        self.assertEqual(P.evidence_text(False, 0.01), "Không thấy nước")

    def test_majority(self):
        s = [{"flooded": f, "water_ratio": 0.5} for f in (True, True, False, False)]
        self.assertTrue(P.segment_info(s)[0])                  # đúng 50% -> NGẬP
        s[1]["flooded"] = False
        self.assertFalse(P.segment_info(s)[0])

    def test_segments_split_and_ratio(self):
        steps = self.steps([(False, 0.1), (True, 0.8), (True, 0.9)])    # mỗi cụm 10 lần x 5 khung = 50 khung
        segs = P.segments_from_steps(steps, 150, 50)
        self.assertEqual([s["flooded"] for s in segs], [False, True, True])
        self.assertEqual(segs[1]["evidence"], "Nước phủ 80% mặt đường")
        self.assertEqual(segs[0]["evidence"], "Nước ít (10%), chưa ngập")

    def test_last_partial_segment_and_gap(self):
        steps = [{"idx": 0, "scene": 0, "water_ratio": 0.9, "lum": 100, "flooded": True}]
        segs = P.segments_from_steps(steps, 130, 50)           # đoạn không có lần suy luận: lấy lần gần nhất trước đó
        self.assertEqual(len(segs), 3)
        self.assertTrue(all(s["flooded"] for s in segs))

    def test_run_rule_marks_steps(self):
        steps = [{"idx": i * 5, "scene": 0, "water_ratio": 0.9, "lum": 100} for i in range(10)]
        out = P.run_rule(steps, 25.0, P.FloodRule(on=0.5, off=0.3, hold_s=0.4, skip_dark=True))
        self.assertEqual([s["flooded"] for s in out], [False, False, True] + [True] * 7)


class SmootherTest(unittest.TestCase):
    def test_hysteresis(self):
        s = P.WaterSmoother(0.6)
        m = np.ones((10, 10), bool)
        for _ in range(3):
            out = s.update(m)
        self.assertTrue(out.all())
        self.assertTrue(s.update(np.zeros((10, 10), bool)).all())   # một lần sót không làm mất vùng nước


class GroundCacheTest(unittest.TestCase):
    def test_cadence_and_cut(self):
        class Fake:
            ground = True
            calls = 0

            def scene_masks(self, f):
                self.calls += 1
                return np.ones((4, 4), bool), np.zeros((4, 4), bool)
        f = Fake()
        c = P.GroundCache(f, every=5)
        for _ in range(10):
            c.update(None)
        self.assertEqual(f.calls, 2)
        c.update(None, cut=True)
        self.assertEqual(f.calls, 3)
        self.assertEqual(c.veto.shape, (4, 4))
        f.ground = None
        self.assertIsNone(c.update(None))


class FlagTest(unittest.TestCase):
    def test_flags_off(self):
        old = {k: os.environ.get(k) for k in ("USE_GROUND", "USE_WATER_VETO", "FLOOD_SKIP_DARK")}
        try:
            for k in old:
                os.environ[k] = "0"
            m = importlib.reload(P)
            self.assertFalse(any((m.USE_GROUND, m.USE_WATER_VETO, m.FLOOD_SKIP_DARK)))
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            importlib.reload(P)

    def test_env_thresholds(self):
        old = {k: os.environ.get(k) for k in ("FLOOD_ON", "FLOOD_OFF", "FLOOD_HOLD_S")}
        try:
            os.environ.update(FLOOD_ON="0.7", FLOOD_OFF="0.6", FLOOD_HOLD_S="3")
            m = importlib.reload(P)
            self.assertEqual((m.FLOOD_ON, m.FLOOD_OFF, m.FLOOD_HOLD_S), (0.7, 0.6, 3.0))
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            importlib.reload(P)


if __name__ == "__main__":
    unittest.main()
