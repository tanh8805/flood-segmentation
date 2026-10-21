"""Unit test luật người, luật cảnh báo, mặt đất, bật/tắt (không cần weights). Chạy: cd backend && python -m unittest discover tests"""
import importlib
import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pipeline as P

H = W = 200


def person_kp(ankle=180, knee=140, hip=100, x=100, conf=0.9, ankle_conf=None):
    kxy = np.zeros((17, 2), np.float32)
    kc = np.zeros(17, np.float32)
    for ids, y in (((11, 12), hip), ((13, 14), knee), ((15, 16), ankle)):
        for i in ids:
            kxy[i], kc[i] = (x, y), (conf if ids != (15, 16) or ankle_conf is None else ankle_conf)
    return kxy, kc


def water_from(top, bottom=200, x1=60, x2=140):
    m = np.zeros((H, W), bool)
    m[top:bottom, x1:x2] = True
    return m


BOX = np.array([70, 20, 130, 185])


class PersonRuleTest(unittest.TestCase):
    def depth(self, top, **kw):
        kxy, kc = person_kp(**kw)
        return P.person_depth(kxy, kc, BOX, water_from(top))

    def test_levels_by_water_edge(self):
        # mắt cá bị nước che (conf thấp) thì luật mép nước quyết định
        self.assertEqual(self.depth(175, ankle_conf=0.3, hip=100, knee=140), 1)   # ngang mắt cá
        self.assertEqual(self.depth(160, ankle_conf=0.3), 2)   # giữa ống chân
        self.assertEqual(self.depth(110, ankle_conf=0.3, conf=0.9), 2)            # gối nhìn rõ: không báo 3
        kxy, kc = person_kp(ankle_conf=0.3)
        kc[13:15] = 0.2   # gối cũng chìm, hông rõ: cả hai tín hiệu cùng chỉ bậc 3
        self.assertEqual(P.person_depth(kxy, kc, BOX, water_from(140)), 3)

    def test_ankle_visible_caps_at_one(self):
        self.assertEqual(self.depth(140), 1)   # mắt cá conf cao nhưng mép nước cao: lấy bậc thấp
        self.assertEqual(self.depth(175), 1)

    def test_ankle_hidden_knee_visible_is_one_or_two(self):
        self.assertEqual(self.depth(175, ankle_conf=0.2), 1)
        self.assertEqual(self.depth(160, ankle_conf=0.2), 2)

    def test_pick_depth_prefers_higher_level(self):
        self.assertEqual(P.pick_depth([0.1, 0.5, 0.36, 0.04]), 2)
        self.assertEqual(P.pick_depth([0.1, 0.55, 0.3, 0.05]), 1)
        self.assertEqual(P.pick_depth([0.1, 0.4, 0.2, 0.35]), 3)

    def test_no_water_at_feet_is_zero(self):
        kxy, kc = person_kp()
        m = np.zeros((H, W), bool)
        m[100:130, 60:140] = True   # nước ở xa, không nối liền đáy box
        self.assertEqual(P.person_depth(kxy, kc, BOX, m), 0)

    def test_needs_knee_or_hip(self):
        kxy, kc = person_kp()
        kc[11:15] = 0
        self.assertIsNone(P.person_depth(kxy, kc, BOX, water_from(150)))

    def test_hip_only_anchor_and_missing_ankle(self):
        kxy, kc = person_kp()
        kc[13:17] = 0   # chỉ còn hông; mắt cá lấy theo đáy box
        self.assertIn(P.person_depth(kxy, kc, BOX, water_from(120)), (2, 3))

    def test_water_behind_person_ignored_outside_leg_columns(self):
        kxy, kc = person_kp(x=100)
        m = np.zeros((H, W), bool)
        m[30:200, 0:20] = True
        m[175:200, 60:140] = True   # nước cao ở cạnh xa người
        self.assertEqual(P.person_depth(kxy, kc, BOX, m), 1)

    def test_person_inside_motorbike_box_is_skipped(self):
        self.assertTrue(P.center_inside(np.array([90, 50, 110, 120]), np.array([60, 40, 140, 190])))
        self.assertFalse(P.center_inside(np.array([0, 0, 20, 20]), np.array([60, 40, 140, 190])))


def obj(kind, level, h=100, cls="person"):
    return {"kind": kind, "cls": cls, "box": np.array([10, 10, 40, 10 + h]), "conf": .9, "level": level}


class DecisionTest(unittest.TestCase):
    def decide(self, frac, objs=(), ground=None):
        water = np.zeros((H, W), bool)
        water[: int(H * frac), :] = True  # frac = tỉ lệ khung hình
        return P.frame_decision(water, ground, list(objs), H, W)

    def ground(self, frac_ground):
        g = np.zeros((H, W), bool)
        g[H - int(H * frac_ground):, :] = True
        return g

    def test_no_water_is_zero(self):
        lv, ev, _ = self.decide(0.0, [obj("person", 3)])
        self.assertEqual((lv, ev), (0, P.NO_WATER_EVIDENCE))

    def test_level3_only_with_level3_object(self):
        self.assertEqual(self.decide(1.0)[0], 2)   # độ phủ kín nhưng không có vật: chỉ 2
        self.assertEqual(self.decide(0.2, [obj("person", 3)])[0], 3)
        self.assertEqual(self.decide(0.2, [obj("vehicle", 2, cls="car")])[0], 2)

    def test_coverage_ratio_uses_ground(self):
        lv, ev, st = self.decide(0.2, ground=self.ground(0.6))
        self.assertEqual(st["ratio_base"], "ground")
        self.assertEqual(lv, 2 if st["flood_ratio"] >= P.COVER_LEVEL2 else 1)
        self.assertEqual(ev, P.NO_REF_EVIDENCE)

    def test_ground_fallback_when_small(self):
        _, _, st = self.decide(0.03, ground=self.ground(0.01))   # nước + mặt đất < 5% khung hình
        self.assertEqual(st["ratio_base"], "frame")
        _, _, st = self.decide(0.5)                              # không có mặt đất (tắt/không nhận được), nước nhiều
        self.assertEqual(st["ratio_base"], "covered")

    def test_small_objects_ignored(self):
        self.assertEqual(self.decide(0.05, [obj("person", 3, h=P.PERSON_MIN_H - 1)])[0], 1)
        self.assertEqual(self.decide(0.05, [obj("vehicle", 3, h=P.VEHICLE_MIN_H - 1, cls="car")])[0], 1)
        self.assertEqual(self.decide(0.05, [obj("vehicle", 3, h=P.VEHICLE_MIN_H, cls="car")])[0], 3)

    def test_evidence_text(self):
        self.assertEqual(self.decide(0.2, [obj("person", 3)])[1], "Người lội nước tới gối")
        self.assertEqual(self.decide(0.2, [obj("vehicle", 2, cls="motorcycle")])[1], "Xe máy ngập quá nửa bánh")


class TrackerTest(unittest.TestCase):
    def test_level_needs_two_consecutive_inferences(self):
        tr = P.ObjectTracker()
        self.assertEqual(tr.update([obj("person", 3)])[0]["level"], 1)   # lần đầu: chỉ tính tới bậc 1
        self.assertEqual(tr.update([obj("person", 3)])[0]["level"], 3)
        self.assertEqual(tr.update([obj("person", 1)])[0]["level"], 1)   # tụt xuống thì theo min
        self.assertEqual(tr.update([obj("person", 3)])[0]["level"], 1)

    def test_miss_once_keeps_track_twice_drops(self):
        tr = P.ObjectTracker()
        tr.update([obj("person", 2)])
        tr.update([obj("person", 2)])
        self.assertEqual(len(tr.update([])), 1)
        self.assertEqual(len(tr.update([])), 0)

    def test_different_kind_not_merged(self):
        tr = P.ObjectTracker()
        tr.update([obj("person", 2)])
        out = tr.update([obj("vehicle", 2, cls="car")])
        self.assertEqual(sorted(o["kind"] for o in out), ["person", "vehicle"])


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

            def ground_mask(self, f):
                self.calls += 1
                return np.ones((4, 4), bool)
        f = Fake()
        c = P.GroundCache(f, every=5)
        for _ in range(10):
            c.update(None)
        self.assertEqual(f.calls, 2)
        c.update(None, cut=True)
        self.assertEqual(f.calls, 3)
        f.ground = None
        self.assertIsNone(c.update(None))


class FlagTest(unittest.TestCase):
    def test_flags_off(self):
        old = {k: os.environ.get(k) for k in ("USE_POSE", "USE_DETECT", "USE_DEPTH_CLS", "USE_GROUND")}
        try:
            for k in old:
                os.environ[k] = "0"
            m = importlib.reload(P)
            self.assertFalse(any((m.USE_POSE, m.USE_DETECT, m.USE_DEPTH_CLS, m.USE_GROUND)))
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            importlib.reload(P)


if __name__ == "__main__":
    unittest.main()
