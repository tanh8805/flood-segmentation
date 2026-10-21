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
        self.assertEqual(self.decide(1.0)[0], 1)   # độ phủ kín nhưng không có vật: tối đa 1
        self.assertEqual(self.decide(0.2, [obj("person", 3)])[0], 3)
        self.assertEqual(self.decide(0.2, [obj("vehicle", 2, cls="car")])[0], 2)

    def test_coverage_ratio_uses_ground(self):
        lv, ev, st = self.decide(0.3, ground=self.ground(0.5))
        self.assertEqual(st["ratio_base"], "ground")
        self.assertEqual(lv, 1)
        self.assertEqual(ev, P.no_ref_evidence(st["flood_ratio"]))
        self.assertIn("không ước lượng được độ sâu (không có vật tham chiếu đủ gần)", ev)

    def test_unreadable_objects_do_not_count_or_justify_shallow(self):
        small = {**obj("vehicle", 0, cls="car"), "read": False}
        lv, ev, st = self.decide(1.0, [small])
        self.assertEqual(lv, 1)
        self.assertTrue(ev.startswith("Nước phủ") and "không ước lượng được độ sâu" in ev)
        self.assertNotIn("nông", ev)
        self.assertIsNone(st["max_object_level"])
        self.assertEqual(st["objects"], 0)
        unread_high = {**obj("person", 3), "read": False}   # bậc đọc không tin cậy không được nâng mức
        self.assertEqual(self.decide(0.2, [unread_high])[0], 1)

    def test_shallow_reference_keeps_level_one_despite_high_coverage(self):
        for o in (obj("person", 1), obj("vehicle", 1, cls="motorcycle")):
            lv, ev, _ = self.decide(1.0, [o])
            self.assertEqual(lv, 1)
            self.assertEqual(ev, P.object_text(o))

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


class BrightenTest(unittest.TestCase):
    def test_dark_frame_is_brightened_bright_frame_unchanged(self):
        rng = np.random.default_rng(0)
        dark = rng.integers(0, 30, (64, 64, 3), dtype=np.uint8)
        self.assertGreater(P.brighten(dark, dark_lum=60).mean(), dark.mean())
        light = rng.integers(120, 200, (64, 64, 3), dtype=np.uint8)
        self.assertIs(P.brighten(light, dark_lum=60), light)
        self.assertIs(P.brighten(dark, dark_lum=0), dark)   # tắt


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


class WeakTrackTest(unittest.TestCase):
    def weak(self):
        return {**obj("vehicle", 2, cls="motorcycle"), "conf": 0.2, "weak": True}

    def test_weak_detection_needs_three_consecutive_steps(self):
        tr = P.ObjectTracker()
        self.assertEqual(tr.update([self.weak()]), [])
        self.assertEqual(tr.update([self.weak()]), [])
        self.assertEqual(len(tr.update([self.weak()])), 1)
        self.assertEqual(len(tr.update([])), 1)   # đã xác nhận thì sót 1 lần vẫn giữ

    def test_weak_streak_resets_on_miss(self):
        tr = P.ObjectTracker()
        tr.update([self.weak()])
        tr.update([self.weak()])
        tr.update([])
        self.assertEqual(tr.update([self.weak()]), [])

    def test_strong_detection_confirms_track(self):
        tr = P.ObjectTracker()
        tr.update([obj("vehicle", 2, cls="motorcycle")])
        self.assertEqual(len(tr.update([self.weak()])), 1)


class DepthMemoryTest(unittest.TestCase):
    def mem(self):
        m = P.DepthMemory(seconds=10, keep=0.5)
        m.update(0.0, 0, [obj("vehicle", 3, cls="motorcycle")], 0.2)
        return m

    def test_remembers_within_window_with_evidence(self):
        m = self.mem()
        rem = m.update(5.0, 0, [], 0.15)
        self.assertEqual([(o["level"], o["age"]) for o in rem], [(3, 5.0)])
        water = np.zeros((H, W), bool)
        water[:60] = True
        lv, ev, _ = P.frame_decision(water, None, rem, H, W)
        self.assertEqual(lv, 3)
        self.assertEqual(ev, "Xe máy ngập hết bánh (đo 5 giây trước, cùng cảnh)")

    def test_expires_after_window(self):
        self.assertEqual(self.mem().update(10.5, 0, [], 0.2), [])

    def test_water_loss_drops_immediately_and_for_good(self):
        m = self.mem()
        self.assertEqual(m.update(2.0, 0, [], 0.05), [])   # < 50% độ phủ lúc đo
        self.assertEqual(m.update(3.0, 0, [], 0.2), [])    # nước quay lại cũng không khôi phục

    def test_scene_cut_drops(self):
        self.assertEqual(self.mem().update(2.0, 1, [], 0.2), [])

    def test_current_object_of_same_or_higher_level_wins(self):
        m = self.mem()
        self.assertEqual(m.update(2.0, 0, [obj("vehicle", 3, cls="motorcycle")], 0.2), [])
        self.assertEqual(len(m.update(3.0, 0, [obj("vehicle", 1, cls="motorcycle")], 0.2)), 1)

    def test_level_one_is_not_remembered(self):
        m = P.DepthMemory(seconds=10, keep=0.5)
        m.update(0.0, 0, [obj("person", 1)], 0.2)
        self.assertEqual(m.update(1.0, 0, [], 0.2), [])


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


class _T:
    def __init__(self, a):
        self.a = np.asarray(a)

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class _Det:
    def __init__(self, boxes, confs, cls):
        self.boxes = type("B", (), {"xyxy": _T(boxes), "conf": _T(confs), "cls": _T(cls)})()

    def predict(self, *a, **k):
        return [self]


class _Depth:
    def __init__(self, probs):
        self.probs = probs

    def predict(self, crops, **k):
        return [type("R", (), {"names": {0: "d0", 1: "d1", 2: "d2", 3: "d3"}, "probs": type("P", (), {"data": _T(self.probs)})()})() for _ in crops]


class ReadingRulesTest(unittest.TestCase):
    FH, FW = 800, 800

    def pipe(self, boxes, depth_probs):
        p = P.FloodPipeline.__new__(P.FloodPipeline)
        p.device, p.pose, p.water, p.ground = "cpu", None, None, None
        p.detect = _Det(boxes, [0.9] * len(boxes), [2] * len(boxes))
        p.depth = _Depth(depth_probs)
        return p

    def run_vehicles(self, box, probs):
        frame = np.zeros((self.FH, self.FW, 3), np.uint8)
        water = np.ones((self.FH, self.FW), bool)
        cars, _ = self.pipe([box], probs).vehicles(frame, water)
        return cars[0]

    def test_small_vehicle_is_drawn_but_not_read(self):
        c = self.run_vehicles([10, 100, 90, 100 + int(P.DEPTH_MIN_FRAC * self.FH) - 3], [0.0, 0.0, 0.0, 1.0])
        self.assertFalse(c["read"])
        self.assertEqual(c["level"], 0)

    def test_big_enough_vehicle_is_read(self):
        c = self.run_vehicles([10, 100, 200, 100 + int(P.DEPTH_MIN_FRAC * self.FH) + 10], [0.0, 0.1, 0.1, 0.8])
        self.assertEqual((c["level"], c.get("read", True)), (3, True))

    def test_level0_with_base_in_water_is_contradiction(self):
        c = self.run_vehicles([10, 100, 200, 250], [0.9, 0.05, 0.03, 0.02])
        self.assertFalse(c["read"])

    def test_dry_vehicle_level0_stays_readable(self):
        frame = np.zeros((self.FH, self.FW, 3), np.uint8)
        cars, _ = self.pipe([[10, 100, 200, 250]], [1, 0, 0, 0]).vehicles(frame, np.zeros((self.FH, self.FW), bool))
        self.assertEqual((cars[0]["level"], cars[0].get("read", True)), (0, True))

    def test_water_veto_removes_water_on_sky_and_buildings(self):
        p = P.FloodPipeline.__new__(P.FloodPipeline)
        p.pose = p.detect = p.depth = None
        wm = np.zeros((20, 20), bool)
        wm[:, :] = True
        p.water_mask = lambda f: wm.copy()
        veto = np.zeros((20, 20), bool)
        veto[:10] = True
        water, _ = p.infer(np.zeros((20, 20, 3), np.uint8), veto=veto)
        self.assertEqual(int(water.sum()), 200)
        self.assertFalse(water[:10].any())
        water, _ = p.infer(np.zeros((20, 20, 3), np.uint8))   # không veto: giữ nguyên
        self.assertTrue(water.all())


class FlagTest(unittest.TestCase):
    def test_flags_off(self):
        old = {k: os.environ.get(k) for k in ("USE_POSE", "USE_DETECT", "USE_DEPTH_CLS", "USE_GROUND", "USE_DEPTH_MEMORY")}
        try:
            for k in old:
                os.environ[k] = "0"
            m = importlib.reload(P)
            self.assertFalse(any((m.USE_POSE, m.USE_DETECT, m.USE_DEPTH_CLS, m.USE_GROUND, m.USE_DEPTH_MEMORY)))
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            importlib.reload(P)


if __name__ == "__main__":
    unittest.main()
