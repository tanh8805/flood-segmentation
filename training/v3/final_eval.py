"""Chấm MỘT lần bộ test mới: v1 / v2 / v3 (cấu hình chọn) / v3 cũ (0.15,0.05,1 s, chỉ tham khảo). Nhãn: review/labels.csv (x bỏ qua).
Ghi final_eval.json; v3 dùng cấu hình mặc định của app (pipeline.FLOOD_*), không chỉnh sau khi xem kết quả."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import sim_binary as S  # noqa: E402

V3 = S.V3
labels = S.load_labels(V3 / "review/labels.csv")
cache = {c: S.load_cache(c) for c in labels}
P = S.P
cfg_new = (P.FLOOD_ON, P.FLOOD_OFF, P.FLOOD_HOLD_S, P.FLOOD_SKIP_DARK)
cfg_old = (0.15, 0.05, 1.0, False)
preds = {"v3": {c: S.predict(c, cfg_new, cache) for c in labels}, "v3_cu_0.15/0.05/1s": {c: S.predict(c, cfg_old, cache) for c in labels}}
for ver in ("v1", "v2"):
    b = json.loads((V3 / "baselines" / f"{ver}.json").read_text())
    preds[ver] = {c: {round(s["t"] / 2) * 2: s["level"] >= 1 for s in b[c]} for c in labels}
res = {"cfg_v3": cfg_new, "n_segments": sum(len(v) for v in labels.values())}
for ver, p in preds.items():
    per = {c: S.counts(p[c], labels[c]) for c in labels}
    res[ver] = {"total": S.metrics(tuple(map(sum, zip(*per.values())))), "per_clip": {c: S.metrics(v) for c, v in per.items()},
                "wrong": {c: sorted(t for t, y in labels[c].items() if p[c].get(t) is not None and bool(p[c][t]) != bool(y)) for c in labels}}
(V3 / "final_eval.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
print("cfg v3", cfg_new, "n =", res["n_segments"])
for ver in preds:
    t = res[ver]["total"]
    print(f"{ver:20s} P={t['precision']} R={t['recall']} F1={t['f1']} FPR={t['fpr']}  TP={t['tp']} FP={t['fp']} FN={t['fn']} TN={t['tn']}")
    print("   sai:", {c: w for c, w in res[ver]["wrong"].items() if w})
    print("   fpr/clip:", {c: v["fpr"] for c, v in res[ver]["per_clip"].items() if c.startswith("khong")})
