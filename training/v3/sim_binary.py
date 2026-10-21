"""Áp luật nhị phân lên cache tín hiệu (không chạy model), chọn ngưỡng trên clip cũ + FRED khô.
Dùng: sim_binary.py  -> work/v3/sim_results.json + bảng in ra.  Quy tắc chọn: xem pick()."""
import csv
import itertools
import json
import sys
from pathlib import Path

ROOT = Path("/home/tuananh/Documents/flood_segmentation")
V3 = ROOT / "work/v3"
sys.path.insert(0, str(ROOT / "flood-segmentation/backend"))
import pipeline as P  # noqa: E402

T_ON = (0.15, 0.25, 0.35, 0.5, 0.65)
GAP = (0.05, 0.10)
HOLD = (1, 2, 3, 4)
SKIP = (False, True)
MIN_RECALL = 0.90


def load_cache(clip):
    steps = [json.loads(l) for l in open(V3 / "cache" / f"{clip}.jsonl", encoding="utf-8")]
    meta = json.loads((V3 / "cache" / f"{clip}.meta.json").read_text())
    return steps, meta


def load_labels(path):
    """-> {clip: {t_s: 0/1}} (bỏ nhãn x)."""
    out = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r["label"] in ("0", "1"):
            out.setdefault(r["clip"], {})[int(r["t_s"])] = int(r["label"])
    return out


def predict(clip, cfg, cache):
    """cfg = (on, off, hold_s, skip_dark) -> {t_s: ngập?} theo đoạn 2 s."""
    steps, meta = cache[clip]
    rule = P.FloodRule(on=cfg[0], off=cfg[1], hold_s=cfg[2], skip_dark=cfg[3])
    steps = P.run_rule(steps, meta["fps"], rule)
    segs = P.segments_from_steps(steps, meta["total_frames"], meta["seg_len"])
    return {round(s["start_idx"] / meta["fps"] / 2) * 2: s["flooded"] for s in segs}


def counts(pred, labels):
    tp = fp = fn = tn = 0
    for t, y in labels.items():
        p = pred.get(t)
        if p is None:
            continue
        tp, fp, fn, tn = tp + (p and y), fp + (p and not y), fn + (not p and y), tn + (not p and not y)
    return int(tp), int(fp), int(fn), int(tn)


def metrics(c):
    tp, fp, fn, tn = c
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": round(prec, 3), "recall": round(rec, 3),
            "f1": round(2 * prec * rec / (prec + rec), 3) if prec + rec else 0.0,
            "fpr": round(fp / (fp + tn), 3) if fp + tn else 0.0}


def grid():
    return [(on, round(on - g, 2), h, s) for on, g, h, s in itertools.product(T_ON, GAP, HOLD, SKIP)]


def pick(cfgs, counts_by_cfg):
    """Quy tắc bảo thủ: trong các cấu hình đạt recall >= 0.90 và báo nhầm = 0 trên dev, chọn T_on CAO nhất, hoà thì H dài nhất,
    rồi F1 cao hơn, rồi không bỏ khung tối, rồi T_off cao hơn. Không có cấu hình nào thì: recall >= 0.90 với báo nhầm thấp nhất
    (hoà: F1 cao, đơn giản); vẫn không có thì recall cao nhất."""
    m = {c: metrics(counts_by_cfg[c]) for c in cfgs}
    safe = [c for c in cfgs if m[c]["recall"] >= MIN_RECALL and m[c]["fpr"] == 0]
    if safe:
        return min(safe, key=lambda c: (-c[0], -c[2], -m[c]["f1"], c[3], -c[1]))
    ok = [c for c in cfgs if m[c]["recall"] >= MIN_RECALL] or [c for c in cfgs if m[c]["recall"] == max(x["recall"] for x in m.values())]
    return min(ok, key=lambda c: (m[c]["fpr"], -m[c]["f1"], c[3], c[2], -(c[0] - c[1])))


def main():
    labels = load_labels(V3 / "labels_dev.csv")
    cache = {c: load_cache(c) for c in labels}
    cfgs = grid()
    per = {c: {cl: counts(predict(cl, c, cache), labels[cl]) for cl in labels} for c in cfgs}
    tot = {c: tuple(map(sum, zip(*per[c].values()))) for c in cfgs}
    best = pick(cfgs, tot)
    res = {"best": best, "best_all_dev": metrics(tot[best]), "n_dev_segments": sum(len(v) for v in labels.values()),
           "grid": {str(c): metrics(tot[c]) for c in cfgs}}
    # leave-one-clip-out: chọn cấu hình trên các clip còn lại, chấm trên clip bị bỏ, gộp lại
    loco, picks = [0, 0, 0, 0], {}
    for hold in labels:
        sub = {c: tuple(map(sum, zip(*[v for k, v in per[c].items() if k != hold]))) for c in cfgs}
        b = pick(cfgs, sub)
        picks[hold] = b
        loco = [a + x for a, x in zip(loco, per[b][hold])]
    res["loco"] = metrics(tuple(loco))
    res["loco_picks"] = {k: list(v) for k, v in picks.items()}
    res["per_clip_best"] = {cl: metrics(per[best][cl]) for cl in labels}
    # v1 / v2 (mức >= 1 = ngập) trên cùng các đoạn
    for ver in ("v1", "v2"):
        f = V3 / "baselines" / f"{ver}.json"
        if f.exists():
            b = json.loads(f.read_text())
            pred = {cl: {round(s["t"] / 2) * 2: s["level"] >= 1 for s in b[cl]} for cl in labels if cl in b}
            res[ver] = metrics(tuple(map(sum, zip(*[counts(pred[cl], labels[cl]) for cl in pred]))))
    (V3 / "sim_results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print("best (on, off, H, skip_dark):", best, res["best_all_dev"])
    print("LOCO:", res["loco"])
    for ver in ("v1", "v2"):
        if ver in res:
            print(ver, res[ver])
    top = sorted(cfgs, key=lambda c: (metrics(tot[c])["recall"] < MIN_RECALL, metrics(tot[c])["fpr"], -metrics(tot[c])["f1"]))[:8]
    for c in top:
        print(c, metrics(tot[c]))


if __name__ == "__main__":
    main()
