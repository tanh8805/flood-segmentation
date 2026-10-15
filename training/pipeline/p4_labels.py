"""Phase 4: turn cached teacher outputs into final instance labels + keep/drop decisions (no model runs; fast, re-runnable).
Output: work/final/labels/<id>.json  (instances with normalised polygons, class ids 0 flood,1 road,2 car_dry,3 car_shallow,4 car_deep)
        work/final/decisions.json (per id: status keep|drop|neg_pool|test, reason, counts)
Rules (user decisions): flood = V-FloodNet ∪ SegFormer-water (regions <0.5% dropped) unless a human mask exists; road = SegFormer road − flood;
street-context filter for ade/old/houston/uploads; car 20%-rule -> dry; STURM: central car human level + consistency checks."""
import json, os, sys, random, collections
import cv2, numpy as np

R = os.environ.get("FLOOD_WORK", os.path.abspath("work"))
OUT = f"{R}/final"; os.makedirs(f"{OUT}/labels", exist_ok=True)
P = dict(min_region=0.005, min_car=0.001, level_conf=0.5, seg_conf=0.35, ctx_ring=0.03, ctx_thr=0.15, touch_thr=0.03,
         ambiguous_car=0.005, sturm_other_car=0.25, per_level_cap=160, neg_frac=0.05)
CLS = dict(flood=0, road=1, car_dry=2, car_shallow=3, car_deep=4)
LEVEL_TO_CLS = {0: 2, 1: 3, 2: 3, 3: 4, 4: 4}

from transformers import AutoConfig
id2 = AutoConfig.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640").id2label
nm = lambda s: s.split(",")[0].strip().lower()
ids_of = lambda names: [int(i) for i, n in id2.items() if nm(n) in names]
ROAD, WATER = ids_of({"road"}), ids_of({"water", "sea", "river", "lake"})
CTX = ids_of({"road", "sidewalk", "building", "car", "person", "pole", "streetlight", "bus", "truck", "van", "fence", "signboard", "traffic light"})


def remove_small(m, frac):
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    out = np.zeros(m.shape, np.uint8); thr = frac * m.size
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= thr: out[lab == i] = 1
    return out


def union_flood(fn, water):
    u = (fn | water).astype(np.uint8)
    u = cv2.morphologyEx(u, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return remove_small(u, P["min_region"])


def touches(flood, box, thr=None):
    H, W = flood.shape; x1, y1, x2, y2 = box; w, h = x2 - x1, y2 - y1
    a, b = max(0, int(x1 - 0.05 * w)), min(W, int(x2 + 0.05 * w)); c, d = max(0, int(y2 - 0.2 * h)), min(H, int(y2 + 0.1 * h))
    reg = flood[c:d, a:b]
    return reg.size > 0 and reg.mean() >= (thr if thr is not None else P["touch_thr"])


def iou_box(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0


def context_ok(water_region, lab):
    """water region adjacent (ring around it) to road/sidewalk/building/car/person/pole etc."""
    H, W = water_region.shape; k = max(3, int(P["ctx_ring"] * min(H, W))) | 1
    ring = cv2.dilate(water_region.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(bool) & ~water_region.astype(bool)
    if ring.sum() == 0: return False, 0.0
    frac = float(np.isin(lab[ring], CTX).mean()); return frac >= P["ctx_thr"], frac


NONB = ids_of({"road", "sidewalk", "car", "bus", "truck", "van", "pole", "streetlight"})
def context_strict(region, lab):
    """ade/old/houston: water must touch road/sidewalk/car/pole (buildings alone = canal/harbour/pool -> reject);
    people+buildings also accepted (wading people in a street)."""
    H, W = region.shape; k = max(3, int(P["ctx_ring"] * min(H, W))) | 1
    ring = cv2.dilate(region.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(bool) & ~region.astype(bool)
    if ring.sum() == 0: return False, 0.0
    nb = float(np.isin(lab[ring], NONB).mean()); bl = float(np.isin(lab[ring], ids_of({"building"})).mean()); pe = float(np.isin(lab[ring], ids_of({"person"})).mean())
    return (nb >= 0.08 or (pe >= 0.10 and bl >= 0.10)), nb


def polys(mask, min_frac):
    """external contours of connected components >= min_frac of the image -> list of normalised flat polygons"""
    H, W = mask.shape; out = []
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cs:
        if cv2.contourArea(c) < min_frac * H * W: continue
        c = cv2.approxPolyDP(c, 1.2, True)[:, 0, :]
        if len(c) < 3: continue
        out.append([(float(x) / W, float(y) / H) for x, y in c])
    return out


def central_car(masks, boxes, H, W):
    best, bs = None, -1e9; cx, cy = W / 2, H / 2
    for i, (m, b) in enumerate(zip(masks, boxes)):
        inside = b[0] <= cx <= b[2] and b[1] <= cy <= b[3]
        d = np.hypot((b[0] + b[2]) / 2 - cx, (b[1] + b[3]) / 2 - cy) / np.hypot(W, H)
        s = m.mean() * (2.0 if inside else 1.0) - d
        if s > bs: best, bs = i, s
    return best


def process(c):
    """returns (instances, info dict). info['status'] in keep|drop|neg_pool"""
    cid, src = c["id"], c["source"]
    z = np.load(f"{R}/cache/{cid}.npz"); meta = json.load(open(f"{R}/cache/{cid}.json"))
    H, W = meta["hw"]; fn = z["fn"]; lab = z["sf"]; cm = z["car_masks"]
    gt = z["gt"] if "gt" in z.files else None
    road_sf = np.isin(lab, ROAD); water_sf = np.isin(lab, WATER)
    info = dict(id=cid, source=src, status="keep", reason="")
    sb, sc = np.array(meta["seg_boxes"]).reshape(-1, 4), np.array(meta["seg_conf"])

    # ---- flood / road ----
    if src in ("deepflood", "sazara", "pixabay"):
        flood = gt.astype(np.uint8); road = road_sf
    elif src == "fred_flooded":
        flood = (gt == 2).astype(np.uint8); road = (gt == 1)
    elif src == "fred_dry":
        flood = np.zeros((H, W), np.uint8); road = road_sf
    elif src == "coco":
        flood = np.zeros((H, W), np.uint8); road = road_sf
    elif src == "sturm":
        flood = union_flood(fn, water_sf); road = road_sf
    else:  # ade, old, houston, uploads  -> street-context filter
        union = union_flood(fn, water_sf)
        region = union.astype(bool) | (gt.astype(bool) if gt is not None else False)
        info["water_frac"] = float(region.mean())
        if region.mean() >= P["min_region"]:
            ok, frac = (context_ok if src == "uploads" else context_strict)(region, lab); info["ctx_frac"] = frac
            if ok and src == "ade":  # ADE real mask must agree with teachers (weird/partial masks rejected)
                tu = (fn | water_sf); agree = float((tu & gt.astype(bool)).sum() / max(1, (tu | gt.astype(bool)).sum()))
                info["agree"] = agree
                if agree < 0.3: info.update(status="drop", reason="ade_mask_disagrees"); return [], info
            if not ok:
                # only clear non-street water (lake/sea/river/pool: nothing street-like around it) becomes a negative; the rest is dropped
                if src != "uploads" and frac < 0.03: info.update(status="neg_pool", reason="water_not_street")
                else: info.update(status="drop", reason="ambiguous_context")
                return [], info
        elif src == "ade":
            info.update(status="drop", reason="ade_no_water"); return [], info
        flood = gt.astype(np.uint8) if gt is not None else union
        road = road_sf
    road = remove_small((road & ~flood.astype(bool)).astype(np.uint8), P["min_region"])
    flood = flood.astype(np.uint8)

    # ---- cars ----
    if src != "sturm" or True:
        pass
    inst = []
    cars = []  # (box, mask, cls)
    if src == "sturm":
        L = c["meta"]["level"]
        if len(sb) == 0: info.update(status="drop", reason="no_car_found"); return [], info
        k = central_car(cm, sb, H, W); box = sb[k]; area = cm[k].sum()
        if not (box[0] <= W / 2 <= box[2] and box[1] <= H / 2 <= box[3]): info.update(status="drop", reason="central_car_not_central"); return [], info
        if c["split"] != "test_car":
            others = [j for j in range(len(sb)) if j != k and sc[j] >= 0.4 and cm[j].sum() >= P["sturm_other_car"] * area]
            if others: info.update(status="drop", reason="second_car"); return [], info
            t = touches(flood, box)
            if L >= 1 and not t: info.update(status="drop", reason="level>=1_but_no_flood_at_car"); return [], info
            if L == 0 and (t or flood[int(box[1]):int(box[3]), int(box[0]):int(box[2])].mean() >= 0.005 or touches(flood, box, 0.005)): info.update(status="drop", reason="level0_but_flood_at_car"); return [], info
        cars.append((box, cm[k], LEVEL_TO_CLS[L]))
    elif src not in ("fred_flooded", "fred_dry") or True:
        dets = [d for d in meta["level_dets"]]
        for j in range(len(sb)):
            if sc[j] < P["seg_conf"]: continue
            frac = cm[j].mean()
            if frac < P["min_car"]: continue
            near = [d for d in dets if iou_box(d["box"], sb[j]) >= 0.5]
            best = max(near, key=lambda d: d["conf"]) if near else None
            if not touches(flood, sb[j]):
                cars.append((sb[j], cm[j], 2)); continue  # 20% rule: bottom not in flood -> dry
            if best is not None and best["conf"] >= P["level_conf"]:
                cars.append((sb[j], cm[j], LEVEL_TO_CLS[best["cls"]]))
            elif frac >= P["ambiguous_car"] and c["split"] != "test_flood":
                info.update(status="drop", reason="ambiguous_car_in_flood"); return [], info
    if cars:  # drop flood components mostly covered by a car (spurious blobs on windows/bodywork)
        allcar = np.any([m for _, m, _ in cars], 0)
        n, cl, st, _ = cv2.connectedComponentsWithStats(flood, 8)
        for q in range(1, n):
            comp = cl == q
            if (comp & allcar).sum() >= 0.5 * comp.sum(): flood[comp] = 0
    for box, m, k in cars:
        for p in polys(m, P["min_car"]): inst.append(dict(cls=k, poly=p))
    for p in polys(flood, P["min_region"]): inst.append(dict(cls=0, poly=p))
    for p in polys(road, P["min_region"]): inst.append(dict(cls=1, poly=p))
    info["n"] = dict(collections.Counter(i["cls"] for i in inst))
    return inst, info


if __name__ == "__main__":
    cands = json.load(open(f"{R}/manifest/candidates.json"))
    res = {}
    for i, c in enumerate(cands):
        if not os.path.exists(f"{R}/cache/{c['id']}.json"): continue
        inst, info = process(c); info["group"] = c["group"]; info["split"] = c["split"]
        res[c["id"]] = info
        json.dump(dict(id=c["id"], source=c["source"], group=c["group"], split=c["split"], instances=inst), open(f"{OUT}/labels/{c['id']}.json", "w"))
        if i % 500 == 0: print(i, len(cands), flush=True)
    json.dump(res, open(f"{OUT}/decisions.json", "w"))
    cnt = collections.Counter((v["source"], v["status"], v["reason"]) for v in res.values())
    for k, v in sorted(cnt.items()): print(k, v)
