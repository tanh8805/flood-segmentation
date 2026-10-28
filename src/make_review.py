"""Xuat review/: contact sheet moi clip (o = 1 doan 2 s, ghi giay) + labels.csv de duyet."""
import csv
import math
import sys

import cv2

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import ROOT, SEG, clip_path, grab_at, n_segments, splits, video_info

MAXW = MAXH = 1280
PER_PAGE = 20
COL = {"ngap": (60, 60, 230), "khong_ngap": (60, 180, 60), "bo_qua": (150, 150, 150)}


def load_labels():
    p = ROOT / "review" / "labels.csv"
    if not p.exists():
        return {}
    with p.open() as f:
        return {(r["clip"], int(r["seg"])): (r["label"], r["note"]) for r in csv.DictReader(f)}


def main():
    out = ROOT / "review"
    out.mkdir(exist_ok=True)
    prev = load_labels()
    rows = []
    for clip, part in sorted(splits().items()):
        path = clip_path(clip)
        info = video_info(path)
        if info is None:
            print("KHONG DOC DUOC", clip)
            continue
        n = n_segments(info["duration"])
        times = [min(i * SEG + SEG / 2, info["duration"] - 0.05) for i in range(n)]
        frames = grab_at(path, times)
        default = "ngap" if clip.startswith("ngap") else "khong_ngap"
        labels = []
        for i, t in enumerate(times):
            lab, note = prev.get((clip, i), (default, ""))
            labels.append(lab)
            rows.append([clip, part, i, i * SEG, min((i + 1) * SEG, info["duration"]), lab, note])
        h0, w0 = next(iter(frames.values())).shape[:2]
        pages = math.ceil(n / PER_PAGE)
        for pg in range(pages):
            idxs = list(range(pg * PER_PAGE, min(n, (pg + 1) * PER_PAGE)))
            best = None
            for cols in range(1, len(idxs) + 1):
                rws = math.ceil(len(idxs) / cols)
                cw = MAXW // cols
                ch = int(cw * h0 / w0)
                if ch * rws > MAXH:
                    ch = MAXH // rws
                    cw = int(ch * w0 / h0)
                if best is None or cw > best[0]:
                    best = (cw, ch, cols, rws)
            cw, ch, cols, rws = best
            sheet = 255 * __import__("numpy").ones((ch * rws, cw * cols, 3), "uint8")
            for k, i in enumerate(idxs):
                fr = frames.get(times[i])
                if fr is None:
                    continue
                cell = cv2.resize(fr, (cw, ch))
                cv2.rectangle(cell, (0, 0), (cw - 1, ch - 1), COL[labels[i]], 4)
                txt = f"{i}:{int(i * SEG)}s"
                cv2.putText(cell, txt, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
                cv2.putText(cell, txt, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                r, c = divmod(k, cols)
                sheet[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw] = cell
            name = f"{clip}.jpg" if pages == 1 else f"{clip}_p{pg + 1}.jpg"
            cv2.imwrite(str(out / name), sheet, [cv2.IMWRITE_JPEG_QUALITY, 82])
        print(clip, part, round(info["duration"], 1), "s", n, "doan", pages, "sheet")
    with (out / "labels.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip", "split", "seg", "t0", "t1", "label", "note"])
        w.writerows(rows)


main()
