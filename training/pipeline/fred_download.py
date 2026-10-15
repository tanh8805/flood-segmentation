"""Download only the needed FRED frames from the public QUT zips via HTTP Range (no full-zip download), in parallel
(the server throttles each connection to ~100 KB/s).
flooded: every STRIDE-th frame that has a human label (front-imgs + front-labels).
dry:     N_DRY frames spread across the 9 dry sequences (no labels exist for dry).
Resumable: skips files already on disk."""
import sys, os, json, zipfile
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(__file__))
from remote_zip import HTTPRangeFile
from fred_list import url

ROOT = os.path.join(os.environ.get("FLOOD_WORK", os.path.abspath("work")), "external", "fred")
STRIDE, N_DRY, WORKERS, CHUNK = 5, 180, 10, 8
idx = json.load(open(f"{ROOT}/index.json"))
dry_keys = [k for k in idx if k.startswith("dry/")]
per_dry = N_DRY // len(dry_keys)
jobs = []
for k, v in idx.items():
    flooded = k.startswith("flooded/")
    if flooded:
        labs = {os.path.basename(n): n for n in v["front_labels"]}
        pairs = [(n, labs[os.path.basename(n)]) for n in v["front_imgs"] if os.path.basename(n) in labs][::STRIDE]
    else:
        imgs = v["front_imgs"]; step = max(1, len(imgs) // per_dry)
        pairs = [(n, None) for n in imgs[::step][:per_dry]]
    od = f"{ROOT}/{'flooded' if flooded else 'dry'}/{k.split('/')[1]}"
    for sub in ("img", "lab"): os.makedirs(f"{od}/{sub}", exist_ok=True)
    for i in range(0, len(pairs), CHUNK): jobs.append((k, od, pairs[i:i + CHUNK]))

def run(job):
    k, od, pairs = job
    todo = [(n, s) for ni, nl in pairs for n, s in ((ni, "img"), (nl, "lab")) if n and not os.path.exists(f"{od}/{s}/{os.path.basename(n)}")]
    if not todo: return 0
    zf = zipfile.ZipFile(HTTPRangeFile(url(k)))
    for n, s in todo:
        dst = f"{od}/{s}/{os.path.basename(n)}"
        with open(dst + ".part", "wb") as f: f.write(zf.read(n))
        os.replace(dst + ".part", dst)
    return len(todo)

done = 0
with ThreadPoolExecutor(WORKERS) as ex:
    for r in ex.map(run, jobs):
        done += r; print("files so far", done, flush=True)
print("DONE jobs", len(jobs))
