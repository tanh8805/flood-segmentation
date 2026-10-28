"""Tai tu HF dgural/bdd100k (BSD, 10k anh val BDD100K) cac anh mua / dem pho lam am tinh kho."""
import glob
import json
import random
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
out = ROOT / "data/bdd"
out.mkdir(parents=True, exist_ok=True)
p = glob.glob(str(Path.home() / ".cache/huggingface/hub/datasets--dgural--bdd100k/snapshots/*/samples.json"))[0]
s = json.load(open(p))["samples"]
g = lambda x, k: (x.get(k) or {}).get("label")
random.seed(0)
pick = []
rainy = [x for x in s if g(x, "weather") == "rainy"]
pick += [(x, "rain") for x in rainy if g(x, "scene") in ("city street", "residential")]
pick += [(x, "rain") for x in random.sample([x for x in rainy if g(x, "scene") == "highway"], 60)]
night = [x for x in s if g(x, "timeofday") == "night" and g(x, "weather") != "rainy" and g(x, "scene") in ("city street", "residential")]
pick += [(x, "night") for x in random.sample(night, 150)]
day = [x for x in s if g(x, "timeofday") == "daytime" and g(x, "weather") in ("clear", "overcast") and g(x, "scene") in ("city street", "residential")]
pick += [(x, "dry") for x in random.sample(day, 100)]
meta = {}


def get(item):
    x, kind = item
    fp = x["filepath"]
    dst = out / Path(fp).name
    for _ in range(15):
        if dst.exists():
            break
        r = subprocess.run(["curl", "-4", "-sfL", "--connect-timeout", "4", "-m", "25", "-o", str(dst) + ".part", f"https://huggingface.co/datasets/dgural/bdd100k/resolve/main/{fp}"])
        if r.returncode == 0:
            Path(str(dst) + ".part").rename(dst)
        else:
            time.sleep(0.5)
    if dst.exists():
        meta[dst.stem] = {"kind": kind, "weather": g(x, "weather"), "tod": g(x, "timeofday"), "scene": g(x, "scene")}


with ThreadPoolExecutor(8) as ex:
    list(ex.map(get, pick))
(out / "meta.json").write_text(json.dumps(meta))
print(len(meta))
