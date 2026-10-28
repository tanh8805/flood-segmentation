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
random.seed(1)
have = set(p.stem for p in out.glob("*.jpg"))
pick = []
rainy = [x for x in s if g(x, "weather") == "rainy" and Path(x["filepath"]).stem not in have]
pick += [(x, "rain") for x in rainy if g(x, "scene") in ("city street", "residential", "highway")]
night = [x for x in s if g(x, "timeofday") == "night" and g(x, "weather") != "rainy" and g(x, "scene") in ("city street", "residential") and Path(x["filepath"]).stem not in have]
pick += [(x, "night") for x in random.sample(night, 400)]
meta = json.loads((out / "meta.json").read_text())


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
