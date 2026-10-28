import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from baselines import baseline_pred
from sim import counts, load_data, metrics
from simlib import load_model

part = sys.argv[1]
data = load_data(load_model(Path(sys.argv[2])), part)
clips = sorted(data)
for ver in ("v1", "v3"):
    pred = {c: baseline_pred(c, ver) for c in clips}
    print(ver, "tong", metrics(*counts(data, pred, clips)))
    for c in clips:
        if data[c]["lab"]:
            print("  ", c, metrics(*counts(data, pred, [c])))
