"""Run one analyzer version over the labelled VN clips (first 32 s) -> JSON of segment levels. usage: e_run.py <backend_dir> <out.json> [env=val ...]"""
import sys, os, json, csv, collections
backend, out = sys.argv[1], sys.argv[2]
for kv in sys.argv[3:]:
    k, v = kv.split("=", 1); os.environ[k] = v
os.environ["MAX_SECONDS"] = "32"
sys.path.insert(0, backend)
import analyzer as A
get = getattr(A, "get_pipeline", None) or A.get_model; get()
clips = collections.OrderedDict()
for r in csv.DictReader(open("/home/tuananh/Documents/flood_segmentation/work/v2/vn_test_labels.csv")): clips[r["clip"]] = r["path"]
res = {}
for k, p in clips.items():
    r = A.process_video(p, "/home/tuananh/Documents/flood_segmentation/work/tmp/eval_out.mp4")
    res[k] = [dict(t=s["start_s"], level=s["raw_level"], evidence=s.get("evidence")) for s in r["segments"]]
json.dump(res, open(out, "w"), ensure_ascii=False)
print("done", {k: len(v) for k, v in res.items()})
