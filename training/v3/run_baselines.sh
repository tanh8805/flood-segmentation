#!/bin/bash
cd /home/tuananh/Documents/flood_segmentation/work/v3
PY=../.venv/bin/python
$PY -I scripts/run_baseline.py v1 ../tmp/wt_main/backend > baselines_v1.log 2>&1
$PY -I scripts/run_baseline.py v2 ../tmp/wt_v2/backend > baselines_v2.log 2>&1
touch baselines_done
