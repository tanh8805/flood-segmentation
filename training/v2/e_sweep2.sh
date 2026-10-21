B=/home/tuananh/Documents/flood_segmentation/flood-segmentation/backend
cd /home/tuananh/Documents/flood_segmentation/work/v2
run() { n=$1; shift; [ -f sweep_$n.json ] || ../.venv/bin/python -I scripts/e_run.py $B sweep_$n.json "$@" > sweep_$n.log 2>&1; }
run all DARK_LUM=60 DARK_SCOPE=all
run all_c20 DARK_LUM=60 DARK_SCOPE=all OBJ_CONF_2W=0.2
run all_c15 DARK_LUM=60 DARK_SCOPE=all OBJ_CONF_2W=0.15
run all_c20_960 DARK_LUM=60 DARK_SCOPE=all OBJ_CONF_2W=0.2 DETECT_IMGSZ=960
touch sweep2_done
