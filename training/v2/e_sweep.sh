B=/home/tuananh/Documents/flood_segmentation/flood-segmentation/backend
cd /home/tuananh/Documents/flood_segmentation/work/v2
run() { n=$1; shift; [ -f sweep_$n.json ] || ../.venv/bin/python -I scripts/e_run.py $B sweep_$n.json "$@" > sweep_$n.log 2>&1; }
run dark DARK_LUM=60
run c20 OBJ_CONF_2W=0.2
run c15 OBJ_CONF_2W=0.15
run dark_c20 DARK_LUM=60 OBJ_CONF_2W=0.2
run dark_c20_960 DARK_LUM=60 OBJ_CONF_2W=0.2 DETECT_IMGSZ=960
touch sweep_done
