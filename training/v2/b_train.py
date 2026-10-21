from ultralytics import YOLO
W = "/home/tuananh/Documents/flood_segmentation/work/v2"
YOLO("yolo11n-cls.pt").train(data=f"{W}/depth_cls", epochs=60, patience=15, imgsz=224, batch=64, device=0, project=f"{W}/runs", name="depth_cls", exist_ok=True,
                             workers=4, seed=0, plots=False, erasing=0.2, degrees=5, fliplr=0.5, verbose=False)
