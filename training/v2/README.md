# Training v2

- `a_build_water_ds.py`: dựng dataset nước 1 class (chỉ mask người gán; ảnh âm ~15%); `a_make_notebook.py`: notebook Kaggle (yolo11s-seg, imgsz 640, 100 epoch, patience 30, batch 16) và đánh giá IoU test_flood → `water_eval_summary.json`, `water_results.csv`.
- `b_*.py`: dựng, huấn luyện, đánh giá `depth_cls` (STURM + crop tự gán, `depth_cls_own_labels.csv`) → `depth_cls_eval.json`, `depth_cls_eval_up.json` (trước/sau ngưỡng 0.35).
- `e_run.py` / `e_score.py`: chạy pipeline trên clip VN và chấm với `vn_test_labels.csv`; `e_time.py`: đo ms/lần suy luận, VRAM.
Đường dẫn trong script là đường dẫn máy tác giả; dữ liệu, crop và video không đưa vào repo.
