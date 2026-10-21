# Training / đánh giá v3 (nhị phân NGẬP / KHÔNG NGẬP)

Cùng `water.pt` của v2; v3 chỉ đổi luật quyết định nên không train lại.

- `cache_signals.py`: chạy model một lần trên mọi clip, lưu tín hiệu từng lần suy luận (`cache/<clip>.jsonl`: t, scene, water_ratio, ground_ratio, water_frac, lum).
- `sim_binary.py`: áp luật lên cache (không chạy model), lưới T_on × T_off × H × bỏ khung tối, chọn cấu hình, leave-one-clip-out → `sim_results.json`.
- `run_baseline.py`, `run_baselines.sh`: chạy v1 (`main`) và v2 (nhánh `v2`) trên các clip, mức ≥ 1 = ngập.
- `app_vs_sim.py`: chạy app thật rồi so từng đoạn với mô phỏng. `final_eval.py`: chấm một lần bộ test mới → `final_eval.json`.
- `review_sheets.py`: contact sheet từng clip bộ test mới + nhãn mặc định theo thư mục.
- `labels_dev.csv` (clip VN cũ + FRED khô), `labels_new_test.csv` (bộ test mới, `x` = bỏ qua).

Đường dẫn trong script là đường dẫn máy tác giả; video, cache và sheet không đưa vào repo.
