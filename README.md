# Phát hiện NGẬP / KHÔNG NGẬP theo thời gian từ video camera đường phố

Upload video → mỗi đoạn 2 s được gán NGẬP / KHÔNG NGẬP, kèm xác suất bộ phân loại `p`, độ phủ nước `water_ratio` và căn cứ tiếng Việt
(ví dụ "Bộ phân loại 0.93, nước phủ 85% mặt đường"). NGẬP = nước tràn mặt đường, không còn lối đi khô. KHÔNG NGẬP = đường khô, ướt sau mưa,
đang mưa, vũng nhỏ, phản chiếu đèn, bóng cây trên mặt nhựa.

## Cơ chế
1. **Bộ phân loại khung hình** (`floodapp/classifier.py`): CLIP ViT-B-32 đóng băng (ảnh ép 224×224, không cắt) + hồi quy logistic → `p`.
2. **Model nước** (`weights/water.pt`, YOLO seg 1 lớp) + SegFormer-B2 (mặt đất, veto nước trên trời/nhà/cây…) → vùng nước vẽ lên video và `water_ratio`.
3. **Luật theo thời gian** (`floodapp/rules.py`, tham số ở `weights/rule.json`): p và water_ratio trung bình trượt trong cùng một cảnh, ngưỡng bật/giữ có trễ;
   đoạn 2 s NGẬP khi ≥ 50% lần suy luận (5/giây) ở trạng thái NGẬP.
Luật chọn: `combo`, W = 2 s: bật khi p ≥ 0.9 hoặc (p ≥ 0.2 và nước ≥ 0.3); giữ với ngưỡng thấp hơn 0.15 / 0.1.

## Dữ liệu
Xem `data_card.md`: 3787 ảnh (1387 ngập, 2400 không ngập, 1011 âm tính khó: mưa, đêm phản chiếu, bóng cây) từ Deepflood, Sazara, flooding_pixabay, FRED,
ADE20K, COCO, BDD100K. Không dùng khung nào của clip DEV/TEST; không có Pexels (không có key). Dataset dẫn xuất để private (có nguồn CC BY-NC-SA).
Bộ phân loại trên ảnh val (chia theo nhóm): F1 0.977; tỉ lệ báo nhầm nhóm âm khó 0.

## Đánh giá
- Chia theo clip (`splits.json`): **DEV** 7 clip (4 ngập, 3 không ngập), 96 đoạn có nhãn; **TEST** 6 clip (4 ngập, 2 không ngập), 102 đoạn (77 ngập, 25 không ngập).
  Không đủ clip không ngập chưa dùng để TEST có ≥ 4 clip không ngập. `ngap_8` trùng nội dung `ngap_6` nên đã xoá.
- Nhãn từng đoạn 2 s xem `review/labels.csv` (người duyệt sửa 3 clip). Đoạn `bo_qua` (logo, màn đen, cận cảnh…) không tính.
- Chọn luật chỉ trên DEV bằng mô phỏng trên cache tín hiệu (294 tổ hợp). Tiêu chí định trước: F1 cao nhất với recall ≥ 0.90 và tỉ lệ báo nhầm ≤ 0.10;
  chênh F1 ≤ 0.01 thì chọn luật đơn giản hơn rồi bảo thủ hơn. Lần chạy đầu (lưới 222 tổ hợp, combo b ≥ 0.4) không luật nào đạt do clip `ngap_4`
  (gara ngập trong nhà, p thấp nhưng model nước thấy nước), nên mở rộng lưới combo (b từ 0.2) trên DEV; TEST chưa được dùng.

| DEV (96 đoạn) | Precision | Recall | F1 | Báo nhầm |
|---|---|---|---|---|
| v1 (mức ≥ 1) | 0.933 | 0.966 | 0.949 | 0.105 |
| v3 | 0.897 | 0.897 | 0.897 | 0.158 |
| Mới, tối ưu trên DEV | 0.947 | 0.931 | 0.939 | 0.079 |
| Mới, leave-one-clip-out | 0.706 | 0.828 | 0.762 | 0.526 |

Leave-one-clip-out kém vì mỗi clip DEV là một miền riêng; luật chọn khi giữ `khong_ngap_01` ra ngoài báo nhầm toàn bộ clip đó.

**TEST** (chấm một lần, luật đã đóng băng; 77 ngập / 25 không ngập; ma trận TP/FP/TN/FN):

| | Precision | Recall | F1 | Báo nhầm | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|
| v1 | 0.747 | 0.922 | 0.826 | 0.96 | 71 | 24 | 1 | 6 |
| v3 | 0.753 | 0.948 | 0.839 | 0.96 | 73 | 24 | 1 | 4 |
| Mới | 0.760 | 0.987 | 0.859 | 0.96 | 76 | 24 | 1 | 1 |

Theo từng clip: xem `results/test_eval.json`. **Cả ba bản đều báo nhầm gần như toàn bộ 2 clip không ngập của TEST** (đường nhựa ướt ban đêm phản chiếu đèn):
bản mới không cải thiện báo nhầm so với v1/v3 trên TEST, chỉ tăng recall nhẹ.

### Lỗi còn lại và nguyên nhân
- `khong_ngap_04` (21 đoạn), `khong_ngap_05` (3 đoạn): model nước cho độ phủ 93–99% trên mặt đường ướt phản chiếu đèn ban đêm; nhánh "p ≥ 0.2 và nước ≥ 0.3"
  của luật kết hợp bật NGẬP dù `p` của bộ phân loại thường chỉ 0.1–0.5. Luật chọn trên DEV không có clip đêm phản chiếu nên không phát hiện rủi ro này.
- `ngap_6`, đoạn 9 (18–20 s): cận cảnh xe trong bãi ngập, p = 0.26 và không có vùng nước → bỏ sót.
- Hạn chế khác: DEV rất nhỏ (96 đoạn, 7 clip); `ngap_4` là cảnh trong nhà nằm ngoài miền huấn luyện; TEST chỉ có 2 clip không ngập; `ngap_6` là video quảng cáo.

## Tốc độ (RTX 3050 4 GB, mỗi lần suy luận = 1 khung, 5/giây)
GPU ~40–100 ms/lần (có ghi video kết quả ~100); CPU ~660–715 ms/lần.

## Kiểm tra
- App thật vs mô phỏng trên cache, 2 clip DEV (`ngap_4`, `khong_ngap_01`): khớp 27/27 đoạn (`src/check_app.py`).
- Unit test luật và đọc video: `python -m pytest floodapp/tests`.

## Chạy
```
python -m venv venv && . venv/bin/activate
pip install -r floodapp/requirements.txt
python floodapp/download_weights.py        # CLIP ViT-B-32, SegFormer-B2
uvicorn floodapp.app:app --port 8000       # mở http://localhost:8000
python -m floodapp.analyzer video.mp4 ket_qua.mp4
```
Video OpenCV không đọc được sẽ được đổi sang H.264 bằng ffmpeg của imageio-ffmpeg. `src/` chứa script dựng dữ liệu, cache tín hiệu, huấn luyện và mô phỏng.
