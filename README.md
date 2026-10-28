# Phát hiện NGẬP / KHÔNG NGẬP theo thời gian từ video camera đường phố

Upload video → mỗi đoạn 2 s được gán NGẬP / KHÔNG NGẬP, kèm xác suất bộ phân loại `p`, độ phủ nước `water_ratio` và căn cứ tiếng Việt
(ví dụ "Bộ phân loại 0.93, nước phủ 85% mặt đường"). NGẬP = nước tràn mặt đường, không còn lối đi khô. KHÔNG NGẬP = đường khô, ướt sau mưa,
đang mưa, vũng nhỏ, phản chiếu đèn, bóng cây trên mặt nhựa.

## Phạm vi
Chỉ **camera đường phố ngoài trời**. Clip `ngap_4` (gara ngập trong nhà) bị loại khỏi mọi tập dữ liệu: nó nằm ngoài miền này và là hạn chế đã biết
(bộ phân loại cho p thấp ở cảnh trong nhà dù có nước).

## Cơ chế
1. **Bộ phân loại khung hình** (`floodapp/classifier.py`): CLIP ViT-B-32 đóng băng (ảnh ép 224×224, không cắt) + hồi quy logistic → `p`.
   Train trên ảnh (Deepflood, Sazara, FRED… + ảnh BDD100K đêm/mưa làm âm tính khó) và trên khung của 18 clip đã duyệt (3 khung mỗi đoạn 2 s, bỏ `bo_qua`).
2. **Model nước** (`weights/water.pt`, YOLO seg 1 lớp) + SegFormer-B2 (mặt đất, veto trời/nhà/cây…) → vùng nước vẽ lên video và `water_ratio` (chỉ để hiển thị, không tham gia quyết định).
3. **Luật theo thời gian** (`floodapp/rules.py`, tham số `weights/rule.json`): `p` trung bình trượt W = 2 s trong cùng một cảnh, trễ **bật khi p ≥ 0.8, tắt khi p < 0.6**;
   đoạn 2 s NGẬP khi ≥ 50% lần suy luận (5/giây) ở trạng thái NGẬP. Nhánh nước (p ≥ b và nước ≥ c) được thử trong lưới nhưng không được chọn.

## Dữ liệu
`data_card.md`: 4045 ảnh train (1387 ngập, 2658 không ngập, 1269 âm tính khó) + khung từ clip. 18 clip ngoài trời, 272 đoạn có nhãn (150 ngập, 122 không ngập):
12 clip `~/Videos/flood` và 6 clip mới `flood_v5` (2 ngập, 4 không ngập, nhiều cảnh đêm ướt phản chiếu). Nhãn từng đoạn: `review/labels.csv` (người duyệt;
5 đoạn có giọt nước che ống kính của `v5_khong_1` được đặt `bo_qua`). Không có Pexels (không có key). Dataset dẫn xuất để private (có nguồn CC BY-NC-SA).

## Đánh giá (out-of-fold, 5 fold chia theo clip)
- `folds.json`: mỗi fold có cả clip ngập và không ngập, 4 clip đêm ướt nằm ở 4 fold khác nhau. Mỗi fold train CLIP+logistic trên ảnh + khung của clip 4 fold còn lại,
  dự đoán clip fold bị giữ lại → có dự đoán out-of-fold cho **mọi** clip (`src/oof.py`).
- Chọn luật bằng mô phỏng trên dự đoán này (138 tổ hợp, `src/sim5.py`). Tiêu chí định trước: recall ≥ 0.90 → tỉ lệ báo nhầm thấp nhất → F1 (chênh ≤ 0.01 thì chọn luật đơn giản hơn, rồi bảo thủ hơn).
- **Ngưỡng được chọn trên chính dự đoán out-of-fold nên các số của "v5 chọn ngưỡng" hơi lạc quan.** Dòng "v5 ngưỡng mặc định" (W = 3 s, bật/tắt p ≥ 0.5, không chỉnh) cho thấy mức lạc quan đó.
- v1/v3/v4 chạy trên đúng các đoạn này (v4 = bộ phân loại ảnh-only + luật combo của bản trước, không thấy clip nào).

| 272 đoạn | Precision | Recall | F1 | Báo nhầm | TP | FP | TN | FN | Báo nhầm nhóm đêm ướt (54 đoạn) |
|---|---|---|---|---|---|---|---|---|---|
| v1 (mức ≥ 1) | 0.644 | 0.927 | 0.760 | 0.631 | 139 | 77 | 45 | 11 | 0.889 |
| v3 | 0.667 | 0.960 | 0.787 | 0.590 | 144 | 72 | 50 | 6 | 0.870 |
| v4 | 0.841 | 0.987 | 0.908 | 0.230 | 148 | 28 | 94 | 2 | 0.444 |
| v5 ngưỡng mặc định (W=3, p ≥ 0.5) | 0.909 | 0.993 | 0.949 | 0.123 | 149 | 15 | 107 | 1 | 0.056 |
| **v5 chọn ngưỡng (W=2, bật 0.8 / tắt 0.6)** | **0.980** | 0.960 | **0.970** | **0.025** | 144 | 3 | 119 | 6 | 0.000 |

Đoạn ngập ban đêm (độ sáng < 70, 53 đoạn): recall v5 chọn ngưỡng 0.925, mặc định 1.000, v4 0.981. Số theo từng clip: `results/v5_oof.json`.
Chọn ngưỡng đổi 12 báo nhầm lấy 5 bỏ sót thêm; chênh F1 giữa hai dòng v5 (0.021) là dấu hiệu mức lạc quan do chọn ngưỡng (không phải cận trên chặt).

### Lỗi còn lại (bản v5 chọn ngưỡng)
- Bỏ sót 6 đoạn ngập: `ngap_01` đoạn 5, 6, 32, 33 (cảnh ngập ban đêm, p trung bình đoạn 0.54–0.92, không vượt ngưỡng bật 0.8 liên tục), `ngap_6` đoạn 9 (cận cảnh xe, p 0.30), `v5_ngap_2` đoạn 1 (p 0.48).
- Báo nhầm 3 đoạn: `ngap_9` đoạn 0 (nhãn không ngập; đã xem lại: đầu time-lapse, nước mới lan tới, còn mảng nhựa khô lớn bên phải nên giữ nhãn; p 0.94), `khong_ngap_01` đoạn 0–1 (mưa, đường ướt ban ngày).
- Hạn chế: 18 clip / 272 đoạn là nhỏ, khoảng tin cậy rộng; các clip đêm ướt chiếm 54 đoạn không ngập; cảnh trong nhà không hỗ trợ; `ngap_6` là video quảng cáo.

## Tốc độ (RTX 3050 4 GB, mỗi lần suy luận = 1 khung, 5/giây)
GPU ~90–135 ms/lần (kèm ghi video kết quả, đo ở `src/check_app.py`); CPU ~660–715 ms/lần.

## Kiểm tra
- App thật vs mô phỏng trên cache với bộ phân loại cuối, 2 clip (`v5_khong_3`, `ngap_7`): khớp 21/21 đoạn (`src/check_app.py`).
- Giao diện: mở bằng trình duyệt headless (Chromium), upload 1 clip ngập + 1 clip đêm ướt, chụp trang kết quả ở 1280 px và 390 px: không lỗi console, không cuộn ngang; ô trạng thái hiển thị đúng Ngập / Không ngập.
- Unit test luật và đọc video: `python -m pytest floodapp/tests` (9 test).

## Chạy
```
python -m venv venv && . venv/bin/activate
pip install -r floodapp/requirements.txt
python floodapp/download_weights.py        # CLIP ViT-B-32, SegFormer-B2
uvicorn floodapp.app:app --port 8000       # mở http://localhost:8000
python -m floodapp.analyzer video.mp4 ket_qua.mp4
```
Video OpenCV không đọc được sẽ được đổi sang H.264 bằng ffmpeg của imageio-ffmpeg. `src/` chứa script dựng dữ liệu, cache tín hiệu, huấn luyện và mô phỏng.
