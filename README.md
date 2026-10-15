# Cảnh báo ngập từ video camera đường phố

Ứng dụng web nhận một video camera đường phố, tô vùng nước, mặt đường và xe trên video, rồi cho biết **mức cảnh báo ngập theo từng giai đoạn** (An toàn, Chú ý, Cảnh báo, Nguy hiểm). Chỉ chạy một model YOLO11n-seg; các model lớn chỉ dùng offline để gán nhãn dữ liệu huấn luyện.

Quy trình sử dụng: tải video lên → phân tích → xem kết quả (video đã tô mask, thước đo mực nước, biểu đồ diễn biến, bảng các giai đoạn).

## Cấu trúc

```
backend/            ứng dụng FastAPI + phân tích video
  app.py            API: /analyze, /jobs/{id}, /jobs/{id}/video, /health
  analyzer.py       suy luận, làm mượt theo thời gian, tính mức cảnh báo, vẽ video kết quả
  static/index.html giao diện
  tests/            unit test cho class_map, luật mức cảnh báo
  weights/best.pt   model YOLO11n-seg đã huấn luyện
training/           tái lập dataset và huấn luyện (xem training/README.md)
```

## Chạy ứng dụng

Yêu cầu Python 3.10+; có GPU thì ultralytics tự dùng, không thì chạy CPU (khoảng 1–2 lần thời lượng video).

```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

- **Trình duyệt**: mở http://localhost:8000, kéo thả hoặc chọn video, bấm **Phân tích**. Giao diện hiện tiến trình tải lên và phân tích, kết luận ở đầu trang, video kết quả cạnh thước đo mực nước, biểu đồ diễn biến theo thời gian (bấm để tua, phím ← → tua 2 giây) và bảng các giai đoạn.
- **Dòng lệnh, không cần server**: `python analyzer.py video.mp4 [ket_qua.mp4]` in kết quả JSON và tạo video đã tô mask.
- **API** (xem thêm http://localhost:8000/docs):
  ```bash
  curl -F "file=@video.mp4" http://localhost:8000/analyze     # -> {"job_id": "...", "status_url": "..."}
  curl http://localhost:8000/jobs/<job_id>                    # status: queued/running/done/error, progress, result
  curl -o ket_qua.mp4 http://localhost:8000/jobs/<job_id>/video
  ```
  Số liệu từng khung hình nằm ở `backend/outputs/<job_id>/result.json`.
- **Test**: `cd backend && python -m unittest tests.test_analyzer`.

Job lưu trong bộ nhớ (tắt server là mất danh sách), video được xử lý lần lượt từng cái (`max_workers=1`; biến `DEPTH_MODEL` trong `analyzer.py` là biến toàn cục nên chỉ an toàn khi xử lý một video một lúc).

## Class của model

| id | class | ý nghĩa |
|---|---|---|
| 0 | `flood` | vùng nước trên mặt đường |
| 1 | `road` | mặt đường |
| 2 | `car_dry` | xe không ngập |
| 3 | `car_shallow` | xe ngập nông (mức 1–2 của STURM: nước tới bánh xe, gầm xe) |
| 4 | `car_deep` | xe ngập sâu (mức 3–4 của STURM: nước tới cửa kính, mái xe) |

Ứng dụng đọc tên class từ `model.names` nên không phụ thuộc thứ tự id. Model cũ với `car_L0…car_L4` vẫn được đọc.

## Cách tính mức cảnh báo

**Độ phủ không phải độ sâu.** Nước phủ kín mặt đường vẫn có thể chỉ nông, nên độ phủ chỉ đẩy tối đa tới mức Cảnh báo; chỉ xe ngập sâu mới lên mức Nguy hiểm.

1. Chạy model 5 lần mỗi giây video (`INFER_FPS`) với `CONF=0.5`, rồi làm mượt theo thời gian:
   - **flood/road**: trung bình trượt mask qua các lần chạy, có ngưỡng trễ (bật khi > 0.5, chỉ tắt khi < 0.25) để vùng không nhấp nháy;
   - **xe**: nối xe giữa các lần chạy theo IoU box; level là level xuất hiện nhiều nhất trong 7 lần gần nhất (hòa thì lấy level cao hơn);
   - khung hình nằm giữa hai lần chạy: mask flood/road chuyển dần từ lần trước sang lần sau.
2. Với mỗi khung hình:
   - `flood_ratio` = diện tích flood / (flood + road). Khi thấy rất ít đường (< 5% khung hình): nếu nước chiếm ≥ 10% khung hình thì coi như đường bị nước phủ kín (vẫn chia cho flood + road, ra gần 100%); nếu nước ít hơn thì chỉ là vũng nước, chia cho cả khung hình để vũng nhỏ không thành 100%.
   - Mức xe: `car_dry` = 0, `car_shallow` = 2, `car_deep` = 3.
   - Khung hình đạt mức nếu thỏa **một trong hai** điều kiện:

   | Mức | Tên | nước phủ ≥ | hoặc xe |
   |---|---|---|---|
   | 0 | An toàn | – | – |
   | 1 | Chú ý | 5% | level ≥ 1 (chỉ model cũ `car_L1`) |
   | 2 | Cảnh báo | 30% | `car_shallow` |
   | 3 | Nguy hiểm | không dùng độ phủ | `car_deep` |

3. Chia video thành đoạn 2 giây (`SEGMENT_SECONDS`). Mức của đoạn là mức cao nhất mà ít nhất 50% khung hình trong đoạn đạt được, nên vài khung hình nhận nhầm không làm mức nhảy lung tung.
4. **Giữ mức** (`LEVEL_HOLD_SECONDS`, mặc định 0 = tắt): mặc định mức luôn là mức thật của đoạn đang phát, đoạn không có nước là An toàn. Đặt > 0 (ví dụ 6) thì mức tăng đổi ngay, còn muốn giảm phải thấp hơn liên tục bấy nhiêu giây (lấp những đoạn model sót nước vì bị che). Chỉ giữ mức trong cùng một cảnh; chuyển cảnh được phát hiện bằng histogram màu (`SCENE_CUT`) và thời điểm nằm trong `scene_cuts`.
5. Các đoạn liền nhau cùng mức được gộp thành **giai đoạn**; mức cả video là mức cao nhất.
6. Đoạn không có xe nào: thanh cảnh báo và JSON (`note` ở đoạn, giai đoạn và kết quả chung) ghi "không có xe để ước lượng độ sâu", nghĩa là mức chỉ dựa vào độ phủ.

Ngưỡng nằm trong `LEVELS` của `analyzer.py`. Các tham số khác chỉnh bằng biến môi trường:

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `MODEL_PATH` | `weights/best.pt` | đường dẫn model |
| `CONF` | 0.5 | confidence tối thiểu |
| `INFER_FPS` | 5 | số lần chạy model mỗi giây video |
| `ROAD_MIN_FRAC` | 0.05 | đường chiếm ít nhất bao nhiêu khung hình mới coi là thấy đường |
| `FLOOD_COVER_MIN_FRAC` | 0.10 | không thấy đường mà nước chiếm ít nhất bấy nhiêu thì coi là đường bị phủ kín |
| `LEVEL_HOLD_SECONDS` | 0 | thời gian thấp liên tục trước khi được hạ mức (0 = tắt) |
| `SCENE_CUT` | 0.6 | ngưỡng khác biệt màu để coi là chuyển cảnh (1 = tắt) |
| `MASK_SMOOTH` | 0.6 | mức làm mượt mask (0 = tắt) |
| `CAR_HISTORY` | 7 | số lần chạy gần nhất dùng để chốt level xe |
| `SEGMENT_SECONDS` / `MIN_FRAME_FRAC` | 2 / 0.5 | độ dài đoạn / tỉ lệ khung hình tối thiểu để đoạn đạt mức |
| `OUT_WIDTH` / `MAX_SECONDS` | 960 / 600 | chiều rộng video kết quả / chỉ xử lý tối đa 10 phút đầu |
| `MAX_UPLOAD_MB` | 500 | dung lượng upload tối đa |
| `FONT_PATH` | tự tìm | font vẽ chữ tiếng Việt lên video |

## Kết quả

Model YOLO11n-seg huấn luyện trên Kaggle (146 epoch, tốt nhất ở epoch 106) với dataset 2877 ảnh gán nhãn bán tự động (xem `training/README.md`). Đánh giá trên tập test giữ riêng, so với model trước đó:

| Chỉ số | Model trước | Model này |
|---|---|---|
| Flood IoU trên test_flood (160 ảnh, mask người) | 0.691 | **0.817** |
| ↳ Deepflood (120) / pixabay (40) | 0.732 / 0.567 | **0.859 / 0.691** |
| Độ chính xác mức xe trên test_car (253 xe) | model trước không có class xe | **84.6%** |

Mask mAP50 từng class trên tập val (406 ảnh): flood 0.693 · road 0.612 · car_dry 0.567 · car_shallow 0.794 · car_deep 0.715 (tổng 0.676). Confusion matrix mức xe và các bảng chi tiết nằm trong `training/README.md`.

Thử trên video thật: người thông cống trên đường ngập (không có xe) cho **Cảnh báo** chứ không phải Nguy hiểm; vũng nước nhỏ trên đường cho **Chú ý**; nước tràn dần cho Chú ý rồi Cảnh báo; đường khô cho **An toàn**.

## Hạn chế

- Class `road` yếu khi đường đã ngập (AP50 trên val 0.612 nhưng gần 0 trên các ảnh ngập của test_flood): khi không thấy đường, độ phủ gần 100% và chỉ bị chặn ở mức Cảnh báo.
- `car_dry` yếu nhất trong các class xe; xe ngập gần hết (chỉ còn mái) hoặc xe bị nước bắn tung có thể không được nhận ra, nên mức Nguy hiểm khó kích hoạt trong các cảnh ngập sâu nhất.
- Thỉnh thoảng có mảng flood nhầm trên tán cây hoặc bóng nước phản chiếu.
- Mức xe trong dữ liệu Deepflood/Sazara là dự đoán của model đã có, không phải nhãn người; nhãn road của tập test_flood cũng là nhãn tự động.
- Chưa được thử với cảnh ban đêm, camera chuyển động nhanh hoặc ngập trong nhà (ví dụ video garage ngập cho mức Cảnh báo vì nước phủ kín khung hình).
- Dataset cuối chứa dữ liệu FRED (CC BY-NC-SA 4.0) nên **không công khai**; kiểm tra giấy phép các nguồn trước khi dùng model vào mục đích thương mại (xem `training/README.md`).
