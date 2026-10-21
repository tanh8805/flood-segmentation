# Cảnh báo ngập từ video camera đường phố

Ứng dụng web nhận một video camera đường phố, tô vùng nước và các vật tham chiếu (người, xe), rồi cho biết **mức cảnh báo ngập theo từng giai đoạn** (0 An toàn, 1 Chú ý, 2 Cảnh báo, 3 Nguy hiểm) kèm **căn cứ** bằng tiếng Việt cho từng đoạn ("Người lội nước tới gối", "Xe máy ngập quá nửa bánh", "Không có vật tham chiếu, chỉ dựa vào độ phủ").

Quy trình: tải video lên → phân tích → xem kết quả (video đã tô mask, thước đo mực nước, biểu đồ diễn biến, bảng các giai đoạn và căn cứ).

## Cơ chế

Độ phủ nước không cho biết độ sâu, nên mức cao chỉ được báo khi có **vật tham chiếu** có kích thước đã biết chạm nước.

| Thành phần | Model | Nguồn |
|---|---|---|
| Vùng nước | `water.pt`, YOLO11s-seg 1 class | tự train (`training/v2`) |
| Người, mắt cá/gối/hông | `yolo11n-pose.pt` | COCO có sẵn |
| Xe đạp, xe máy, ô tô, xe buýt, xe tải | `yolo11n.pt` | COCO có sẵn |
| Bậc ngập của xe | `depth_cls.pt`, YOLO11n-cls 4 lớp | tự train |
| Mặt đất (mẫu số độ phủ) | SegFormer-b2 ADE20K (road, sidewalk, path, earth, floor) ∪ vùng nước | có sẵn, tải từ Hugging Face |

Bậc độ sâu 0–3: 0 không chạm nước · 1 mắt cá / dưới nửa bánh (~10–20 cm) · 2 giữa ống chân / quá nửa bánh xe máy (~20–45 cm) · 3 tới gối trở lên / ngập hết bánh (≥ 50 cm).

**Người** (luật hình học, `person_depth`): mép nước = hàng trên cùng của vùng nước nối liền đáy box, trong cột của chân; so với y mắt cá, gối, hông (cần gối hoặc hông đủ tin cậy để neo). Kết hợp thêm conf keypoint: mắt cá conf cao (≥ 0.7) → bậc ≤ 1; mắt cá thấp + gối cao → bậc 1 hoặc 2 tuỳ mép nước; mắt cá và gối cùng thấp + hông cao → tín hiệu bậc 3. Hai tín hiệu mâu thuẫn thì lấy bậc **thấp hơn**, trừ khi cả hai cùng chỉ bậc 3. Người nằm trong box xe hai bánh bị bỏ.

**Xe**: chỉ xe có ≥ 3% phần dưới box chạm nước mới chạy `depth_cls` (crop nới box 20%); xe không chạm nước là bậc 0. Chọn bậc cao hơn top-1 nếu xác suất của nó ≥ 0.35 (`DEPTH_UP_P`, thiên về an toàn).

**Mức của khung hình** (độ phủ một mình tối đa mức 1; mức 2, 3 chỉ đến từ vật tham chiếu):

| Mức | Điều kiện | Căn cứ (`evidence`) |
|---|---|---|
| 0 | không có nước đáng kể (nước < 0.5% khung hoặc độ phủ < 5%) và không có vật bậc ≥ 1 | "Không có nước đáng kể" |
| 1 | có nước, hoặc vật đọc được bậc 1 | tên vật ("Người lội nước tới mắt cá") |
| 1 | không có vật nào đọc được (không vật, vật quá nhỏ/xa, hoặc đọc mâu thuẫn) | "Nước phủ X% mặt đường, không ước lượng được độ sâu (không có vật tham chiếu đủ gần)" |
| 2 | có vật bậc 2 | "Người lội nước tới giữa ống chân", "Xe máy ngập quá nửa bánh"… |
| 3 | **chỉ** khi có vật bậc 3 | "Người lội nước tới gối", "Ô tô ngập hết bánh"… |

Vật hợp lệ: người cao ≥ 60 px, xe cao ≥ 40 px, giữ bậc ≥ 2 lần suy luận liên tiếp. **Vật đọc được tin cậy** (chỉ loại này tính vào mức và căn cứ; vật còn lại vẫn được vẽ box, nhãn `?`): xe cao ≥ 8% chiều cao khung (`DEPTH_MIN_FRAC`); người cao ≥ 12% khung (`PERSON_READ_MIN_FRAC`) và có gối hoặc hông tin cậy; xe có đáy box chạm nước mà `depth_cls` đọc bậc 0 là mâu thuẫn nên bỏ kết quả đọc. **Mask nước** bị xoá trên các lớp SegFormer sky, building, house, tree, wall, ceiling, signboard (`USE_WATER_VETO`). Khung không có nước thì bỏ qua pose/detect/cls. Mặt đất < 5% khung → logic dự phòng (nước ≥ 10% khung coi là phủ kín).

**Theo thời gian**: 5 lần suy luận/giây, mask nước làm mượt có ngưỡng trễ, đoạn 2 giây (mức đoạn = mức cao nhất mà ≥ 50% khung hình đạt), phát hiện chuyển cảnh bằng histogram màu, mặt đất chạy ~1 lần/giây hoặc khi chuyển cảnh. Các đoạn liền nhau cùng mức gộp thành giai đoạn; mức cả video là mức cao nhất.

## Cấu trúc

```
backend/
  app.py              API FastAPI: /analyze, /jobs/{id}, /jobs/{id}/video, /health
  pipeline.py         FloodPipeline (nạp model một lần) và các luật thuần: người, xe, cảnh báo, tracker, mặt đất
  analyzer.py         đọc video, chia đoạn, làm mượt, vẽ video kết quả
  download_weights.py tải weights công khai
  static/index.html   giao diện
  tests/              unit test luật người, luật cảnh báo, mặt đất, bật/tắt
  weights/            water.pt, depth_cls.pt (tự train, ~20 MB) + yolo11n-pose.pt, yolo11n.pt
training/             tái lập dataset và huấn luyện (training/v2: model nước, depth_cls, đánh giá)
```

## Chạy ứng dụng

Python 3.10+; có GPU thì tự dùng, không thì chạy CPU.

```bash
cd backend
pip install -r requirements.txt
python download_weights.py        # yolo11n(-pose), SegFormer; kiểm tra water.pt, depth_cls.pt
uvicorn app:app --host 0.0.0.0 --port 8000
```

- **Trình duyệt**: http://localhost:8000, chọn video, bấm **Phân tích**.
- **Dòng lệnh**: `python analyzer.py video.mp4 [ket_qua.mp4]` in kết quả JSON và tạo video đã tô mask.
- **API** (http://localhost:8000/docs): `curl -F "file=@video.mp4" http://localhost:8000/analyze` → `job_id`; `GET /jobs/<id>` (status, progress, result, mỗi đoạn có `evidence`); `GET /jobs/<id>/video`.
- **Test**: `cd backend && python -m unittest discover tests`.

Bật/tắt từng phần bằng biến môi trường `USE_POSE`, `USE_DETECT`, `USE_DEPTH_CLS`, `USE_GROUND` (`0` để tắt); thiếu weights của phần nào thì chạy với phần còn lại. Thiếu `water.pt` thì không chạy được.

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `WATER_MODEL_PATH` / `POSE_MODEL_PATH` / `DETECT_MODEL_PATH` / `DEPTH_MODEL_PATH` | `weights/…` | đường dẫn model |
| `GROUND_MODEL` | `nvidia/segformer-b2-finetuned-ade-512-512` | model mặt đất |
| `WATER_CONF` / `OBJ_CONF` / `KP_CONF` / `KP_HIGH` | 0.4 / 0.35 / 0.5 / 0.7 | ngưỡng conf: nước / vật / keypoint / mắt cá "cao" |
| `OBJ_CONF_2W` / `DARK_LUM` / `DARK_SCOPE` / `DETECT_IMGSZ` | = `OBJ_CONF` / 0 / `det` / 0 | tuỳ chọn thử cho cảnh tối (tắt mặc định): conf riêng xe máy/xe đạp; CLAHE khi độ sáng xám < ngưỡng (`det`: chỉ detect/pose, `all`: cả mask nước và crop depth_cls); imgsz detect |
| `USE_DEPTH_MEMORY` / `DEPTH_MEMORY_S` / `MEMORY_WATER_KEEP` / `MEMORY_MIN_STEPS` | 0 / 10 / 0.5 / 1 | tuỳ chọn thử, **tắt mặc định, chưa có số liệu ủng hộ**: nhớ bậc ≥ 2 của vật tin cậy tối đa 10 giây khi cùng cảnh và nước còn (evidence ghi "đo X giây trước"); trên test VN nó làm báo dư tăng 0.08 → 0.35 |
| `DETECT_LOW_CONF` / `TRACK_CONFIRM_STEPS` | 0.15 / 3 | xe conf trong [0.15, ngưỡng) chỉ nhận khi cùng track thấy ≥ 3 lần suy luận liên tiếp (0 = tắt); không đổi kết quả test VN |
| `USE_WATER_VETO` | 1 | xoá nước trên sky/building/house/tree/wall/ceiling/signboard (cần SegFormer) |
| `DEPTH_MIN_FRAC` / `PERSON_READ_MIN_FRAC` | 0.08 / 0.12 | chiều cao box tối thiểu (theo khung) để đọc độ sâu xe / người |
| `DEPTH_UP_P` | 0.35 | chọn bậc cao hơn của depth_cls nếu xác suất ≥ giá trị này |
| `SHALLOW_F` / `KNEE_F` | 0.25 / 0.9 | tỉ lệ ống chân bị nước phủ: bậc 1 / bậc 3 |
| `INFER_FPS` / `SEGMENT_SECONDS` / `MIN_FRAME_FRAC` | 5 / 2 / 0.5 | tốc độ suy luận, độ dài đoạn, tỉ lệ khung đạt mức |
| `LEVEL_HOLD_SECONDS` | 0 | > 0: mức chỉ giảm khi đã thấp liên tục bấy nhiêu giây |
| `OUT_WIDTH` / `MAX_SECONDS` / `MAX_UPLOAD_MB` | 960 / 600 / 500 | video kết quả, thời lượng tối đa, upload tối đa |

Job lưu trong bộ nhớ, video xử lý lần lượt từng cái.

## Kết quả

**Model nước** (YOLO11s-seg, 100 epoch, chỉ mask người gán; train/val 1308/228 ảnh gồm ảnh âm). IoU trên test_flood (160 ảnh, cùng tập với các lần trước):

| Model | IoU | Deepflood (120) | pixabay (40) |
|---|---|---|---|
| model đầu tiên | 0.691 | 0.732 | 0.567 |
| model 5 class (v1) | 0.817 | 0.859 | 0.691 |
| **model nước v2** | **0.820** | 0.860 | 0.699 |

Cải thiện so với v1 là rất nhỏ (+0.003); v2 đơn giản hơn (1 class, không phụ thuộc nhãn giáo viên) nhưng không chính xác hơn rõ rệt.

**depth_cls** (519 crop test, nhãn xe tay + STURM; ngưỡng chọn bậc cao `DEPTH_UP_P = 0.35`):

| | top-1 | có ngưỡng 0.35 |
|---|---|---|
| đúng bậc | 0.653 | 0.645 |
| sai ≤ 1 bậc | 0.948 | 0.956 |
| đoán thấp hơn nhãn | 0.202 | 0.170 |
| đoán cao hơn nhãn | 0.145 | 0.185 |

Trên crop xe máy/xe đạp/bus/tải tự gán (153): đúng bậc 0.529 → 0.575, đoán thấp 0.353 → 0.281. Trên STURM (366): đúng bậc 0.705 → 0.675.

**Mask nước và veto SegFormer** (IoU test_flood, conf 0.25 như lúc train): 0.8195 → **0.8158** khi xoá nước trên sky/building/house/tree/wall/ceiling/signboard (deepflood 0.860 → 0.856, pixabay 0.699 → 0.695); 3 ảnh giảm > 0.05, 3 ảnh tăng > 0.05; 2.8% diện tích nước thật nằm trên các lớp bị xoá. Đổi lại, video đêm tuoitre_dem ở ~1:23 hết tràn lên mặt tiền và vòm cây (nước trên khung hình giảm 0.90 → 0.75). Ảnh tối (độ sáng xám TB < 70) chỉ chiếm 4.1% train (53/1308), 2.2% val, 2.5% test (4 ảnh, IoU 0.679 → 0.680): model chưa được train đủ ảnh đêm, chưa train lại.

**Test video VN** (51 đoạn 2 giây trong clip0, 1, 2, 5 và tuoitre_dem [66–98 s]; clip6 đã bị mất khỏi máy nên không còn trong bảng này). Nhãn tay theo contact sheet; clip2 và clip5 đã được người duyệt, các đoạn tuoitre_dem **mới gán, chưa duyệt** (cả 16 đoạn nhãn 2 theo toàn cảnh "đường ngập như sông", không có vật tham chiếu đo được ở phần lớn đoạn). `training/v2/vn_test_labels.csv`:

| | đúng mức | lệch ≤ 1 | báo thiếu | báo dư |
|---|---|---|---|---|
| v1 | 0.608 | 0.980 | 0.078 | 0.314 |
| v2 đã push | 0.412 | 0.941 | 0.569 | 0.020 |
| **v2 + đọc tin cậy + veto nước** | 0.412 | 0.941 | 0.588 | 0.000 |
| v2 + đọc tin cậy + veto + nhớ bật | 0.373 | 0.843 | 0.431 | 0.196 |

Chỉ trên clip0, 1, 2, 5 (35 đoạn, nhãn đã duyệt): v1 0.429 đúng / thiếu 0.114 / dư 0.457; v2 0.600 / 0.400 / 0.000; v2 + nhớ 0.486 / 0.371 / 0.143. Ma trận nhầm v2 hiện tại (hàng = nhãn, cột = dự đoán, 51 đoạn): `[[3,0,0,0],[0,17,0,0],[0,27,1,0],[0,3,0,0]]`.

Việc sửa đọc tin cậy làm đúng căn cứ (không còn "vật tham chiếu chỉ ngập nông" khi vật quá nhỏ) nhưng không thêm mức cao: khi không có vật đọc được, mức tối đa là 1, nên báo thiếu cao (27/28 đoạn nhãn 2 bị báo 1). Bật nhớ giảm báo thiếu nhưng báo dư tăng lên 0.14–0.20, ngay cả khi đọc đã chặt hơn. Báo dư của v1 chủ yếu là luật "độ phủ ≥ 30% → mức 2" (nay đã bỏ).

Đã thử (đo trên clip0, 1, 2, 5, 6 trước khi clip6 bị mất): tăng sáng CLAHE khi khung tối (`DARK_LUM`, `DARK_SCOPE`), conf riêng cho xe máy/xe đạp 0.15–0.2 (`OBJ_CONF_2W`), detect imgsz 960 (`DETECT_IMGSZ`, +6 ms). Không cách nào sửa được clip2 (đúng mức 1/11 trước và sau); chỉ có imgsz 960 làm các clip khác thêm 1 đoạn đúng, quá ít để kết luận. Vì vậy các tuỳ chọn này **tắt mặc định**. Tập nhỏ (3 đoạn nhãn 3).

**Tốc độ** (RTX 3050 Laptop 4 GB, khung 1 clip, không tính SegFormer mặt đất chạy ~1 lần/giây): GPU **29 ms/lần suy luận** (VRAM cấp phát tối đa ~200 MB, chưa tính SegFormer-b2 và CUDA context), CPU **221 ms/lần**. 5 lần/giây video thì CPU chậm hơn thời gian thực khoảng 1.1×.

## Hạn chế

- Độ phủ một mình tối đa mức 1: cảnh nước phủ rộng không có người/xe được báo Chú ý dù có thể sâu hơn (báo thiếu 0.37 trên test VN).
- Cảnh tối, trong nhà, nhìn từ trên xuống, xe chỉ lộ một phần: xe máy thường bị detect conf thấp hoặc không thấy (xem trên).
- Mức 3 hiếm: cần người tới gối hoặc xe ngập hết bánh được nhận ra; xe ngập gần hết có thể không được detect.
- Luật người phụ thuộc chất lượng mask nước và keypoint; mask nước kém (phản chiếu, tán cây) làm lệch mép nước.
- `depth_cls` trên crop video VN chỉ đúng ~55%, thiên về đoán bậc 1 cho nhãn 2; nhãn tay theo contact sheet, không đo thực tế.
- Bộ test VN 51 đoạn/5 clip đã được dùng để chỉnh luật (độ phủ, ngưỡng, tuỳ chọn cảnh tối) nên số liệu trên lạc quan; cần đánh giá lại trên clip mới. Test nhỏ, đa số bậc 1–2; chưa thử ban đêm, camera chuyển động nhanh, ngập trong nhà.
- Dữ liệu train nước gồm FRED (CC BY-NC-SA 4.0): dataset không công khai; kiểm tra giấy phép các nguồn trước khi dùng thương mại (xem `training/README.md`).
