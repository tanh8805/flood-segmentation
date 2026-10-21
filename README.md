# Phát hiện ngập từ video camera đường phố

Ứng dụng web nhận một video camera đường phố, tô vùng nước, rồi cho biết **đường có NGẬP hay KHÔNG NGẬP ở từng thời điểm**,
tổng số giây ngập và các khoảng ngập, kèm căn cứ ("Nước phủ 85% mặt đường").

- **NGẬP**: nước tràn mặt đường, không còn lối đi khô.
- **KHÔNG NGẬP**: đường khô, ướt sau mưa, vũng nhỏ rải rác, phản chiếu.

Quy trình: tải video lên → phân tích → xem kết quả (video tô mask có banner đỏ/xanh, ô trạng thái đổi theo thời điểm đang phát,
dải thời gian 2 màu bấm để tua, một dòng tổng kết).

## Cơ chế

| Thành phần | Model | Nguồn |
|---|---|---|
| Vùng nước | `water.pt`, YOLO11s-seg 1 class | tự train (`training/v2`) |
| Mặt đất (mẫu số độ phủ) + vùng cấm nước | SegFormer-b2 ADE20K | có sẵn, tải từ Hugging Face |

1. Mỗi 1/5 giây: model nước cho mask; xoá nước trên sky/building/house/tree/wall/ceiling/signboard (SegFormer); làm mượt mask theo thời gian.
2. **Độ phủ** `water_ratio` = diện tích nước / (mặt đất ∪ nước), mặt đất = road, sidewalk, path, earth, floor (SegFormer, ~1 lần/giây hoặc khi chuyển cảnh).
   Mặt đất < 5% khung thì dùng dự phòng: nước ≥ 10% khung → độ phủ 1.0, ngược lại độ phủ = nước / khung. Nước < 0.5% khung coi như không có.
3. **Luật nhị phân** (`FloodRule`, hàm thuần): ngưỡng trễ + giữ H giây. KHÔNG NGẬP → NGẬP khi `water_ratio ≥ FLOOD_ON` liên tục `FLOOD_HOLD_S` giây;
   NGẬP → KHÔNG NGẬP khi `water_ratio < FLOOD_OFF` liên tục `FLOOD_HOLD_S` giây. Khung tối (sáng TB < 40) chỉ bị bỏ qua nếu bật `FLOOD_SKIP_DARK`.
   Chuyển cảnh được phát hiện (dùng cho làm mượt mask và mặt đất) nhưng **mặc định không reset trạng thái ngập**: có video là slideshow ảnh tĩnh đổi mỗi ~0.4 s,
   reset + giữ H giây làm clip đó không bao giờ được nhận là ngập (`FLOOD_RESET_ON_CUT=1` để bật lại).
4. **Đoạn 2 giây**: NGẬP khi ≥ 50% lần suy luận trong đoạn ở trạng thái NGẬP. Mỗi đoạn trả `flooded`, `water_ratio` (trung bình), `evidence`:
   "Nước phủ 85% mặt đường" / "Nước ít (12%), chưa ngập" (độ phủ ≥ 5%) / "Không thấy nước". Kết quả chung: `flooded`, `flooded_seconds`, `flood_intervals`.

## Chọn ngưỡng

Mô phỏng trên cache tín hiệu (`training/v3/sim_binary.py`, không chạy lại model), chỉ trên **clip VN cũ + 6 chuỗi FRED đường khô**
(111 đoạn 2 s: 48 ngập, 63 không ngập; FRED khô đã được dùng làm ảnh âm khi train nên chỉ dùng để chọn ngưỡng). Lưới: T_on ∈ {0.15, 0.25, 0.35, 0.5, 0.65},
T_off = T_on − {0.05, 0.10}, H ∈ {1, 2, 3, 4} s, bỏ khung tối ∈ {có, không}.

**Quy tắc chọn (bảo thủ với đường ướt chưa có trong dev)**: trong các cấu hình đạt recall ngập ≥ 0.90 **và** báo nhầm = 0 trên dev,
chọn **T_on cao nhất**; hoà thì **H dài nhất**; hoà nữa thì F1 cao hơn, rồi không bỏ khung tối, rồi T_off cao hơn.
(Quy tắc ban đầu là "báo nhầm thấp nhất, hoà thì F1 cao" cho cấu hình 0.15/0.05/1 s, nằm sát rìa lưới; đổi sang quy tắc bảo thủ **trước khi** chấm bộ test mới.)

Chọn được: **`FLOOD_ON=0.25`, `FLOOD_OFF=0.20`, `FLOOD_HOLD_S=1`, không bỏ khung tối**. Chỉ 4/80 cấu hình đạt cả hai điều kiện.

| Dev (111 đoạn) | recall | báo nhầm | F1 |
|---|---|---|---|
| v3 chọn được (trên toàn dev) | 0.938 | 0 | 0.968 |
| v3 leave-one-clip-out (chọn lại trên các clip còn lại) | 0.792 | 0 | 0.884 |
| v1, mức ≥ 1 | 0.979 | 0.016 | 0.979 |
| v2, mức ≥ 1 | 1.0 | 0 | 1.0 |

v2 đã được chỉnh trên chính các clip này nên số của v2 trên dev không công bằng.

## Kết quả trên bộ test mới

Bộ chấm cuối, chỉ chấm **một lần**, không dùng để chọn ngưỡng: 5 clip (ngập 2 clip 92 s; không ngập 3 clip 88 s, đa số TikTok dọc), 82 đoạn 2 s
(44 ngập, 38 không ngập; bỏ 10 đoạn thẻ TikTok/cận cảnh mặt nước, người dùng duyệt). Không clip nào dùng ở v2. Nhãn mặc định theo thư mục, đã duyệt bằng contact sheet.
**Bộ test nhỏ, vài clip, nên sai số lớn**: một đoạn sai đổi recall ~2%, báo nhầm ~3%.

| Bản | precision | recall | F1 | báo nhầm | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|
| v1, mức ≥ 1 | 0.915 | 0.977 | **0.945** | 0.105 | 43 | 4 | 1 | 34 |
| v2, mức ≥ 1 | 0.880 | 1.0 | 0.936 | 0.158 | 44 | 6 | 0 | 32 |
| **v3** (0.25 / 0.20 / 1 s) | 0.878 | 0.977 | 0.925 | 0.158 | 43 | 6 | 1 | 32 |
| v3 cấu hình cũ (0.15 / 0.05 / 1 s), chỉ tham khảo | 0.878 | 0.977 | 0.925 | 0.158 | 43 | 6 | 1 | 32 |

Ma trận 2×2 của v3 (hàng = thật, cột = dự đoán): ngập → 43 ngập, 1 không ngập; không ngập → 6 ngập, 32 không ngập. Hai cấu hình v3 cho đúng cùng kết quả trên bộ này.

Theo từng clip (TP / FP / FN / TN):

| Clip | v1 | v2 | v3 |
|---|---|---|---|
| ngap_01 | 35 / 0 / 1 / 0 | 36 / 0 / 0 / 0 | 36 / 0 / 0 / 0 |
| ngap_02 | 8 / 0 / 0 / 0 | 8 / 0 / 0 / 0 | 7 / 0 / 1 / 0 |
| khong_ngap_01 (mưa to, đường ướt) | 0 / 2 / 0 / 12 | 0 / 2 / 0 / 12 | 0 / 2 / 0 / 12 |
| khong_ngap_02 (đêm) | 0 / 2 / 0 / 10 | 0 / 1 / 0 / 11 | 0 / 0 / 0 / 12 |
| khong_ngap_03 (phố có cây) | 0 / 0 / 0 / 12 | 0 / 3 / 0 / 9 | 0 / 4 / 0 / 8 |

v3 không cao hơn v1 về F1 trên bộ này, nên không merge vào `main`.

### Báo nhầm của v3

- **khong_ngap_03, báo nhầm 0.333 (> 0.15)**, các đoạn bắt đầu ở 6, 10, 14, 16 s. Nguyên nhân: model nước tô mặt nhựa khô dưới bóng cây thành nước
  (độ phủ ~0.9) thành từng đợt dài 1–2 s (4.2–6.0 s, 9–10 s, 15–16.4 s; đoạn 14 s nước tới 27% khung). Đợt dài hơn H = 1 s nên luật không lọc được,
  và mỗi lần còn kéo trạng thái NGẬP thêm 1 s (đoạn 6 s chỉ là phần đuôi của đợt 4.2–6.0 s).
- **khong_ngap_01, báo nhầm 0.143 (≤ 0.15)**, các đoạn 2, 4 s: mưa to, mặt đường ướt phản chiếu bị tô thành nước (độ phủ 0.8–0.9).
- **ngap_02, bỏ sót đoạn 0 s**: model thấy nước từ 0.4 s nhưng trạng thái chỉ bật sau 1 s giữ (tại 1.4 s), tức dưới 50% đoạn.
- **Đề xuất** (chưa làm, không tự train lại): thêm ảnh âm là đường ướt/mưa to, phản chiếu, nhựa khô dưới bóng cây (và video đường phố Việt Nam) vào dataset rồi train lại model nước;
  sau đó chọn lại ngưỡng với dev có đường ướt. Tăng H chỉ lọc được lần nhầm ngắn hơn H, không sửa được đợt nhầm dài 1–2 s, và làm chậm khi ngập thật.

## Cấu trúc

```
backend/
  app.py              API FastAPI: /analyze, /jobs/{id}, /jobs/{id}/video, /health
  pipeline.py         FloodPipeline (nạp model một lần) và các luật thuần: tín hiệu nước, FloodRule, đoạn, mặt đất
  analyzer.py         đọc video, chia đoạn, làm mượt, vẽ video kết quả
  download_weights.py nạp trước SegFormer
  static/index.html   giao diện
  tests/              unit test luật nhị phân, tín hiệu nước, đoạn, mặt đất, bật/tắt
  weights/            water.pt (tự train, ~20 MB); best.pt là model cũ của v1, không dùng
training/             tái lập dataset và huấn luyện (training/v2: model nước; training/v3: chọn ngưỡng và chấm)
```

## Chạy ứng dụng

Python 3.10+; có GPU thì tự dùng, không thì chạy CPU.

```bash
cd backend
pip install -r requirements.txt
python download_weights.py        # nạp SegFormer; kiểm tra water.pt
uvicorn app:app --host 0.0.0.0 --port 8000
```

- **Trình duyệt**: http://localhost:8000, chọn video, bấm **Phân tích**.
- **Dòng lệnh**: `python analyzer.py video.mp4 [ket_qua.mp4]` in kết quả JSON và tạo video đã tô mask.
- **API** (http://localhost:8000/docs): `curl -F "file=@video.mp4" http://localhost:8000/analyze` → `job_id`; `GET /jobs/<id>` (status, progress, result); `GET /jobs/<id>/video`.
- **Test**: `cd backend && python -m unittest discover tests`.

Thiếu `water.pt` thì không chạy được; thiếu SegFormer thì chạy bằng logic dự phòng.

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `FLOOD_ON` / `FLOOD_OFF` / `FLOOD_HOLD_S` | 0.25 / 0.2 / 1 | ngưỡng bật / tắt NGẬP (độ phủ) và số giây giữ |
| `FLOOD_SKIP_DARK` / `DARK_LUM` | 0 / 40 | 1: bỏ qua khung có sáng TB < `DARK_LUM` |
| `FLOOD_RESET_ON_CUT` | 0 | 1: đổi cảnh thì về KHÔNG NGẬP và đếm lại |
| `WATER_MODEL_PATH` / `GROUND_MODEL` | `weights/water.pt` / `nvidia/segformer-b2-finetuned-ade-512-512` | model nước / mặt đất |
| `WATER_CONF` / `IMG_SIZE` | 0.4 / 640 | ngưỡng conf và kích thước suy luận của model nước |
| `USE_GROUND` / `USE_WATER_VETO` | 1 / 1 | mặt đất SegFormer / xoá nước trên trời, nhà, cây... |
| `WATER_MIN_FRAME_FRAC` / `GROUND_MIN_FRAC` / `FLOOD_COVER_MIN_FRAC` | 0.005 / 0.05 / 0.10 | nước tối thiểu (theo khung) / mặt đất tối thiểu / ngưỡng dự phòng "phủ kín" |
| `INFER_FPS` / `SEGMENT_SECONDS` / `MIN_FRAME_FRAC` | 5 / 2 / 0.5 | tốc độ suy luận, độ dài đoạn, tỉ lệ lần suy luận ngập để đoạn là NGẬP |
| `OUT_WIDTH` / `MAX_SECONDS` / `MAX_UPLOAD_MB` | 960 / 600 / 500 | video kết quả, thời lượng tối đa, upload tối đa |

Job lưu trong bộ nhớ, video xử lý lần lượt từng cái.

## Hạn chế

- Bộ test mới rất nhỏ (5 clip, 82 đoạn), phần lớn là video TikTok; số liệu chỉ để tham khảo, không đủ kết luận v3 hơn hay kém v1/v2.
- Dev dùng chọn ngưỡng không có đường ướt, mưa to hay vũng nước (âm chỉ là đường khô FRED + vài đoạn logo/đen), nên ngưỡng chưa được kiểm trên đúng loại ca khó nhất.
- Độ phủ không phân biệt nước thật với mặt đường ướt phản chiếu hoặc bóng cây; lỗi của model nước đi thẳng vào kết quả.
- Không còn ước lượng độ sâu: "ngập" chỉ nghĩa là nước phủ phần lớn mặt đường, không biết nông hay sâu.
- Cảnh trong nhà, cận cảnh mặt nước (không thấy đường) hoặc camera không thấy mặt đường cho độ phủ không đáng tin.
- Trạng thái ngập trễ H giây khi bật/tắt; clip ngắn hơn vài giây có thể bị sót đoạn đầu.
