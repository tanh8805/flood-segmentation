# Tạo dataset và huấn luyện model

Không có nhãn tay: nhãn đến từ các **model giáo viên chạy offline** (pseudo-label) kết hợp mask/nhãn người sẵn có trong một số bộ dữ liệu công khai. Model đưa vào ứng dụng chỉ là một YOLO11n-seg.

Class cuối: `0 flood`, `1 road`, `2 car_dry`, `3 car_shallow`, `4 car_deep` (flood và road không chồng nhau).

## Nguồn dữ liệu và giấy phép

| Nguồn | Ảnh dùng | Nhãn lấy từ | Giấy phép |
|---|---|---|---|
| Deepflood (Urban Flood Image Dataset, HydroShare) | 944 | mask flood của người; xe do giáo viên | CC BY 4.0 |
| Sazara = Roadway Flooding Image Dataset (cùng bộ HydroShare) | 228 | mask flood của người; xe do giáo viên | CC BY 4.0 |
| STURM-FloodDepth Flooded Cars (Zenodo 14833532) | 589 crop cho train/val (cân bằng ≤ 160 mỗi mức gốc) + 224 crop test | mức ngập xe là nhãn người; mask xe + flood + road do giáo viên | CC BY 4.0 |
| FRED (QUT) | 157 ảnh ngập + 177 ảnh khô | flood/road là nhãn người (ảnh khô: flood rỗng, road do giáo viên) | **CC BY-NC-SA 4.0** |
| COCO val2017 | 156 ảnh đường khô có xe | road + xe do giáo viên (lọc tay 39 ảnh road sai) | ảnh theo giấy phép từng ảnh trên Flickr |
| flooding_pixabay (WaterDataset) | 148 | mask flood của người | theo bộ WaterDataset / ảnh Pixabay |
| ADE20K_water (WaterDataset) | 77 (đã lọc ngữ cảnh đường phố) + ảnh hồ/biển làm negative | mask nước của người | theo giấy phép ADE20K |
| Video quay của nhóm và ảnh dataset cũ của nhóm | 20 + 10 | giáo viên | của nhóm |
| Houston (WaterDataset) | 4 | giáo viên | theo bộ WaterDataset |

Giấy phép của các bộ WaterDataset/ADE20K cần được kiểm tra lại trước khi phát hành dataset. Không dùng nhãn cũ của dataset trước đó, chỉ dùng ảnh. **Dataset cuối chứa FRED (CC BY-NC-SA 4.0) nên chỉ dùng riêng tư**, không công khai; model suy ra từ nó nên kiểm tra điều khoản trước khi dùng thương mại.

Các nguồn đã thử rồi loại: WebCOOS (một camera cố định, mask đáng ngờ), FloodIMG (chỉ 25 ảnh có nhãn và không có lớp nước; phần lớn là ảnh sông/ảnh trên cao), AlleyFloodNet (chỉ có nhãn phân loại), Boston/LSU/lake_exp/river_segs (không phải đường phố).

## Model giáo viên (chỉ dùng offline)

| Model | Dùng cho |
|---|---|
| V-FloodNet (LinkNet-EfficientNet-b4) | vùng nước |
| SegFormer-b5 ADE20K 640 (`nvidia/segformer-b5-finetuned-ade-640-640`) | road, nước, và ngữ cảnh đường phố |
| `yolo11x-seg` (Ultralytics) | mask từng chiếc xe |
| `best_car_YOLOv11` (YOLO11n detect, nhóm tự huấn luyện) | mức ngập của xe (Level 0–4) |

## Pipeline (thư mục `pipeline/`)

Đặt `FLOOD_WORK` là thư mục làm việc (mặc định `./work`); dữ liệu thô đặt trong `$FLOOD_WORK/raw/…` hoặc chỉ bằng biến môi trường (xem đầu `p2_select.py` và `teachers.py`).

| Bước | Script | Việc làm |
|---|---|---|
| tải nguồn | `fred_list.py`, `fred_download.py`, `remote_zip.py`, `coco_select.py` | liệt kê và tải đúng các khung hình FRED cần (HTTP Range, không tải cả zip), lọc ảnh COCO có road + xe, không có nước |
| 2 | `p2_select.py` | dedupe bằng dHash, chia nhóm (cụm ảnh gần giống, chuỗi, video), giữ riêng tập test |
| 3 | `p3_teachers.py` | chạy giáo viên một lần cho mỗi ảnh, cache kết quả thô ra đĩa (tiếp tục được khi bị ngắt) |
| 4 | `p4_labels.py` | áp luật gán nhãn và lọc (mô tả dưới), sinh polygon |
| 5 | `p5_export.py` | cân bằng, tách train/val theo nhóm 85/15, xuất YOLO-seg, vẽ 100 overlay ngẫu nhiên để kiểm tra |
| 6 | `p6_check.py` | kiểm tra định dạng bằng ultralytics (số ảnh/nhãn mỗi split, polygon lỗi), không train |
| huấn luyện | `make_notebook.py` → `train_kaggle.ipynb` | sinh notebook train trên Kaggle |

Trình tự: `python pipeline/fred_list.py` → `fred_download.py` → `coco_select.py` → `p2_select.py` → `p3_teachers.py` → `p4_labels.py` → `p5_export.py --review` → `p6_check.py`. Cài môi trường bằng `requirements.txt` (phiên bản được ghim vì V-FloodNet là pickle của cả model).

### Luật gán nhãn (`p4_labels.py`)

- **flood** = mask người nếu có; ngược lại `V-FloodNet ∪ SegFormer-water`, đóng 5×5, bỏ vùng nhỏ hơn 0.5% ảnh; không dùng hậu xử lý "giữ vùng lớn nhất", không loại ảnh chỉ vì hai giáo viên lệch nhau.
- **road** = `SegFormer road − flood` (FRED ngập: nhãn người; FRED khô: flood rỗng).
- **xe**: mask từ `yolo11x-seg`, mức từ `best_car_YOLOv11` (conf ≥ 0.5). Nếu 20% dưới box xe không chạm vùng flood thì ép thành `car_dry`. Xe lớn chạm flood mà không có mức đủ tin cậy thì loại cả ảnh. Mức gộp: L0 → `car_dry`, L1–L2 → `car_shallow`, L3–L4 → `car_deep`.
- **Lọc ngữ cảnh đường phố** (ADE20K, Houston, ảnh dataset cũ): vùng nước phải kề road/sidewalk/xe/cột (≥ 8% vành quanh vùng nước), hoặc người + toà nhà. Nước không có gì giống đường phố quanh (hồ, biển, sông, bể) thành ảnh negative (flood rỗng, tối đa 5% dataset); còn lại bị loại. Ảnh ADE phải khớp giáo viên (IoU ≥ 0.3).
- **STURM**: mức ngập là nhãn người của chiếc xe ở giữa crop (chọn bằng mask `yolo11x-seg`); flood/road do giáo viên. Loại crop không thấy xe, có xe thứ hai lớn, xe giữa không chứa tâm crop, mức ≥ 1 mà đáy xe không chạm flood, hoặc mức 0 mà có flood ở xe. Tối đa 160 crop mỗi mức gốc. Dùng split gốc của STURM cho test và loại khỏi train/val mọi chiếc xe (id + mức) có trong `test.txt`, vì split gốc có rò rỉ cùng xe giữa các tập.
- **FRED**: dedupe bằng dHash, tối đa 60 khung hình mỗi chuỗi ngập và 20 mỗi chuỗi khô; nhãn id 0 = other, 1 = road, 2 = water hazard (→ flood).
- **COCO**: danh sách ảnh bị loại tay (road sai: sân bóng, cận cảnh vật thể…) ở `coco_exclude.txt`.

### Kiểm tra chất lượng nhãn

Xem ngẫu nhiên 100 overlay của train+val: vòng 1 sai 11/100 (kênh đào/bến cảng bị coi là ngập trong ADE20K, blob flood sai trên thân xe STURM, chọn nhầm xe giữa). Sau khi siết bộ lọc ADE, ép xe giữa phải chứa tâm crop và loại flood nằm trong xe, vòng 2 (mẫu khác) sai **7/100**: 2 ADE (bến cảng/cầu), 3 STURM có blob flood trên thân xe, 1 COCO có road phủ nắp capô, 1 flood lẻ trên cây.

### Dataset cuối

2877 ảnh: train 2087, val 406, `test_flood` 160 (120 Deepflood + 40 pixabay, mask flood của người), `test_car` 224 (STURM test, nhãn mức người). Số instance: flood 2872, road 1205, car_dry 1139, car_shallow 1180, car_deep 413. Có 143 ảnh negative (hồ/biển, flood rỗng).

## Huấn luyện

`train_kaggle.ipynb` (YOLO11n-seg, `device=0`, batch 16, optimizer `auto`, 150 epoch, patience 40, imgsz 640, `plots=True`, resume từ `last.pt` nếu có). Quy trình: nén thư mục `dataset` thành zip, tạo dataset **riêng tư** trên Kaggle (`kaggle datasets create`), push notebook bằng `kaggle kernels push` (private, bật GPU và internet, `dataset_sources` trỏ tới dataset). Sau khi train, notebook đánh giá trên `test_flood` và `test_car`, so với model cũ đặt trong dataset ở `old_model/best.pt`, rồi chép `best.pt`, `results.csv`, ảnh plots ra `/kaggle/working`.

Lần chạy cho `backend/weights/best.pt`: dừng sớm ở epoch 146, tốt nhất ở epoch 106 (khoảng 1.1 giờ).

## Kết quả

**Flood IoU trên test_flood (160 ảnh, mask người)**

| | Model cũ | Model mới |
|---|---|---|
| Trung bình | 0.691 | **0.817** |
| Median | 0.845 | 0.886 |
| Deepflood (120) | 0.732 | **0.859** |
| pixabay (40) | 0.567 | **0.691** |

**Mức xe trên test_car (253 xe)**: recall 0.846, độ chính xác mức **84.6%** trên các xe khớp được.

| Thật \ Dự đoán | car_dry | car_shallow | car_deep |
|---|---|---|---|
| car_dry (46) | 36 | 10 | 0 |
| car_shallow (124) | 7 | 105 | 12 |
| car_deep (44) | 0 | 4 | 40 |

**Mask mAP50 từng class**

| Class | val (406 ảnh) | test_flood | test_car |
|---|---|---|---|
| flood | 0.693 | 0.837 | 0.444 |
| road | 0.612 | 0.001 | 0.056 |
| car_dry | 0.567 | 0.334 | 0.761 |
| car_shallow | 0.794 | 0.872 | 0.789 |
| car_deep | 0.715 | 0.535 | 0.744 |
| tổng (mAP50 / mAP50-95) | 0.676 / 0.477 | | |

Nhãn road của `test_flood` và `test_car` là nhãn tự động (SegFormer) và rất thưa trong ảnh ngập nên con số road ở hai cột đó không đáng tin; model cũ không có class xe nên chỉ so sánh được flood.
