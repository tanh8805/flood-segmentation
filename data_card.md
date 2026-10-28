# Data card — bo phan loai NGAP / KHONG NGAP (anh)

Dataset dan xuat chi dung noi bo/rieng tu (co FRED CC BY-NC-SA, BDD100K, ADE20K); khong dua anh vao git. Chi commit script dung va file nay.

## Quy tac gan nhan
- Ti le nuoc tren mat dat = nuoc / (SegFormer-B2 road|sidewalk|path|earth|floor  hop  nuoc); nuoc = mask nguoi gan, bo vung SegFormer cho la sky/building/house/tree/wall/ceiling/signboard.
- >= 0.3 -> NGAP; 0.02-0.3 -> bo (mo ho); < 0.02 -> KHONG NGAP. Nguon khong co mask (FRED kho, COCO, BDD): KHONG NGAP neu SegFormer khong thay nuoc.
- Loai: khung 'up*' (cung video voi clip DEV), houston/old (khong phai duong pho), STURM (khong can), anh co do giong CLIP >= 0.93 voi bat ky khung clip DEV/TEST (khong co anh nao vuot nguong; cao nhat 0.92 = canh tuong tu, khong phai khung trung).
- Tu cum anh giong nhau (CLIP cos >= 0.9) va chuoi FRED thanh nhom; 20% nhom cho val (hat 1); khong nhom nao nam o ca hai phia.

## So anh

| Nguon | License | Ngap | Khong ngap | trong do 'kho' | Ghi chu |
|---|---|---|---|---|---|
| Deepflood (HydroShare, Wang 2024) | CC BY 4.0 | 942 | 0 | 0 | mask nguoi gan |
| Sazara / Roadway Flooding | CC BY 4.0 | 228 | 0 | 0 | mask nguoi gan |
| flooding_pixabay (WaterDataset) | Pixabay Content License (anh) | 145 | 0 | 0 | mask nguoi gan |
| FRED (QUT) chuoi ngap | CC BY-NC-SA 4.0 -> dataset dan xuat de PRIVATE | 11 | 106 | 106 | mask nguoi gan; khung it nuoc/chua ngap -> am |
| FRED (QUT) chuoi kho | CC BY-NC-SA 4.0 -> PRIVATE | 0 | 177 | 177 | duong kho nong thon, bong cay |
| ADE20K water | ADE20K (nghien cuu, phi thuong mai) | 61 | 140 | 0 | mask nguoi gan; anh nuoc ngoai duong -> am |
| COCO duong pho (work/dataset) | CC BY 4.0 (anh: theo tung anh) | 0 | 153 | 0 | am, duong kho |
| COCO val2017 (mat dat >= 10%, khong nuoc) | CC BY 4.0 (anh: theo tung anh) | 0 | 1000 | 0 | am, ngau nhien 1000 |
| BDD100K val qua HF dgural/bdd100k (mua, dem, ngay) | BDD100K (BSD, nghien cuu) | 0 | 1082 | 986 | am; mua/dem = am kho; v5: them toan bo anh mua + 300 anh dem xep hang 'dem uot' theo CLIP zero-shot (co the lan vai anh dem kho) |
| **Tong (anh, v5)** | | **1387** | **2658** | **1269** | |

'Kho' = mua (BDD rainy, ngay + dem), dem phan chieu (BDD night), bong cay tren mat nhua (FRED). Muc tieu >= 400: dat.

Pexels: khong co PEXELS_API_KEY nen khong tai video; ~/Videos/hard_negatives/ khong ton tai.
v5: ngoai anh, bo phan loai cuoi con train tren 3 khung/doan (1/6, 3/6, 5/6) cua 18 clip ngoai troi da duyet (bo bo_qua), nhan = nhan doan. Danh gia dung out-of-fold theo clip (loai fold chua clip do), nen khong ro ri khung vao diem danh gia. Clip ngap_4 (trong nha) bi loai.
Kiem tra bang contact sheet 60 anh ngau nhien trong nhom 'kho' (mua/dem/bong cay): dung loai canh.
