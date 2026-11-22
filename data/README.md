# Tài liệu Dữ liệu & Nhật ký Thao tác trên Thư mục `data/`

Tài liệu này mô tả chi tiết cấu trúc thư mục `data/`, bảng ánh xạ toàn bộ **42 chuyên mục thể loại gốc $\rightarrow$ 22 nhãn chuẩn**, quá trình tiền xử lý tiếng Việt và các bảng thống kê thực tế từ [webtruyen_raw.csv](raw/webtruyen_raw.csv) cho đồ án **NLP Đề 11: Phân loại thể loại truyện/văn học tiếng Việt**.

---

## 1. Cây thư mục `data/` và vai trò từng file

```text
data/
├── README.md                              # Tài liệu này
├── raw/                                   # Dữ liệu crawl gốc (webtruyen_raw.csv là nguồn chính đang chạy)
│   ├── webtruyen_raw.csv                  # [ĐANG DÙNG] 55.684 bản ghi thô (42 thẻ thể loại × đủ số trang) từ webtruyen.vn (wetruyen.com)
│   ├── webnovel_raw.csv                   # [Lưu trữ] 247 truyện/sách từ webnovel.vn (CSV UTF-8)
│   ├── webnovel_raw.json                  # [Lưu trữ] 247 truyện/sách từ webnovel.vn (JSON)
│   ├── metruyenchu_raw.csv                # [Lưu trữ] 374 truyện từ metruyenchu.com.vn
│   └── wiki_raw.csv                       # [Lưu trữ] 255 tác phẩm văn học & sách từ Wikipedia / Wikisource
├── labeled/                               # Dữ liệu định dạng Label Studio (lưu trữ)
│   ├── webnovel_label_studio.json         # 247 bản ghi webnovel.vn định dạng Label Studio JSON
│   └── wiki_label_studio.json             # 255 bản ghi Wiki định dạng Label Studio JSON
├── processed/                             # Dữ liệu sau lọc trùng tên/tóm tắt & chuẩn hóa 22 nhãn
│   ├── stories.csv                        # 18.603 mẫu duy nhất, 22 cột label_* (36 cột)
│   └── stories_clean.csv                  # 18.603 mẫu, thêm summary_clean, tokens, split (39 cột)
└── stopwords/
    └── vietnamese_stopwords.txt           # 120 từ dừng tiếng Việt & từ đặc thù web truyện
```

---

## 2. Lọc dữ liệu & Bảng ánh xạ 42 thể loại gốc $\rightarrow$ 22 nhãn chuẩn

### 2.1. Kết quả cào & lọc từ `webtruyen_raw.csv` (`src/crawl/crawler.py`, `src/crawl/merge_data.py` & `notebooks/01_eda.ipynb`)
1. **Cào toàn bộ 42 chuyên mục (`crawl_webtruyen`):** Duyệt hết **2.269 trang** thuộc 42 thẻ thể loại trên `wetruyen.com` (đích chuyển hướng của `webtruyen.vn`), giữ nguyên bản ghi thô theo từng thẻ (`category_crawled`), thu được **`55.684` bản ghi thô** (`12` cột) vào [webtruyen_raw.csv](raw/webtruyen_raw.csv).
2. **Loại truyện trùng tên & trùng văn án (`deduplicate_stories`):** Gom các bản ghi trùng `title` (chuẩn hóa chữ thường, NFC, khoảng trắng) và trùng `summary`, đồng thời **gộp toàn bộ thẻ `genres_raw` bằng `|`**, thu được **`18.621` truyện duy nhất** (loại `37.063` bản ghi trùng giữa các thẻ; vượt ngưỡng cấu hình `MIN_SUMMARY_WORDS = 10` từ, độ dài thực tế ngắn nhất là `15` từ).
3. **Chuẩn hóa 22 nhãn (`GENRE_MAPPING` & `NON_GENRE_TAGS` từ [src/crawl/sources.py](../src/crawl/sources.py)):** Tách riêng các thể loại giàu dữ liệu (`đam mỹ`, `bách hợp`, `linh dị`, `võng du`, `trinh thám`, `cung đấu / gia đấu`, `sủng`, `ngược`) và gộp một số thẻ gần nghĩa/ít mẫu (`khoa huyễn / mạt thế / dị năng`, `ngôn tình / nữ phụ / tổng tài`, `cổ đại / đông phương`, `đô thị / hiện đại / truyện teen`, `huyền huyễn / dị giới / thám hiểm / light novel`, `lịch sử / quân sự / dã sử`, `xuyên không / xuyên nhanh`). Chỉ loại **`18` truyện** không còn nhãn hợp lệ (chỉ gắn các thẻ bị loại: `Khác`, `Đoản Văn`, `Tiểu Thuyết`, `Phương Tây`, `Quan Trường`, `Việt Nam`), giữ lại **`18.603 / 18.621` (`99.90%`) truyện hợp lệ**.
4. Sinh mã `ST_0001` $\rightarrow$ `ST_18603`, cột `genres`, `n_labels`, `n_words` cùng **22 cột one-hot `label_*`**, lưu ra [stories.csv](processed/stories.csv) (`18.603` dòng × `36` cột).

### 2.2. Bảng ánh xạ đầy đủ 42 chuyên mục thể loại gốc trên `webtruyen.vn` (`18.621` truyện duy nhất)

| STT | Thẻ gốc (`genres_raw`) | Số truyện duy nhất | Nhãn chuẩn (`GENRE_LABELS`) |
|---|---|---|---|
| 1 | `Ngôn Tình` | 8.556 | `ngon_tinh` |
| 2 | `Cổ Đại` | 5.791 | `co_dai` |
| 3 | `Sủng` | 4.568 | `sung` |
| 4 | `Đam Mỹ` | 3.921 | `dam_my` |
| 5 | `Đô Thị` | 3.771 | `hien_dai` |
| 6 | `Nữ Cường` | 3.524 | `nu_cuong` |
| 7 | `Hiện Đại` | 2.459 | `hien_dai` |
| 8 | `Xuyên Không` | 2.213 | `xuyen_khong` |
| 9 | `Khác` | 2.144 | *Bỏ (`NON_GENRE_TAGS` — thẻ tạp)* |
| 10 | `Trọng Sinh` | 2.135 | `trong_sinh` |
| 11 | `Hài Hước` | 1.979 | `hai_huoc` |
| 12 | `Huyền Huyễn` | 1.734 | `huyen_huyen` |
| 13 | `Ngược` | 1.667 | `nguoc` |
| 14 | `Cung Đấu` | 1.227 | `cung_dau_gia_dau` |
| 15 | `Tiên Hiệp` | 1.217 | `tien_hiep` |
| 16 | `Điền Văn` | 1.069 | `dien_van` |
| 17 | `Đoản Văn` | 983 | *Bỏ (`NON_GENRE_TAGS` — độ dài)* |
| 18 | `Gia Đấu` | 802 | `cung_dau_gia_dau` |
| 19 | `Hệ Thống` | 789 | `he_thong` |
| 20 | `Linh Dị` | 619 | `linh_di` |
| 21 | `Tổng tài` | 586 | `ngon_tinh` |
| 22 | `Tiểu Thuyết` | 584 | *Bỏ (`NON_GENRE_TAGS` — hình thức)* |
| 23 | `Dị Năng` | 435 | `khoa_huyen_mat_the` |
| 24 | `Bách Hợp` | 416 | `bach_hop` |
| 25 | `Kiếm Hiệp` | 320 | `kiem_hiep` |
| 26 | `Dị Giới` | 259 | `huyen_huyen` |
| 27 | `Trinh Thám` | 233 | `trinh_tham` |
| 28 | `Mạt Thế` | 204 | `khoa_huyen_mat_the` |
| 29 | `Khoa Huyễn` | 203 | `khoa_huyen_mat_the` |
| 30 | `Truyện Teen` | 180 | `hien_dai` |
| 31 | `Xuyên Nhanh` | 153 | `xuyen_khong` |
| 32 | `Võng Du` | 148 | `vong_du` |
| 33 | `Lịch Sử` | 134 | `lich_su_quan_su` |
| 34 | `Nữ Phụ` | 119 | `ngon_tinh` |
| 35 | `Quân Sự` | 90 | `lich_su_quan_su` |
| 36 | `Đông Phương` | 55 | `co_dai` |
| 37 | `Phương Tây` | 54 | *Bỏ (`NON_GENRE_TAGS` — ít mẫu)* |
| 38 | `Quan Trường` | 36 | *Bỏ (`NON_GENRE_TAGS` — ít mẫu)* |
| 39 | `Thám Hiểm` | 18 | `huyen_huyen` |
| 40 | `Dã Sử` | 9 | `lich_su_quan_su` |
| 41 | `Việt Nam` | 5 | *Bỏ (`NON_GENRE_TAGS` — ít mẫu)* |
| 42 | `Light Novel` | 4 | `huyen_huyen` |

---

## 3. Tiền xử lý tiếng Việt & Chia tập (`data/processed/stories_clean.csv`)

1. **Làm sạch & chống rò rỉ nhãn (`clean.py`, `leakage.py`):** Chuẩn hóa Unicode NFC, giải mã HTML, bỏ URL/thẻ HTML/ký tự đặc biệt, cắt cụm header `"Thể loại: ... Văn án:"` và xóa từ khóa chuyên biệt trong `LEAKAGE_KEYWORDS` (không xóa từ đơn thông thường như `"sủng"`, `"ngược"`).
2. **Tách từ & lọc stopwords (`tokenizer.py`):** Tách từ bằng **`underthesea`**, loại bỏ **120 từ dừng** trong [vietnamese_stopwords.txt](stopwords/vietnamese_stopwords.txt).
3. **Chia tập phân tầng đa nhãn (`70% / 15% / 15%`, `IterativeStratification`, `RANDOM_STATE = 42`):**
   - `train`: **`13.297`** mẫu (`71.48%`) | `val`: **`2.634`** mẫu (`14.16%`) | `test`: **`2.672`** mẫu (`14.36%`), lưu tại [stories_clean.csv](processed/stories_clean.csv) (`18.603` dòng × `39` cột; `0` ô rỗng ở `split`, `1` mẫu `ST_12710` thuộc `train` bị rỗng `summary_clean`/`tokens` do toàn bộ đoạn nằm trong cụm header `"Thể loại:"`).

---

## 4. Bảng thống kê bộ dữ liệu hiện tại (`18.603` mẫu, `22` nhãn)

### 4.1. Chỉ số tổng quan
- **Tổng số mẫu hợp lệ:** `18.603` truyện (`100%` nguồn `webtruyen.vn`).
- **Số nhãn:** `22` nhãn đa nhãn; mọi nhãn có ít nhất `148` mẫu (`vong_du`), tỷ lệ mất cân bằng max/min là `58.8 : 1`.
- **Label cardinality:** `2.6704` nhãn/truyện | **Label density:** `0.1214`.
- **Độ dài `summary` (`n_words`):** nhỏ nhất `15` từ, trung bình `202.76` từ, trung vị `157` từ, lớn nhất `3.598` từ.

### 4.2. Phân bố chi tiết 22 nhãn và tập `train` / `val` / `test`

| Nhãn (`label_*`) | Tên hiển thị | Tổng (`18.603`) | Tỷ lệ (%) | Độ dài TB (từ) | `train` (`13.297`) | `val` (`2.634`) | `test` (`2.672`) |
|---|---|---|---|---|---|---|---|
| `ngon_tinh` | Ngôn tình (Nam–Nữ) | 8.702 | 46.78% | 208.0 | 5.542 | 1.584 | 1.576 |
| `co_dai` | Cổ đại | 5.833 | 31.36% | 186.2 | 4.083 | 875 | 875 |
| `hien_dai` | Hiện đại / Đô thị | 5.613 | 30.17% | 213.0 | 3.929 | 842 | 842 |
| `sung` | Sủng / Ngọt | 4.568 | 24.56% | 244.0 | 3.198 | 704 | 666 |
| `dam_my` | Đam mỹ | 3.921 | 21.08% | 248.0 | 2.745 | 588 | 588 |
| `nu_cuong` | Nữ cường | 3.524 | 18.94% | 163.8 | 2.514 | 505 | 505 |
| `xuyen_khong` | Xuyên không / Xuyên nhanh | 2.299 | 12.36% | 242.1 | 1.609 | 345 | 345 |
| `trong_sinh` | Trọng sinh | 2.135 | 11.48% | 210.3 | 1.494 | 320 | 321 |
| `hai_huoc` | Hài hước | 1.979 | 10.64% | 198.1 | 1.385 | 297 | 297 |
| `huyen_huyen` | Huyền huyễn / Dị giới | 1.891 | 10.17% | 200.8 | 1.324 | 283 | 284 |
| `cung_dau_gia_dau` | Cung đấu / Gia đấu | 1.830 | 9.84% | 172.9 | 1.300 | 275 | 255 |
| `nguoc` | Ngược | 1.667 | 8.96% | 186.8 | 1.061 | 300 | 306 |
| `tien_hiep` | Tiên hiệp | 1.217 | 6.54% | 182.0 | 852 | 182 | 183 |
| `dien_van` | Điền văn | 1.069 | 5.75% | 218.7 | 748 | 155 | 166 |
| `he_thong` | Hệ thống | 789 | 4.24% | 241.4 | 552 | 118 | 119 |
| `khoa_huyen_mat_the` | Khoa huyễn / Mạt thế / Dị năng | 728 | 3.91% | 215.5 | 510 | 109 | 109 |
| `linh_di` | Linh dị / Kinh dị | 619 | 3.33% | 209.3 | 430 | 94 | 95 |
| `bach_hop` | Bách hợp | 416 | 2.24% | 323.4 | 305 | 55 | 56 |
| `kiem_hiep` | Kiếm hiệp | 320 | 1.72% | 194.0 | 224 | 48 | 48 |
| `trinh_tham` | Trinh thám | 233 | 1.25% | 209.9 | 163 | 35 | 35 |
| `lich_su_quan_su` | Lịch sử / Quân sự | 176 | 0.95% | 219.3 | 123 | 26 | 27 |
| `vong_du` | Võng du / Game | 148 | 0.80% | 222.6 | 104 | 22 | 22 |

---

## 5. Lệnh tái tạo dữ liệu

```powershell
.\.venv\Scripts\python.exe -m src.crawl.crawler --sources webtruyen --resume
.\.venv\Scripts\python.exe -m src.crawl.merge_data
.\.venv\Scripts\python.exe -c "from src.config import STORIES_CSV, STORIES_CLEAN_CSV; from src.pipeline import prepare_clean_dataset; prepare_clean_dataset(STORIES_CSV, STORIES_CLEAN_CSV, preserve_existing_split=False)"
```
