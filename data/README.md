# Tài liệu Dữ liệu & Nhật ký Thao tác trên Thư mục `data/`

Tài liệu này mô tả cấu trúc thư mục `data/`, nguồn thu thập đa trang (`webtruyen.vn`, `truyenfull.live`, `metruyenchu.com.vn`, `webnovel.vn`), bảng định nghĩa **42 thẻ gốc $\rightarrow$ 35 nhãn chuẩn** cùng cơ sở logic, và thống kê bộ dữ liệu **`20.852` truyện** cho đồ án **NLP Đề 11**.

---

## 1. Cây thư mục `data/` và vai trò từng file

```text
data/
├── README.md                              # Tài liệu này
├── raw/                                   # Dữ liệu crawl thô từ 4 nguồn đang kích hoạt (ACTIVE_RAW_CSVS)
│   ├── webtruyen_raw.csv                  # [ĐANG DÙNG - GỐC] 55.684 bản ghi thô (42 thẻ × 2.269 trang) từ webtruyen.vn
│   ├── truyenfull_raw.csv                 # [ĐANG DÙNG - BỔ SUNG] 2.400 bản ghi thô (10 thể loại thiếu mẫu) từ truyenfull.live
│   ├── metruyenchu_raw.csv                # [ĐANG DÙNG - BỔ SUNG] 420 truyện có văn án thật (145 Dã sử & nhãn hiếm) từ metruyenchu.com.vn
│   ├── webnovel_raw.csv                   # [ĐANG DÙNG - BỔ SUNG] 247 bản ghi từ webnovel.vn (CSV)
│   ├── webnovel_raw.json                  # [Lưu trữ] 247 bản ghi webnovel.vn (JSON)
│   └── wiki_raw.csv                       # [Lưu trữ cũ - không gộp train] 255 bản ghi từ Wikipedia / Wikisource
├── labeled/                               # Dữ liệu định dạng Label Studio (lưu trữ)
│   ├── webnovel_label_studio.json         # 247 bản ghi webnovel.vn
│   └── wiki_label_studio.json             # 255 bản ghi Wiki
├── processed/                             # Dữ liệu sau lọc trùng & chuẩn hóa 35 nhãn
│   ├── stories.csv                        # 20.852 truyện duy nhất, 35 cột label_* (49 cột)
│   └── stories_clean.csv                  # 20.852 mẫu, thêm summary_clean, tokens, split (52 cột)
└── stopwords/
    └── vietnamese_stopwords.txt           # 120 từ dừng tiếng Việt & từ đặc thù web truyện
```

---

## 2. Nguồn thu thập, Cào bổ sung & Chuẩn hóa `42 thẻ gốc -> 35 nhãn chuẩn`

### 2.1. Quy trình cào bổ sung và gộp 4 nguồn (`src/crawl/crawler.py`, `src/crawl/merge_data.py`)
1. **Nguồn gốc chủ lực (`webtruyen_raw.csv`):** Cào đủ 42 chuyên mục (`2.269` trang) trên `wetruyen.com` (`55.684` bản ghi thô $\rightarrow$ `18.621` truyện duy nhất $\rightarrow$ `18.617` mẫu hợp lệ theo 35 nhãn). Do các nhãn như `Thám Hiểm` (`18`), `Quan Trường` (`36`), `Quân Sự` (`90`), `Nữ Phụ` (`119`), `Lịch Sử / Dã Sử` (`143`), `Võng Du` (`148`), `Học đường / Truyện teen` (`180`) bị thiếu dữ liệu nặng, nhóm tiến hành cào bổ sung có kiểm soát.
2. **Cào bổ sung 10 thể loại thiếu mẫu trên `truyenfull.live` (`crawl_truyenfull`):** Trang công khai hợp lệ theo `robots.txt`, dùng chung hệ thẻ đa nhãn và cung cấp định nghĩa chuyên mục (`div.cat-desc`). Thu được **`2.400` bản ghi thô** (`2.116` truyện duy nhất nội bộ) vào [truyenfull_raw.csv](raw/truyenfull_raw.csv).
3. **Mở rộng `metruyenchu.com.vn` (`crawl_metruyenchu(..., merge_existing=True)`):** Kết hợp `374` truyện có văn án thật từ đợt thu thập tổng hợp (gồm `145` truyện `Dã Sử`) với `46` truyện cào bổ sung ở 10 chuyên mục hiếm $\rightarrow$ **`420` truyện** trong [metruyenchu_raw.csv](raw/metruyenchu_raw.csv).
4. **Kết hợp `webnovel.vn` ([webnovel_raw.csv](raw/webnovel_raw.csv)):** `247` bản ghi (`245` duy nhất); `NON_GENRE_TAGS` tự động lọc sách phi hư cấu (`Phát triển bản thân`, `Tâm linh`, `Kinh doanh`), giữ lại **`137` truyện mạng**.
5. **Khử trùng và tạo `stories.csv`:** Từ `58.751` bản ghi thô $\rightarrow$ `deduplicate_stories()` thu được `20.963` truyện duy nhất $\rightarrow$ lọc nhãn hợp lệ thu được **`20.852` truyện** (`webtruyen.vn`: `18.230`, `truyenfull.live`: `2.069`, `metruyenchu.com.vn`: `418`, `webnovel.vn`: `135`) với **35 cột `label_*`** tại [stories.csv](processed/stories.csv) (`49` cột).

### 2.2. Bảng định nghĩa 42 thẻ gốc & Cơ sở logic giữ `35` nhãn chuẩn

> **Lưu ý về định nghĩa:** `wetruyen.com` không đăng mô tả định nghĩa trên trang chuyên mục; phần lớn định nghĩa (`38/42` thẻ) được trích trực tiếp từ `<div class="cat-desc">` của `truyenfull.live`, `4` thẻ còn lại (`Hiện Đại`, `Tổng tài`, `Tiểu Thuyết`, `Dã Sử`) dùng định nghĩa chuẩn chuyên ngành webnovel. Chi tiết phân tích Jaccard và đồng xuất hiện xem thêm tại [01_tv1_thu_thap_va_eda.md](../bao_cao_ket_qua/01_tv1_thu_thap_va_eda.md).

- **Hợp nhất 3 cặp thẻ đồng trục ngữ nghĩa:** `Đô Thị + Hiện Đại` $\rightarrow$ `hien_dai`; `Xuyên Không + Xuyên Nhanh` $\rightarrow$ `xuyen_khong`; `Lịch Sử + Dã Sử` $\rightarrow$ `lich_su`.
- **Duy trì 32 nhãn riêng biệt** (gồm `Học đường / Truyện teen` (`hoc_duong`), `Dị Năng`, `Mạt Thế`, `Khoa Huyễn`, `Dị Giới`, `Thám Hiểm`, `Cung Đấu`, `Gia Đấu`, `Tổng tài`, `Nữ Phụ`, `Quân Sự`, `Quan Trường`, `Đoản Văn`, `Tiểu Thuyết`, `Khác`...).
- **Loại bỏ 4 thẻ phân loại theo nguồn gốc quốc gia / khu vực địa lý (`NON_GENRE_TAGS`):** `Việt Nam` (`5`), `Light Novel` (`4`), `Đông Phương` (`55`), `Phương Tây` (`54`).

| STT | Thẻ gốc (`genres_raw`) | Định nghĩa thể loại | Số truyện (`webtruyen.vn`) | Nhãn chuẩn (`35` nhãn) |
|---|---|---|---:|---|
| 1 | `Ngôn Tình` | Truyện lãng mạn về tình yêu nam–nữ của nhân vật chính | 8.556 | `ngon_tinh` |
| 2 | `Cổ Đại` | Truyện lấy bối cảnh xã hội thời cổ đại phong kiến | 5.791 | `co_dai` |
| 3 | `Sủng` | Ngọt ngào vui vẻ, nhân vật chính được cưng chiều hết mình | 4.568 | `sung` |
| 4 | `Đam Mỹ` | Truyện tình cảm giữa nam và nam | 3.921 | `dam_my` |
| 5 | `Đô Thị` | Truyện diễn ra chủ yếu trong môi trường đô thị hiện đại | 3.771 | `hien_dai` (gộp) |
| 6 | `Nữ Cường` | Nhân vật nữ sở hữu tính cách, năng lực mạnh mẽ | 3.524 | `nu_cuong` |
| 7 | `Hiện Đại` | Truyện lấy bối cảnh xã hội thời kỳ hiện đại | 2.459 | `hien_dai` (gộp) |
| 8 | `Xuyên Không` | Nhân vật chính bị đưa tới một không gian hoặc thời gian khác | 2.213 | `xuyen_khong` (gộp) |
| 9 | `Khác` | Truyện đa chủ đề không rơi vào một chuyên mục hẹp duy nhất | 2.144 | `khac` |
| 10 | `Trọng Sinh` | Nhân vật chính sống lại sau khi chết hoặc quay về quá khứ | 2.135 | `trong_sinh` |
| 11 | `Hài Hước` | Truyện có nội dung hài hước, đem lại tiếng cười vui vẻ | 1.979 | `hai_huoc` |
| 12 | `Huyền Huyễn` | Truyện có yếu tố huyền bí, pháp thuật mà khoa học khó giải thích | 1.734 | `huyen_huyen` |
| 13 | `Ngược` | Truyện nhiều tình tiết bi thương, hành hạ thể xác hoặc tinh thần | 1.667 | `nguoc` |
| 14 | `Cung Đấu` | Đấu đá, âm mưu tính toán trong hoàng cung phong kiến | 1.227 | `cung_dau` |
| 15 | `Tiên Hiệp` | Hành trình tu luyện tu chân thành tiên với các cảnh giới tu sĩ | 1.217 | `tien_hiep` |
| 16 | `Điền Văn` | Cuộc sống điền viên nhẹ nhàng, làm ruộng, buôn bán và gia đình | 1.069 | `dien_van` |
| 17 | `Đoản Văn` | Truyện có dung lượng ngắn, thường từ một đến vài chương | 983 | `doan_van` |
| 18 | `Gia Đấu` | Đấu đá, mưu tính giữa các thành viên trong gia tộc/dòng họ | 802 | `gia_dau` |
| 19 | `Hệ Thống` | Nhân vật chính mang hệ thống cộng sinh giao nhiệm vụ – ban thưởng | 789 | `he_thong` |
| 20 | `Linh Dị` | Truyện có yếu tố ma quỷ, tâm linh rùng rợn, kinh dị | 619 | `linh_di` |
| 21 | `Tổng tài` | Ngôn tình lấy nam chính là chủ tịch, giám đốc tập đoàn quyền lực | 586 | `tong_tai` |
| 22 | `Tiểu Thuyết` | Văn xuôi tự sự dài hơi có cốt truyện tâm lý – xã hội hoàn chỉnh | 584 | `tieu_thuyet` |
| 23 | `Dị Năng` | Nhân vật sở hữu năng lực siêu nhiên đặc biệt vượt người thường | 435 | `di_nang` |
| 24 | `Bách Hợp` | Truyện tình cảm giữa nữ và nữ | 416 | `bach_hop` |
| 25 | `Kiếm Hiệp` | Rèn luyện võ công, hành tẩu giang hồ trong giới võ lâm | 320 | `kiem_hiep` |
| 26 | `Dị Giới` | Bối cảnh diễn ra ở một thế giới kỳ dị khác thế giới thực | 259 | `di_gioi` |
| 27 | `Trinh Thám` | Truyện điều tra, suy luận phá án hình sự ly kỳ | 233 | `trinh_tham` |
| 28 | `Mạt Thế` | Bối cảnh ngày tận thế, thế giới sụp đổ vì thảm họa / đại dịch | 204 | `mat_the` |
| 29 | `Khoa Huyễn` | Truyện giả tưởng có yếu tố khoa học công nghệ, vũ trụ, tinh tế | 203 | `khoa_huyen` |
| 30 | `Truyện Teen` | Truyện viết cho lứa tuổi học trò thanh thiếu niên, bối cảnh học đường | 180 | `hoc_duong` |
| 31 | `Xuyên Nhanh` | Xuyên qua nhiều thế giới liên tiếp nhau để hoàn thành nhiệm vụ | 153 | `xuyen_khong` (gộp) |
| 32 | `Võng Du` | Bối cảnh game online, thực tế ảo trên mạng với kỹ xảo kỳ ảo | 148 | `vong_du` |
| 33 | `Lịch Sử` | Bối cảnh lịch sử tương đồng hoặc bám sát lịch sử thế giới thực | 134 | `lich_su` (gộp) |
| 34 | `Nữ Phụ` | Nhân vật chính là vai nữ phụ trong tiểu thuyết tìm cách đổi vận | 119 | `nu_phu` |
| 35 | `Quân Sự` | Truyện yếu tố quân sự, chiến tranh, môi trường quân đội | 90 | `quan_su` |
| 36 | `Đông Phương` | Truyện có bối cảnh thuộc các nước phương Đông | 55 | *Bỏ (Địa lý)* |
| 37 | `Phương Tây` | Tác phẩm của tác giả / văn hóa phương Tây | 54 | *Bỏ (Địa lý)* |
| 38 | `Quan Trường` | Quá trình đấu trí chính trị và thăng quan tiến chức | 36 | `quan_truong` |
| 39 | `Thám Hiểm` | Phiêu lưu thám hiểm các địa danh kỳ bí, di tích, vùng đất hiểm trở | 18 | `tham_hiem` |
| 40 | `Dã Sử` | Truyện phỏng theo giai thoại lịch sử dân gian ngoài chính sử | 9 | `lich_su` (gộp) |
| 41 | `Việt Nam` | Tác phẩm thuộc nền văn học Việt Nam | 5 | *Bỏ (Quốc gia)* |
| 42 | `Light Novel` | Tiểu thuyết nhẹ xuất xứ từ Nhật Bản hướng tới độc giả trẻ | 4 | *Bỏ (Quốc gia)* |

---

## 3. Tiền xử lý tiếng Việt & Chia tập (`data/processed/stories_clean.csv`)

1. **Làm sạch & chống rò rỉ nhãn (`clean.py`, `leakage.py`):** Chuẩn hóa Unicode `NFC` (`1.036 -> 0` lỗi `NFD`), `split_case_boundaries()` tách từ dính hoa/thường (`6.408 -> 0`), giải mã HTML, bỏ URL/ký tự đặc biệt, cắt `2.881 -> 0` cụm header `"Thể loại: ... Văn án/Giới thiệu:"` và xóa từ khóa chuyên biệt trong `LEAKAGE_KEYWORDS` (`9.552 -> 2`).
2. **Tách từ & lọc stopwords (`tokenizer.py`):** Tách từ bằng **`underthesea`**, lọc **120 từ dừng** trong [vietnamese_stopwords.txt](stopwords/vietnamese_stopwords.txt).
3. **Chia tập phân tầng đa nhãn (`70% / 15% / 15%`, `IterativeStratification`, `RANDOM_STATE = 42`):**
   - `train`: **`14.613`** (`70.08%`) | `val`: **`3.125`** (`14.99%`) | `test`: **`3.114`** (`14.93%`), lưu tại [stories_clean.csv](processed/stories_clean.csv) (`20.852` dòng × `52` cột; `0` ô rỗng ở `split`; `1` mẫu `ST_12713` trong `train` bị rỗng `tokens` do toàn bộ văn án nằm trong header `"Thể loại:"`; mọi nhãn trong cả `val` và `test` đều có $\ge 10$ mẫu).

---

## 4. Thống kê tổng quan bộ dữ liệu (`20.852` mẫu, `35` nhãn)

- **Label cardinality:** `3.0601` nhãn/truyện | **Label density:** `0.0874`.
- **Độ dài `summary` (`n_words`):** nhỏ nhất `15` từ, trung bình `205.90` từ, trung vị `161` từ, lớn nhất `3.598` từ.
- Xem bảng phân bố chi tiết đầy đủ `35` nhãn trên `train / val / test` và kết quả tăng cường 10 nhãn hiếm tại [01_tv1_thu_thap_va_eda.md](../bao_cao_ket_qua/01_tv1_thu_thap_va_eda.md).

---

## 5. Lệnh tái tạo dữ liệu

```powershell
.\.venv\Scripts\python.exe -m src.crawl.crawler --sources truyenfull metruyenchu
.\.venv\Scripts\python.exe -m src.crawl.merge_data
.\.venv\Scripts\python.exe -c "from src.config import STORIES_CSV, STORIES_CLEAN_CSV; from src.pipeline import prepare_clean_dataset; prepare_clean_dataset(STORIES_CSV, STORIES_CLEAN_CSV, preserve_existing_split=False)"
```
