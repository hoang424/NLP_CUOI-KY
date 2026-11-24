# Báo cáo Kết quả Phần 2: Tiền xử lý tiếng Việt & Trích xuất đặc trưng — _TV2_

- **Phụ trách:** Thành viên 2 (`feat/preprocess-features`)
- **Trạng thái:** **Đã hoàn thành khử trùng tên truyện, tiền xử lý (chuẩn hóa NFC, tách từ dính hoa/thường), chống rò rỉ nhãn, tách từ bằng `underthesea`, chia tập phân tầng và trích xuất đặc trưng TF-IDF trên `20.852` mẫu (`35` nhãn chuẩn) từ [stories.csv](../data/processed/stories.csv).**

---

## 1. Các công việc đã hoàn thành

### 1.1. Khử trùng liên nguồn, Làm sạch văn bản & Chống rò rỉ nhãn ([merge_data.py](../src/crawl/merge_data.py), [clean.py](../src/preprocessing/clean.py), [leakage.py](../src/preprocessing/leakage.py), [crawler.py](../src/crawl/crawler.py))
- `deduplicate_stories()`: gộp dữ liệu từ 4 nguồn (`webtruyen.vn`, `truyenfull.live`, `metruyenchu.com.vn`, `webnovel.vn`), loại bản ghi trùng tên (`normalize_title`) và trùng `summary`, đồng thời hợp nhất toàn bộ `genres_raw` từ **`58.751`** bản ghi thô xuống `20.963` truyện duy nhất trước khi lọc ra **`20.852`** mẫu hợp lệ trên `35` nhãn.
- `clean_text()` và `split_case_boundaries()`: chuẩn hóa Unicode `NFC`, giải mã HTML, xóa URL/thẻ HTML, tách từ dính liền tại ranh giới hoa/thường *trước* khi đưa về chữ thường (`tráiĐại` $\rightarrow$ `trái Đại`, `HEEditor` $\rightarrow$ `HE Editor`), giữ nguyên ngoại lệ `CAMELCASE_WHITELIST` (`WeChat`, `iPhone`, `YouTube`, `TikTok`...).
- `remove_label_leakage()`: cắt cụm `"Thể loại: ... (Văn án|Giới thiệu|Tóm tắt|Nội dung):"` ở đầu văn án và xóa các từ khóa chuyên biệt trong `LEAKAGE_KEYWORDS` ([src/config.py](../src/config.py) — bao phủ các cụm từ đa âm tiết đặc trưng của 35 nhãn như *thám hiểm, học đường, truyện teen, nữ phụ, dã sử, đoản văn*, đồng thời giữ nguyên các từ đơn thông dụng như *sủng, ngược, khác*).

#### Hiệu quả đo được trong [02_preprocessing.ipynb](../notebooks/02_preprocessing.ipynb) (`20.852` mẫu)

| Chỉ tiêu kiểm tra | Trước xử lý | Sau xử lý | Mức giảm |
|---|---:|---:|---|
| Văn án chứa ký tự tổ hợp NFD (`summary` $\rightarrow$ `summary_clean`) | 1.036 | **0** | 100% |
| Truyện có từ dính ranh giới hoa/thường ngoài `CAMELCASE_WHITELIST` | 6.408 | **0** | 100% |
| Văn án có cụm header `"Thể loại: ... Văn án/Giới thiệu:"` | 2.881 | **0** | 100% |
| Văn án có từ khóa rò rỉ nhãn `LEAKAGE_KEYWORDS` (`summary_clean`) | 9.552 | **2** | 99,98% |

### 1.2. Tách từ tiếng Việt & Stopwords ([tokenizer.py](../src/preprocessing/tokenizer.py))
- Tách từ ghép tiếng Việt bằng **`underthesea`** và lọc **120 từ dừng** từ [vietnamese_stopwords.txt](../data/stopwords/vietnamese_stopwords.txt).

### 1.3. Thực thi Notebook tiền xử lý & xuất `stories_clean.csv` ([02_preprocessing.ipynb](../notebooks/02_preprocessing.ipynb), [pipeline.py](../src/pipeline.py))
- Đã thực thi toàn bộ [02_preprocessing.ipynb](../notebooks/02_preprocessing.ipynb) từ [stories.csv](../data/processed/stories.csv) (`20.852` mẫu, `35` nhãn), xuất [stories_clean.csv](../data/processed/stories_clean.csv) (`20.852` dòng × `52` cột) và `models/tfidf_vectorizer.pkl` (`10.000` chiều), chia phân tầng đa nhãn `IterativeStratification` (`70% / 15% / 15%`, `RANDOM_STATE = 42`):
  - **`train = 14.613`** (`70,08%`) | **`val = 3.125`** (`14,99%`) | **`test = 3.114`** (`14,93%`), `0` ô rỗng ở `split`, `1` mẫu thuộc `train` (`ST_12713` — *Nhập Thụy Chỉ Nam*) bị rỗng `summary_clean`/`tokens` do toàn bộ đoạn nằm trong cụm header `"Thể loại:"`; mọi nhãn trong 35 nhãn ở cả `val` và `test` đều có $\ge 10$ mẫu.

| Nhãn chuẩn | `train` (`14.613`) | `val` (`3.125`) | `test` (`3.114`) | Nhãn chuẩn | `train` (`14.613`) | `val` (`3.125`) | `test` (`3.114`) |
|---|---:|---:|---:|---|---:|---:|---:|
| `ngon_tinh` | 6.222 | 1.721 | 1.721 | `trinh_tham` | 330 | 70 | 71 |
| `dam_my` | 2.792 | 598 | 598 | `tham_hiem` | 212 | 11 | 12 |
| `bach_hop` | 369 | 53 | 54 | `vong_du` | 375 | 56 | 57 |
| `tong_tai` | 388 | 93 | 107 | `he_thong` | 769 | 155 | 156 |
| `nu_phu` | 326 | 28 | 30 | `xuyen_khong` | 2.073 | 444 | 445 |
| `nu_cuong` | 2.683 | 544 | 545 | `trong_sinh` | 1.707 | 365 | 366 |
| `sung` | 3.255 | 781 | 779 | `co_dai` | 4.251 | 911 | 911 |
| `nguoc` | 997 | 382 | 335 | `hien_dai` | 4.378 | 938 | 938 |
| `hai_huoc` | 1.521 | 326 | 326 | `hoc_duong` | 420 | 45 | 45 |
| `tien_hiep` | 987 | 211 | 212 | `cung_dau` | 901 | 188 | 189 |
| `kiem_hiep` | 289 | 62 | 62 | `gia_dau` | 584 | 120 | 133 |
| `huyen_huyen` | 1.702 | 365 | 365 | `dien_van` | 850 | 176 | 188 |
| `di_gioi` | 297 | 64 | 64 | `lich_su` | 352 | 62 | 63 |
| `di_nang` | 442 | 94 | 95 | `quan_su` | 172 | 37 | 37 |
| `mat_the` | 325 | 52 | 40 | `quan_truong` | 50 | 10 | 11 |
| `khoa_huyen` | 369 | 75 | 76 | `doan_van` | 684 | 196 | 177 |
| `linh_di` | 534 | 114 | 115 | `tieu_thuyet` | 364 | 119 | 101 |
| `khac` | 2.065 | 442 | 443 | — | — | — | — |

### 1.4. Trích xuất đặc trưng (`src/features/`)
- [bow_tfidf.py](../src/features/bow_tfidf.py): Fit `TfidfVectorizer` (`ngram_range=(1, 2)`, `max_features=10000`, `sublinear_tf=True`) **chỉ trên tập `train`** (`(14613, 10000)`), transform `val` (`(3125, 10000)`) và `test` (`(3114, 10000)`), lưu tại `models/tfidf_vectorizer.pkl`.

---

## 2. Kết quả kiểm thử ([tests/test_preprocessing.py](../tests/test_preprocessing.py))
- Toàn bộ **10/10 tests tiền xử lý PASS** (và tổng **25/25 tests toàn dự án PASS**, bao gồm các kiểm thử crawl & chuẩn hóa 35 nhãn).
