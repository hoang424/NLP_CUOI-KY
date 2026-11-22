# Báo cáo Kết quả Phần 2: Tiền xử lý tiếng Việt & Trích xuất đặc trưng — _TV2_

- **Phụ trách:** Thành viên 2 (`feat/preprocess-features`)
- **Trạng thái:** **Đã hoàn thành khử trùng tên truyện, tiền xử lý (tách từ dính ranh giới hoa/thường), chống rò rỉ nhãn, tách từ bằng `underthesea`, chia tập phân tầng và trích xuất đặc trưng TF-IDF trên `18.603` mẫu (`22` nhãn chuẩn) từ [stories.csv](file:///c:/Users/ACER/Downloads/NLP_CU%E1%BB%90I%20K%E1%BB%B2/data/processed/stories.csv).**

---

## 1. Các công việc đã hoàn thành

### 1.1. Khử trùng tên truyện, Làm sạch văn bản & Chống rò rỉ nhãn ([merge_data.py](file:///c:/Users/ACER/Downloads/NLP_CU%E1%BB%90I%20K%E1%BB%B2/src/crawl/merge_data.py), [clean.py](file:///c:/Users/ACER/Downloads/NLP_CU%E1%BB%90I%20K%E1%BB%B2/src/preprocessing/clean.py), [leakage.py](file:///c:/Users/ACER/Downloads/NLP_CU%E1%BB%90I%20K%E1%BB%B2/src/preprocessing/leakage.py), [crawler.py](file:///c:/Users/ACER/Downloads/NLP_CU%E1%BB%90I%20K%E1%BB%B2/src/crawl/crawler.py))
- `deduplicate_stories()`: loại bản ghi trùng tên (`normalize_title`) và trùng `summary`, đồng thời gộp toàn bộ `genres_raw` từ `55.684` dòng thô xuống `18.621` truyện duy nhất trước khi ánh xạ 22 nhãn (`prepare_clean_dataset()` cũng kiểm tra tự động bước này nếu đầu vào còn trùng tên).
- `clean_text()` và `split_case_boundaries()`: chuẩn hóa Unicode `NFC` (xử lý `970` văn án có dấu tổ hợp NFD), giải mã HTML, xóa URL/thẻ HTML, **tách từ dính liền tại ranh giới hoa/thường** *trước* khi đưa về chữ thường (`tráiĐại` $\rightarrow$ `trái Đại`, `CănThể` $\rightarrow$ `Căn Thể`, `HEEditor` $\rightarrow$ `HE Editor`), đồng thời giữ nguyên danh sách ngoại lệ `CAMELCASE_WHITELIST` (`WeChat`, `iPhone`, `YouTube`, `TikTok`...), giữ chữ tiếng Việt có dấu và chữ số.
- `_block_text()` trong [crawler.py](file:///c:/Users/ACER/Downloads/NLP_CU%E1%BB%90I%20K%E1%BB%B2/src/crawl/crawler.py): khắc phục nguyên nhân gốc ở bước cào dữ liệu bằng cách chèn khoảng trắng tại các thẻ khối (`<br>`, `<p>`, `<div>`, `<li>`...) trước khi gọi `.text_content()` cho các lần thu thập sau.
- `remove_label_leakage()`: cắt bỏ cụm `"Thể loại: ... (Văn án|Giới thiệu|Tóm tắt|Nội dung):"` ở đầu văn án và xóa các từ khóa chuyên biệt trong `LEAKAGE_KEYWORDS` ([src/config.py](file:///c:/Users/ACER/Downloads/NLP_CU%E1%BB%90I%20K%E1%BB%B2/src/config.py); không xóa từ đơn thông thường như `"sủng"`, `"ngược"`).

#### Hiệu quả đo được trước / sau khi áp dụng tách từ dính hoa/thường (`18.603` mẫu)

| Chỉ tiêu kiểm tra | Trước xử lý | Sau xử lý | Ghi chú |
|---|---:|---:|---|
| Truyện còn token dính hoa/thường trong `tokens` | `4.513` (`24,26%`) | **`9`** (`0,05%`) | Giảm `99,8%` (từ `6.334` dạng token xuống `21` dạng ngoại lệ/mojibake/từ 2 ký tự như `zZ` hoặc từ xuất hiện lặp dạng viết thường) |
| Truyện còn cụm metadata dính tiền tố (`...thể loại`, `...giới thiệu`, `...editor`...) | `3.285` (`17,66%`) | **`487`** (`2,62%`) | Giảm `85,2%` (chủ yếu còn `username/số/chữ Hán + chữ thường` như `ngocquynh520editor`, `1v1giới thiệu`, `兔thể loại`) |
| Truyện còn từ khóa `LEAKAGE_KEYWORDS` dạng chuỗi con trong `summary_clean` | `678` (`3,64%`) | **`39`** (`0,21%`) | Giảm `94,2%` (chủ yếu là trùng chuỗi con ngẫu nhiên như `linh dịch`/`do thiên`, dính số như `trùng sinh2`, hoặc 2 ca nối liền sau khi xóa từ kề như `ST_10103`, `ST_12110`) |
| Truyện có token chữ cái `>= 9` ký tự trong `summary_clean` | `3.123` (`16,79%`) | **`1.100`** (`5,91%`) | Phần còn lại đa số là từ tiếng Anh hợp lệ (`livestream`, `pheromone`, `converter`...) |

### 1.2. Tách từ tiếng Việt & Stopwords ([tokenizer.py](file:///c:/Users/ACER/Downloads/NLP_CU%E1%BB%90I%20K%E1%BB%B2/src/preprocessing/tokenizer.py))
- Tách từ ghép tiếng Việt bằng **`underthesea`** và lọc **120 từ dừng** từ [vietnamese_stopwords.txt](file:///c:/Users/ACER/Downloads/NLP_CU%E1%BB%B2/data/stopwords/vietnamese_stopwords.txt).

### 1.3. Thực thi Notebook tiền xử lý & xuất `stories_clean.csv` ([02_preprocessing.ipynb](file:///c:/Users/ACER/Downloads/NLP_CU%E1%BB%90I%20K%E1%BB%B2/notebooks/02_preprocessing.ipynb), [pipeline.py](file:///c:/Users/ACER/Downloads/NLP_CU%E1%BB%90I%20K%E1%BB%B2/src/pipeline.py))
- Đã chạy lại toàn bộ 6 code cells của [02_preprocessing.ipynb](file:///c:/Users/ACER/Downloads/NLP_CU%E1%BB%90I%20K%E1%BB%B2/notebooks/02_preprocessing.ipynb) từ đầu vào `stories.csv` (**`18.603` mẫu**) sau khi bổ sung bước tách từ dính hoa/thường, xuất lại [stories_clean.csv](file:///c:/Users/ACER/Downloads/NLP_CU%E1%BB%90I%20K%E1%BB%B2/data/processed/stories_clean.csv) (`18.603` dòng × `39` cột) và `models/tfidf_vectorizer.pkl` (`10.000` chiều), giữ nguyên cấu trúc chia phân tầng đa nhãn `IterativeStratification` (`70% / 15% / 15%`, `RANDOM_STATE = 42`):
  - **`train = 13.297`**, **`val = 2.634`**, **`test = 2.672`** (`0` ô rỗng ở `split`; có `1` mẫu thuộc `train` là `ST_12710` rỗng `summary_clean`/`tokens` do toàn bộ văn bản nằm trong cụm header `"Thể loại:"`; mọi nhãn trong cả 3 tập đều có `>= 22` mẫu).

| Nhãn chuẩn | `train` (`13.297`) | `val` (`2.634`) | `test` (`2.672`) |
|---|---|---|---|
| `ngon_tinh` | 5.542 | 1.584 | 1.576 |
| `dam_my` | 2.745 | 588 | 588 |
| `bach_hop` | 305 | 55 | 56 |
| `tien_hiep` | 852 | 182 | 183 |
| `kiem_hiep` | 224 | 48 | 48 |
| `huyen_huyen` | 1.324 | 283 | 284 |
| `linh_di` | 430 | 94 | 95 |
| `khoa_huyen_mat_the` | 510 | 109 | 109 |
| `vong_du` | 104 | 22 | 22 |
| `trinh_tham` | 163 | 35 | 35 |
| `dien_van` | 748 | 155 | 166 |
| `lich_su_quan_su` | 123 | 26 | 27 |
| `co_dai` | 4.083 | 875 | 875 |
| `hien_dai` | 3.929 | 842 | 842 |
| `cung_dau_gia_dau` | 1.300 | 275 | 255 |
| `xuyen_khong` | 1.609 | 345 | 345 |
| `trong_sinh` | 1.494 | 320 | 321 |
| `he_thong` | 552 | 118 | 119 |
| `nu_cuong` | 2.514 | 505 | 505 |
| `sung` | 3.198 | 704 | 666 |
| `nguoc` | 1.061 | 300 | 306 |
| `hai_huoc` | 1.385 | 297 | 297 |

### 1.4. Trích xuất đặc trưng (`src/features/`)
- [bow_tfidf.py](file:///c:/Users/ACER/Downloads/NLP_CU%E1%BB%90I%20K%E1%BB%B2/src/features/bow_tfidf.py): Fit `TfidfVectorizer` (`ngram_range=(1, 2)`, `max_features=10000`, `sublinear_tf=True`) **chỉ trên tập `train`** (`(13297, 10000)`), transform `val` (`(2634, 10000)`) và `test` (`(2672, 10000)`), lưu tại `models/tfidf_vectorizer.pkl`.

---

## 2. Kết quả kiểm thử ([tests/test_preprocessing.py](file:///c:/Users/ACER/Downloads/NLP_CU%E1%BB%90I%20K%E1%BB%B2/tests/test_preprocessing.py))
- Toàn bộ **10/10 tests tiền xử lý PASS** (và tổng **20/20 tests toàn dự án PASS**, bao gồm test `_block_text()` trong [tests/test_crawler_and_merge.py](file:///c:/Users/ACER/Downloads/NLP_CU%E1%BB%90I%20K%E1%BB%B2/tests/test_crawler_and_merge.py)).

---

## 3. Hạn chế còn tồn đọng chưa xử lý

1. **Từ dính liền cùng kiểu chữ (`thường+thường` và `HOA+HOA`) và chữ Hán dính chữ thường:**
   - Khoảng **`141` truyện (`0,76%`)** có từ dính kiểu `chữ_thường + chữ_thường` (ví dụ: `đờingười`, `câuchuyện`, `canhthuốc`, `nuôicủa`, `thiếunữ`, `làkhông`) và **`20` truyện (`0,11%`)** có từ dính kiểu `CHỮ_HOA + CHỮ_HOA` (ví dụ: `HEGIỚI`, `CHƯƠNGTHỂ`), cộng thêm một số ít ca ký tự chữ Hán (không phân biệt hoa/thường) dính liền chữ thường tiếng Việt (ví dụ: `兔thể loại`, `长恨thể loại`). Do không có ranh giới hoa/thường, hàm `split_case_boundaries()` không tách được các trường hợp này.
   - *Hướng khắc phục tương lai:* bổ sung bước tách âm tiết bằng từ điển tần suất âm tiết tiếng Việt + quy hoạch động hoặc crawl lại bằng `_block_text()`.
2. **Chuỗi dính không dấu, tên miền, username và chữ+số:**
   - Các chuỗi như `truyenfullvn`, `wikidicheditor`, `ngocquynh520editor`, `thiệu1`, `1v1giới thiệu` được cố ý giữ nguyên vì việc tách tự động chuỗi không dấu hoặc chữ+số rất dễ cắt nhầm các từ tiếng Anh và thuật ngữ truyện mạng hợp lệ (`livestream`, `18t`, `nam9`, `2d`, `1v1`).
3. **Từ metadata đã tách nhưng chưa lọc khỏi stopwords:**
   - Sau khi `split_case_boundaries()` tách riêng thành công (ví dụ `HEEditor` $\rightarrow$ `he editor`, `chươngEdit` $\rightarrow$ `chương edit`), các từ metadata như `editor`, `edit`, `convert`, `converter`, `beta` trở thành token sạch nhưng vẫn tồn tại trong cột `tokens` do chưa được đưa vào [vietnamese_stopwords.txt](file:///c:/Users/ACER/Downloads/NLP_CU%E1%BB%90I%20K%E1%BB%B2/data/stopwords/vietnamese_stopwords.txt).
4. **Nhiễu văn bản khác còn tồn đọng:**
   - Vẫn còn một số ít mẫu bị lỗi mã hóa font (mojibake, ví dụ: `khĂŽng`, `tĂŽi`, `lĂ`, `vÄ`) và từ bị chèn dấu chấm giữa để lách kiểm duyệt (ví dụ: `g·iết` $\rightarrow$ `g iết`).
5. **Trạng thái đồng bộ toàn bộ pipeline:**
   - Sau khi cập nhật [stories_clean.csv](file:///c:/Users/ACER/Downloads/NLP_CU%E1%BB%90I%20K%E1%BB%B2/data/processed/stories_clean.csv), toàn bộ các notebook `02` $\rightarrow$ `03` $\rightarrow$ `04`, `models/tfidf_vectorizer.pkl`, các mô hình ML/DL trong `models/`, bảng [experiments.csv](file:///c:/Users/ACER/Downloads/NLP_CU%E1%BB%90I%20K%E1%BB%B2/outputs/tables/experiments.csv), biểu đồ `outputs/figures/` và các báo cáo TV3/TV4 **đều đã được chạy lại và đồng bộ hoàn toàn** với dữ liệu mới.
