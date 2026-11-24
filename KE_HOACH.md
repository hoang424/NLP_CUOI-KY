# Kế hoạch đồ án NLP: Phân loại thể loại truyện tiếng Việt

## Yêu cầu đề bài (đề số 11: Văn học × Text classification)

- Nhóm tối đa 3–4 thành viên; đăng ký nhóm và đề qua **LMS** (mỗi đề tối đa 2 nhóm).
- Dữ liệu **tiếng Việt**, thuộc lĩnh vực **văn học**; tự crawl hoặc dùng dataset có sẵn.
- Có đủ quy trình: **crawl đa nguồn → EDA → tiền xử lý → chuẩn hóa nhãn → xây dựng/huấn luyện mô hình → đánh giá → phân tích lỗi → ứng dụng**.
- Dùng ít nhất **3 mô hình ML truyền thống** (`Multinomial Naive Bayes`, `Linear SVM`, `Logistic Regression`) và **1 mô hình Deep Learning** (`TextCNN`).
- Có **ứng dụng** (`app/streamlit_app.py`) chạy trực quan để nhập tóm tắt và dự đoán đa nhãn.
- Nộp: code trên **GitHub** (có mô tả + phân công từng thành viên), **báo cáo PDF** theo mẫu LMS (tỷ lệ trùng lặp < 25%), **slide**.
- Hạn nộp: **2 ngày trước lịch thi**. Chấm: Chương trình 40%, Báo cáo 40%, Thuyết trình + QA 20%.

---

## Định nghĩa bài toán

- **Loại bài toán:** phân loại **đa nhãn (multi-label)**. Một truyện có thể thuộc nhiều thể loại cùng lúc.
- **Input:** phần giới thiệu/tóm tắt của một truyện tiếng Việt (`summary`).
- **Output:** tập các thể loại chuẩn của truyện, ví dụ `{tien_hiep, huyen_huyen, xuyen_khong}`; biểu diễn bằng vector nhị phân độ dài **35** (`GENRE_LABELS`).
- **Nhãn:** xây dựng từ 42 thẻ thể loại trên `webtruyen.vn`, cào bổ sung 10 nhãn thiếu mẫu từ `truyenfull.live`, `metruyenchu.com.vn`, `webnovel.vn`, khử trùng và chuẩn hóa thành **35 nhãn chuẩn** (`20.852` truyện hợp lệ, mỗi nhãn có ít nhất `71` mẫu `quan_truong`).
- **Lưu ý rò rỉ nhãn:** cắt cụm metadata `"Thể loại: ... Văn án:"` ở đầu văn bản và xóa các từ khóa chuyên biệt trong `LEAKAGE_KEYWORDS` ở bước tiền xử lý.

---

## Cấu trúc thư mục

```text
NLP_CUỐI KỲ/
├── README.md                     # Hướng dẫn cài đặt, chạy pipeline, chạy ứng dụng
├── KE_HOACH.md                   # File này
├── PHAN_CONG.md                  # Phân công công việc từng thành viên
├── requirements.txt              # Thư viện dùng chung
├── .gitignore                    # __pycache__, data thô, môi trường ảo (giữ sẵn models/*.pkl & textcnn.pt)
│
├── data/
│   ├── README.md                 # Tài liệu bộ dữ liệu, định nghĩa 42 thẻ & bảng chuẩn hóa 35 nhãn
│   ├── raw/                      # Dữ liệu crawl thô (webtruyen, truyenfull, metruyenchu, webnovel)
│   ├── labeled/                  # Dữ liệu định dạng Label Studio (lưu trữ)
│   ├── processed/
│   │   ├── stories.csv           # 20.852 truyện đã gộp, khử trùng & chuẩn hóa 35 nhãn (49 cột)
│   │   └── stories_clean.csv     # + summary_clean, tokens, split (52 cột)
│   └── stopwords/
│       └── vietnamese_stopwords.txt
│
├── src/
│   ├── config.py                 # Đường dẫn, RANDOM_STATE=42, 35 GENRE_LABELS, tham số chung
│   ├── crawl/                    # crawler.py (4 nguồn), sources.py (42 -> 35), merge_data.py
│   ├── preprocessing/            # clean.py, tokenizer.py, leakage.py
│   ├── features/                 # bow_tfidf.py, embedding.py
│   ├── models/                   # ml_models.py, textcnn.py
│   ├── evaluation/               # metrics.py
│   └── pipeline.py               # Chạy toàn bộ: data -> features -> TN1..TN5 -> kết quả
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_ml_baselines.ipynb
│   └── 04_deep_learning.ipynb
│
├── bao_cao_ket_qua/              # Báo cáo kết quả chi tiết từng thành viên (TV1 -> TV4)
├── models/                       # Vectorizer (.pkl), ML models (.pkl), textcnn.pt, *_thresholds.json
│
├── outputs/
│   ├── figures/                  # 10 biểu đồ: EDA, ML, DL learning curve, f1 per label, error slices, coef_
│   └── tables/                   # experiments.csv, error_analysis_by_slice.csv, error_cases_sample.csv
│
├── app/
│   └── streamlit_app.py          # Ứng dụng demo Streamlit (ML + TextCNN)
│
├── reports/
│   ├── bao_cao.pdf
│   ├── slides.pdf
│   └── demo_backup.mp4           # Video dự phòng
│
└── tests/                        # 25 unit tests (crawler_and_merge, preprocessing, metrics, models)
```

**Quy ước chung**

- Định dạng dữ liệu: CSV UTF-8, các cột mô tả ở phần **Dữ liệu** bên dưới.
- Cột nhãn đa nhãn lưu dạng chuỗi ngăn cách bằng `|` (ví dụ `tien_hiep|huyen_huyen`), kèm 35 cột one-hot `label_*`.
- Cố định `RANDOM_STATE = 42` ở mọi nơi (đặt trong `src/config.py`).
- Chia dữ liệu **train/val/test** (`14.613` / `3.125` / `3.114`) **trước** khi fit vectorizer hay embedding, tránh rò rỉ dữ liệu. Với đa nhãn dùng **iterative stratification** (`IterativeStratification`) để giữ phân bố nhãn; lưu kết quả chia vào cột `split`.
- Mỗi thành viên làm trên branch riêng (`feat/crawl-eda`, `feat/preprocess-features`, `feat/ml-models`, `feat/dl-app`), merge vào `main` qua Pull Request.

---

## Dữ liệu

### Dữ liệu thô (`data/raw/`)

Tổng hợp từ 4 nguồn: `webtruyen_raw.csv` (`55.684` dòng từ 42 chuyên mục), `truyenfull_raw.csv` (`2.400` dòng bổ sung 10 nhãn thiếu), `metruyenchu_raw.csv` (`420` dòng), `webnovel_raw.csv` (`247` dòng) $\rightarrow$ **`58.751` dòng thô**.

| Cột | Kiểu | Mô tả |
|---|---|---|
| `id` | str | Mã duy nhất của truyện (`ST_0001` $\rightarrow$ `ST_20852`) |
| `title` | str | Tên truyện |
| `author` | str | Tác giả (nếu có) |
| `summary` | str | Phần giới thiệu/tóm tắt truyện, **là input chính của mô hình** |
| `genres_raw` | str | Các thẻ thể loại gốc trên web, ngăn cách bằng `\|` |
| `url` | str | Đường dẫn trang truyện |
| `source` | str | Tên website nguồn (`webtruyen.vn`, `truyenfull.live`, `metruyenchu.com.vn`, `webnovel.vn`) |
| `crawl_date` | date | Ngày crawl |

### Dữ liệu đã xử lý (`data/processed/stories.csv`)

Gồm các cột trên, thêm:

| Cột | Kiểu | Mô tả |
|---|---|---|
| `genres` | str | Nhãn sau chuẩn hóa, ngăn cách bằng `\|` (ví dụ `tien_hiep\|huyen_huyen`) |
| `n_labels` | int | Số nhãn của truyện (Label cardinality trung bình `3.0601`) |
| `n_words` | int | Số từ của `summary` |
| `label_<tên_nhãn>` | 0/1 | Một cột one-hot cho mỗi nhãn trong 35 nhãn chuẩn (`label_ngon_tinh`, ..., `label_khac`) |

### Dữ liệu sau tiền xử lý (`data/processed/stories_clean.csv`)

Gồm các cột của `stories.csv`, thêm:

| Cột | Kiểu | Mô tả |
|---|---|---|
| `summary_clean` | str | `summary` sau khi làm sạch, tách từ dính hoa/thường và bỏ rò rỉ nhãn |
| `tokens` | str | Văn bản đã tách từ bằng `underthesea`, bỏ stopwords |
| `split` | str | `train` (`14.613`) / `val` (`3.125`) / `test` (`3.114`) |

### Quy tắc chuẩn hóa nhãn

- Chuẩn hóa 42 thẻ gốc $\rightarrow$ **35 nhãn chuẩn** qua `GENRE_MAPPING` (`src/crawl/sources.py`):
  - Hợp nhất **3 cặp đồng trục ngữ nghĩa**: `Đô Thị + Hiện Đại -> hien_dai`, `Xuyên Không + Xuyên Nhanh -> xuyen_khong`, `Lịch Sử + Dã Sử -> lich_su`.
  - Duy trì **32 nhãn riêng biệt** (chuẩn hóa `Truyện Teen -> hoc_duong`, giữ riêng `Dị Năng`/`Mạt Thế`/`Khoa Huyễn`, `Cung Đấu`/`Gia Đấu`, `Huyền Huyễn`/`Dị Giới`/`Thám Hiểm`, `Tổng tài`/`Nữ Phụ`, `Quân Sự`/`Quan Trường`, `Đoản Văn`/`Tiểu Thuyết`/`Khác`).
  - Loại bỏ **4 thẻ nguồn gốc quốc gia / địa lý** (`Việt Nam`, `Light Novel`, `Đông Phương`, `Phương Tây`) và thẻ phi hư cấu (`Phát triển bản thân`, `Tâm linh`, `Kinh doanh`, `Review sách`).
- Loại `111` / `20.963` bản ghi không còn nhãn hợp lệ, giữ lại **`20.852` mẫu hợp lệ**.

---

## Các bước thực hiện chi tiết

### Bước 1. Thu thập dữ liệu
- **Nguồn chính:** `webtruyen.vn` (`55.684` thô $\rightarrow$ `18.621` duy nhất).
- **Nguồn bổ sung nhãn thiếu:** `truyenfull.live` (`2.400` thô từ 10 nhãn thiếu), `metruyenchu.com.vn` (`420` truyện có văn án thật), `webnovel.vn` (`247` bản ghi). Tổng hợp `58.751` bản ghi thô $\rightarrow$ `20.963` duy nhất.

### Bước 2–3. Chuẩn hóa 35 nhãn & EDA (`notebooks/01_eda.ipynb`)
- Xuất `20.852` mẫu (`35` nhãn) vào `data/processed/stories.csv`; `label cardinality = 3.0601`, `label density = 0.0874`, ma trận đồng xuất hiện `35×35` và biểu đồ `outputs/figures/eda_overview.png`.

### Bước 4–5. Tiền xử lý & Trích xuất đặc trưng (`notebooks/02_preprocessing.ipynb`)
- Làm sạch Unicode NFC (`1.036 -> 0`), tách dính hoa/thường (`6.408 -> 0`), cắt header (`2.881 -> 0`), xóa `LEAKAGE_KEYWORDS` (`9.552 -> 2`), tách từ `underthesea` + `120` stopwords.
- Fit `CountVectorizer` / `TfidfVectorizer` (`10.000` chiều, 1–2 gram) và `PPMI-SVD` (`128` chiều, `max_len=750`, Dynamic Padding) **chỉ trên tập `train` (`14.613`)**.

### Bước 6–7. Mô hình ML, Deep Learning & Phân tích lỗi (`notebooks/03`, `04`)
- `TN1_nb_bow` (`alpha=0.25`), `TN2_multinomial_nb` (`alpha=0.05`), `TN3_linear_svm` (`C=0.25, cw=balanced`), `TN4_logistic_regression` (`C=2.0, cw=balanced`), `TN5_textcnn` (`kernel=(2,3,4), f=64, emb=128 PPMI-SVD, max_len=750`).
- Tối ưu ngưỡng trên `val` (`3.125`), đánh giá 1 lần trên `test` (`3.114`); phân tích cặp nhãn `35×35`, Bootstrap 95% CI, phân tích lát cắt và độ nhạy ngưỡng `trong_sinh`.

### Bước 8. Ứng dụng demo (`app/streamlit_app.py`)
- Tự nhập tóm tắt $\rightarrow$ dự đoán đa nhãn **35 thể loại** với 4 mô hình, hiển thị mọi nhãn đạt ngưỡng xác suất và trang khám phá EDA/bảng so sánh mô hình.

---

## Luồng dữ liệu giữa các bước

```text
4 nguồn raw CSV (webtruyen + truyenfull + metruyenchu + webnovel = 58.751 dòng thô)
   └─(merge_data: khử trùng + chuẩn hóa 42 -> 35 nhãn)→ data/processed/stories.csv (20.852 mẫu)
        └─(preprocessing: NFC + split_case_boundaries + leakage + underthesea)→ data/processed/stories_clean.csv
             ├─ BoW (10.000) ─────────── Multinomial NB (TN1) ──────────┐
             ├─ TF-IDF (10.000) ──────── NB / Linear SVM / LogReg ──────┼─→ metrics + error analysis ─→ experiments.csv
             └─ Emb (PPMI-SVD 128d) ──── TextCNN (TN5) ───────────────────┘                                    │
                                                                                                             └─→ models/ (.pkl + textcnn.pt) ─→ app/streamlit_app.py
```
