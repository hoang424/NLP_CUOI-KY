# Kế hoạch đồ án NLP: Phân loại thể loại truyện tiếng Việt

## Yêu cầu đề bài (đề số 11: Văn học × Text classification)

- Nhóm tối đa 3–4 thành viên; đăng ký nhóm và đề qua **LMS** (mỗi đề tối đa 2 nhóm).
- Dữ liệu **tiếng Việt**, thuộc lĩnh vực **văn học**; tự crawl hoặc dùng dataset có sẵn.
- Có đủ quy trình: **crawl → EDA → tiền xử lý → chuẩn hóa nhãn → xây dựng/huấn luyện mô hình → đánh giá → phân tích lỗi → ứng dụng**.
- Dùng ít nhất **3 mô hình ML truyền thống** (`Multinomial Naive Bayes`, `Linear SVM`, `Logistic Regression`) và **1 mô hình Deep Learning** (`TextCNN`).
- Có **ứng dụng** (`app/streamlit_app.py`) chạy trực quan để nhập tóm tắt và dự đoán đa nhãn.
- Nộp: code trên **GitHub** (có mô tả + phân công từng thành viên), **báo cáo PDF** theo mẫu LMS (tỷ lệ trùng lặp < 25%), **slide**.
- Hạn nộp: **2 ngày trước lịch thi**. Chấm: Chương trình 40%, Báo cáo 40%, Thuyết trình + QA 20%.

---

## Định nghĩa bài toán

- **Loại bài toán:** phân loại **đa nhãn (multi-label)**. Một truyện có thể thuộc nhiều thể loại cùng lúc.
- **Input:** phần giới thiệu/tóm tắt của một truyện tiếng Việt (`summary`).
- **Output:** tập các thể loại chuẩn của truyện, ví dụ `{tien_hiep, huyen_huyen, xuyen_khong}`; biểu diễn bằng vector nhị phân độ dài **22** (`GENRE_LABELS`).
- **Nhãn:** thu thập từ 42 thẻ thể loại trên `webtruyen.vn`, khử trùng và chuẩn hóa thành **22 nhãn chuẩn** (`18.603` truyện hợp lệ, mỗi nhãn có ít nhất `148` mẫu `vong_du`).
- **Lưu ý rò rỉ nhãn:** cắt cụm metadata `"Thể loại: ... Văn án:"` ở đầu văn bản và xóa các từ khóa chuyên biệt trong `LEAKAGE_KEYWORDS` ở bước tiền xử lý.

---

## Cấu trúc thư mục

```text
NLP_CUỐI KỲ/
├── README.md                     # Hướng dẫn cài đặt, chạy pipeline, chạy ứng dụng
├── KE_HOACH.md                   # File này
├── PHAN_CONG.md                  # Phân công công việc từng thành viên
├── requirements.txt              # Thư viện dùng chung
├── .gitignore                    # __pycache__, models nặng, data thô
│
├── data/
│   ├── README.md                 # Tài liệu bộ dữ liệu & bảng ánh xạ 42 -> 22 nhãn
│   ├── raw/                      # Dữ liệu crawl gốc (webtruyen_raw.csv)
│   ├── labeled/                  # Dữ liệu định dạng Label Studio (lưu trữ)
│   ├── processed/
│   │   ├── stories.csv           # 18.603 truyện đã gộp, khử trùng & chuẩn hóa 22 nhãn
│   │   └── stories_clean.csv     # + summary_clean, tokens, split (train/val/test)
│   └── stopwords/
│       └── vietnamese_stopwords.txt
│
├── src/
│   ├── config.py                 # Đường dẫn, RANDOM_STATE=42, 22 GENRE_LABELS, tham số chung
│   ├── crawl/
│   │   ├── crawler.py            # requests + lxml/BeautifulSoup (_block_text tránh dính từ)
│   │   ├── sources.py            # URL nguồn truyện & bảng ánh xạ GENRE_MAPPING (42 -> 22)
│   │   └── merge_data.py         # Khử trùng, lọc ngắn, ánh xạ nhãn -> stories.csv
│   ├── preprocessing/
│   │   ├── clean.py              # Chuẩn hóa NFC, split_case_boundaries, bỏ HTML/URL
│   │   ├── tokenizer.py          # Tách từ tiếng Việt (underthesea) & lọc stopwords
│   │   └── leakage.py            # Cắt header Thể loại & xóa LEAKAGE_KEYWORDS
│   ├── features/
│   │   ├── bow_tfidf.py          # CountVectorizer, TfidfVectorizer (ngram 1-2, 10.000 chiều)
│   │   └── embedding.py          # build_vocab (20.002), texts_to_sequences, PPMI-SVD Embedding (128d)
│   ├── models/
│   │   ├── ml_models.py          # Multinomial NB, Linear SVM, Logistic Regression + tuned thresholds
│   │   └── textcnn.py            # TextCNNClassifier (Multi-scale 1D CNN) + checkpoint PyTorch
│   ├── evaluation/
│   │   └── metrics.py            # Subset Accuracy, Macro/Micro F1, Hamming Loss, bảng experiments
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
│   ├── figures/                  # Phân bố lớp, learning curve, f1 per label, error slices, coef_
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
└── tests/                        # 21 unit tests (crawler_and_merge, preprocessing, metrics, models)
```

**Quy ước chung**

- Định dạng dữ liệu: CSV UTF-8, các cột mô tả ở phần **Dữ liệu** bên dưới.
- Cột nhãn đa nhãn lưu dạng chuỗi ngăn cách bằng `|` (ví dụ `tien_hiep|huyen_huyen`), kèm 22 cột one-hot `label_*`.
- Cố định `RANDOM_STATE = 42` ở mọi nơi (đặt trong `src/config.py`).
- Chia dữ liệu **train/val/test** (`13.297` / `2.634` / `2.672`) **trước** khi fit vectorizer hay embedding, tránh rò rỉ dữ liệu. Với đa nhãn dùng **iterative stratification** (`IterativeStratification`) để giữ phân bố nhãn; lưu kết quả chia vào cột `split`.
- Mỗi thành viên làm trên branch riêng (`feat/crawl-eda`, `feat/preprocess-features`, `feat/ml-models`, `feat/dl-app`), merge vào `main` qua Pull Request.

---

## Dữ liệu

### Dữ liệu thô (`data/raw/`)

Mỗi bản ghi là một truyện, giữ nguyên những gì crawl được trong `webtruyen_raw.csv` (`55.684` dòng từ 42 chuyên mục).

| Cột | Kiểu | Mô tả |
|---|---|---|
| `id` | int/str | Mã duy nhất của truyện trong bộ dữ liệu (`ST_0001` $\rightarrow$ `ST_18603`) |
| `title` | str | Tên truyện |
| `author` | str | Tác giả (nếu có) |
| `summary` | str | Phần giới thiệu/tóm tắt truyện, **là input chính của mô hình** |
| `genres_raw` | str | Các thẻ thể loại gốc trên web, ngăn cách bằng `\|` |
| `url` | str | Đường dẫn trang truyện |
| `source` | str | Tên website nguồn (`webtruyen.vn`) |
| `crawl_date` | date | Ngày crawl |

### Dữ liệu đã xử lý (`data/processed/stories.csv`)

Gồm các cột trên, thêm:

| Cột | Kiểu | Mô tả |
|---|---|---|
| `genres` | str | Nhãn sau chuẩn hóa, ngăn cách bằng `\|` (ví dụ `tien_hiep\|huyen_huyen`) |
| `n_labels` | int | Số nhãn của truyện (Label cardinality trung bình `2.6704`) |
| `n_words` | int | Số từ của `summary` |
| `label_<tên_nhãn>` | 0/1 | Một cột one-hot cho mỗi nhãn trong 22 nhãn chuẩn (`label_ngon_tinh`, ..., `label_hai_huoc`) |

### Dữ liệu sau tiền xử lý (`data/processed/stories_clean.csv`)

Gồm các cột của `stories.csv`, thêm:

| Cột | Kiểu | Mô tả |
|---|---|---|
| `summary_clean` | str | `summary` sau khi làm sạch, tách từ dính hoa/thường và bỏ rò rỉ nhãn |
| `tokens` | str | Văn bản đã tách từ bằng `underthesea`, bỏ stopwords |
| `split` | str | `train` (`13.297`) / `val` (`2.634`) / `test` (`2.672`) |

### Quy tắc chuẩn hóa nhãn

- Ánh xạ 42 thẻ gốc $\rightarrow$ **22 nhãn chuẩn** qua `GENRE_MAPPING` (`src/crawl/sources.py`).
- Loại bỏ 6 thẻ không thuộc thể loại hoặc quá ít mẫu (`NON_GENRE_TAGS`: `Khác`, `Đoản Văn`, `Tiểu Thuyết`, `Phương Tây`, `Quan Trường`, `Việt Nam`).
- Giữ `genres_raw` để truy vết, `genres` là nhãn dùng để huấn luyện.
- Loại các truyện không còn nhãn hợp lệ sau chuẩn hóa (`18` / `18.621` truyện, giữ lại `18.603` mẫu hợp lệ).

### Thống kê cần có trong EDA (đa nhãn)

- Số mẫu theo từng nhãn trong 22 lớp; **label cardinality** (`2.6704`) và **label density** (`0.1214`).
- Phân bố số nhãn trên mỗi truyện (`1`–`11` nhãn).
- Ma trận **đồng xuất hiện nhãn `22×22`**.
- Phân bố độ dài `summary` và Wordcloud / từ khóa nổi bật theo thể loại.

---

## Các bước thực hiện chi tiết

### Bước 1. Thu thập dữ liệu

- **Nguồn chính:** `webtruyen.vn` (cào đủ 42 chuyên mục × `2.269` trang).
- **Công cụ:** `requests` + `lxml`/`BeautifulSoup` (có hàm `_block_text()` giữ khoảng trắng giữa các thẻ HTML).
- **Số lượng:** `55.684` bản ghi thô $\rightarrow$ `18.621` truyện duy nhất sau khi gộp/khử trùng.

### Bước 2. Chuẩn hóa nhãn

- Ánh xạ 42 thẻ gốc sang 22 nhãn chuẩn, tạo cột `genres`, `n_labels`, `n_words` và 22 cột `label_*` trong `data/processed/stories.csv` (`18.603` mẫu).

### Bước 3. EDA (`notebooks/01_eda.ipynb`)

- Phân tích mất cân bằng lớp (`max/min = 58.8 : 1`), đồng xuất hiện nhãn, phân bố độ dài văn bản và xuất `outputs/figures/eda_overview.png`.

### Bước 4. Tiền xử lý tiếng Việt (`notebooks/02_preprocessing.ipynb`)

- **Làm sạch:** chuẩn hóa NFC, giải mã HTML, bỏ URL/ký tự đặc biệt, tách từ dính hoa/thường (`split_case_boundaries`), lowercase.
- **Loại rò rỉ nhãn:** cắt header `"Thể loại: ... Văn án:"` và xóa các cụm trong `LEAKAGE_KEYWORDS`.
- **Tách từ & Stopwords:** `underthesea` + `120` từ dừng (`vietnamese_stopwords.txt`).
- **Mất cân bằng nhãn:** dùng `class_weight="balanced"` cho `LinearSVC` / `LogisticRegression` và `sqrt pos_weight` trong `BCEWithLogitsLoss` cho `TextCNN`.

### Bước 5. Biểu diễn văn bản (`src/features/`)

| Cách | Công cụ | Dùng cho |
|---|---|---|
| Bag-of-Words | `CountVectorizer` (1–2 gram, `10.000` chiều) | `Multinomial NB (BoW)` (`TN1`) |
| TF-IDF | `TfidfVectorizer` (1–2 gram, `10.000` chiều, `sublinear_tf=True`) | `Multinomial NB`, `Linear SVM`, `Logistic Regression` (`TN2`–`TN4`) |
| Word Embedding | `build_vocab` (`20.002`) + `PPMI-SVD` (`128` chiều, `max_len=750`, Dynamic Padding theo batch) | `TextCNN` (`TN5`) |

### Bước 6. Xây dựng và huấn luyện mô hình (`notebooks/03_ml_baselines.ipynb`, `04_deep_learning.ipynb`)

**ML truyền thống (`src/models/ml_models.py`):** bọc bằng `OneVsRestClassifier` với `RANDOM_STATE = 42`:
- `MultinomialNB` trên BoW (`alpha=0.1` — `TN1`) và TF-IDF (`alpha=0.05` — `TN2`)
- `LinearSVC` trên TF-IDF (`C=0.25`, `class_weight="balanced"` — `TN3`)
- `LogisticRegression` trên TF-IDF (`C=1.0`, `class_weight="balanced"` — `TN4`)

**Deep Learning (`src/models/textcnn.py`):**
- **`TextCNNClassifier` (`TN5`):** Multi-scale 1D CNN (`kernel_sizes=(2, 3, 4)`, `num_filters=64`, `embedding_dim=128` khởi tạo `PPMI-SVD`, `SpatialDropout(0.15)`, `BatchNorm1d`, `GELU`, Global Max + Masked Mean Pooling, `Dropout(0.35)`), đầu ra `22` logit, loss `BCEWithLogitsLoss` + `sqrt pos_weight`, tối ưu `AdamW` + `OneCycleLR`.

**Ngưỡng quyết định:** tối ưu ngưỡng riêng cho từng nhãn trên tập `val` (`tune_per_label_thresholds`), lưu tại `models/ml_thresholds.json` và `models/dl_thresholds.json`; chỉ đánh giá tập `test` (`2.672` mẫu) một lần duy nhất.

| Mô hình | BoW | TF-IDF | Embedding |
|---|---|---|---|
| Multinomial Naive Bayes | `TN1_nb_bow` | `TN2_multinomial_nb` | |
| Linear SVM | | `TN3_linear_svm` | |
| Logistic Regression | | `TN4_logistic_regression` | |
| TextCNN | | | `TN5_textcnn` (`PPMI-SVD` 128d) |

### Bước 7. Đánh giá và phân tích lỗi (`notebooks/03_ml_baselines.ipynb` & `04_deep_learning.ipynb`)

- **Độ đo đa nhãn:** `Subset Accuracy`, `Macro-F1`, `Micro-F1`, `Hamming Loss`, cùng Precision/Recall/F1 cho 22 nhãn.
- **Phân tích lỗi & Giải thích nhánh ML (`03_ml_baselines.ipynb`):** ma trận cặp nhãn `22×22` (`error_confusion_pairs.png`), trích xuất 12 mẫu lỗi trọng tâm (`error_cases_sample.csv`) và top-8 từ khóa trọng số `coef_` (`top_features_per_genre.png`).
- **Đối chiếu & phân tích lỗi toàn hệ thống (`04_deep_learning.ipynb`):** kiểm định Bootstrap 95% CI, hiệu năng theo lát cắt độ dài văn án & số nhãn (`error_analysis_by_slice.csv`, `error_analysis_slices.png`), phân loại lỗi thừa/thiếu nhãn và độ nhạy ngưỡng `TextCNN`.
- **Bảng tổng hợp:** `outputs/tables/experiments.csv` (chọn `Logistic Regression` làm mô hình chính và cung cấp thêm `Linear SVM`, `TextCNN`, `Multinomial NB` trên ứng dụng).

### Bước 8. Ứng dụng demo (`app/streamlit_app.py`)

- Nhập đoạn tóm tắt truyện $\rightarrow$ làm sạch & suy luận đa nhãn bằng mô hình đã chọn (**Logistic Regression**, **Linear SVM**, **TextCNN**, **Multinomial NB**).
- Mặc định chế độ tự nhập văn bản mới; hiển thị toàn bộ thể loại đạt ngưỡng kèm xác suất, biểu đồ cột xác suất và bảng 22 nhãn.
- Chuẩn bị video quay màn hình dự phòng tại `reports/demo_backup.mp4`.

### Bước 9. Báo cáo, slide, thuyết trình, nộp bài

- **Báo cáo:** tổng hợp từ 4 nhật ký trong `bao_cao_ket_qua/` ra PDF theo mẫu LMS (< 25% trùng lặp).
- **GitHub:** `README.md` + `PHAN_CONG.md` + `KE_HOACH.md` đầy đủ và đồng bộ với mã nguồn.
- **Slide:** khoảng 15–20 trang, nắm chắc toàn bộ pipeline để trả lời QA.

---

## Luồng dữ liệu giữa các bước

```text
data/raw/webtruyen_raw.csv (55.684 bản ghi thô)
   └─(merge_data: khử trùng + ánh xạ 42 -> 22 nhãn)→ data/processed/stories.csv (18.603 mẫu)
        └─(preprocessing: NFC + split_case_boundaries + leakage + underthesea)→ data/processed/stories_clean.csv
             ├─ BoW (10.000) ─────────── Multinomial NB (TN1) ──────────┐
             ├─ TF-IDF (10.000) ──────── NB / Linear SVM / LogReg ──────┼─→ metrics + error analysis ─→ experiments.csv
             └─ Emb (PPMI-SVD 128d) ──── TextCNN (TN5) ───────────────────┘                                    │
                                                                                                             └─→ models/ (.pkl + textcnn.pt) ─→ app/streamlit_app.py
```
