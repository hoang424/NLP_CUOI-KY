# Đồ án NLP Cuối kỳ (Đề 11): Phân loại thể loại văn học / truyện tiếng Việt

Dự án xây dựng pipeline xử lý ngôn ngữ tự nhiên (NLP) hoàn chỉnh để phân loại **đa nhãn (multi-label)** thể loại cho phần giới thiệu/tóm tắt truyện và tác phẩm văn học tiếng Việt trên bộ **22 nhãn chuẩn (`18.603` truyện)** từ `webtruyen.vn`, phục vụ môn học Xử lý Ngôn ngữ Tự nhiên.

## Tổng quan Pipeline (6 giai đoạn)

1. **Giai đoạn 1 – Thu thập dữ liệu (`src/crawl/`):**
   - Thu thập toàn bộ 42 chuyên mục thể loại trên **`webtruyen.vn`** (`55.684` bản ghi thô vào `data/raw/webtruyen_raw.csv`); lưu trữ thêm các nguồn tham khảo (`webnovel`, `metruyenchu`, `wiki`).
2. **Giai đoạn 2 – Khử trùng & Chuẩn hóa 22 nhãn (`src/crawl/merge_data.py`, `src/crawl/sources.py`):**
   - Khử trùng tên truyện và tóm tắt (`55.684` $\rightarrow$ `18.621` truyện duy nhất), ánh xạ 42 thẻ gốc $\rightarrow$ **22 nhãn chuẩn (`GENRE_LABELS`)**, loại thẻ rác (`NON_GENRE_TAGS`), xuất `18.603` truyện hợp lệ ra `data/processed/stories.csv`.
3. **Giai đoạn 3 – Khám phá dữ liệu / EDA (`notebooks/01_eda.ipynb`):**
   - Thống kê phân bố 22 lớp, độ dài văn bản trung bình theo lớp, label cardinality (`2.6704`) và label density (`0.1214`), ma trận đồng xuất hiện nhãn `22×22`, Wordcloud và biểu đồ tổng quan `outputs/figures/eda_overview.png`.
4. **Giai đoạn 4 – Tiền xử lý tiếng Việt & Vector hóa dùng chung (`src/preprocessing/`, `src/features/`):**
   - Chuẩn hóa Unicode (`NFC`) $\rightarrow$ tách từ dính ranh giới hoa/thường (`split_case_boundaries`) $\rightarrow$ xóa HTML/URL/ký tự đặc biệt $\rightarrow$ lowercase $\rightarrow$ loại bỏ rò rỉ tên thể loại (`LEAKAGE_KEYWORDS`) $\rightarrow$ tách từ tiếng Việt (`underthesea`) & lọc `120` stopwords (`data/stopwords/vietnamese_stopwords.txt`).
   - Chia tập phân tầng đa nhãn **Train / Val / Test** (`13.297` / `2.634` / `2.672`, `IterativeStratification`, `RANDOM_STATE = 42`) **trước** khi fit vectorizer, lưu ra `data/processed/stories_clean.csv`.
   - Trích xuất **TF-IDF (`10.000` chiều, 1–2 gram)** và **BoW (`10.000` chiều)** fit duy nhất trên tập Train; đồng thời xây dựng từ điển (`20.002` token, `min_freq=2`, `max_len=750` tại phân vị `p99.9` kết hợp **Dynamic Padding** theo batch) và **Word Embedding `PPMI + TruncatedSVD` (`128` chiều)** trên tập Train cho mô hình Deep Learning `TextCNN`.
5. **Giai đoạn 5 – Huấn luyện, Đánh giá & Phân tích lỗi (`src/models/`, `src/evaluation/`, `notebooks/03`–`04`):**
   - So sánh **5 thí nghiệm chính (`TN1`–`TN5`)** thuộc 2 nhánh phương pháp (kèm tối ưu ngưỡng từng nhãn trên tập `val`):
     1. `TN1_nb_bow`: **Multinomial Naive Bayes** + One-vs-Rest trên BoW (`alpha=0.1`)
     2. `TN2_multinomial_nb`: **Multinomial Naive Bayes** + One-vs-Rest trên TF-IDF (`alpha=0.05`)
     3. `TN3_linear_svm`: **Linear SVM** (`LinearSVC`) + One-vs-Rest trên TF-IDF (`C=0.25`, `class_weight="balanced"`)
     4. `TN4_logistic_regression`: **Logistic Regression** + One-vs-Rest trên TF-IDF (`C=1.0`, `class_weight="balanced"`)
     5. `TN5_textcnn`: **TextCNN (`TextCNNClassifier`)** — Multi-scale 1D CNN (`kernel_sizes=(2, 3, 4)`, `num_filters=64`, `BatchNorm1d`, `GELU`, Global Max + Masked Mean Pooling) trên `PPMI-SVD 128 chiều` Word Embedding
   - Phân tích cặp nhãn nhầm lẫn `22×22`, mẫu lỗi và top đặc trưng `coef_` trong `notebooks/03_ml_baselines.ipynb`; kiểm định độ tin cậy Bootstrap 95% CI, phân tích theo phân khúc dữ liệu và độ nhạy ngưỡng giữa ML vs `TextCNN` trong `notebooks/04_deep_learning.ipynb`.
6. **Giai đoạn 6 – Ứng dụng Demo (`app/streamlit_app.py`):**
   - Giao diện Streamlit cho phép chọn một trong **4 mô hình** (`Logistic Regression`, `Linear SVM`, `TextCNN`, `Multinomial Naive Bayes`), mặc định chế độ tự nhập văn bản mới, dự đoán và hiển thị toàn bộ thể loại đạt ngưỡng xác suất, xem biểu đồ phân tích và bảng so sánh mô hình.

## Kết quả thực nghiệm trên tập Test (`2.672` mẫu — `22` nhãn)

| Experiment ID | Mô hình | Biểu diễn đặc trưng | Subset Accuracy | Macro-F1 | Micro-F1 | Hamming Loss |
|---|---|---|---:|---:|---:|---:|
| `TN1_nb_bow` | Multinomial NB (BoW) | BoW (1–2 gram) | 0.0603 | 0.4319 | 0.5641 | 0.1300 |
| `TN2_multinomial_nb` | Multinomial NB (TF-IDF) | TF-IDF (1–2 gram) | 0.0584 | 0.4031 | 0.5877 | 0.1207 |
| `TN3_linear_svm` | Linear SVM (TF-IDF) | TF-IDF (1–2 gram) | **0.0618** | 0.4767 | 0.6006 | **0.1193** |
| `TN4_logistic_regression` | **Logistic Regression (TF-IDF)** | TF-IDF (1–2 gram) | 0.0561 | **0.4897** | **0.6035** | 0.1230 |
| `TN5_textcnn` | **TextCNN (Word Embedding)** | PPMI-SVD Emb + Multi-scale CNN (`max_len=750`, Dynamic Pad) | 0.0483 | 0.4237 | 0.5657 | 0.1388 |

## Cấu trúc thư mục

```text
NLP_CUỐI KỲ/
├── README.md                     # Hướng dẫn cài đặt, chạy pipeline, chạy ứng dụng
├── KE_HOACH.md                   # Kế hoạch kỹ thuật chi tiết
├── PHAN_CONG.md                  # Phân công công việc & hợp đồng dữ liệu giữa các thành viên
├── requirements.txt              # Danh sách thư viện Python dùng chung
├── .gitignore                    # Chặn file tạm, model nặng, môi trường ảo
├── data/
│   ├── README.md                 # Tài liệu cấu trúc dữ liệu & bảng ánh xạ 42 -> 22 nhãn
│   ├── raw/                      # Dữ liệu crawl thô (webtruyen_raw.csv là nguồn chính)
│   ├── labeled/                  # Dữ liệu định dạng Label Studio (lưu trữ)
│   ├── processed/                # stories.csv, stories_clean.csv (18.603 mẫu, 22 nhãn)
│   └── stopwords/                # vietnamese_stopwords.txt (120 từ dừng)
├── src/
│   ├── config.py                 # Đường dẫn, RANDOM_STATE=42, 22 GENRE_LABELS, tham số chung
│   ├── crawl/                    # crawler.py, sources.py, merge_data.py
│   ├── preprocessing/            # clean.py, tokenizer.py, leakage.py
│   ├── features/                 # bow_tfidf.py, embedding.py
│   ├── models/                   # ml_models.py, textcnn.py
│   ├── evaluation/               # metrics.py
│   └── pipeline.py               # Chạy toàn bộ pipeline đầu cuối (TN1 -> TN5)
├── notebooks/                    # 01_eda.ipynb -> 04_deep_learning.ipynb (đủ 4 notebook cho TV1 -> TV4)
├── bao_cao_ket_qua/              # Nhật ký kết quả từng thành viên (TV1 -> TV4)
├── models/                       # bow/tfidf_vectorizer.pkl, *.pkl, textcnn.pt, *_thresholds.json
├── outputs/
│   ├── figures/                  # Biểu đồ EDA, ML, DL learning curves, error analysis
│   └── tables/                   # experiments.csv, error_analysis_by_slice.csv, error_cases_sample.csv
├── app/
│   └── streamlit_app.py          # Ứng dụng demo Streamlit (ML + TextCNN)
├── reports/                      # bao_cao.pdf, slides.pdf, demo_backup.mp4
└── tests/                        # 21 unit tests (crawler_and_merge, preprocessing, metrics, models)
```

## Hướng dẫn cài đặt & sử dụng

### 1. Tạo môi trường ảo & cài đặt thư viện

> **Khuyến nghị Python 3.10+.** Nếu máy dùng đúng Python `3.9.7`, tệp `requirements.txt` có điều kiện loại trừ `streamlit` (`python_full_version != "3.9.7"`), hãy cài bổ sung `pip install streamlit` riêng hoặc ưu tiên nâng lên Python 3.10+.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Chạy toàn bộ pipeline trên dữ liệu hiện tại

```powershell
.\.venv\Scripts\python.exe -m src.pipeline
```

Kết quả mô hình (`.pkl`, `textcnn.pt`, `ml_thresholds.json`, `dl_thresholds.json`) được lưu trong `models/` và bảng đối chiếu 5 thí nghiệm được ghi vào `outputs/tables/experiments.csv`.

### 3. Chạy ứng dụng Demo Streamlit

> **Lưu ý khi mới clone repo:** Do `.gitignore` không theo dõi các file nhị phân trong `models/` (`*.pkl`, `*.pt`), hãy chạy `python -m src.pipeline` ở Bước 2 để sinh đầy đủ mô hình trước khi khởi chạy ứng dụng.

```powershell
# Cách 1: Chạy trực tiếp qua Python (tự chọn .venv)
.\.venv\Scripts\python.exe app/streamlit_app.py

# Cách 2: Chạy qua CLI của Streamlit
.\.venv\Scripts\streamlit.exe run app/streamlit_app.py
```

### 4. Chạy kiểm thử đơn vị (21 Unit Tests)

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

## Quy ước làm việc nhóm trên Git

- Mỗi thành viên làm việc trên nhánh riêng theo quy ước: `feat/crawl-eda`, `feat/preprocess-features`, `feat/ml-models`, `feat/dl-app`.
- Tạo **Pull Request** vào nhánh `main`, không push trực tiếp lên `main`.
- Cố định `RANDOM_STATE = 42` trong `src/config.py` và tuyệt đối không hardcode đường dẫn cá nhân.
