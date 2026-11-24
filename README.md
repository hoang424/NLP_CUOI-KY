# Đồ án NLP Cuối kỳ (Đề 11): Phân loại thể loại văn học / truyện tiếng Việt

Dự án xây dựng pipeline xử lý ngôn ngữ tự nhiên (NLP) hoàn chỉnh để phân loại **đa nhãn (multi-label)** thể loại cho phần giới thiệu/tóm tắt truyện tiếng Việt trên bộ **35 nhãn chuẩn (`20.852` truyện)** thu thập từ 4 nguồn (`webtruyen.vn`, `truyenfull.live`, `metruyenchu.com.vn`, `webnovel.vn`), phục vụ môn học Xử lý Ngôn ngữ Tự nhiên.

## Tổng quan Pipeline (6 giai đoạn)

1. **Giai đoạn 1 – Thu thập đa nguồn & Cào bổ sung (`src/crawl/`):**
   - Thu thập toàn bộ 42 chuyên mục thể loại trên **`webtruyen.vn`** (`55.684` bản ghi thô $\rightarrow$ `18.621` truyện duy nhất) làm nền tảng, đồng thời cào bổ sung 10 thể loại thiếu mẫu từ **`truyenfull.live`** (`2.400` bản ghi thô), **`metruyenchu.com.vn`** (`420` truyện có văn án thật, gồm `145` `Dã Sử`) và **`webnovel.vn`** (`247` bản ghi). Tổng hợp **`58.751` bản ghi thô**.
2. **Giai đoạn 2 – Khử trùng & Chuẩn hóa 35 nhãn (`src/crawl/merge_data.py`, `src/crawl/sources.py`):**
   - Khử trùng tên truyện và văn án (`58.751` $\rightarrow$ `20.963` truyện duy nhất), chuẩn hóa **42 thẻ gốc $\rightarrow$ 35 nhãn chuẩn (`GENRE_LABELS`)**: hợp nhất 3 cặp đồng trục ngữ nghĩa (`Đô Thị + Hiện Đại -> hien_dai`, `Xuyên Không + Xuyên Nhanh -> xuyen_khong`, `Lịch Sử + Dã Sử -> lich_su`), chuẩn hóa `Truyện Teen -> hoc_duong`, duy trì 32 nhãn riêng biệt theo định nghĩa thể loại (`di_nang`, `mat_the`, `khoa_huyen`, `di_gioi`, `tham_hiem`, `cung_dau`, `gia_dau`, `tong_tai`, `nu_phu`, `quan_su`, `quan_truong`, `doan_van`, `tieu_thuyet`, `khac`...) và loại bỏ 4 thẻ nguồn gốc quốc gia/địa lý (`Việt Nam`, `Light Novel`, `Đông Phương`, `Phương Tây`) cùng sách phi hư cấu. Xuất **`20.852` truyện hợp lệ** ra `data/processed/stories.csv`.
3. **Giai đoạn 3 – Khám phá dữ liệu / EDA (`notebooks/01_eda.ipynb`):**
   - Thống kê phân bố 35 lớp (mọi lớp đều tăng mạnh sau bổ sung, từ `71` đến `9.664` mẫu), `label cardinality = 3.0601`, `label density = 0.0874`, ma trận đồng xuất hiện `35×35` và biểu đồ `outputs/figures/eda_overview.png`.
4. **Giai đoạn 4 – Tiền xử lý tiếng Việt & Vector hóa dùng chung (`src/preprocessing/`, `src/features/`):**
   - Chuẩn hóa Unicode (`NFC`) $\rightarrow$ tách từ dính ranh giới hoa/thường (`split_case_boundaries`) $\rightarrow$ xóa HTML/URL/ký tự đặc biệt $\rightarrow$ lowercase $\rightarrow$ loại bỏ rò rỉ tên thể loại (`LEAKAGE_KEYWORDS`) $\rightarrow$ tách từ tiếng Việt (`underthesea`) & lọc `120` stopwords (`data/stopwords/vietnamese_stopwords.txt`).
   - Chia tập phân tầng đa nhãn **Train / Val / Test** (`14.613` / `3.125` / `3.114`, `IterativeStratification`, `RANDOM_STATE = 42`) **trước** khi fit vectorizer, lưu ra `data/processed/stories_clean.csv`.
   - Trích xuất **TF-IDF (`10.000` chiều, 1–2 gram)** và **BoW (`10.000` chiều)** fit duy nhất trên tập Train; đồng thời xây dựng từ điển (`20.002` token, `min_freq=2`, `max_len=750` tại phân vị `p99.9` kết hợp **Dynamic Padding** theo batch) và **Word Embedding `PPMI + TruncatedSVD` (`128` chiều)** trên tập Train cho mô hình Deep Learning `TextCNN`.
5. **Giai đoạn 5 – Huấn luyện, Đánh giá & Phân tích lỗi (`src/models/`, `src/evaluation/`, `notebooks/03`–`04`):**
   - So sánh **5 thí nghiệm chính (`TN1`–`TN5`)** thuộc 2 nhánh phương pháp (kèm tối ưu ngưỡng từng nhãn trên tập `val`):
     1. `TN1_nb_bow`: **Multinomial Naive Bayes** + One-vs-Rest trên BoW (`alpha=0.25`)
     2. `TN2_multinomial_nb`: **Multinomial Naive Bayes** + One-vs-Rest trên TF-IDF (`alpha=0.05`)
     3. `TN3_linear_svm`: **Linear SVM** (`LinearSVC`) + One-vs-Rest trên TF-IDF (`C=0.25`, `class_weight="balanced"`)
     4. `TN4_logistic_regression`: **Logistic Regression** + One-vs-Rest trên TF-IDF (`C=2.0`, `class_weight="balanced"`)
     5. `TN5_textcnn`: **TextCNN (`TextCNNClassifier`)** — Multi-scale 1D CNN (`kernel_sizes=(2, 3, 4)`, `num_filters=64`, `BatchNorm1d`, `GELU`, Global Max + Masked Mean Pooling) trên `PPMI-SVD 128 chiều` Word Embedding
   - Phân tích cặp nhãn nhầm lẫn `35×35`, mẫu lỗi và top đặc trưng `coef_` trong `notebooks/03_ml_baselines.ipynb`; kiểm định độ tin cậy Bootstrap 95% CI, phân tích theo phân khúc dữ liệu và độ nhạy ngưỡng giữa ML vs `TextCNN` trong `notebooks/04_deep_learning.ipynb`.
6. **Giai đoạn 6 – Ứng dụng Demo (`app/streamlit_app.py`):**
   - Giao diện Streamlit cho phép chọn một trong **4 mô hình** (`Logistic Regression`, `Linear SVM`, `TextCNN`, `Multinomial Naive Bayes`), mặc định chế độ tự nhập văn bản mới, dự đoán và hiển thị toàn bộ thể loại đạt ngưỡng xác suất trong **35 thể loại**, xem biểu đồ phân tích và bảng so sánh mô hình.

## Kết quả thực nghiệm trên tập Test (`3.114` mẫu — `35` nhãn)

| Experiment ID | Mô hình | Biểu diễn đặc trưng | Subset Accuracy | Macro-F1 | Micro-F1 | Hamming Loss |
|---|---|---|---:|---:|---:|---:|
| `TN1_nb_bow` | Multinomial NB (BoW) | BoW (1–2 gram) | 0.0193 | 0.3501 | 0.4889 | 0.1186 |
| `TN2_multinomial_nb` | Multinomial NB (TF-IDF) | TF-IDF (1–2 gram) | 0.0286 | 0.3347 | 0.5376 | **0.0947** |
| `TN3_linear_svm` | Linear SVM (TF-IDF) | TF-IDF (1–2 gram) | 0.0260 | 0.3868 | 0.5276 | 0.1041 |
| `TN4_logistic_regression` | **Logistic Regression (TF-IDF)** | TF-IDF (1–2 gram) | **0.0302** | **0.3944** | **0.5413** | 0.1008 |
| `TN5_textcnn` | **TextCNN (Word Embedding)** | PPMI-SVD Emb + Multi-scale CNN (`max_len=750`, Dynamic Pad) | 0.0218 | 0.3503 | 0.4968 | 0.1255 |

## Cấu trúc thư mục

```text
NLP_CUỐI KỲ/
├── README.md                     # Hướng dẫn cài đặt, chạy pipeline, chạy ứng dụng
├── KE_HOACH.md                   # Kế hoạch kỹ thuật chi tiết
├── PHAN_CONG.md                  # Phân công công việc & hợp đồng dữ liệu giữa các thành viên
├── requirements.txt              # Danh sách thư viện Python dùng chung
├── .gitignore                    # Chặn file tạm, dữ liệu crawl thô, môi trường ảo (giữ sẵn models/*.pkl & textcnn.pt để chạy ngay)
├── data/
│   ├── README.md                 # Tài liệu dữ liệu, bảng định nghĩa 42 thẻ -> 35 nhãn & cào đa nguồn
│   ├── raw/                      # Dữ liệu crawl thô (webtruyen, truyenfull, metruyenchu, webnovel)
│   ├── labeled/                  # Dữ liệu định dạng Label Studio (lưu trữ)
│   ├── processed/                # stories.csv, stories_clean.csv (20.852 mẫu, 35 nhãn)
│   └── stopwords/                # vietnamese_stopwords.txt (120 từ dừng)
├── src/
│   ├── config.py                 # Đường dẫn, RANDOM_STATE=42, 35 GENRE_LABELS, tham số chung
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
│   ├── figures/                  # 10 biểu đồ EDA, ML, DL learning curves, error analysis
│   └── tables/                   # experiments.csv, error_analysis_by_slice.csv, error_cases_sample.csv
├── app/
│   └── streamlit_app.py          # Ứng dụng demo Streamlit (ML + TextCNN)
├── reports/                      # bao_cao.pdf, slides.pdf, demo_backup.mp4
└── tests/                        # 25 unit tests (crawler_and_merge, preprocessing, metrics, models)
```

## Hướng dẫn cài đặt & sử dụng

### 1. Tạo môi trường ảo & cài đặt thư viện

> **Khuyến nghị Python 3.10+.** Nếu máy dùng đúng Python `3.9.7`, tệp `requirements.txt` có điều kiện loại trừ `streamlit` (`python_full_version != "3.9.7"`), hãy cài bổ sung `pip install streamlit` riêng hoặc ưu tiên nâng lên Python 3.10+.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Chạy ngay ứng dụng Demo Streamlit (không cần huấn luyện lại)

Toàn bộ mô hình đã huấn luyện sẵn (`models/*.pkl`, `models/textcnn.pt`, `models/*_thresholds.json`) đã được đóng gói sẵn trong repository. Sau khi `git clone` / `git pull` và cài thư viện ở Bước 1, bạn có thể chạy thẳng giao diện dự đoán ngay:

```powershell
# Cách 1: Chạy trực tiếp qua Python (tự chọn .venv)
.\.venv\Scripts\python.exe app/streamlit_app.py

# Cách 2: Chạy qua CLI của Streamlit
.\.venv\Scripts\streamlit.exe run app/streamlit_app.py
```

### 3. Huấn luyện lại toàn bộ pipeline (Tùy chọn)

Chỉ cần chạy bước này nếu bạn thay đổi dữ liệu hoặc muốn huấn luyện lại cả 5 thí nghiệm (`TN1`–`TN5`):

```powershell
.\.venv\Scripts\python.exe -m src.pipeline
```

Kết quả mô hình (`.pkl`, `textcnn.pt`, `ml_thresholds.json`, `dl_thresholds.json`) được lưu trong `models/` và bảng đối chiếu 5 thí nghiệm được ghi vào `outputs/tables/experiments.csv`.

### 4. Chạy kiểm thử đơn vị (25 Unit Tests)

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

## Quy ước làm việc nhóm trên Git

- Mỗi thành viên làm việc trên nhánh riêng theo quy ước: `feat/crawl-eda`, `feat/preprocess-features`, `feat/ml-models`, `feat/dl-app`.
- Tạo **Pull Request** vào nhánh `main`, không push trực tiếp lên `main`.
- Cố định `RANDOM_STATE = 42` trong `src/config.py` và tuyệt đối không hardcode đường dẫn cá nhân.
