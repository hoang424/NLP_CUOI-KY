# Phân công công việc nhóm (Đề 11: Phân loại thể loại văn học / truyện tiếng Việt)

Tài liệu này quy định phạm vi công việc, hợp đồng dữ liệu đầu vào/đầu ra và quy trình bàn giao để **4 thành viên** phát triển song song mà không phụ thuộc chéo.

## Bảng tổng quan phân công

| Thành viên | Mảng phụ trách chính | Nhánh Git | Thư mục / File sở hữu chính | Báo cáo kết quả |
|---|---|---|---|---|
| **TV1** | Thu thập dữ liệu `webtruyen.vn`, Chuẩn hóa 22 nhãn & EDA | `feat/crawl-eda` | `src/crawl/*`, `data/raw/*`, `data/labeled/*`, `data/processed/stories.csv`, `notebooks/01_eda.ipynb`, `tests/test_crawler_and_merge.py` | `bao_cao_ket_qua/01_tv1_thu_thap_va_eda.md` |
| **TV2** | Tiền xử lý tiếng Việt, Chống rò rỉ nhãn & Trích xuất đặc trưng | `feat/preprocess-features` | `src/preprocessing/*`, `src/features/*`, `data/stopwords/*`, `data/processed/stories_clean.csv`, `notebooks/02_preprocessing.ipynb`, `tests/test_preprocessing.py` | `bao_cao_ket_qua/02_tv2_tien_xu_ly_va_dac_trung.md` |
| **TV3** | Mô hình ML truyền thống, Giải thích & Phân tích lỗi ML | `feat/ml-models` | `src/models/ml_models.py`, `notebooks/03_ml_baselines.ipynb`, `src/evaluation/metrics.py`, `tests/test_metrics.py` | `bao_cao_ket_qua/03_tv3_mo_hinh_ml_truyen_thong.md` |
| **TV4** | Mô hình Deep Learning (`TextCNN`), Đối chiếu ML–DL & Ứng dụng Streamlit | `feat/dl-app` | `src/models/textcnn.py`, `src/pipeline.py`, `notebooks/04_deep_learning.ipynb`, `app/streamlit_app.py`, `tests/test_models.py` | `bao_cao_ket_qua/04_tv4_deep_learning_va_ung_dung.md` |

> **Lưu ý chung:** Mọi thành viên tổng hợp kết quả thực nghiệm vào `bao_cao_ket_qua/` (xem `bao_cao_ket_qua/README.md`) trước khi ghép thành báo cáo PDF (`reports/bao_cao.pdf`) và slide thuyết trình (`reports/slides.pdf`).

---

## 1. Thu thập dữ liệu, Chuẩn hóa 22 nhãn & EDA — _TV1_

| Công việc | Đầu ra (file) |
|---|---|
| Thu thập dữ liệu 42 chuyên mục từ `webtruyen.vn` (`55.684` bản ghi thô) | `src/crawl/sources.py`, `src/crawl/crawler.py`, `data/raw/webtruyen_raw.csv` |
| Khử trùng tên/tóm tắt (`18.621`), ánh xạ 42 thẻ $\rightarrow$ **22 nhãn chuẩn** (`18.603` mẫu) | `src/crawl/merge_data.py`, `data/processed/stories.csv` |
| Thực hiện EDA đa nhãn (`cardinality=2.6704`, đồng xuất hiện, độ dài văn bản) | `notebooks/01_eda.ipynb`, `outputs/figures/eda_overview.png` |
| Kiểm thử đơn vị phần crawl/khử trùng/chuẩn hóa thẻ | `tests/test_crawler_and_merge.py` |
| Viết báo cáo kết quả Phần 1 | `bao_cao_ket_qua/01_tv1_thu_thap_va_eda.md` |

**Chi tiết thực hiện:**
- Ánh xạ 42 thẻ gốc $\rightarrow$ 22 nhãn chuẩn qua `GENRE_MAPPING`, loại bỏ 6 thẻ rác/quá hiếm (`NON_GENRE_TAGS`) trong `src/crawl/sources.py`.
- Bảo toàn khoảng trắng giữa các thẻ block HTML trong `_block_text()` (`src/crawl/crawler.py`) và giữ nguyên `summary`, `genres_raw` để truy vết.

**Bàn giao:**

| Đầu ra của TV1 | Ai dùng |
|---|---|
| `data/processed/stories.csv` | TV2 đọc để tiền xử lý & tạo `stories_clean.csv`; TV4 hiển thị thống kê EDA trên Streamlit |
| `outputs/figures/eda_overview.png` | TV4 nhúng vào tab EDA của `app/streamlit_app.py` và cả nhóm đưa vào báo cáo/slide |

---

## 2. Tiền xử lý tiếng Việt & Trích xuất đặc trưng — _TV2_

| Công việc | Đầu ra (file) |
|---|---|
| Làm sạch Unicode NFC, tách từ dính hoa/thường (`split_case_boundaries`) & chống rò rỉ nhãn | `src/preprocessing/clean.py`, `src/preprocessing/leakage.py` |
| Tách từ tiếng Việt (`underthesea`) và lọc `120` từ dừng | `src/preprocessing/tokenizer.py`, `data/stopwords/vietnamese_stopwords.txt` |
| Chia tập phân tầng đa nhãn (`train=13.297`, `val=2.634`, `test=2.672`, `RANDOM_STATE=42`) | `data/processed/stories_clean.csv`, `notebooks/02_preprocessing.ipynb` |
| Xây dựng bộ vector hóa TF-IDF/BoW (`10.000` chiều) & từ điển/embedding `PPMI-SVD` (`128` chiều) cho `TextCNN` | `src/features/bow_tfidf.py`, `src/features/embedding.py` |
| Viết unit test kiểm tra làm sạch, tách dính hoa/thường, chống rò rỉ nhãn & stopwords | `tests/test_preprocessing.py` |
| Viết báo cáo kết quả Phần 2 | `bao_cao_ket_qua/02_tv2_tien_xu_ly_va_dac_trung.md` |

**Chi tiết thực hiện:**
- Đảm bảo `clean_text()` và `remove_label_leakage()` xử lý an toàn chuỗi rỗng/`NaN`, giữ whitelist `CAMELCASE_WHITELIST` và không làm rò rỉ nhãn web.
- Chia `split` (`train`/`val`/`test`) **trước** khi fit `TfidfVectorizer` và `build_corpus_embeddings()` (`PPMI + TruncatedSVD`) để loại trừ hoàn toàn data leakage.
- Lưu `tfidf_vectorizer.pkl` (`10.000` chiều) vào `models/` qua `fit_transform_tfidf()`.

**Bàn giao:**

| Đầu ra của TV2 | Ai dùng |
|---|---|
| `data/processed/stories_clean.csv` | TV3 và TV4 đọc cột `tokens` và `split` để train/val/test |
| `fit_transform_tfidf()`, `load_tfidf_vectorizer()` trong `src/features/bow_tfidf.py` | TV3 (`Multinomial NB`, `Linear SVM`, `Logistic Regression`) và TV4 (`app/streamlit_app.py`) |
| `build_vocab()`, `texts_to_sequences()`, `build_corpus_embeddings()` trong `src/features/embedding.py` | TV4 dùng để huấn luyện và suy luận mô hình `TextCNN` |

---

## 3. Mô hình Machine Learning truyền thống, Giải thích & Phân tích lỗi ML — _TV3_

| Công việc | Đầu ra (file) |
|---|---|
| Xây dựng 3 thuật toán ML đa nhãn (`Multinomial NB` BoW/TF-IDF, `Linear SVM`, `Logistic Regression`) | `src/models/ml_models.py` |
| Tinh chỉnh siêu tham số & tối ưu ngưỡng từng nhãn trên tập `val` (`TN1`–`TN4`) | `notebooks/03_ml_baselines.ipynb`, `models/ml_thresholds.json` |
| Phân tích ma trận cặp nhãn `22×22`, trích xuất 12 mẫu lỗi & giải thích từ khóa `coef_` của `Logistic Regression` | `outputs/figures/error_confusion_pairs.png`, `outputs/tables/error_cases_sample.csv`, `outputs/figures/top_features_per_genre.png` |
| Chuẩn hóa module tính độ đo đa nhãn & lưu bảng thí nghiệm | `src/evaluation/metrics.py`, `outputs/tables/experiments.csv` |
| Viết unit test kiểm tra hàm đánh giá và ghi bảng thí nghiệm | `tests/test_metrics.py` |
| Viết báo cáo kết quả Phần 3 | `bao_cao_ket_qua/03_tv3_mo_hinh_ml_truyen_thong.md` |

**Chi tiết thực hiện:**
- Sử dụng `OneVsRestClassifier` bọc `MultinomialNB`, `LinearSVC` và `LogisticRegression` (cố định `RANDOM_STATE = 42` từ `src/config.py`).
- Chuẩn hóa điểm quyết định của `LinearSVC` qua sigmoid để tương thích ngưỡng quyết định và giao diện hiển thị xác suất và toàn bộ thể loại đạt ngưỡng trên Streamlit.
- Đọc đúng cột `split` từ `stories_clean.csv`, không tự chia lại dữ liệu.

**Bàn giao:**

| Đầu ra của TV3 | Ai dùng |
|---|---|
| `get_linear_svm()`, `get_logistic_regression()`, `get_naive_bayes()`, `tune_per_label_thresholds()` | TV4 gọi từ `src/pipeline.py` và nạp `.pkl` + `ml_thresholds.json` trong Streamlit |
| `evaluate_model()`, `save_experiment_result()` | TV4 dùng chung cho `TextCNN` để ghi `TN5_textcnn` vào `outputs/tables/experiments.csv` |

---

## 4. Deep Learning (`TextCNN`), Đối chiếu ML–DL, Pipeline tổng & Ứng dụng Demo — _TV4_

| Công việc | Đầu ra (file) |
|---|---|
| Cài đặt **`TextCNN` (`TextCNNClassifier`)** (Multi-scale 1D CNN) | `src/models/textcnn.py`, `notebooks/04_deep_learning.ipynb` |
| Huấn luyện `TN5_textcnn` trên `PPMI-SVD 128d` Word Embedding, lưu trọng số & ngưỡng tối ưu | `models/textcnn.pt`, `models/dl_thresholds.json` |
| Ghép pipeline chạy đầu cuối từ dữ liệu $\rightarrow$ feature $\rightarrow$ `TN1`–`TN5` $\rightarrow$ biểu đồ & CSV | `src/pipeline.py` |
| Kiểm định Bootstrap 95% CI, phân tích lát cắt dữ liệu, tỷ lệ lỗi thừa/thiếu & độ nhạy ngưỡng `TextCNN` | `notebooks/04_deep_learning.ipynb`, `outputs/tables/error_analysis_by_slice.csv`, `outputs/figures/error_analysis_slices.png` |
| Viết unit test kiểm tra forward/padding, embedding PPMI-SVD và checkpoint `TextCNN` | `tests/test_models.py` |
| Xây dựng ứng dụng demo Streamlit + quay video dự phòng | `app/streamlit_app.py`, `reports/demo_backup.mp4` |
| Viết báo cáo kết quả Phần 4 | `bao_cao_ket_qua/04_tv4_deep_learning_va_ung_dung.md` |

**Chi tiết thực hiện:**
- Đảm bảo `.\.venv\Scripts\python.exe -m src.pipeline` chạy thông suốt toàn bộ 5 thí nghiệm (`TN1`–`TN5`), lưu `.pkl`, `textcnn.pt`, `*_thresholds.json` vào `models/`.
- `app/streamlit_app.py` nạp trực tiếp các model ML (`.pkl`), checkpoint `TextCNN` (`models/textcnn.pt`), bộ ngưỡng tối ưu và các bảng/biểu đồ trong `outputs/`.

**Bàn giao:**

| Đầu ra của TV4 | Ai dùng |
|---|---|
| `src/pipeline.py` & `outputs/tables/experiments.csv` | Cả nhóm dùng kiểm thử tích hợp và lấy số liệu tổng hợp báo cáo PDF / slide |
| `app/streamlit_app.py` | Cả nhóm dùng khi thuyết trình trực tiếp và quay `reports/demo_backup.mp4` |

---

## Quy ước phối hợp & Kiểm thử tích hợp

1. Mọi hằng số (`RANDOM_STATE = 42`, đường dẫn, `GENRE_LABELS`, `TFIDF_PARAMS`) lấy từ `src/config.py`.
2. Trước khi tạo Pull Request vào `main`, mỗi thành viên tự chạy `.\.venv\Scripts\python.exe -m pytest tests/ -v` (`21 tests`).
3. Trước buổi bảo vệ, kiểm tra toàn bộ pipeline và giao diện `.\.venv\Scripts\streamlit.exe run app/streamlit_app.py`.
