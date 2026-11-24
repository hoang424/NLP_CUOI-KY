# Phân công công việc nhóm (Đề 11: Phân loại thể loại văn học / truyện tiếng Việt)

Tài liệu này quy định phạm vi công việc, hợp đồng dữ liệu đầu vào/đầu ra và quy trình bàn giao để **4 thành viên** phát triển song song mà không phụ thuộc chéo.

## Bảng tổng quan phân công

| Thành viên | Mảng phụ trách chính | Nhánh Git | Thư mục / File sở hữu chính | Báo cáo kết quả |
|---|---|---|---|---|
| **TV1** | Thu thập đa nguồn, Bảng định nghĩa 42 thẻ, Chuẩn hóa 35 nhãn & EDA | `feat/crawl-eda` | `src/crawl/*`, `data/raw/*`, `data/labeled/*`, `data/processed/stories.csv`, `notebooks/01_eda.ipynb`, `tests/test_crawler_and_merge.py` | `bao_cao_ket_qua/01_tv1_thu_thap_va_eda.md` |
| **TV2** | Tiền xử lý tiếng Việt, Chống rò rỉ nhãn & Trích xuất đặc trưng | `feat/preprocess-features` | `src/preprocessing/*`, `src/features/*`, `data/stopwords/*`, `data/processed/stories_clean.csv`, `notebooks/02_preprocessing.ipynb`, `tests/test_preprocessing.py` | `bao_cao_ket_qua/02_tv2_tien_xu_ly_va_dac_trung.md` |
| **TV3** | Mô hình ML truyền thống, Giải thích & Phân tích lỗi ML | `feat/ml-models` | `src/models/ml_models.py`, `notebooks/03_ml_baselines.ipynb`, `src/evaluation/metrics.py`, `tests/test_metrics.py` | `bao_cao_ket_qua/03_tv3_mo_hinh_ml_truyen_thong.md` |
| **TV4** | Mô hình Deep Learning (`TextCNN`), Đối chiếu ML–DL & Ứng dụng Streamlit | `feat/dl-app` | `src/models/textcnn.py`, `src/pipeline.py`, `notebooks/04_deep_learning.ipynb`, `app/streamlit_app.py`, `tests/test_models.py` | `bao_cao_ket_qua/04_tv4_deep_learning_va_ung_dung.md` |

> **Lưu ý chung:** Mọi thành viên tổng hợp kết quả thực nghiệm vào `bao_cao_ket_qua/` (xem `bao_cao_ket_qua/README.md`) trước khi ghép thành báo cáo PDF (`reports/bao_cao.pdf`) và slide thuyết trình (`reports/slides.pdf`).

---

## 1. Thu thập đa nguồn, Chuẩn hóa 35 nhãn & EDA — _TV1_

| Công việc | Đầu ra (file) |
|---|---|
| Cào 42 chuyên mục `webtruyen.vn` (`55.684`) + bổ sung 10 nhãn thiếu từ `truyenfull.live` (`2.400`), `metruyenchu` (`420`), `webnovel` (`247`) | `src/crawl/sources.py`, `src/crawl/crawler.py`, `data/raw/*.csv` |
| Lập bảng định nghĩa 42 thẻ, khử trùng (`20.963`), chuẩn hóa 42 $\rightarrow$ **35 nhãn** (`20.852` mẫu) | `src/crawl/merge_data.py`, `data/processed/stories.csv` |
| Thực hiện EDA đa nhãn (`cardinality=3.0601`, ma trận `35×35`, độ dài văn bản) | `notebooks/01_eda.ipynb`, `outputs/figures/eda_overview.png` |
| Kiểm thử đơn vị phần crawl, parser `truyenfull.live`, khử trùng & chuẩn hóa 35 nhãn | `tests/test_crawler_and_merge.py` |
| Viết báo cáo kết quả Phần 1 (bảng 42 thẻ gốc, quá trình thu thập & cơ sở chuẩn hóa 35 nhãn) | `bao_cao_ket_qua/01_tv1_thu_thap_va_eda.md` |

**Chi tiết thực hiện:**
- Chuẩn hóa 42 thẻ gốc $\rightarrow$ **35 nhãn chuẩn** qua `GENRE_MAPPING`: hợp nhất 3 cặp đồng trục (`Đô Thị + Hiện Đại`, `Xuyên Không + Xuyên Nhanh`, `Lịch Sử + Dã Sử`), loại 4 thẻ nguồn gốc địa lý/quốc gia (`Việt Nam`, `Light Novel`, `Đông Phương`, `Phương Tây`) và duy trì 32 nhãn riêng biệt còn lại.
- Bảo toàn khoảng trắng giữa các thẻ block HTML trong `_block_text()` (`src/crawl/crawler.py`).

**Bàn giao:**

| Đầu ra của TV1 | Ai dùng |
|---|---|
| `data/processed/stories.csv` | TV2 đọc để tiền xử lý & tạo `stories_clean.csv`; TV4 hiển thị thống kê EDA đa nguồn trên Streamlit |
| `outputs/figures/eda_overview.png` | TV4 nhúng vào tab EDA của `app/streamlit_app.py` và cả nhóm đưa vào báo cáo/slide |

---

## 2. Tiền xử lý tiếng Việt & Trích xuất đặc trưng — _TV2_

| Công việc | Đầu ra (file) |
|---|---|
| Làm sạch Unicode NFC, tách từ dính hoa/thường (`split_case_boundaries`) & chống rò rỉ nhãn (`LEAKAGE_KEYWORDS`) | `src/preprocessing/clean.py`, `src/preprocessing/leakage.py` |
| Tách từ tiếng Việt (`underthesea`) và lọc `120` từ dừng | `src/preprocessing/tokenizer.py`, `data/stopwords/vietnamese_stopwords.txt` |
| Chia tập phân tầng đa nhãn (`train=14.613`, `val=3.125`, `test=3.114`, `RANDOM_STATE=42`) | `data/processed/stories_clean.csv`, `notebooks/02_preprocessing.ipynb` |
| Xây dựng bộ vector hóa TF-IDF/BoW (`10.000` chiều) & từ điển/embedding `PPMI-SVD` (`128` chiều) cho `TextCNN` | `src/features/bow_tfidf.py`, `src/features/embedding.py` |
| Viết unit test kiểm tra làm sạch, tách dính hoa/thường, chống rò rỉ nhãn & stopwords | `tests/test_preprocessing.py` |
| Viết báo cáo kết quả Phần 2 | `bao_cao_ket_qua/02_tv2_tien_xu_ly_va_dac_trung.md` |

---

## 3. Mô hình Machine Learning truyền thống, Giải thích & Phân tích lỗi ML — _TV3_

| Công việc | Đầu ra (file) |
|---|---|
| Xây dựng 3 thuật toán ML đa nhãn (`Multinomial NB` BoW/TF-IDF, `Linear SVM`, `Logistic Regression`) | `src/models/ml_models.py` |
| Tinh chỉnh siêu tham số & tối ưu ngưỡng từng nhãn trên tập `val` (`TN1`–`TN4`) | `notebooks/03_ml_baselines.ipynb`, `models/ml_thresholds.json` |
| Phân tích ma trận cặp nhãn `35×35`, trích xuất 12 mẫu lỗi & giải thích từ khóa `coef_` của `Logistic Regression` | `outputs/figures/error_confusion_pairs.png`, `outputs/tables/error_cases_sample.csv`, `outputs/figures/top_features_per_genre.png` |
| Chuẩn hóa module tính độ đo đa nhãn & lưu bảng thí nghiệm | `src/evaluation/metrics.py`, `outputs/tables/experiments.csv` |
| Viết unit test kiểm tra hàm đánh giá và ghi bảng thí nghiệm | `tests/test_metrics.py` |
| Viết báo cáo kết quả Phần 3 | `bao_cao_ket_qua/03_tv3_mo_hinh_ml_truyen_thong.md` |

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

---

## Quy ước phối hợp & Kiểm thử tích hợp

1. Mọi hằng số (`RANDOM_STATE = 42`, đường dẫn, `GENRE_LABELS`, `TFIDF_PARAMS`) lấy từ `src/config.py`.
2. Trước khi tạo Pull Request vào `main`, mỗi thành viên tự chạy `.\.venv\Scripts\python.exe -m pytest tests/ -v` (**25 tests**).
3. Trước buổi bảo vệ, kiểm tra toàn bộ pipeline và giao diện `.\.venv\Scripts\streamlit.exe run app/streamlit_app.py`.
