# Báo cáo Kết quả Phần 4: Deep Learning, Pipeline tổng & Ứng dụng Demo — _TV4_

- **Phụ trách:** Thành viên 4 (`feat/dl-app`)
- **Trạng thái:** **Đã hoàn thành chạy lại Giai đoạn 5b (Deep Learning), 5c (Phân tích lỗi) và Giai đoạn 6 (Ứng dụng Streamlit) theo bộ 22 nhãn chuẩn (`18.603` mẫu `webtruyen.vn`) với độ dài chuẩn `max_len = 750` (p99.9) kết hợp Dynamic Padding theo batch.**

---

## 1. Các công việc đã hoàn thành

### 1.1. Cơ sở định lượng chọn `max_len = 750` (p99.9) & Dynamic Padding ([04_deep_learning.ipynb](../notebooks/04_deep_learning.ipynb), [embedding.py](../src/features/embedding.py))
- Để bảo đảm **so sánh công bằng** với các mô hình BoW/TF-IDF (vốn đọc 100% văn bản) mà không bị chậm vì các ngoại lệ siêu dài (tối đa `1.967` token ở `ST_6834`), cấu hình `TEXTCNN_MAX_LEN = 750` trong [src/config.py](../src/config.py) được chọn dựa trên phân vị **`p99.9`** của cột `tokens` sau tách từ `underthesea` và lọc stopwords:

| Tập | Số mẫu | `p50` | `p90` | `p95` | `p99` | `p99.5` | `p99.9` | `max` | Phủ truyện `<= 750` | Token giữ lại `<= 750` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `train` | 13.297 | 80 | 201 | 267 | 416 | 478 | 664 | 1.316 | 99.92% | 99.81% |
| `val` | 2.634 | 85 | 211 | 271 | 399 | 497 | 748 | 1.104 | 99.89% | 99.79% |
| `test` | 2.672 | 82 | 210 | 275 | 435 | 544 | 759 | 1.967 | 99.89% | 99.41% |
| **Toàn tập** | **18.603** | **81** | **203** | **269** | **419** | **488** | **721** | **1.967** | **99.91%** | **99.75%** |

- **Đường cong độ phủ, tỷ lệ `<PAD>` và bão hòa thông tin phân loại (`LogReg TF-IDF` trên `val`):**

| Ngưỡng cắt | Truyện giữ trọn | Số truyện bị cắt | Token giữ lại | `<PAD>` cố định | `<PAD>` Dynamic Padding | `LogReg val Macro-F1` | `LogReg val Micro-F1` |
|---|---:|---:|---:|---:|---:|---:|---:|
| `100` | 63.24% | 6.838 | 71.32% | 25.36% | 4.05% | 0.4773 | 0.6075 |
| `150` | 81.86% | 3.375 | 83.81% | 41.53% | 5.48% | 0.4898 | 0.6117 |
| `250` | 93.88% | 1.139 | 94.23% | 60.55% | 8.51% | 0.5011 | 0.6156 |
| `300` | 96.56% | 640 | 96.50% | 66.34% | 9.55% | 0.5037 | 0.6160 |
| `400` | 98.80% | 224 | 98.56% | 74.21% | 13.53% | 0.5038 | 0.6160 |
| `500` | 99.58% | 79 | 99.29% | 79.22% | 18.16% | 0.5038 | 0.6161 |
| **`750` (Chọn)** | **99.91%** | **17** | **99.75%** | 86.08% | **26.29%** | **0.5045** | **0.6161** |
| `FULL` | 100.00% | 0 | 100.00% | 94.68% | 36.65% | 0.5044 | 0.6161 |

- **3 lý do chốt `max_len = 750` + Dynamic Padding:**
  1. **Bao phủ 99.91% truyện (vượt ngưỡng `p99.9 = 721`):** Chỉ cắt `17 / 18.603` truyện ngoại lệ siêu dài, bảo toàn **`99.75%`** tổng lượng token; kể cả những nhãn có văn án dài nhất như `bach_hop` (`p99 = 547`, cắt **`0.00%`** tại `750`), `dam_my` (`0.05%`), `sung` (`0.02%`), `xuyen_khong` (`0.09%`), `he_thong` (`0.13%`) gần như không mất ngữ cảnh.
  2. **Công bằng tuyệt đối về thông tin với BoW/TF-IDF:** Tại `750`, đường cong bão hòa thông tin của `Logistic Regression (TF-IDF)` trên `val` đạt **`Macro-F1 = 0.5045`** (`Micro-F1 = 0.6161`), tương đương hoàn toàn mức đọc 100% văn bản (`FULL`: `0.5044` / `0.6161`).
  3. **Dynamic Padding theo batch (`LengthBucketBatchSampler` + `make_dynamic_dataloader`):** Thay vì pad cứng mọi mẫu về `750` (lãng phí **`86.22%`** `<PAD>` trên `train`), cơ chế gom nhóm độ dài và pad động theo batch giảm tỷ lệ `<PAD>` trên `train` xuống còn **`26.29%`**, giúp huấn luyện CPU nhanh (`239.71s`) mà vẫn đọc trọn 99.91% dữ liệu. Các bảng phân tích định lượng được xuất trực tiếp trong [04_deep_learning.ipynb](../notebooks/04_deep_learning.ipynb).

### 1.2. Huấn luyện Deep Learning ([04_deep_learning.ipynb](../notebooks/04_deep_learning.ipynb))
- Nạp dữ liệu chuẩn hóa `train=13.297`, `val=2.634`, `test=2.672`; xây dựng từ điển `build_vocab` (`min_freq=2`, tối đa `20.000` $\rightarrow$ **`20.002` token** gồm `<PAD>`, `<UNK>`) và khởi tạo ma trận nhúng `PPMI-SVD Embedding` (`20.002 × 128`, tạo trong `2.64s`) qua [src/features/embedding.py](../src/features/embedding.py) với `max_len=750` và Dynamic Padding (`batch_size=256`).
- **`TN5_textcnn` (`TextCNNClassifier` trong [src/models/textcnn.py](../src/models/textcnn.py)):** Kiến trúc Multi-scale 1D CNN (`kernel_sizes=(2, 3, 4)`, `num_filters=64` mỗi kích thước, `BatchNorm1d`, `GELU`, Global Max + Masked Mean Pooling, `spatial_dropout=0.15`, `dropout=0.35`, `sqrt pos_weight` capped `3.5`, `OneCycleLR`). Huấn luyện `239.71s` (`10 epochs`, đỉnh epoch 10: `val_loss = 0.4608`, `val Macro-F1 = 0.4138`, `val Micro-F1 = 0.5759`, `Accuracy = 0.0763`, `Hamming = 0.1132`; sau tối ưu ngưỡng 71 điểm trên `val` đạt `Macro-F1 = 0.4488`, `Micro-F1 = 0.5889`, `Accuracy = 0.0535`, `Hamming = 0.1334`). Trên `test`: **`Accuracy = 0.0483`**, **`Macro-F1 = 0.4237`**, **`Micro-F1 = 0.5657`**, **`Hamming Loss = 0.1388`** (`infer = 4649.40 ms`). Lưu trọng số và ngưỡng tối ưu tại [models/textcnn.pt](../models/textcnn.pt) và [models/dl_thresholds.json](../models/dl_thresholds.json).

### 1.3. Đối chiếu toàn bộ 5 thí nghiệm trên tập Test (`2.672` mẫu — `22` nhãn)

| Experiment ID | Mô hình | Biểu diễn đặc trưng | Subset Accuracy | Macro-F1 | Micro-F1 | Hamming Loss |
|---|---|---|---|---|---|---|
| `TN1_nb_bow` | Multinomial NB (BoW) | BoW (1-2 gram) | 0.0603 | 0.4319 | 0.5641 | 0.1300 |
| `TN2_multinomial_nb` | Multinomial NB (TF-IDF) | TF-IDF (1-2 gram) | 0.0584 | 0.4031 | 0.5877 | 0.1207 |
| `TN3_linear_svm` | Linear SVM (TF-IDF) | TF-IDF (1-2 gram) | **0.0618** | 0.4767 | 0.6006 | **0.1193** |
| `TN4_logistic_regression` | **Logistic Regression (TF-IDF)** | TF-IDF (1-2 gram) | 0.0561 | **0.4897** | **0.6035** | 0.1230 |
| `TN5_textcnn` | TextCNN (Word Embedding) | PPMI-SVD Emb + Multi-scale CNN (`max_len=750`) | 0.0483 | 0.4237 | 0.5657 | 0.1388 |

- Trên `TextCNN (Word Embedding)`, các nhãn dẫn đầu gồm `ngon_tinh` (`0.8288`), `co_dai` (`0.8055`), `dam_my` (`0.7732`), `hien_dai` (`0.6607`), `tien_hiep` (`0.6461`), `huyen_huyen` (`0.4834`), `nu_cuong` (`0.4764`), `kiem_hiep` (`0.4694`), `xuyen_khong` (`0.4692`), `sung` (`0.4418`), `khoa_huyen_mat_the` (`0.4405`), `trong_sinh` (`0.4281`).
- Biểu đồ đã xuất: [dl_learning_curves.png](../outputs/figures/dl_learning_curves.png), [all_models_comparison.png](../outputs/figures/all_models_comparison.png), [dl_f1_per_label.png](../outputs/figures/dl_f1_per_label.png).

### 1.4. Đối chiếu Bootstrap 95% CI, Phân tích theo lát cắt & Độ nhạy ngưỡng ([04_deep_learning.ipynb](../notebooks/04_deep_learning.ipynb))
- **Bootstrap 95% CI (`1.000` lần trên `2.672` mẫu `test`):** `Logistic Regression` đạt `Macro-F1 = 0.4897 [0.4681, 0.5065]`, `Linear SVM` đạt `0.4767 [0.4571, 0.4938]`, `TextCNN (Word Embedding)` đạt `0.4237 [0.4117, 0.4359]` (`Micro-F1 = 0.5657 [0.5574, 0.5749]`). Khoảng tin cậy hiệu số `LogReg - SVM` (`[+0.0022, +0.0237]`) và `LogReg - TextCNN` (`[+0.0450, +0.0823]`) đều dương hoàn toàn, khẳng định `Logistic Regression` dẫn đầu có ý nghĩa thống kê ở mức 95%.
- **Slice-based Error Analysis ([error_analysis_by_slice.csv](../outputs/tables/error_analysis_by_slice.csv)):**
  - Nhóm độ dài `>250` từ (`n=682`) đạt hiệu năng cao nhất trên cả 3 mô hình (`LogReg Micro-F1 = 0.6262 / Macro-F1 = 0.5105`, `SVM = 0.6288 / 0.4987`, `TextCNN = 0.5756 / 0.4385`), vượt nhóm `<100` từ (`n=597`, `Micro-F1 = 0.5387–0.5703`).
  - Nhóm truyện `3 nhãn` (`n=717`) đạt `LogReg Micro-F1 = 0.6668` (`SVM Exact Acc = 10.88%`, `TextCNN Micro-F1 = 0.6177`), còn truyện `>=4 nhãn` (`n=829`, `avg_labels = 4.65`) đạt `LogReg Macro-F1 = 0.4972` (`Micro-F1 = 0.6572`, `TextCNN Micro-F1 = 0.6302`).
- **Phân loại lỗi thừa/thiếu nhãn & Độ nhạy ngưỡng `TextCNN`:** Số nhãn TB thực tế trên `test` là `2.89`, các mô hình dự đoán TB `3.94` (`LogReg`), `3.68` (`SVM`), `4.14` (`TextCNN`); `Exact Match` của `SVM` đạt `165 (6.2%)`, `LogReg` đạt `150 (5.6%)`, `TextCNN` đạt `129 (4.8%)`. Kiểm tra độ nhạy ngưỡng nhãn `trong_sinh` của `TextCNN`: ngưỡng `0.35` đạt `test_f1 = 0.4398` (`Precision = 0.3657`, `Recall = 0.5514`), ngưỡng `0.50` đạt đỉnh `val_f1 = 0.4946` (`test_f1 = 0.4281`, `Precision = 0.4295`, `Recall = 0.4268`), tăng lên `0.60` làm `test_f1` giảm xuống `0.4190` (`Precision = 0.4818`, `Recall = 0.3707`).
- **Biểu đồ đã xuất:** [error_analysis_slices.png](../outputs/figures/error_analysis_slices.png). *(Phần ma trận cặp nhãn `22×22`, 12 mẫu lỗi tiêu biểu và giải thích từ khóa `coef_` của `Logistic Regression` đã chuyển về [03_ml_baselines.ipynb](../notebooks/03_ml_baselines.ipynb) — báo cáo TV3.)*

### 1.5. Ứng dụng Demo Streamlit ([app/streamlit_app.py](../app/streamlit_app.py))
- Đã cập nhật giao diện theo bộ **18.603 truyện (`22` nhãn)** với **`Logistic Regression (TF-IDF)`** làm mô hình mặc định hàng đầu cùng `Linear SVM (TF-IDF)`, `TextCNN (Word Embedding)` (`max_len=750`) và `Multinomial NB (TF-IDF)`; mặc định chọn `"Tự nhập văn bản mới"`, hiển thị toàn bộ thể loại đạt ngưỡng xác suất (bỏ giới hạn Top-K), biểu đồ cột xác suất và bảng chi tiết 22 nhãn; đồng thời bỏ hình `dl_max_len_analysis.png` ở tab so sánh (chỉ giữ bảng phân tích trong [04_deep_learning.ipynb](../notebooks/04_deep_learning.ipynb)).
- **Unit tests ([tests/test_models.py](../tests/test_models.py)):** **4/4 PASS** (bao gồm kiểm thử Dynamic Padding & `LengthBucketBatchSampler`).

---

## 2. Việc còn lại trước khi thuyết trình
1. Chạy `.\.venv\Scripts\streamlit.exe run app/streamlit_app.py` trên máy để demo trực tiếp và quay video dự phòng lưu vào [reports/demo_backup.mp4](../reports/demo_backup.mp4).
2. Xuất lại [reports/bao_cao.pdf](../reports/bao_cao.pdf) và [reports/slides.pdf](../reports/slides.pdf) từ nội dung cập nhật trong `bao_cao_ket_qua/`.
