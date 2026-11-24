# Báo cáo Kết quả Phần 4: Deep Learning, Pipeline tổng & Ứng dụng Demo — _TV4_

- **Phụ trách:** Thành viên 4 (`feat/dl-app`)
- **Trạng thái:** **Đã hoàn thành Giai đoạn 5b (Deep Learning), 5c (Phân tích lỗi) và Giai đoạn 6 (Ứng dụng Streamlit) theo bộ 35 nhãn chuẩn (`20.852` mẫu đa nguồn) với độ dài chuẩn `max_len = 750` (p99.9) kết hợp Dynamic Padding theo batch.**

---

## 1. Các công việc đã hoàn thành

### 1.1. Cơ sở định lượng chọn `max_len = 750` (p99.9) & Dynamic Padding ([04_deep_learning.ipynb](../notebooks/04_deep_learning.ipynb), [embedding.py](../src/features/embedding.py))
- Để so sánh công bằng với BoW/TF-IDF (đọc 100% văn bản) mà vẫn giữ tốc độ CPU, `TEXTCNN_MAX_LEN = 750` trong [src/config.py](../src/config.py) được chọn theo phân vị **`p99.9`** trên cột `tokens` (`20.852` truyện):

| Tập | Số mẫu | `p50` | `p90` | `p95` | `p99` | `p99.5` | `p99.9` | `max` | Phủ truyện `<= 750` | Token giữ lại `<= 750` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `train` | 14.613 | 81 | 203 | 269 | 432 | 494 | 690 | 1.967 | 99.92% | 99.74% |
| `val` | 3.125 | 84 | 204 | 279 | 413 | 476 | 647 | 1.158 | 99.94% | 99.86% |
| `test` | 3.114 | 82 | 209 | 266 | 405 | 487 | 845 | 1.151 | 99.84% | 99.75% |
| **Toàn tập** | **20.852** | **82** | **204** | **270** | **424** | **494** | **700** | **1.967** | **99.91%** | **99.76%** |

- **Đường cong độ phủ, tỷ lệ `<PAD>` và bão hòa thông tin (`LogReg TF-IDF` trên `val`):**

| Ngưỡng cắt | Truyện giữ trọn | Số truyện bị cắt | Token giữ lại | `<PAD>` cố định | `<PAD>` Dynamic Padding | `LogReg val Macro-F1` | `LogReg val Micro-F1` |
|---|---:|---:|---:|---:|---:|---:|---:|
| `100` | 62.68% | 7.782 | 71.13% | 25.08% | 3.83% | 0.3940 | 0.5358 |
| `150` | 81.72% | 3.811 | 83.69% | 41.23% | 5.14% | 0.3973 | 0.5407 |
| `250` | 93.84% | 1.284 | 94.12% | 60.35% | 7.88% | 0.4068 | 0.5431 |
| `300` | 96.47% | 737 | 96.40% | 66.15% | 8.73% | 0.4108 | 0.5443 |
| `400` | 98.73% | 264 | 98.52% | 74.06% | 12.29% | 0.4114 | 0.5448 |
| `500` | 99.55% | 94 | 99.29% | 79.08% | 15.97% | **0.4117** | **0.5452** |
| **`750` (Chọn)** | **99.91%** | **18** | **99.76%** | 85.99% | **23.80%** | 0.4116 | 0.5451 |
| `FULL` | 100.00% | 0 | 100.00% | 94.65% | 39.72% | 0.4115 | 0.5451 |

- Mốc `750` bảo toàn **`99.91%`** truyện (chỉ cắt `18` ngoại lệ siêu dài), giữ `0.00%` bị cắt với `bach_hop`, `0.02%` với `sung`, `0.05%` với `dam_my`, `0.21%` với `lich_su`, `0.81%` với `quan_su`, đồng thời `LengthBucketBatchSampler` + Dynamic Padding giảm tỷ lệ `<PAD>` trên `train` từ **`86.01%`** xuống còn **`23.80%`**.

### 1.2. Huấn luyện Deep Learning ([04_deep_learning.ipynb](../notebooks/04_deep_learning.ipynb))
- Nạp tập chia `14.613 / 3.125 / 3.114`; từ điển **`20.002` token** (`min_freq=2`), khởi tạo `PPMI-SVD Embedding` (`20.002 × 128`, tạo trong `3.33s`) qua [src/features/embedding.py](../src/features/embedding.py).
- **`TN5_textcnn` (`TextCNNClassifier` trong [src/models/textcnn.py](../src/models/textcnn.py)):** Multi-scale 1D CNN (`kernel_sizes=(2, 3, 4)`, `num_filters=64`, `BatchNorm1d`, `GELU`, Global Max + Masked Mean Pooling, `spatial_dropout=0.15`, `dropout=0.35`, `sqrt pos_weight` max `3.5`, `OneCycleLR`). Huấn luyện `294.54s` (`10 epochs`, đỉnh `val Macro-F1 = 0.3315` ở epoch 9; sau tối ưu ngưỡng 71 điểm trên `val` đạt `Macro-F1 = 0.3795`, `Micro-F1 = 0.5067`). Trên `test`: **`Accuracy = 0.0218`**, **`Macro-F1 = 0.3503`**, **`Micro-F1 = 0.4968`**, **`Hamming Loss = 0.1255`** (`infer = 5803.76 ms`). Lưu tại [models/textcnn.pt](../models/textcnn.pt) và [models/dl_thresholds.json](../models/dl_thresholds.json).

### 1.3. Đối chiếu toàn bộ 5 thí nghiệm trên tập Test (`3.114` mẫu — `35` nhãn)

| Experiment ID | Mô hình | Biểu diễn đặc trưng | Subset Accuracy | Macro-F1 | Micro-F1 | Hamming Loss |
|---|---|---|---:|---:|---:|---:|
| `TN1_nb_bow` | Multinomial NB (BoW) | BoW (1-2 gram) | 0.0193 | 0.3501 | 0.4889 | 0.1186 |
| `TN2_multinomial_nb` | Multinomial NB (TF-IDF) | TF-IDF (1-2 gram) | 0.0286 | 0.3347 | 0.5376 | **0.0947** |
| `TN3_linear_svm` | Linear SVM (TF-IDF) | TF-IDF (1-2 gram) | 0.0260 | 0.3868 | 0.5276 | 0.1041 |
| `TN4_logistic_regression` | **Logistic Regression (TF-IDF)** | TF-IDF (1-2 gram) | **0.0302** | **0.3944** | **0.5413** | 0.1008 |
| `TN5_textcnn` | TextCNN (Word Embedding) | PPMI-SVD Emb + Multi-scale CNN (`max_len=750`) | 0.0218 | 0.3503 | 0.4968 | 0.1255 |

- Trên `TextCNN (Word Embedding)`, nhóm nhãn dẫn đầu gồm `ngon_tinh` (`0.8251` — cao nhất toàn hệ thống), `co_dai` (`0.8013`), `dam_my` (`0.7798`), `hien_dai` (`0.6483`), `tien_hiep` (`0.5867`), `xuyen_khong` (`0.5079`), `nu_cuong` (`0.4994`), `huyen_huyen` (`0.4806`), `sung` (`0.4780`), `kiem_hiep` (`0.4593`), `trong_sinh` (`0.4326`), `vong_du` (`0.4299`), `trinh_tham` (`0.4110`).
- Biểu đồ đã xuất: [dl_learning_curves.png](../outputs/figures/dl_learning_curves.png), [all_models_comparison.png](../outputs/figures/all_models_comparison.png), [dl_f1_per_label.png](../outputs/figures/dl_f1_per_label.png).

### 1.4. Đối chiếu Bootstrap 95% CI, Phân tích theo lát cắt & Độ nhạy ngưỡng ([04_deep_learning.ipynb](../notebooks/04_deep_learning.ipynb))
- **Bootstrap 95% CI (`1.000` lần trên `3.114` mẫu `test`):** `Logistic Regression` đạt `Macro-F1 = 0.3944 [0.3786, 0.4067]` (`Micro-F1 = 0.5413 [0.5337, 0.5495]`), `Linear SVM` đạt `0.3868 [0.3699, 0.4009]`, `TextCNN` đạt `0.3503 [0.3376, 0.3644]` (`Micro-F1 = 0.4968 [0.4895, 0.5045]`). Hiệu số `LogReg - TextCNN` có 95% CI **`[+0.0267, +0.0611]`** (dương hoàn toàn ở mức 95%), còn `LogReg - SVM` đạt **`[-0.0012, +0.0162]`**.
- **Slice-based Error Analysis ([error_analysis_by_slice.csv](../outputs/tables/error_analysis_by_slice.csv)):**
  - Nhóm độ dài `>250` từ (`n=749`) đạt hiệu năng cao nhất trên cả 3 mô hình (`LogReg Micro-F1 = 0.5934 / Macro-F1 = 0.4217`, `SVM = 0.5837 / 0.4135`, `TextCNN = 0.5183 / 0.3605`), vượt xa nhóm `<100` từ (`n=723`, `LogReg = 0.4843 / 0.3561`).
  - Nhóm `Có header Thể loại` (`n=779`) cao hơn `Không có header` (`n=2.335`) ở `LogReg` (`Micro-F1 = 0.5626` vs `0.5343`); truyện `>=4 nhãn` (`n=1.141`) đạt `Micro-F1` cao nhất (`LogReg = 0.6032`, `TextCNN = 0.5746`).
- **Phân loại lỗi thừa/thiếu & Độ nhạy ngưỡng `TextCNN`:** Số nhãn TB thực tế trên `test` là `3.17`, dự đoán TB `4.53` (`LogReg`), `4.54` (`SVM`), `5.56` (`TextCNN`). Khảo sát độ nhạy ngưỡng trên nhãn `trong_sinh` của `TextCNN`: ngưỡng `0.45` đạt đỉnh `val_f1 = 0.4305` (`test_f1 = 0.4350`, `Precision = 0.4009`, `Recall = 0.4754`), mốc `0.50` đạt `test_f1 = 0.4530` (`Precision = 0.4732`, `Recall = 0.4344`), tăng lên `0.60` làm `test_f1` giảm còn `0.4303`.
- **Biểu đồ đã xuất:** [error_analysis_slices.png](../outputs/figures/error_analysis_slices.png).

### 1.5. Ứng dụng Demo Streamlit ([app/streamlit_app.py](../app/streamlit_app.py))
- Cập nhật đồng bộ theo bộ **`20.852` truyện đa nguồn (`35` nhãn)**: đọc động chỉ số mô hình từ [experiments.csv](../outputs/tables/experiments.csv), mặc định `"Tự nhập văn bản mới"`, hiển thị **toàn bộ thể loại đạt ngưỡng xác suất** (không giới hạn Top-K) cùng biểu đồ cột xác suất và bảng chi tiết 35 nhãn; Tab 2 hiển thị thống kê theo nguồn, WordCloud 35 nhãn và đủ 10 biểu đồ (không chứa `dl_max_len_analysis.png`).
- **Unit tests ([tests/test_models.py](../tests/test_models.py)):** **4/4 PASS** (tổng toàn dự án **25/25 PASS**).

---

## 2. Việc còn lại trước khi thuyết trình
1. Chạy `.\.venv\Scripts\streamlit.exe run app/streamlit_app.py` trên máy để demo trực tiếp và quay video dự phòng lưu vào [reports/demo_backup.mp4](../reports/demo_backup.mp4).
2. Xuất lại [reports/bao_cao.pdf](../reports/bao_cao.pdf) và [reports/slides.pdf](../reports/slides.pdf) từ nội dung cập nhật trong `bao_cao_ket_qua/`.
