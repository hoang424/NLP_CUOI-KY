# Báo cáo Kết quả Phần 4: Deep Learning, Pipeline tổng & Ứng dụng Demo — _TV4_

- **Phụ trách:** Thành viên 4 (`feat/dl-app`)
- **Trạng thái:** **Đã hoàn thành chạy lại Giai đoạn 5b (Deep Learning), 5c (Phân tích lỗi) và Giai đoạn 6 (Ứng dụng Streamlit) theo bộ 22 nhãn chuẩn (`18.603` mẫu `webtruyen.vn`).**

---

## 1. Các công việc đã hoàn thành

### 1.1. Huấn luyện Deep Learning ([04_deep_learning.ipynb](file:///c:/Users/ACER/Downloads/NLP_CU%E1%BB%90I%20K%E1%BB%B2/notebooks/04_deep_learning.ipynb))
- Nạp dữ liệu chuẩn hóa `train=13.297`, `val=2.634`, `test=2.672`; xây dựng từ điển `build_vocab` (`min_freq=2`, tối đa `20.000` $\rightarrow$ **`20.002` token** gồm `<PAD>`, `<UNK>`) và khởi tạo ma trận nhúng `PPMI-SVD Embedding` (`20.002 × 128`, tạo trong `2.24s`) qua [src/features/embedding.py](file:///c:/Users/ACER/Downloads/NLP_CU%E1%BB%90I%20K%E1%BB%B2/src/features/embedding.py) với độ dài chuỗi `max_len=200`.
- **`TN5_textcnn` (`TextCNNClassifier` trong [src/models/textcnn.py](file:///c:/Users/ACER/Downloads/NLP_CU%E1%BB%90I%20K%E1%BB%B2/src/models/textcnn.py)):** Kiến trúc Multi-scale 1D CNN (`kernel_sizes=(2, 3, 4)`, `num_filters=64` mỗi kích thước, `BatchNorm1d`, `GELU`, Global Max + Masked Mean Pooling, `spatial_dropout=0.15`, `dropout=0.35`, `sqrt pos_weight` capped `3.5`, `OneCycleLR`, `batch_size=256`). Huấn luyện `284.83s` (`10 epochs`, đỉnh epoch 9: `val_loss = 0.4624`, `val Macro-F1 = 0.4194`, `val Micro-F1 = 0.5728`; sau tối ưu ngưỡng trên `val` đạt `Macro-F1 = 0.4505`, `Micro-F1 = 0.5918`, `Hamming = 0.1340`). Trên `test`: `Accuracy = 0.0423`, `Macro-F1 = 0.4381`, `Micro-F1 = 0.5742`, `Hamming Loss = 0.1383` (`infer = 1694.25 ms`). Lưu trọng số và ngưỡng tối ưu tại [models/textcnn.pt](file:///c:/Users/ACER/Downloads/NLP_CU%E1%BB%90I%20K%E1%BB%B2/models/textcnn.pt) và [models/dl_thresholds.json](file:///c:/Users/ACER/Downloads/NLP_CU%E1%BB%90I%20K%E1%BB%B2/models/dl_thresholds.json).

### 1.2. Đối chiếu toàn bộ 5 thí nghiệm trên tập Test (`2.672` mẫu — `22` nhãn)

| Experiment ID | Mô hình | Biểu diễn đặc trưng | Subset Accuracy | Macro-F1 | Micro-F1 | Hamming Loss |
|---|---|---|---|---|---|---|
| `TN1_nb_bow` | Multinomial NB (BoW) | BoW (1-2 gram) | 0.0603 | 0.4319 | 0.5641 | 0.1300 |
| `TN2_multinomial_nb` | Multinomial NB (TF-IDF) | TF-IDF (1-2 gram) | 0.0584 | 0.4031 | 0.5877 | 0.1207 |
| `TN3_linear_svm` | Linear SVM (TF-IDF) | TF-IDF (1-2 gram) | **0.0618** | 0.4767 | 0.6006 | **0.1193** |
| `TN4_logistic_regression` | **Logistic Regression (TF-IDF)** | TF-IDF (1-2 gram) | 0.0561 | **0.4897** | **0.6035** | 0.1230 |
| `TN5_textcnn` | TextCNN (Word Embedding) | PPMI-SVD Emb + Multi-scale CNN (`max_len=200`) | 0.0423 | 0.4381 | 0.5742 | 0.1383 |

- Trên `TextCNN (Word Embedding)`, các nhãn dẫn đầu gồm `ngon_tinh` (`0.8363`), `co_dai` (`0.8213`), `dam_my` (`0.7624`), `hien_dai` (`0.6647`), `tien_hiep` (`0.6518`), `huyen_huyen` (`0.4954`), `xuyen_khong` (`0.4897`), `nu_cuong` (`0.4853`), `sung` (`0.4409`), `kiem_hiep` (`0.4310`), `khoa_huyen_mat_the` (`0.4220`), `trong_sinh` (`0.4128`).
- Biểu đồ đã xuất: [dl_learning_curves.png](file:///c:/Users/ACER/Downloads/NLP_CU%E1%BB%90I%20K%E1%BB%B2/outputs/figures/dl_learning_curves.png), [all_models_comparison.png](file:///c:/Users/ACER/Downloads/NLP_CU%E1%BB%90I%20K%E1%BB%B2/outputs/figures/all_models_comparison.png), [dl_f1_per_label.png](file:///c:/Users/ACER/Downloads/NLP_CU%E1%BB%90I%20K%E1%BB%B2/outputs/figures/dl_f1_per_label.png).

### 1.3. Đối chiếu Bootstrap 95% CI, Phân tích theo lát cắt & Độ nhạy ngưỡng ([04_deep_learning.ipynb](file:///c:/Users/ACER/Downloads/NLP_CU%E1%BB%90I%20K%E1%BB%B2/notebooks/04_deep_learning.ipynb))
- **Bootstrap 95% CI (`1.000` lần trên `2.672` mẫu `test`):** `Logistic Regression` đạt `Macro-F1 = 0.4897 [0.4681, 0.5065]`, `Linear SVM` đạt `0.4767 [0.4571, 0.4938]`, `TextCNN (Word Embedding)` đạt `0.4381 [0.4221, 0.4537]`. Khoảng tin cậy hiệu số `LogReg - SVM` (`[+0.0022, +0.0237]`) và `LogReg - TextCNN` (`[+0.0312, +0.0714]`) đều dương hoàn toàn, khẳng định `Logistic Regression` dẫn đầu có ý nghĩa thống kê ở mức 95%.
- **Slice-based Error Analysis ([error_analysis_by_slice.csv](file:///c:/Users/ACER/Downloads/NLP_CU%E1%BB%90I%20K%E1%BB%B2/outputs/tables/error_analysis_by_slice.csv)):**
  - Nhóm độ dài `>250` từ (`n=682`) đạt hiệu năng cao nhất trên cả 3 mô hình (`LogReg Micro-F1 = 0.6262 / Macro-F1 = 0.5105`, `SVM = 0.6288 / 0.4987`, `TextCNN = 0.5826 / 0.4385`), vượt nhóm `<100` từ (`n=597`, `Micro-F1 = 0.5447–0.5703`).
  - Nhóm truyện `3 nhãn` (`n=717`) đạt `LogReg Micro-F1 = 0.6668` (`SVM Exact Acc = 10.88%`, `TextCNN Micro-F1 = 0.6253`), còn truyện `>=4 nhãn` (`n=829`, `avg_labels = 4.65`) đạt `LogReg Macro-F1 = 0.4972` (`Micro-F1 = 0.6572`, `TextCNN Micro-F1 = 0.6433`).
- **Phân loại lỗi thừa/thiếu nhãn & Độ nhạy ngưỡng `TextCNN`:** Số nhãn TB thực tế trên `test` là `2.89`, các mô hình dự đoán TB `3.94` (`LogReg`), `3.68` (`SVM`), `4.26` (`TextCNN`); `Exact Match` của `SVM` đạt `165 (6.2%)`, `LogReg` đạt `150 (5.6%)`, `TextCNN` đạt `113 (4.2%)`. Kiểm tra độ nhạy ngưỡng nhãn `trong_sinh` của `TextCNN`: ngưỡng `0.50` cho `test_f1 = 0.4434` (`Precision = 0.4354`, `Recall = 0.4517`), tăng lên `0.60` làm `test_f1` giảm xuống `0.4092` (`Recall = 0.3614`).
- **Biểu đồ đã xuất:** [error_analysis_slices.png](file:///c:/Users/ACER/Downloads/NLP_CU%E1%BB%90I%20K%E1%BB%B2/outputs/figures/error_analysis_slices.png). *(Phần ma trận cặp nhãn `22×22`, 12 mẫu lỗi tiêu biểu và giải thích từ khóa `coef_` của `Logistic Regression` đã chuyển về [03_ml_baselines.ipynb](file:///c:/Users/ACER/Downloads/NLP_CU%E1%BB%90I%20K%E1%BB%B2/notebooks/03_ml_baselines.ipynb) — báo cáo TV3.)*

### 1.4. Ứng dụng Demo Streamlit ([app/streamlit_app.py](file:///c:/Users/ACER/Downloads/NLP_CU%E1%BB%90I%20K%E1%BB%B2/app/streamlit_app.py))
- Đã cập nhật giao diện theo bộ **18.603 truyện (`22` nhãn)** với **`Logistic Regression (TF-IDF)`** làm mô hình mặc định hàng đầu cùng `Linear SVM (TF-IDF)`, `TextCNN (Word Embedding)` và `Multinomial NB (TF-IDF)`, hiển thị thể loại dự đoán, Top-K xác suất cao nhất, biểu đồ xác suất và bảng chi tiết 22 nhãn.
- **Unit tests ([tests/test_models.py](file:///c:/Users/ACER/Downloads/NLP_CU%E1%BB%90I%20K%E1%BB%B2/tests/test_models.py)):** **3/3 PASS**.

---

## 2. Việc còn lại trước khi thuyết trình
1. Chạy `.\.venv\Scripts\streamlit.exe run app/streamlit_app.py` trên máy để demo trực tiếp và quay video dự phòng lưu vào [reports/demo_backup.mp4](file:///c:/Users/ACER/Downloads/NLP_CU%E1%BB%90I%20K%E1%BB%B2/reports/demo_backup.mp4).
