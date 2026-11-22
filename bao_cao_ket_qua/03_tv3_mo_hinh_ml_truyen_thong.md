# Báo cáo Kết quả Phần 3: Mô hình Machine Learning truyền thống & Đánh giá — _TV3_

- **Phụ trách:** Thành viên 3 (`feat/ml-models`)
- **Trạng thái:** **Đã hoàn thành huấn luyện, tinh chỉnh siêu tham số, tối ưu ngưỡng quyết định và đánh giá 4 thí nghiệm ML trên tập Test (`2.672` mẫu, `22` nhãn chuẩn).**

---

## 1. Các công việc đã hoàn thành

### 1.1. Mô hình ML & Tối ưu ngưỡng ([src/models/ml_models.py](../src/models/ml_models.py))
- Áp dụng **Binary Relevance (`OneVsRestClassifier`)** với `RANDOM_STATE = 42` cho `MultinomialNB` (BoW `TN1`, TF-IDF `TN2`), `LinearSVC` (`TN3`) và `LogisticRegression` (`TN4`).
- Tối ưu ngưỡng từng nhãn trên tập `val` (`tune_per_label_thresholds()`) và lưu vào [ml_thresholds.json](../models/ml_thresholds.json) (đủ 22 khóa nhãn).

### 1.2. Thực thi Notebook ML Baselines ([03_ml_baselines.ipynb](../notebooks/03_ml_baselines.ipynb))
- Nạp từ [stories_clean.csv](../data/processed/stories_clean.csv) (`train=13.297`, `val=2.634`, `test=2.672`) trên không gian TF-IDF/BoW `10.000` chiều.
- Baseline mặc định: `Linear SVM` overfit (`train_macro_f1 = 0.9200` vs `val = 0.3595`), còn `Multinomial NB (TF-IDF)` (`val = 0.1780`) và `Logistic Regression` (`val = 0.2376`) underfit ở ngưỡng `0.5`; sau tinh chỉnh + tối ưu ngưỡng trên `val`: `Logistic Regression` đạt **`val_macro_f1 = 0.5044`** (`val_micro_f1 = 0.6161`), `Linear SVM` đạt `0.4849` (`0.6084`), `NB BoW` đạt `0.4559` (`0.5818`) và `NB TF-IDF` đạt `0.4077` (`0.6046`).

---

## 2. Kết quả thực nghiệm trên tập Test (`2.672` mẫu — `22` nhãn)

### 2.1. Bảng so sánh các mô hình ([experiments.csv](../outputs/tables/experiments.csv))

| Experiment ID | Mô hình | Đặc trưng | Cấu hình tối ưu (từ `val`) | Subset Accuracy | Macro-F1 | Micro-F1 | Hamming Loss | Suy luận Test |
|---|---|---|---|---|---|---|---|---|
| `TN0_naive_baseline` | Naive Majority Baseline | Label Prior | Most frequent label | 0.0000 | 0.0337 | 0.3033 | 0.1232 | — |
| `TN1_nb_bow` | Multinomial NB (BoW) | BoW (1–2 gram) | `alpha=0.1` + tuned thresholds | 0.0603 | 0.4319 | 0.5641 | 0.1300 | 48.89 ms |
| `TN2_multinomial_nb` | Multinomial NB (TF-IDF) | TF-IDF (1–2 gram) | `alpha=0.05` + tuned thresholds | 0.0584 | 0.4031 | 0.5877 | 0.1207 | 39.32 ms |
| `TN3_linear_svm` | Linear SVM (TF-IDF) | TF-IDF (1–2 gram) | `C=0.25, cw=balanced` + tuned thresholds | **0.0618** | 0.4767 | 0.6006 | **0.1193** | **14.83 ms** |
| `TN4_logistic_regression` | **Logistic Regression (TF-IDF)** | TF-IDF (1–2 gram) | `C=1.0, cw=balanced` + tuned thresholds | 0.0561 | **0.4897** | **0.6035** | 0.1230 | 24.24 ms |

### 2.2. Hiệu năng 22 nhãn của mô hình ML tốt nhất (`Logistic Regression TF-IDF` trên tập Test)

| Thể loại | Precision | Recall | F1-score | Support (`test`) |
|---|---|---|---|---|
| `co_dai` (Cổ đại) | 0.8081 | 0.8617 | **0.8341** | 875 |
| `ngon_tinh` (Ngôn tình Nam–Nữ) | 0.7408 | 0.9156 | **0.8190** | 1.576 |
| `dam_my` (Đam mỹ) | 0.7569 | 0.7891 | **0.7727** | 588 |
| `hien_dai` (Hiện đại / Đô thị) | 0.5808 | 0.8409 | **0.6870** | 842 |
| `tien_hiep` (Tiên hiệp) | 0.6425 | 0.7268 | **0.6821** | 183 |
| `trong_sinh` (Trọng sinh) | 0.5642 | 0.4517 | **0.5017** | 321 |
| `vong_du` (Võng du / Game) | 0.4615 | 0.5455 | **0.5000** | 22 |
| `nu_cuong` (Nữ cường) | 0.3813 | 0.6871 | **0.4905** | 505 |
| `sung` (Sủng / Ngọt) | 0.3588 | 0.7417 | **0.4836** | 666 |
| `khoa_huyen_mat_the` (Khoa huyễn / Mạt thế / Dị năng) | 0.5625 | 0.4128 | **0.4762** | 109 |
| `kiem_hiep` (Kiếm hiệp) | 0.5938 | 0.3958 | **0.4750** | 48 |
| `xuyen_khong` (Xuyên không / Xuyên nhanh) | 0.4439 | 0.4928 | **0.4670** | 345 |
| `huyen_huyen` (Huyền huyễn / Dị giới) | 0.3719 | 0.5880 | **0.4557** | 284 |
| `cung_dau_gia_dau` (Cung đấu / Gia đấu) | 0.3536 | 0.5922 | **0.4428** | 255 |
| `linh_di` (Linh dị / Kinh dị) | 0.4368 | 0.4000 | **0.4176** | 95 |
| `trinh_tham` (Trinh thám) | 0.3571 | 0.4286 | 0.3896 | 35 |
| `dien_van` (Điền văn) | 0.4655 | 0.3253 | 0.3830 | 166 |
| `nguoc` (Ngược) | 0.2707 | 0.5229 | 0.3567 | 306 |
| `bach_hop` (Bách hợp) | 0.4167 | 0.2679 | 0.3261 | 56 |
| `lich_su_quan_su` (Lịch sử / Quân sự) | 0.3684 | 0.2593 | 0.3043 | 27 |
| `hai_huoc` (Hài hước) | 0.2020 | 0.4747 | 0.2834 | 297 |
| `he_thong` (Hệ thống) | 0.3134 | 0.1765 | 0.2258 | 119 |

### 2.3. Giải thích đặc trưng & Phân tích cặp nhãn nhầm của `Logistic Regression` ([03_ml_baselines.ipynb](../notebooks/03_ml_baselines.ipynb))
- **Ma trận cặp nhãn nhầm `22×22` ([error_confusion_pairs.png](../outputs/figures/error_confusion_pairs.png)) & 12 mẫu lỗi ([error_cases_sample.csv](../outputs/tables/error_cases_sample.csv)):** Các cặp `False Positive` lớn nhất trên `Logistic Regression` là `ngon_tinh -> sung: 491`, `ngon_tinh -> nu_cuong: 479`, `co_dai -> nu_cuong: 324`, `hien_dai -> sung: 302`, `ngon_tinh -> hai_huoc: 299`; đồng thời lọc `773` mẫu thuộc 4 nhóm lỗi trọng tâm và xuất 12 mẫu đại diện.
- **Interpretability (`coef_` — [top_features_per_genre.png](../outputs/figures/top_features_per_genre.png)):** Trọng số `Logistic Regression` bắt chuẩn từ khóa đặc trưng cho cả các nhãn mới tách: `vong_du` (*game, trò_chơi, chơi, chiến_đội*), `trinh_tham` (*cảnh_sát, vụ, phá_án, án*), `linh_di` (*quỷ, thần_quái, ma*), `khoa_huyen_mat_the` (*không_gian, zombie, tận thế*), `sung` (*sủng, ngọt*), `nguoc` (*ngược, ngược_luyến, ngược_tâm*), `cung_dau_gia_dau` (*cung_đình, quý_phi, hoàng_đế*).

### 2.4. Biểu đồ đã xuất (`outputs/figures/`)
- [ml_model_comparison.png](../outputs/figures/ml_model_comparison.png) — So sánh 4 thí nghiệm ML truyền thống & Naive Baseline trên 22 nhãn.
- [ml_f1_per_label.png](../outputs/figures/ml_f1_per_label.png) — Precision / Recall / F1 của `Logistic Regression (TF-IDF)` theo 22 thể loại.
- [ml_confusion_matrices.png](../outputs/figures/ml_confusion_matrices.png) — Lưới ma trận nhầm lẫn nhị phân 22 thể loại (`Logistic Regression`).
- [error_confusion_pairs.png](../outputs/figures/error_confusion_pairs.png) và [top_features_per_genre.png](../outputs/figures/top_features_per_genre.png) — Ma trận nhầm lẫn cặp nhãn và Top-8 từ khóa `coef_` theo thể loại.

---

## 3. Nhận xét chính & Kiểm thử
- **`Logistic Regression (TF-IDF, C=1.0, cw=balanced)`** dẫn đầu nhóm ML về **`Macro-F1 = 0.4897`** và **`Micro-F1 = 0.6035`**, gấp hơn 14 lần Naive Baseline (`Macro-F1 = 0.0337`), trong khi **`Linear SVM`** đạt **`Subset Accuracy = 0.0618`**, **`Hamming Loss = 0.1193`** và tốc độ suy luận nhanh nhất (`14.83 ms`).
- Các nhãn đặc thù đạt kết quả ổn định dù đã loại bỏ triệt để rò rỉ nhãn ở các đoạn văn án dính từ: `dam_my` (`0.7727`), `trong_sinh` (`0.5017`), `vong_du` (`0.5000`, dù chỉ `22` mẫu test), `sung` (`0.4836`), `khoa_huyen_mat_the` (`0.4762`), `cung_dau_gia_dau` (`0.4428`), `linh_di` (`0.4176`), `trinh_tham` (`0.3896`), `nguoc` (`0.3567`), `bach_hop` (`0.3261`), `lich_su_quan_su` (`0.3043`).
- **Unit tests ([tests/test_metrics.py](../tests/test_metrics.py)):** **3/3 PASS**.
