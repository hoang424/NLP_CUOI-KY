# Báo cáo Kết quả Phần 3: Mô hình Machine Learning truyền thống & Đánh giá — _TV3_

- **Phụ trách:** Thành viên 3 (`feat/ml-models`)
- **Trạng thái:** **Đã hoàn thành huấn luyện, tinh chỉnh siêu tham số, tối ưu ngưỡng quyết định và đánh giá 4 thí nghiệm ML trên tập Test (`3.114` mẫu, `35` nhãn chuẩn).**

---

## 1. Các công việc đã hoàn thành

### 1.1. Mô hình ML & Tối ưu ngưỡng ([src/models/ml_models.py](../src/models/ml_models.py))
- Áp dụng **Binary Relevance (`OneVsRestClassifier`)** với `RANDOM_STATE = 42` cho `MultinomialNB` (BoW `TN1`, TF-IDF `TN2`), `LinearSVC` (`TN3`) và `LogisticRegression` (`TN4`).
- Tối ưu ngưỡng từng nhãn trên tập `val` (`tune_per_label_thresholds()`) và lưu vào [ml_thresholds.json](../models/ml_thresholds.json) (đủ 35 khóa nhãn).

### 1.2. Thực thi Notebook ML Baselines ([03_ml_baselines.ipynb](../notebooks/03_ml_baselines.ipynb))
- Nạp từ [stories_clean.csv](../data/processed/stories_clean.csv) (`train=14.613`, `val=3.125`, `test=3.114`) trên không gian TF-IDF/BoW `10.000` chiều.
- Baseline mặc định: `Linear SVM` overfit (`train_macro_f1 = 0.8994` vs `val = 0.2840`), còn `Multinomial NB (TF-IDF)` (`val = 0.1113`) và `Logistic Regression` (`val = 0.1644`) underfit ở ngưỡng `0.5`; sau tinh chỉnh siêu tham số + tối ưu ngưỡng trên `val`:
  - `Logistic Regression (C=2.0, cw=balanced)`: `val_macro_f1` tăng `0.3796` $\rightarrow$ **`0.4115`** (`val_micro_f1 = 0.5451`, `Hamming = 0.0999`).
  - `Linear SVM (C=0.25, cw=balanced)`: `0.3747` $\rightarrow$ `0.3981` (`0.5274`, `0.1042`).
  - `Multinomial NB (BoW, alpha=0.25)`: `0.3389` $\rightarrow$ `0.3516` (`0.4931`, `0.1177`).
  - `Multinomial NB (TF-IDF, alpha=0.05)`: `0.2117` $\rightarrow$ `0.3393` (`0.5394`, `0.0942`).

---

## 2. Kết quả thực nghiệm trên tập Test (`3.114` mẫu — `35` nhãn)

### 2.1. Bảng so sánh các mô hình ([experiments.csv](../outputs/tables/experiments.csv))

| Experiment ID | Mô hình | Đặc trưng | Cấu hình tối ưu (từ `val`) | Subset Accuracy | Macro-F1 | Micro-F1 | Hamming Loss | Suy luận Test |
|---|---|---|---|---:|---:|---:|---:|---:|
| `TN0_naive_baseline` | Naive Majority Baseline | Label Prior | Most frequent label | 0.0000 | 0.0203 | 0.2652 | 0.0875 | — |
| `TN1_nb_bow` | Multinomial NB (BoW) | BoW (1–2 gram) | `alpha=0.25` + tuned thresholds | 0.0193 | 0.3501 | 0.4889 | 0.1186 | 79.64 ms |
| `TN2_multinomial_nb` | Multinomial NB (TF-IDF) | TF-IDF (1–2 gram) | `alpha=0.05` + tuned thresholds | 0.0286 | 0.3347 | 0.5376 | **0.0947** | 83.35 ms |
| `TN3_linear_svm` | Linear SVM (TF-IDF) | TF-IDF (1–2 gram) | `C=0.25, cw=balanced` + tuned thresholds | 0.0260 | 0.3868 | 0.5276 | 0.1041 | **27.23 ms** |
| `TN4_logistic_regression` | **Logistic Regression (TF-IDF)** | TF-IDF (1–2 gram) | `C=2.0, cw=balanced` + tuned thresholds | **0.0302** | **0.3944** | **0.5413** | 0.1008 | 35.19 ms |

### 2.2. Hiệu năng đầy đủ 35 nhãn của mô hình ML tốt nhất (`Logistic Regression TF-IDF` trên tập Test)

| STT | Thể loại | Precision | Recall | F1-score | Support (`test`) |
|---|---|---:|---:|---:|---:|
| 1 | `co_dai` (Cổ đại) | 0.7973 | 0.8332 | **0.8148** | 911 |
| 2 | `ngon_tinh` (Ngôn tình Nam–Nữ) | 0.7000 | 0.9274 | **0.7978** | 1.721 |
| 3 | `dam_my` (Đam mỹ) | 0.7630 | 0.8127 | **0.7870** | 598 |
| 4 | `hien_dai` (Hiện đại / Đô thị) | 0.5304 | 0.8475 | **0.6524** | 938 |
| 5 | `tien_hiep` (Tiên hiệp) | 0.5120 | 0.7028 | **0.5924** | 212 |
| 6 | `trinh_tham` (Trinh thám) | 0.4706 | 0.5634 | **0.5128** | 71 |
| 7 | `kiem_hiep` (Kiếm hiệp) | 0.5254 | 0.5000 | **0.5124** | 62 |
| 8 | `xuyen_khong` (Xuyên không / Xuyên nhanh) | 0.4193 | 0.6360 | **0.5054** | 445 |
| 9 | `nu_cuong` (Nữ cường) | 0.4198 | 0.6294 | **0.5037** | 545 |
| 10 | `sung` (Sủng / Ngọt) | 0.3920 | 0.6829 | **0.4981** | 779 |
| 11 | `trong_sinh` (Trọng sinh) | 0.5430 | 0.4481 | **0.4910** | 366 |
| 12 | `huyen_huyen` (Huyền huyễn) | 0.3850 | 0.6055 | **0.4707** | 365 |
| 13 | `vong_du` (Võng du / Game) | 0.4375 | 0.4912 | **0.4628** | 57 |
| 14 | `mat_the` (Mạt thế) | 0.3770 | 0.5750 | **0.4554** | 40 |
| 15 | `tham_hiem` (Thám hiểm) | 0.3571 | 0.4167 | **0.3846** | 12 |
| 16 | `linh_di` (Linh dị / Kinh dị) | 0.3796 | 0.3565 | **0.3677** | 115 |
| 17 | `cung_dau` (Cung đấu) | 0.3922 | 0.3175 | **0.3509** | 189 |
| 18 | `khoa_huyen` (Khoa huyễn) | 0.3676 | 0.3289 | **0.3472** | 76 |
| 19 | `nguoc` (Ngược) | 0.2645 | 0.4896 | **0.3435** | 335 |
| 20 | `khac` (Khác) | 0.2840 | 0.4289 | **0.3417** | 443 |
| 21 | `dien_van` (Điền văn) | 0.4364 | 0.2553 | **0.3221** | 188 |
| 22 | `nu_phu` (Nữ phụ) | 0.2453 | 0.4333 | **0.3133** | 30 |
| 23 | `quan_su` (Quân sự) | 0.2708 | 0.3514 | **0.3059** | 37 |
| 24 | `he_thong` (Hệ thống) | 0.2480 | 0.3974 | **0.3054** | 156 |
| 25 | `hai_huoc` (Hài hước) | 0.2168 | 0.4908 | **0.3008** | 326 |
| 26 | `lich_su` (Lịch sử / Dã sử) | 0.3077 | 0.2540 | **0.2783** | 63 |
| 27 | `di_nang` (Dị năng) | 0.1944 | 0.3684 | **0.2545** | 95 |
| 28 | `hoc_duong` (Học đường / Truyện teen) | 0.1644 | 0.5333 | **0.2513** | 45 |
| 29 | `gia_dau` (Gia đấu) | 0.1602 | 0.4962 | **0.2422** | 133 |
| 30 | `bach_hop` (Bách hợp) | 0.3913 | 0.1667 | **0.2338** | 54 |
| 31 | `tong_tai` (Tổng tài) | 0.2155 | 0.2336 | **0.2242** | 107 |
| 32 | `di_gioi` (Dị giới) | 0.1605 | 0.2031 | **0.1793** | 64 |
| 33 | `doan_van` (Đoản văn) | 0.1202 | 0.2486 | **0.1621** | 177 |
| 34 | `tieu_thuyet` (Tiểu thuyết) | 0.0929 | 0.2079 | **0.1284** | 101 |
| 35 | `quan_truong` (Quan trường) | 0.1429 | 0.0909 | **0.1111** | 11 |

### 2.3. Giải thích đặc trưng & Phân tích cặp nhãn nhầm của `Logistic Regression` ([03_ml_baselines.ipynb](../notebooks/03_ml_baselines.ipynb))
- **Ma trận cặp nhãn nhầm `35×35` ([error_confusion_pairs.png](../outputs/figures/error_confusion_pairs.png)) & 12 mẫu lỗi ([error_cases_sample.csv](../outputs/tables/error_cases_sample.csv)):** Các cặp `False Positive` lớn nhất trên `Logistic Regression` là `ngon_tinh -> sung: 482`, `ngon_tinh -> nu_cuong: 379`, `ngon_tinh -> hai_huoc: 336`, `ngon_tinh -> nguoc: 312`, `hien_dai -> sung: 305`; lọc `1.187` mẫu thuộc 4 nhóm lỗi trọng tâm và xuất 12 mẫu đại diện.
- **Interpretability (`coef_` — [top_features_per_genre.png](../outputs/figures/top_features_per_genre.png)):** Trọng số `Logistic Regression` bắt trúng từ khóa phân biệt giữa các nhóm thể loại gần bối cảnh hoặc dễ nhầm lẫn:
  - `mat_the` (*tận thế 10.60, zombie 9.78, tang_thi 7.54*) vs `khoa_huyen` (*thế_giới 7.63, vũ_trụ 5.86, viễn_tưởng 5.82, tinh_tế 5.06*) vs `di_nang` (*không_gian 7.96, năng_lực 6.86, biến_dị 4.80*).
  - `tham_hiem` (*phiêu_lưu 8.39, vùng_đất 5.49, hành_trình 4.97*).
  - `cung_dau` (*cung_đình 7.83, hoàng_hậu 6.10, thái_tử 5.36*) vs `gia_dau` (*hầu_phủ 4.99, phu_quân 4.03*).
  - `quan_su` (*chiến_tranh 5.50, quân_đội 4.71*) vs `quan_truong` (*cán_bộ 4.94, triều_đình 4.76, chính_trị 4.40*).
  - `vong_du` (*game 14.98, trò_chơi 12.40*), `trinh_tham` (*cảnh_sát 8.96, phá_án 7.41*), `linh_di` (*quỷ 12.17, thần_quái 7.05*).

### 2.4. Biểu đồ đã xuất (`outputs/figures/`)
- [ml_model_comparison.png](../outputs/figures/ml_model_comparison.png) — So sánh 4 thí nghiệm ML truyền thống & Naive Baseline trên 35 nhãn.
- [ml_f1_per_label.png](../outputs/figures/ml_f1_per_label.png) — Precision / Recall / F1 của `Logistic Regression (TF-IDF)` theo 35 thể loại.
- [ml_confusion_matrices.png](../outputs/figures/ml_confusion_matrices.png) — Lưới ma trận nhầm lẫn nhị phân `7×5` cho 35 thể loại (`Logistic Regression`).
- [error_confusion_pairs.png](../outputs/figures/error_confusion_pairs.png) và [top_features_per_genre.png](../outputs/figures/top_features_per_genre.png) — Ma trận cặp nhãn `35×35` và Top-8 từ khóa `coef_`.

---

## 3. Nhận xét chính & Kiểm thử
- **`Logistic Regression (TF-IDF, C=2.0, cw=balanced)`** dẫn đầu nhóm ML về **Accuracy (`0.0302`)**, **Macro-F1 (`0.3944`)** và **Micro-F1 (`0.5413`)**, gấp gần 20 lần Naive Baseline (`Macro-F1 = 0.0203`), trong khi **`Multinomial NB (TF-IDF)`** đạt **Hamming Loss thấp nhất (`0.0947`)** và **`Linear SVM`** có tốc độ suy luận nhanh nhất (`27.23 ms`).
- Nhờ cào bổ sung dữ liệu từ `truyenfull.live` và `metruyenchu.com.vn`, các thể loại ít mẫu ban đầu đều đạt hiệu năng khả quan trên tập Test độc lập (`trinh_tham = 0.5128`, `vong_du = 0.4628`, `mat_the = 0.4554`, `tham_hiem = 0.3846`, `nu_phu = 0.3133`, `quan_su = 0.3059`), trong khi các nhãn hình thức (`doan_van = 0.1621`, `tieu_thuyet = 0.1284`) vẫn thấp do dung lượng/cách phân loại không phụ thuộc nhiều vào từ vựng nội dung.
- **Unit tests ([tests/test_metrics.py](../tests/test_metrics.py)):** **3/3 PASS**.
