# Thư mục Báo cáo Kết quả theo từng Thành viên (`bao_cao_ket_qua/`)

Thư mục này lưu lại **toàn bộ phần việc đã làm, cơ sở logic chuẩn hóa nhãn, phương pháp kỹ thuật, sản phẩm đầu ra và số liệu thực tế** của từng thành viên (TV1–TV4) sau khi cào bổ sung đa nguồn (`webtruyen.vn`, `truyenfull.live`, `metruyenchu.com.vn`, `webnovel.vn`) và chuẩn hóa **42 thẻ gốc $\rightarrow$ 35 nhãn chuẩn** trên **`20.852` truyện hợp lệ**, dùng để tổng hợp vào báo cáo PDF ([reports/bao_cao.pdf](../reports/bao_cao.pdf)) và slide thuyết trình ([reports/slides.pdf](../reports/slides.pdf)).

## Danh sách báo cáo & trạng thái thực hiện

| Thành viên | File báo cáo | Mảng phụ trách | Trạng thái hiện tại |
|---|---|---|---|
| **TV1** | [01_tv1_thu_thap_va_eda.md](01_tv1_thu_thap_va_eda.md) | Bảng 42 thẻ + định nghĩa, Cào bổ sung 4 nguồn, Cơ sở logic 35 nhãn & EDA | **Đã hoàn thành & chạy thực tế** (`58.751` thô $\rightarrow$ `20.852` mẫu trong `stories.csv`, `eda_overview.png`) |
| **TV2** | [02_tv2_tien_xu_ly_va_dac_trung.md](02_tv2_tien_xu_ly_va_dac_trung.md) | Tiền xử lý tiếng Việt (NFC, tách dính hoa/thường), Chống rò rỉ nhãn, Chia tập & TF-IDF | **Đã hoàn thành & chạy đồng bộ** (`stories_clean.csv`: `train=14.613`, `val=3.125`, `test=3.114`; `tfidf_vectorizer.pkl` 10.000 chiều) |
| **TV3** | [03_tv3_mo_hinh_ml_truyen_thong.md](03_tv3_mo_hinh_ml_truyen_thong.md) | Mô hình ML truyền thống (`LogReg`, `Linear SVM`, `Multinomial NB`) & Bộ độ đo 35 nhãn | **Đã hoàn thành thực nghiệm (`TN1`–`TN4`)** (`Logistic Regression Test Macro-F1 = 0.3944`, `Micro-F1 = 0.5413`, `Accuracy = 0.0302`) |
| **TV4** | [04_tv4_deep_learning_va_ung_dung.md](04_tv4_deep_learning_va_ung_dung.md) | Deep Learning (`TextCNN`), Đối chiếu Bootstrap/Slice & Streamlit App 35 nhãn | **Đã hoàn thành toàn bộ (Giai đoạn 5b, 5c & 6)** (`TextCNN Macro-F1 = 0.3503 / Micro-F1 = 0.4968`, `max_len=750 + Dynamic Padding`, `app/streamlit_app.py`) |

## Lệnh kiểm tra nhanh hệ thống

```powershell
# 1. Chạy toàn bộ kiểm thử đơn vị (25 tests)
.\.venv\Scripts\python.exe -m pytest tests/ -v

# 2. Khởi chạy ứng dụng Demo Streamlit
.\.venv\Scripts\streamlit.exe run app/streamlit_app.py
```

Kết quả các mô hình được lưu tại [outputs/tables/experiments.csv](../outputs/tables/experiments.csv) và các biểu đồ tại `outputs/figures/`.
