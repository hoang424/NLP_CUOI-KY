# Thư mục Báo cáo Kết quả theo từng Thành viên (`bao_cao_ket_qua/`)

Thư mục này lưu lại **toàn bộ phần việc đã làm, phương pháp kỹ thuật, sản phẩm đầu ra và số liệu thực tế** của từng thành viên (TV1–TV4) sau khi ánh xạ lại toàn bộ **42 chuyên mục thể loại gốc $\rightarrow$ 22 nhãn chuẩn** trên tệp [webtruyen_raw.csv](file:///c:/Users/ACER/Downloads/NLP_CU%E1%BB%90I%20K%E1%BB%B2/data/raw/webtruyen_raw.csv) (`55.684` bản ghi thô $\rightarrow$ **`18.603` truyện hợp lệ**), dùng để tổng hợp vào báo cáo PDF ([reports/bao_cao.pdf](file:///c:/Users/ACER/Downloads/NLP_CU%E1%BB%90I%20K%E1%BB%B2/reports/bao_cao.pdf)) và slide thuyết trình ([reports/slides.pdf](file:///c:/Users/ACER/Downloads/NLP_CU%E1%BB%90I%20K%E1%BB%B2/reports/slides.pdf)).

## Danh sách báo cáo & trạng thái thực hiện

| Thành viên | File báo cáo | Mảng phụ trách | Trạng thái hiện tại |
|---|---|---|---|
| **TV1** | [01_tv1_thu_thap_va_eda.md](file:///c:/Users/ACER/Downloads/NLP_CU%E1%BB%90I%20K%E1%BB%B2/bao_cao_ket_qua/01_tv1_thu_thap_va_eda.md) | Dữ liệu `webtruyen.vn` (42 thẻ), Chuẩn hóa 22 nhãn & EDA | **Đã hoàn thành & chạy thực tế** (`55.684` bản ghi thô $\rightarrow$ `18.603` mẫu trong `stories.csv`, `eda_overview.png`) |
| **TV2** | [02_tv2_tien_xu_ly_va_dac_trung.md](file:///c:/Users/ACER/Downloads/NLP_CU%E1%BB%90I%20K%E1%BB%B2/bao_cao_ket_qua/02_tv2_tien_xu_ly_va_dac_trung.md) | Tiền xử lý tiếng Việt (chuẩn hóa NFC, tách từ dính hoa/thường), Chống rò rỉ nhãn, Chia tập & TF-IDF/Embedding | **Đã hoàn thành & chạy đồng bộ** (`stories_clean.csv`: `train=13.297`, `val=2.634`, `test=2.672`; `tfidf_vectorizer.pkl` 10.000 chiều) |
| **TV3** | [03_tv3_mo_hinh_ml_truyen_thong.md](file:///c:/Users/ACER/Downloads/NLP_CU%E1%BB%90I%20K%E1%BB%B2/bao_cao_ket_qua/03_tv3_mo_hinh_ml_truyen_thong.md) | Mô hình ML truyền thống (`LogReg`, `Linear SVM`, `Multinomial NB`) & Bộ độ đo | **Đã hoàn thành thực nghiệm (`TN1`–`TN4`)** (`Logistic Regression Test Macro-F1 = 0.4897`, `Micro-F1 = 0.6035`, `Hamming = 0.1230`) |
| **TV4** | [04_tv4_deep_learning_va_ung_dung.md](file:///c:/Users/ACER/Downloads/NLP_CU%E1%BB%90I%20K%E1%BB%B2/bao_cao_ket_qua/04_tv4_deep_learning_va_ung_dung.md) | Deep Learning (`TextCNN`), Phân tích lỗi & Streamlit App | **Đã hoàn thành toàn bộ (Giai đoạn 5b, 5c & 6)** (`TextCNN Macro-F1 = 0.4381 / Micro-F1 = 0.5742`, `app/streamlit_app.py`) |

## Lệnh kiểm tra nhanh hệ thống

```powershell
# 1. Chạy toàn bộ kiểm thử đơn vị (20 tests)
.\.venv\Scripts\python.exe -m pytest tests/ -v

# 2. Khởi chạy ứng dụng Demo Streamlit
.\.venv\Scripts\streamlit.exe run app/streamlit_app.py
```

Kết quả các mô hình được lưu tại [outputs/tables/experiments.csv](file:///c:/Users/ACER/Downloads/NLP_CU%E1%BB%90I%20K%E1%BB%B2/outputs/tables/experiments.csv) và các biểu đồ tại `outputs/figures/`.
