# Báo cáo Kết quả Phần 1: Thu thập dữ liệu, Chuẩn hóa nhãn & EDA — _TV1_

- **Phụ trách:** Thành viên 1 (`feat/crawl-eda`)
- **Trạng thái:** **Đã ánh xạ lại toàn bộ 42 thẻ gốc $\rightarrow$ 22 nhãn chuẩn cân bằng và chạy xong [01_eda.ipynb](../notebooks/01_eda.ipynb) trên toàn bộ tập dữ liệu `webtruyen.vn` (`18.603` truyện hợp lệ).**

---

## 1. Các công việc đã hoàn thành

### 1.1. Tái thiết kế bảng ánh xạ 42 thẻ gốc $\rightarrow$ 22 nhãn chuẩn (`src/config.py`, `src/crawl/sources.py`)
Tận dụng quy mô `18.621` truyện duy nhất sau khi cào đủ 42 chuyên mục, bộ nhãn được chuẩn hóa thành **22 nhãn** cân bằng (mọi nhãn đều có `>= 148` mẫu, tỷ lệ max/min `58.8 : 1`):
- `ngon_tinh` (*ngôn tình, nữ phụ, tổng tài*), `dam_my` (*đam mỹ*), `bach_hop` (*bách hợp*), `tien_hiep` (*tiên hiệp*), `kiem_hiep` (*kiếm hiệp*), `huyen_huyen` (*huyền huyễn, dị giới, thám hiểm, light novel*), `linh_di` (*linh dị*), `khoa_huyen_mat_the` (*khoa huyễn, mạt thế, dị năng*), `vong_du` (*võng du*), `trinh_tham` (*trinh thám*), `dien_van` (*điền văn*), `lich_su_quan_su` (*lịch sử, quân sự, dã sử*).
- `co_dai` (*cổ đại, đông phương*), `hien_dai` (*đô thị, hiện đại, truyện teen*), `cung_dau_gia_dau` (*cung đấu, gia đấu*), `xuyen_khong` (*xuyên không, xuyên nhanh*), `trong_sinh` (*trọng sinh*), `he_thong` (*hệ thống*), `nu_cuong` (*nữ cường*), `sung` (*sủng*), `nguoc` (*ngược*), `hai_huoc` (*hài hước*).
- **Loại bỏ 6 thẻ (`NON_GENRE_TAGS`):** `khác` (`2.144`), `đoản văn` (`983`), `tiểu thuyết` (`584`), `phương tây` (`54`), `quan trường` (`36`), `việt nam` (`5`).

### 1.2. Kết quả lọc & xuất `stories.csv` (`src/crawl/merge_data.py`, `notebooks/01_eda.ipynb`)
- Từ **`55.684` bản ghi thô** (cào đủ 42 chuyên mục × 2.269 trang trong [webtruyen_raw.csv](../data/raw/webtruyen_raw.csv)), `deduplicate_stories()` loại trùng `title` + `summary` kèm gộp thẻ thu được **`18.621` truyện duy nhất** (loại `37.063` bản ghi trùng).
- Chỉ loại **`18` bản ghi** không còn nhãn hợp lệ trong 22 nhãn, giữ lại **`18.603 / 18.621` (`99.90%`) truyện hợp lệ**, xuất ra [stories.csv](../data/processed/stories.csv) (`18.603` dòng × `36` cột) và biểu đồ [eda_overview.png](../outputs/figures/eda_overview.png).

---

## 2. Kết quả EDA đa nhãn thực tế trên `stories.csv` (`18.603` mẫu, `22` nhãn)

- **Chỉ số đa nhãn:** `Label cardinality = 2.6704` nhãn/truyện | `Label density = 0.1214`.
- **Phân bố số nhãn trên mỗi truyện:** 1 nhãn (`3.959`), 2 nhãn (`5.338`), 3 nhãn (`4.906`), 4 nhãn (`2.678`), 5 nhãn (`1.079`), 6 nhãn (`402`), 7 nhãn (`156`), 8 nhãn (`64`), 9 nhãn (`11`), 10 nhãn (`8`), 11 nhãn (`2`).
- **Độ dài `summary` (`n_words`):** tối thiểu `15` từ, trung bình `202.76` từ, trung vị `157` từ, tối đa `3.598` từ.

| Nhãn chuẩn | Tên hiển thị | Số mẫu | Tỷ lệ (%) | Độ dài TB (từ) |
|---|---|---|---|---|
| `ngon_tinh` | Ngôn tình (Nam–Nữ) | 8.702 | 46.78% | 208.0 |
| `co_dai` | Cổ đại | 5.833 | 31.36% | 186.2 |
| `hien_dai` | Hiện đại / Đô thị | 5.613 | 30.17% | 213.0 |
| `sung` | Sủng / Ngọt | 4.568 | 24.56% | 244.0 |
| `dam_my` | Đam mỹ | 3.921 | 21.08% | 248.0 |
| `nu_cuong` | Nữ cường | 3.524 | 18.94% | 163.8 |
| `xuyen_khong` | Xuyên không / Xuyên nhanh | 2.299 | 12.36% | 242.1 |
| `trong_sinh` | Trọng sinh | 2.135 | 11.48% | 210.3 |
| `hai_huoc` | Hài hước | 1.979 | 10.64% | 198.1 |
| `huyen_huyen` | Huyền huyễn / Dị giới | 1.891 | 10.17% | 200.8 |
| `cung_dau_gia_dau` | Cung đấu / Gia đấu | 1.830 | 9.84% | 172.9 |
| `nguoc` | Ngược | 1.667 | 8.96% | 186.8 |
| `tien_hiep` | Tiên hiệp | 1.217 | 6.54% | 182.0 |
| `dien_van` | Điền văn | 1.069 | 5.75% | 218.7 |
| `he_thong` | Hệ thống | 789 | 4.24% | 241.4 |
| `khoa_huyen_mat_the` | Khoa huyễn / Mạt thế / Dị năng | 728 | 3.91% | 215.5 |
| `linh_di` | Linh dị / Kinh dị | 619 | 3.33% | 209.3 |
| `bach_hop` | Bách hợp | 416 | 2.24% | 323.4 |
| `kiem_hiep` | Kiếm hiệp | 320 | 1.72% | 194.0 |
| `trinh_tham` | Trinh thám | 233 | 1.25% | 209.9 |
| `lich_su_quan_su` | Lịch sử / Quân sự | 176 | 0.95% | 219.3 |
| `vong_du` | Võng du / Game | 148 | 0.80% | 222.6 |

- **Các cặp nhãn đồng xuất hiện nhiều nhất (`22×22`):** (`ngon_tinh`, `sung`): `3.055`; (`ngon_tinh`, `hien_dai`): `3.037`; (`ngon_tinh`, `co_dai`): `2.867`; (`co_dai`, `nu_cuong`): `2.167`; (`hien_dai`, `sung`): `1.881`; (`ngon_tinh`, `nu_cuong`): `1.737`; (`co_dai`, `cung_dau_gia_dau`): `1.543`; (`dam_my`, `sung`): `1.183`.
- **Unit tests ([tests/test_crawler_and_merge.py](../tests/test_crawler_and_merge.py)):** **4/4 PASS**.
