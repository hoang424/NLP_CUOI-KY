# Báo cáo Kết quả Phần 1: Thu thập dữ liệu, Chuẩn hóa 35 nhãn & EDA — _TV1_

- **Phụ trách:** Thành viên 1 (`feat/crawl-eda`)
- **Trạng thái:** **Hoàn thành thu thập dữ liệu đa nguồn, chuẩn hóa từ 42 thẻ thể loại gốc sang 35 nhãn chuẩn và phân tích khám phá dữ liệu ([01_eda.ipynb](../notebooks/01_eda.ipynb)) trên `20.852` truyện hợp lệ.**

---

## 1. Quá trình thu thập và mở rộng nguồn dữ liệu (`src/crawl/`)

### 1.1. Nguồn dữ liệu nền tảng và nhu cầu bổ sung mẫu
Nguồn dữ liệu chính được thu thập từ **`webtruyen.vn`** (chuyển hướng chính thức về `wetruyen.com`) bằng cách duyệt toàn bộ **42 chuyên mục thể loại** qua `2.269` trang, thu được **`55.684` bản ghi thô**. Sau khi khử trùng các bản ghi xuất hiện lặp lại ở nhiều chuyên mục, tập nền tảng còn **`18.621` truyện duy nhất** (trong đó `18.617` truyện có nhãn hợp lệ theo hệ 35 nhãn).

Khi thống kê phân bố trên tập nền tảng, một số thể loại chuyên biệt có số lượng truyện hoàn thành còn thấp để huấn luyện và kiểm định ổn định trên mô hình đa nhãn, gồm: *Thám Hiểm* (`18` mẫu), *Quan Trường* (`36` mẫu), *Quân Sự* (`90` mẫu), *Nữ Phụ* (`119` mẫu), *Lịch Sử / Dã Sử* (`143` mẫu), *Võng Du* (`148` mẫu), *Học đường / Truyện teen* (`180` mẫu), *Khoa Huyễn* (`203` mẫu), *Mạt Thế* (`204` mẫu) và *Trinh Thám* (`233` mẫu). Do đó, nhóm tiến hành khảo sát và thu thập bổ sung từ các trang truyện tiếng Việt công khai, không chặn truy cập tự động và tuân thủ `robots.txt`.

### 1.2. Đánh giá và thu thập từ các nguồn bổ sung
- **`truyenfull.live` (Nguồn bổ sung chính):** Cho phép truy cập công khai, dùng chung hệ thẻ thể loại đa nhãn với `webtruyen.vn`, bao gồm cả truyện đang cập nhật lẫn hoàn thành nên tỷ lệ truyện mới chưa xuất hiện trong `webtruyen_raw.csv` đạt khoảng `56%–78%`, đồng thời trang chuyên mục có sẵn mô tả thể loại (`div.cat-desc`). Hàm `crawl_truyenfull()` và `parse_truyenfull_story_html()` được xây dựng để cào **10 chuyên mục thiếu mẫu** (cào toàn bộ các mục ít mẫu như *Thám Hiểm* `9` trang, *Quan Trường* `3` trang, *Quân Sự* `8` trang, *Lịch Sử* `10` trang, *Võng Du* `13` trang; cào giới hạn trang có kiểm soát với *Nữ Phụ*, *Truyện Teen*, *Khoa Huyễn*, *Mạt Thế*, *Trinh Thám*), thu về **`2.400` bản ghi thô** (`2.116` truyện duy nhất nội bộ) lưu tại [truyenfull_raw.csv](../data/raw/truyenfull_raw.csv).
- **`metruyenchu.com.vn`:** Tập hợp dữ liệu từ đợt cào tổng hợp trước đó (`374` truyện có văn án thật, nổi bật với `145` truyện thuộc nhóm *Dã Sử*) kết hợp cào bổ sung 10 chuyên mục thiếu mẫu qua chế độ `merge_existing=True` trong `crawl_metruyenchu()` (lọc bỏ các trang chỉ chứa văn bản SEO mặc định), thu thêm `46` truyện hợp lệ, nâng tổng số lên **`420` truyện** tại [metruyenchu_raw.csv](../data/raw/metruyenchu_raw.csv).
- **`webnovel.vn`:** Thu thập qua `sitemap.xml` và các trang `/all/` được **`247` bản ghi** ([webnovel_raw.csv](../data/raw/webnovel_raw.csv)). Do trang này chứa cả đầu sách kỹ năng – phi hư cấu (*Phát triển bản thân*, *Tâm linh*, *Kinh doanh*), bước chuẩn hóa tự động loại bỏ các đầu sách ngoài phạm vi truyện hư cấu và giữ lại **`137` truyện mạng** có nhãn hợp lệ.
- **Các nguồn đã khảo sát nhưng không sử dụng:** `truyenhoan.com` (dữ liệu và ID trùng khớp hoàn toàn với `wetruyen.com`), `truyenfull.vision` / `.bio` / `.vn` (lỗi máy chủ HTTP 521) và một số trang gặp lỗi kết nối/Cloudflare.

### 1.3. Tổng hợp dữ liệu đa nguồn và hiệu quả bổ sung mẫu (`src/crawl/merge_data.py`)

Hàm `deduplicate_stories()` hợp nhất 4 tệp thô theo tên truyện (chuẩn hóa NFC, chữ thường, khoảng trắng) và văn án, giữ bản tóm tắt đầy đủ hơn và gộp toàn bộ thẻ `genres_raw` phân cách bằng `|`. Từ **`58.751` bản ghi thô**, bước khử trùng thu được **`20.963` truyện duy nhất** (loại `37.788` bản ghi trùng); sau khi áp dụng bảng chuẩn hóa 35 nhãn và loại các sách phi hư cấu/ngoại lệ, tập dữ liệu hoàn chỉnh gồm **`20.852` truyện hợp lệ** trong [stories.csv](../data/processed/stories.csv) (`49` cột).

| Nguồn dữ liệu | Bản ghi thô | Truyện duy nhất nội bộ | Hợp lệ theo 35 nhãn | Giữ lại sau khử trùng liên nguồn | Tỷ lệ đóng góp |
|---|---:|---:|---:|---:|---:|
| `webtruyen.vn` (`wetruyen.com`) | 55.684 | 18.621 | 18.617 | **18.230** | 87.43% |
| `truyenfull.live` | 2.400 | 2.116 | 2.116 | **2.069** | 9.92% |
| `metruyenchu.com.vn` | 420 | 420 | 420 | **418** | 2.00% |
| `webnovel.vn` | 247 | 245 | 137 | **135** | 0.65% |
| **Tổng cộng (`4` nguồn)** | **58.751** | — | — | **20.852** | **100.00%** |

Bảng đối chiếu số lượng mẫu ở 10 thể loại thiếu dữ liệu trước và sau khi mở rộng nguồn:

| Thể loại được bổ sung | Tập nền tảng (`webtruyen.vn`) | Sau khi gộp đa nguồn | Mức tăng |
|---|---:|---:|---|
| `tham_hiem` (Thám hiểm) | 18 | **235** | +217 (**×13.1**) |
| `quan_truong` (Quan trường) | 36 | **71** | +35 (**×2.0**) |
| `quan_su` (Quân sự) | 90 | **246** | +156 (**×2.7**) |
| `nu_phu` (Nữ phụ) | 119 | **384** | +265 (**×3.2**) |
| `lich_su` (Lịch sử / Dã sử) | 143 | **477** | +334 (**×3.3**) |
| `vong_du` (Võng du / Game) | 148 | **488** | +340 (**×3.3**) |
| `hoc_duong` (Học đường / Truyện teen) | 180 | **510** | +330 (**×2.8**) |
| `khoa_huyen` (Khoa huyễn) | 203 | **520** | +317 (**×2.6**) |
| `mat_the` (Mạt thế) | 204 | **417** | +213 (**×2.0**) |
| `trinh_tham` (Trinh thám) | 233 | **471** | +238 (**×2.0**) |

---

## 2. Bảng 42 thẻ thể loại gốc & Cơ sở chuẩn hóa thành 35 nhãn

### 2.1. Định nghĩa và thống kê toàn bộ 42 thẻ trên `webtruyen.vn`

> **Lưu ý về nguồn định nghĩa:** Qua kiểm tra HTML thực tế tại `wetruyen.com`, các trang chuyên mục không hiển thị phần mô tả thể loại mà chỉ sinh thẻ meta SEO theo mẫu chung (`"Danh sách 50 Truyện <Tên> full hay nhất năm 2025..."`). Do đó, bảng dưới đây tham chiếu trực tiếp mô tả từ thẻ `<div class="cat-desc">` của nguồn đồng hệ nhãn **`truyenfull.live`** (`38/42` thẻ) kết hợp định nghĩa chuẩn tiểu thuyết mạng cho `4` thẻ còn lại (*Hiện Đại*, *Tổng tài*, *Tiểu Thuyết*, *Dã Sử*).

| STT | Thẻ gốc (`genres_raw`) | Định nghĩa thể loại | Số truyện (`webtruyen.vn`) | Quy tắc xử lý | Nhãn chuẩn (`35` nhãn) |
|---|---|---|---:|---|---|
| 1 | `Ngôn Tình` | Truyện thuộc kiểu lãng mạn, kể về những sự kiện vui buồn trong tình yêu nam–nữ của nhân vật chính | 8.556 | Giữ nguyên | `ngon_tinh` |
| 2 | `Cổ Đại` | Truyện có nội dung, bối cảnh và xã hội xảy ra ở thời cổ đại phong kiến | 5.791 | Giữ nguyên | `co_dai` |
| 3 | `Sủng` | Ngược với truyện ngược, đem lại cảm giác ngọt ngào vui vẻ khi nhân vật chính được nửa kia quan tâm, cưng chiều hết mình | 4.568 | Giữ nguyên | `sung` |
| 4 | `Đam Mỹ` | Truyện tình cảm lãng mạn giữa nhân vật nam và nam | 3.921 | Giữ nguyên | `dam_my` |
| 5 | `Đô Thị` | Những truyện có nội dung chủ yếu diễn ra trong môi trường đô thị hiện đại (cuộc sống thành thị, công sở, thương trường) | 3.771 | Gộp (`Đô Thị + Hiện Đại`) | `hien_dai` |
| 6 | `Nữ Cường` | Truyện có nhân vật nữ sở hữu tính cách, bản lĩnh, trí tuệ và thực lực mạnh mẽ, độc lập | 3.524 | Giữ nguyên | `nu_cuong` |
| 7 | `Hiện Đại` | Truyện lấy bối cảnh xã hội, văn hóa, công nghệ và lối sống thời kỳ hiện đại | 2.459 | Gộp (`Đô Thị + Hiện Đại`) | `hien_dai` |
| 8 | `Xuyên Không` | Nhân vật chính vì lý do nào đó mà bị đưa đến sinh sống ở một không gian hay một khoảng thời gian khác | 2.213 | Gộp (`Xuyên Không + Xuyên Nhanh`) | `xuyen_khong` |
| 9 | `Khác` | Những truyện có chủ đề pha trộn hoặc không thuộc riêng biệt các nhóm chuyên mục hẹp có sẵn trên web | 2.144 | Giữ nguyên | `khac` |
| 10 | `Trọng Sinh` | Nhân vật chính sau khi chết đi được quay trở lại quá khứ hoặc sống lại bằng chính kiếp trước để làm lại cuộc đời | 2.135 | Giữ nguyên | `trong_sinh` |
| 11 | `Hài Hước` | Những bộ truyện có nội dung, lời thoại và tình huống hài hước, đem lại tiếng cười vui vẻ cho độc giả | 1.979 | Giữ nguyên | `hai_huoc` |
| 12 | `Huyền Huyễn` | Truyện có các yếu tố huyền bí, pháp thuật, siêu nhiên, thế giới tưởng tượng mà khoa học khó giải thích | 1.734 | Giữ nguyên | `huyen_huyen` |
| 13 | `Ngược` | Truyện có tình tiết bi thương khiến người đọc xúc động mạnh, thường là nhân vật bị tổn thương thể xác hoặc tinh thần | 1.667 | Giữ nguyên | `nguoc` |
| 14 | `Cung Đấu` | Thể loại có các nhân vật trong hoàng cung thời phong kiến đấu đá, tranh sủng, âm mưu tính toán lẫn nhau | 1.227 | Giữ nguyên | `cung_dau` |
| 15 | `Tiên Hiệp` | Truyện kể về quá trình tu luyện (Luyện Khí, Trúc Cơ, Kim Đan, Nguyên Anh...) và khám phá thế giới tu chân thành tiên | 1.217 | Giữ nguyên | `tien_hiep` |
| 16 | `Điền Văn` | Thể loại nhẹ nhàng, kể về cuộc sống điền viên, canh nông, chăm sóc gia đình, buôn bán nhỏ và sự nghiệp bình dị | 1.069 | Giữ nguyên | `dien_van` |
| 17 | `Đoản Văn` | Những truyện có dung lượng ngắn gọn, cô đọng, thường chỉ kéo dài từ một đến vài chương | 983 | Giữ nguyên | `doan_van` |
| 18 | `Gia Đấu` | Còn gọi là trạch đấu; các tình tiết đấu đá, mưu tính giữa các thành viên trong gia đình hoặc dòng họ thế gia | 802 | Giữ nguyên | `gia_dau` |
| 19 | `Hệ Thống` | Nhân vật chính mang trong người một "hệ thống" cộng sinh, hoàn thành nhiệm vụ/yêu cầu để nhận phần thưởng hoặc năng lực | 789 | Giữ nguyên | `he_thong` |
| 20 | `Linh Dị` | Truyện có yếu tố ma quỷ, tâm linh, hiện tượng siêu nhiên rùng rợn, kinh dị | 619 | Giữ nguyên | `linh_di` |
| 21 | `Tổng tài` | Tiểu thuyết ngôn tình lấy nhân vật nam chính là chủ tịch, giám đốc điều hành tập đoàn quyền lực, giàu có (bá đạo tổng tài) | 586 | Giữ nguyên | `tong_tai` |
| 22 | `Tiểu Thuyết` | Các tác phẩm văn xuôi tự sự dài hơi có cốt truyện hoàn chỉnh, thiên về tâm lý – xã hội hoặc văn học đại chúng | 584 | Giữ nguyên | `tieu_thuyet` |
| 23 | `Dị Năng` | Nhân vật sở hữu các năng lực siêu nhiên đặc biệt (điều khiển nguyên tố, ngoại cảm, không gian...) mà người thường không có | 435 | Giữ nguyên | `di_nang` |
| 24 | `Bách Hợp` | Truyện tình cảm lãng mạn giữa nhân vật nữ và nữ | 416 | Giữ nguyên | `bach_hop` |
| 25 | `Kiếm Hiệp` | Truyện xoay quanh nhân vật chính rèn luyện bí kíp võ công, phiêu lưu hành hiệp trong giới võ lâm giang hồ | 320 | Giữ nguyên | `kiem_hiep` |
| 26 | `Dị Giới` | Trong truyện xuất hiện những thế giới khác biệt hoàn toàn với Trái Đất (luật lệ, chủng tộc, ma pháp hoặc nền văn minh riêng) | 259 | Giữ nguyên | `di_gioi` |
| 27 | `Trinh Thám` | Truyện mang yếu tố điều tra, suy luận logic, phá án hình sự ly kỳ và hấp dẫn | 233 | Giữ nguyên | `trinh_tham` |
| 28 | `Mạt Thế` | Truyện lấy bối cảnh ngày tận thế, thế giới sụp đổ vì thảm họa, thiên tai hoặc đại dịch (zombie/tang thi, sinh tồn) | 204 | Giữ nguyên | `mat_the` |
| 29 | `Khoa Huyễn` | Những bộ truyện giả tưởng có yếu tố khoa học công nghệ tiên tiến, vũ trụ, tinh tế, cơ giáp, AI | 203 | Giữ nguyên | `khoa_huyen` |
| 30 | `Truyện Teen` | Truyện viết cho lứa tuổi thanh thiếu niên, thường kể về tình cảm, tình bạn tuổi học đường hồn nhiên, trong sáng | 180 | Đổi tên hiển thị | `hoc_duong` (*Học đường / Truyện teen*) |
| 31 | `Xuyên Nhanh` | Còn gọi là mau xuyên / khoái xuyên; xuyên qua nhiều thế giới liên tiếp nhau để thực hiện nhiệm vụ | 153 | Gộp (`Xuyên Không + Xuyên Nhanh`) | `xuyen_khong` |
| 32 | `Võng Du` | Thể loại khoa học viễn tưởng lấy bối cảnh game online, thực tế ảo, tổ đội phó bản và thế giới trò chơi trực tuyến | 148 | Giữ nguyên | `vong_du` |
| 33 | `Lịch Sử` | Nội dung xảy ra trong bối cảnh lịch sử có thật hoặc gần sát với dòng chảy lịch sử thế giới thực | 134 | Gộp (`Lịch Sử + Dã Sử`) | `lich_su` |
| 34 | `Nữ Phụ` | Nhân vật chính xuyên vào hoặc thức tỉnh với thân phận nữ phụ (thường là pháo hôi/phản diện trong nguyên tác) để đổi vận | 119 | Giữ nguyên | `nu_phu` |
| 35 | `Quân Sự` | Truyện có yếu tố quân sự, binh pháp, chiến tranh, môi trường quân đội và người lính | 90 | Giữ nguyên | `quan_su` |
| 36 | `Đông Phương` | Thẻ chỉ bối cảnh địa lý – văn hóa thuộc các nước phương Đông | 55 | Loại bỏ (`NON_GENRE_TAGS`) | — |
| 37 | `Phương Tây` | Thẻ chỉ xuất xứ tác phẩm do các tác giả phương Tây sáng tác hoặc thuộc văn hóa phương Tây | 54 | Loại bỏ (`NON_GENRE_TAGS`) | — |
| 38 | `Quan Trường` | Truyện kể về quá trình đấu trí chính trị, điều hành bộ máy và thăng quan tiến chức trong chính trường | 36 | Giữ nguyên | `quan_truong` |
| 39 | `Thám Hiểm` | Nội dung truyện kể về các cuộc phiêu lưu thám hiểm địa danh kỳ bí, di tích cổ, vùng đất hiểm trở | 18 | Giữ nguyên | `tham_hiem` |
| 40 | `Dã Sử` | Truyện phỏng theo giai thoại, truyền thuyết lịch sử dân gian không ghi chính thức trong chính sử | 9 | Gộp (`Lịch Sử + Dã Sử`) | `lich_su` |
| 41 | `Việt Nam` | Thẻ phân loại tác phẩm thuộc nền văn học Việt Nam / tác giả Việt Nam | 5 | Loại bỏ (`NON_GENRE_TAGS`) | — |
| 42 | `Light Novel` | Dạng tiểu thuyết nhẹ có xuất xứ từ Nhật Bản hướng tới độc giả trẻ (chỉ xuất xứ văn hóa – xuất bản Nhật Bản) | 4 | Loại bỏ (`NON_GENRE_TAGS`) | — |

### 2.2. Cơ sở logic xây dựng hệ thống 35 nhãn chuẩn

Việc chuẩn hóa từ 42 thẻ gốc về 35 nhãn dựa trên sự đồng nhất về ngữ nghĩa thể loại, tránh gộp cơ học chỉ vì chênh lệch số lượng mẫu.

#### 2.2.1. Các cặp thẻ đồng trục ngữ nghĩa được hợp nhất (3 cặp)
1. **`Đô Thị + Hiện Đại` $\rightarrow$ `hien_dai` (*Hiện đại / Đô thị*):** Hai thẻ cùng mô tả bối cảnh xã hội hiện đại (đối lập trực tiếp với `co_dai`); `Đô Thị` nhấn mạnh không gian thành thị trong bối cảnh đương đại. Trên dữ liệu `webtruyen.vn`, có `749` truyện gắn đồng thời cả hai thẻ (`30,5%` truyện *Hiện Đại* đồng xuất hiện với *Đô Thị*, hệ số Jaccard đạt `13,7%`).
2. **`Xuyên Không + Xuyên Nhanh` $\rightarrow$ `xuyen_khong` (*Xuyên không / Xuyên nhanh*):** Mang quan hệ bao hàm (thể loại cha – nhánh biến thể), trong đó *Xuyên Nhanh* (mau xuyên) là mô-típ *Xuyên Không* lặp lại qua nhiều thế giới liên tiếp. Trên `webtruyen.vn`, `43,8%` truyện *Xuyên Nhanh* được gắn đồng thời thẻ *Xuyên Không*.
3. **`Lịch Sử + Dã Sử` $\rightarrow$ `lich_su` (*Lịch sử / Dã sử*):** Cùng khai thác bối cảnh lịch sử, triều đại và biến cố quá khứ. Ở cấp độ tóm tắt văn án, ranh giới giữa chính sử và dã sử/giai thoại khó tách biệt rành mạch; đồng thời *Dã Sử* trên `webtruyen.vn` chỉ có `9` mẫu nhưng được bổ sung mạnh từ `metruyenchu.com.vn` (`145` mẫu), tạo thành cụm *Lịch sử / Dã sử* với `477` mẫu.

#### 2.2.2. Phân định ranh giới giữa các nhóm thể loại lân cận (32 nhãn duy trì riêng)
- **Nhóm `Học đường / Truyện teen` (`hoc_duong`) vs `Hiện đại / Đô thị` (`hien_dai`):** Dù bối cảnh diễn ra ở thời hiện đại, *Truyện Teen* tập trung vào lứa tuổi học trò, môi trường trường lớp và tình cảm thanh xuân. Trên `webtruyen.vn`, Jaccard giữa *Truyện Teen* và *Hiện Đại* chỉ là **`0,3%`** (chỉ `4,4%` đồng xuất hiện), do đó được giữ làm một nhãn riêng với tên gọi *Học đường / Truyện teen* để phân biệt rõ ràng.
- **Nhóm `Dị năng` (`di_nang`), `Mạt thế` (`mat_the`) và `Khoa huyễn` (`khoa_huyen`):** Ba thể loại này khác biệt rõ về trục nội dung và trường từ vựng (Jaccard giữa *Dị Năng*–*Khoa Huyễn* là **`3,7%`**, giữa *Mạt Thế*–*Khoa Huyễn* là **`5,4%`**): *Mạt thế* gắn với thảm họa tận thế, sinh tồn và zombie (*tận thế, zombie, tang_thi*); *Khoa huyễn* tập trung vào khoa học công nghệ, vũ trụ, tinh tế (*vũ_trụ, viễn_tưởng, tinh_tế*); còn *Dị năng* xoay quanh năng lực siêu nhiên cá nhân (*không_gian, năng_lực, biến_dị*).
- **Nhóm `Cung đấu` (`cung_dau`) vs `Gia đấu` (`gia_dau`):** Dù cùng mang mô-típ đấu trí nhưng không gian xã hội và hệ từ khóa khác biệt rõ rệt giữa hoàng cung – triều chính (*cung_đình, hoàng_hậu, thái_tử*) và gia tộc – phủ đệ (*hầu_phủ, phu_quân*), với Jaccard chỉ **`10,9%`**.
- **Nhóm `Huyền huyễn` (`huyen_huyen`), `Dị giới` (`di_gioi`) và `Thám hiểm` (`tham_hiem`):** *Dị giới* đặt trọng tâm vào việc đặt chân sang một thế giới có quy luật khác lạ (*dị_thế*), *Huyền huyễn* nhấn mạnh yếu tố phép thuật – huyền bí, còn *Thám hiểm* mô tả hành trình khám phá vùng đất bí ẩn hoặc cổ mộ (*phiêu_lưu, vùng_đất, hành_trình*, với `50%` mẫu gốc trên `webtruyen.vn` gắn kèm *Linh Dị*).
- **Nhóm `Ngôn tình` (`ngon_tinh`), `Tổng tài` (`tong_tai`) và `Nữ phụ` (`nu_phu`):** *Tổng tài* (*sếp, lạnh_lùng, bá_đạo*) và *Nữ phụ* (*xuyên, tiểu_thuyết, nữ phụ*) là các phân nhóm có mô-típ nhân vật đặc thù cao nên được giữ làm nhãn dự đoán riêng.
- **Nhóm `Quân sự` (`quan_su`) vs `Lịch sử / Dã sử` (`lich_su`):** *Quân sự* bao quát cả quân đội, chiến tranh hiện đại và đời sống quân nhân (*chiến_tranh, quân_đội*), khác với phạm vi lịch sử các triều đại quá khứ của `lich_su`.
- **Các nhãn hình thức và nhóm mở (`khac`, `tieu_thuyet`, `doan_van`):** Đây là những thẻ có tần suất cao trong kho truyện (`2.144`, `584`, `983` truyện gốc); việc duy trì giúp bộ nhãn phản ánh đầy đủ phân bố thực tế trên trang nguồn.

#### 2.2.3. Loại bỏ các thẻ phân loại theo nguồn gốc quốc gia, khu vực và phi hư cấu
Bốn thẻ **`Việt Nam`** (*văn học Việt Nam*), **`Light Novel`** (*tiểu thuyết nhẹ xuất xứ Nhật Bản*), **`Đông Phương`** (*các nước phương Đông*) và **`Phương Tây`** (*tác giả phương Tây*) phản ánh **xuất xứ quốc gia hoặc bối cảnh địa lý – văn hóa** thay vì thể loại nội dung văn học. Số liệu thực nghiệm trên `webtruyen.vn` cũng cho thấy tính không thuần nhất về thể loại của nhóm thẻ này (chỉ `23,6%` truyện gắn *Đông Phương* mang thẻ *Cổ Đại*, trong khi `56%` thuộc nhóm *Huyền Huyễn*). Khi đưa 4 thẻ này vào `NON_GENRE_TAGS` (cùng các thẻ sách phi hư cấu như *Phát triển bản thân*, *Tâm linh*, *Kinh doanh*, *Review sách* từ nguồn phụ), tỷ lệ giữ lại trên tập gốc `webtruyen.vn` vẫn đạt **`18.617 / 18.621` (`99,98%`)**.

---

## 3. Kết quả EDA đa nhãn trên `stories.csv` (`20.852` mẫu, `35` nhãn)

- **Chỉ số đa nhãn:** `Label cardinality = 3.0601` nhãn/truyện | `Label density = 0.0874`.
- **Phân bố số nhãn trên mỗi truyện:** 1 nhãn (`3.470`), 2 nhãn (`5.023`), 3 nhãn (`5.460`), 4 nhãn (`3.546`), 5 nhãn (`1.755`), 6 nhãn (`871`), 7 nhãn (`401`), 8 nhãn (`166`), 9 nhãn (`89`), 10 nhãn (`33`), `>=11` nhãn (`38`, tối đa `16` nhãn).
- **Độ dài `summary` (`n_words`):** tối thiểu `15` từ, trung bình `205.90` từ, trung vị `161` từ, tối đa `3.598` từ.
- **Biểu đồ EDA:** [eda_overview.png](../outputs/figures/eda_overview.png).

| STT | Nhãn chuẩn | Tên hiển thị | Tập nền (`webtruyen.vn`) | Tổng sau gộp (`20.852`) | Tỷ lệ (%) | Độ dài TB (từ) | `train` (`14.613`) | `val` (`3.125`) | `test` (`3.114`) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `ngon_tinh` | Ngôn tình (Nam–Nữ) | 8.556 | **9.664** | 46.35% | 210.9 | 6.222 | 1.721 | 1.721 |
| 2 | `hien_dai` | Hiện đại / Đô thị | 5.481 | **6.254** | 29.99% | 216.2 | 4.378 | 938 | 938 |
| 3 | `co_dai` | Cổ đại | 5.791 | **6.073** | 29.12% | 188.7 | 4.251 | 911 | 911 |
| 4 | `sung` | Sủng / Ngọt | 4.568 | **4.815** | 23.09% | 245.5 | 3.255 | 781 | 779 |
| 5 | `dam_my` | Đam mỹ | 3.921 | **3.988** | 19.13% | 251.6 | 2.792 | 598 | 598 |
| 6 | `nu_cuong` | Nữ cường | 3.524 | **3.772** | 18.09% | 169.3 | 2.683 | 544 | 545 |
| 7 | `xuyen_khong` | Xuyên không / Xuyên nhanh | 2.299 | **2.962** | 14.20% | 240.0 | 2.073 | 444 | 445 |
| 8 | `khac` | Khác | 2.144 | **2.950** | 14.15% | 283.8 | 2.065 | 442 | 443 |
| 9 | `trong_sinh` | Trọng sinh | 2.135 | **2.438** | 11.69% | 214.1 | 1.707 | 365 | 366 |
| 10 | `huyen_huyen` | Huyền huyễn | 1.734 | **2.432** | 11.66% | 208.7 | 1.702 | 365 | 365 |
| 11 | `hai_huoc` | Hài hước | 1.979 | **2.173** | 10.42% | 201.9 | 1.521 | 326 | 326 |
| 12 | `nguoc` | Ngược | 1.667 | **1.714** | 8.22% | 188.0 | 997 | 382 | 335 |
| 13 | `tien_hiep` | Tiên hiệp | 1.217 | **1.410** | 6.76% | 186.3 | 987 | 211 | 212 |
| 14 | `cung_dau` | Cung đấu | 1.227 | **1.278** | 6.13% | 181.2 | 901 | 188 | 189 |
| 15 | `dien_van` | Điền văn | 1.069 | **1.214** | 5.82% | 225.4 | 850 | 176 | 188 |
| 16 | `he_thong` | Hệ thống | 789 | **1.080** | 5.18% | 241.0 | 769 | 155 | 156 |
| 17 | `doan_van` | Đoản văn | 983 | **1.057** | 5.07% | 136.5 | 684 | 196 | 177 |
| 18 | `gia_dau` | Gia đấu | 802 | **837** | 4.01% | 168.1 | 584 | 120 | 133 |
| 19 | `linh_di` | Linh dị / Kinh dị | 619 | **763** | 3.66% | 214.0 | 534 | 114 | 115 |
| 20 | `di_nang` | Dị năng | 435 | **631** | 3.03% | 222.2 | 442 | 94 | 95 |
| 21 | `tong_tai` | Tổng tài | 586 | **588** | 2.82% | 151.9 | 388 | 93 | 107 |
| 22 | `tieu_thuyet` | Tiểu thuyết | 584 | **584** | 2.80% | 159.6 | 364 | 119 | 101 |
| 23 | `khoa_huyen` | Khoa huyễn | 203 | **520** | 2.49% | 224.3 | 369 | 75 | 76 |
| 24 | `hoc_duong` | Học đường / Truyện teen | 180 | **510** | 2.45% | 161.8 | 420 | 45 | 45 |
| 25 | `vong_du` | Võng du / Game | 148 | **488** | 2.34% | 217.5 | 375 | 56 | 57 |
| 26 | `lich_su` | Lịch sử / Dã sử | 143 | **477** | 2.29% | 260.4 | 352 | 62 | 63 |
| 27 | `bach_hop` | Bách hợp | 416 | **476** | 2.28% | 326.3 | 369 | 53 | 54 |
| 28 | `trinh_tham` | Trinh thám | 233 | **471** | 2.26% | 233.3 | 330 | 70 | 71 |
| 29 | `di_gioi` | Dị giới | 259 | **425** | 2.04% | 216.4 | 297 | 64 | 64 |
| 30 | `mat_the` | Mạt thế | 204 | **417** | 2.00% | 227.9 | 325 | 52 | 40 |
| 31 | `kiem_hiep` | Kiếm hiệp | 320 | **413** | 1.98% | 200.6 | 289 | 62 | 62 |
| 32 | `nu_phu` | Nữ phụ | 119 | **384** | 1.84% | 246.5 | 326 | 28 | 30 |
| 33 | `quan_su` | Quân sự | 90 | **246** | 1.18% | 296.4 | 172 | 37 | 37 |
| 34 | `tham_hiem` | Thám hiểm | 18 | **235** | 1.13% | 230.5 | 212 | 11 | 12 |
| 35 | `quan_truong` | Quan trường | 36 | **71** | 0.34% | 254.0 | 50 | 10 | 11 |

- **Các cặp nhãn đồng xuất hiện nhiều nhất (`35×35`):** (`ngon_tinh`, `hien_dai`): `3.332`; (`ngon_tinh`, `sung`): `3.222`; (`ngon_tinh`, `co_dai`): `3.018`; (`nu_cuong`, `co_dai`): `2.213`; (`sung`, `hien_dai`): `1.964`; (`ngon_tinh`, `nu_cuong`): `1.908`; (`ngon_tinh`, `khac`): `1.445`; (`ngon_tinh`, `xuyen_khong`): `1.416`.
- **Unit tests ([tests/test_crawler_and_merge.py](../tests/test_crawler_and_merge.py)):** **8/8 PASS** (kiểm thử bộ cào `webtruyen`, `truyenfull`, `metruyenchu`, `webnovel`, khử trùng và quy tắc chuẩn hóa `42 -> 35` nhãn).
