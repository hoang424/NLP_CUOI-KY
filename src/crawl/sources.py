"""Định nghĩa nguồn dữ liệu và bảng ánh xạ chuẩn hóa thể loại văn học."""

DATA_SOURCES: dict[str, dict[str, str]] = {
    "webnovel_vn": {
        "name": "Webnovel.vn",
        "description": "Kho truyện chữ & sách hay tiếng Việt (247 đầu truyện/sách có gắn thẻ thể loại)",
        "url": "https://webnovel.vn/",
        "sitemap": "https://webnovel.vn/sitemap/stories/sitemap_story_0.xml.gz",
    },
    "metruyenchu": {
        "name": "MeTruyenChu (metruyenchu.com.vn)",
        "description": "Kho truyện chữ đa thể loại tiếng Việt có gán nhãn đa nhãn và phần giới thiệu",
        "url": "https://metruyenchu.com.vn/",
    },
    "webtruyen": {
        "name": "WebTruyen (webtruyen.vn -> wetruyen.com)",
        "description": "Kho truyện chữ phân loại theo đầy đủ chuyên mục thể loại",
        "url": "https://wetruyen.com/",
    },
    "wikipedia_vi": {
        "name": "Wikipedia Tiếng Việt",
        "description": "Tóm tắt tác phẩm văn học, tiểu thuyết lịch sử và sách xuất bản qua MediaWiki API",
        "url": "https://vi.wikipedia.org/w/api.php",
    },
    "uvb_dataset": {
        "name": "UVB Vietnamese Books Dataset",
        "description": "Bộ dữ liệu sách tiếng Việt (447 sách, 230 mẫu có sẵn nhãn)",
        "url": "https://github.com/uvb-dataset",
    },
    "wikisource": {
        "name": "Wikisource Tiếng Việt",
        "description": "Các đoạn văn bản văn học Việt Nam và tác phẩm kinh điển",
        "url": "https://vi.wikisource.org/w/api.php",
    },
    "sgk_nguvan": {
        "name": "Ngữ liệu SGK Ngữ văn",
        "description": "Đoạn trích tóm tắt tác phẩm văn học trong chương trình Ngữ văn",
        "url": "https://vi.wikisource.org/wiki/Thể_loại:Văn_học_Việt_Nam",
    },
}

# Ánh xạ 42 thẻ thể loại gốc sang 22 nhãn chuẩn
GENRE_MAPPING: dict[str, str] = {
    # 1. Ngôn tình (gồm ngôn tình, nữ phụ, tổng tài)
    "ngôn tình": "ngon_tinh",
    "ngon tinh": "ngon_tinh",
    "nữ phụ": "ngon_tinh",
    "nu phu": "ngon_tinh",
    "tổng tài": "ngon_tinh",
    "tong tai": "ngon_tinh",
    "romance": "ngon_tinh",
    "tình cảm": "ngon_tinh",
    # 2. Đam mỹ (tách riêng)
    "đam mỹ": "dam_my",
    "dam my": "dam_my",
    # 3. Bách hợp (tách riêng)
    "bách hợp": "bach_hop",
    "bach hop": "bach_hop",
    # 4. Tiên hiệp
    "tiên hiệp": "tien_hiep",
    "tien hiep": "tien_hiep",
    "tu tiên": "tien_hiep",
    # 5. Kiếm hiệp
    "kiếm hiệp": "kiem_hiep",
    "kiem hiep": "kiem_hiep",
    # 6. Huyền huyễn / Dị giới
    "huyền huyễn": "huyen_huyen",
    "huyen huyen": "huyen_huyen",
    "dị giới": "huyen_huyen",
    "di gioi": "huyen_huyen",
    "thám hiểm": "huyen_huyen",
    "tham hiem": "huyen_huyen",
    "light novel": "huyen_huyen",
    # 7. Linh dị / Kinh dị (tách riêng)
    "linh dị": "linh_di",
    "linh di": "linh_di",
    "kinh dị": "linh_di",
    "kinh di": "linh_di",
    # 8. Khoa huyễn / Mạt thế / Dị năng (gộp nhóm giả tưởng khoa học - năng lực)
    "khoa huyễn": "khoa_huyen_mat_the",
    "khoa huyen": "khoa_huyen_mat_the",
    "mạt thế": "khoa_huyen_mat_the",
    "mat the": "khoa_huyen_mat_the",
    "dị năng": "khoa_huyen_mat_the",
    "di nang": "khoa_huyen_mat_the",
    # 9. Võng du / Game (tách riêng)
    "võng du": "vong_du",
    "vong du": "vong_du",
    # 10. Trinh thám (tách riêng)
    "trinh thám": "trinh_tham",
    "trinh tham": "trinh_tham",
    # 11. Điền văn
    "điền văn": "dien_van",
    "dien van": "dien_van",
    "thập niên 60 70 80": "dien_van",
    "thập niên": "dien_van",
    "làm ruộng": "dien_van",
    # 12. Lịch sử / Quân sự (gồm lịch sử, quân sự, dã sử)
    "lịch sử": "lich_su_quan_su",
    "lich su": "lich_su_quan_su",
    "quân sự": "lich_su_quan_su",
    "quan su": "lich_su_quan_su",
    "dã sử": "lich_su_quan_su",
    "da su": "lich_su_quan_su",
    "historical": "lich_su_quan_su",
    "classics": "lich_su_quan_su",

    # 13. Cổ đại (gồm cổ đại, đông phương, cổ trang)
    "cổ đại": "co_dai",
    "co dai": "co_dai",
    "cổ trang": "co_dai",
    "đông phương": "co_dai",
    "dong phuong": "co_dai",
    # 14. Hiện đại / Đô thị (gồm đô thị, hiện đại, truyện teen)
    "đô thị": "hien_dai",
    "do thi": "hien_dai",
    "hiện đại": "hien_dai",
    "hien dai": "hien_dai",
    "truyện teen": "hien_dai",
    "truyen teen": "hien_dai",

    # 15. Cung đấu / Gia đấu (tách khỏi cổ đại)
    "cung đấu": "cung_dau_gia_dau",
    "cung dau": "cung_dau_gia_dau",
    "gia đấu": "cung_dau_gia_dau",
    "gia dau": "cung_dau_gia_dau",
    # 16. Xuyên không / Xuyên nhanh
    "xuyên không": "xuyen_khong",
    "xuyen khong": "xuyen_khong",
    "xuyên việt": "xuyen_khong",
    "xuyên nhanh": "xuyen_khong",
    "xuyen nhanh": "xuyen_khong",
    # 17. Trọng sinh
    "trọng sinh": "trong_sinh",
    "trong sinh": "trong_sinh",
    "trùng sinh": "trong_sinh",
    # 18. Hệ thống
    "hệ thống": "he_thong",
    "he thong": "he_thong",
    # 19. Nữ cường
    "nữ cường": "nu_cuong",
    "nu cuong": "nu_cuong",
    # 20. Sủng / Ngọt (tách riêng vì xuất hiện mạnh cả trên Ngôn tình lẫn Đam mỹ/Bách hợp)
    "sủng": "sung",
    "sung": "sung",
    # 21. Ngược (tách riêng vì xuất hiện mạnh cả trên Ngôn tình lẫn Đam mỹ/Bách hợp)
    "ngược": "nguoc",
    "nguoc": "nguoc",
    # 22. Hài hước
    "hài hước": "hai_huoc",
    "hai huoc": "hai_huoc",
}

# Các thẻ không phải thể loại nội dung hoặc quá ít mẫu (bỏ khác, đoản văn, tiểu thuyết, phương tây, quan trường, việt nam)
NON_GENRE_TAGS: set[str] = {
    "full",
    "hot",
    "hoàn thành",
    "đã hoàn",
    "đang ra",
    "mới cập nhật",
    "truyện full",
    "truyện hot",
    "vip",
    "đề cử",
    "sách hay dành cho hội viên",
    "khác",
    "đoản văn",
    "doan van",
    "tiểu thuyết",
    "tieu thuyet",
    "phương tây",
    "phuong tay",
    "quan trường",
    "quan truong",
    "việt nam",
    "viet nam",
}
