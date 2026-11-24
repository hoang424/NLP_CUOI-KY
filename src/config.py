"""Cấu hình tập trung cho toàn bộ pipeline phân loại thể loại văn học tiếng Việt."""

from pathlib import Path

# Thư mục gốc dự án
BASE_DIR: Path = Path(__file__).resolve().parent.parent

# Đường dẫn dữ liệu
DATA_DIR: Path = BASE_DIR / "data"
DATA_RAW_DIR: Path = DATA_DIR / "raw"
WEBTRUYEN_RAW_CSV: Path = DATA_RAW_DIR / "webtruyen_raw.csv"
TRUYENFULL_RAW_CSV: Path = DATA_RAW_DIR / "truyenfull_raw.csv"
METRUYENCHU_RAW_CSV: Path = DATA_RAW_DIR / "metruyenchu_raw.csv"
WEBNOVEL_RAW_CSV: Path = DATA_RAW_DIR / "webnovel_raw.csv"
ACTIVE_RAW_CSV: Path = WEBTRUYEN_RAW_CSV
ACTIVE_RAW_CSVS: tuple[Path, ...] = (
    WEBTRUYEN_RAW_CSV,
    TRUYENFULL_RAW_CSV,
    METRUYENCHU_RAW_CSV,
    WEBNOVEL_RAW_CSV,
)
DATA_LABELED_DIR: Path = DATA_DIR / "labeled"
DATA_PROCESSED_DIR: Path = DATA_DIR / "processed"
STORIES_CSV: Path = DATA_PROCESSED_DIR / "stories.csv"
STORIES_CLEAN_CSV: Path = DATA_PROCESSED_DIR / "stories_clean.csv"
STOPWORDS_PATH: Path = DATA_DIR / "stopwords" / "vietnamese_stopwords.txt"

# Đường dẫn lưu mô hình & đầu ra
MODELS_DIR: Path = BASE_DIR / "models"
TFIDF_VECTORIZER_PATH: Path = MODELS_DIR / "tfidf_vectorizer.pkl"
OUTPUTS_DIR: Path = BASE_DIR / "outputs"
FIGURES_DIR: Path = OUTPUTS_DIR / "figures"
TABLES_DIR: Path = OUTPUTS_DIR / "tables"
EXPERIMENTS_CSV: Path = TABLES_DIR / "experiments.csv"

# Tham số tái lập & chia tập dữ liệu
RANDOM_STATE: int = 42
TRAIN_RATIO: float = 0.70
VAL_RATIO: float = 0.15
TEST_RATIO: float = 0.15
MIN_SUMMARY_WORDS: int = 10
MIN_SAMPLES_PER_LABEL: int = 5
DEFAULT_THRESHOLD: float = 0.5
TEXTCNN_MAX_LEN: int = 750

# Danh sách 35 nhãn thể loại chuẩn (từ 42 thẻ gốc webtruyen.vn, bỏ 4 thẻ nguồn gốc địa lý, giữ 3 cặp gộp hợp lý)
GENRE_LABELS: list[str] = [
    "ngon_tinh",
    "dam_my",
    "bach_hop",
    "tong_tai",
    "nu_phu",
    "nu_cuong",
    "sung",
    "nguoc",
    "hai_huoc",
    "tien_hiep",
    "kiem_hiep",
    "huyen_huyen",
    "di_gioi",
    "di_nang",
    "mat_the",
    "khoa_huyen",
    "linh_di",
    "trinh_tham",
    "tham_hiem",
    "vong_du",
    "he_thong",
    "xuyen_khong",
    "trong_sinh",
    "co_dai",
    "hien_dai",
    "hoc_duong",
    "cung_dau",
    "gia_dau",
    "dien_van",
    "lich_su",
    "quan_su",
    "quan_truong",
    "doan_van",
    "tieu_thuyet",
    "khac",
]

GENRE_DISPLAY_NAMES: dict[str, str] = {
    "ngon_tinh": "Ngôn tình (Nam–Nữ)",
    "dam_my": "Đam mỹ",
    "bach_hop": "Bách hợp",
    "tong_tai": "Tổng tài",
    "nu_phu": "Nữ phụ",
    "nu_cuong": "Nữ cường",
    "sung": "Sủng / Ngọt",
    "nguoc": "Ngược",
    "hai_huoc": "Hài hước",
    "tien_hiep": "Tiên hiệp",
    "kiem_hiep": "Kiếm hiệp",
    "huyen_huyen": "Huyền huyễn",
    "di_gioi": "Dị giới",
    "di_nang": "Dị năng",
    "mat_the": "Mạt thế",
    "khoa_huyen": "Khoa huyễn",
    "linh_di": "Linh dị / Kinh dị",
    "trinh_tham": "Trinh thám",
    "tham_hiem": "Thám hiểm",
    "vong_du": "Võng du / Game",
    "he_thong": "Hệ thống",
    "xuyen_khong": "Xuyên không / Xuyên nhanh",
    "trong_sinh": "Trọng sinh",
    "co_dai": "Cổ đại",
    "hien_dai": "Hiện đại / Đô thị",
    "hoc_duong": "Học đường / Truyện teen",
    "cung_dau": "Cung đấu",
    "gia_dau": "Gia đấu",
    "dien_van": "Điền văn",
    "lich_su": "Lịch sử / Dã sử",
    "quan_su": "Quân sự",
    "quan_truong": "Quan trường",
    "doan_van": "Đoản văn",
    "tieu_thuyet": "Tiểu thuyết",
    "khac": "Khác",
}

# Từ khóa dễ gây rò rỉ nhãn khi xuất hiện trực tiếp trong phần giới thiệu truyện
LEAKAGE_KEYWORDS: list[str] = [
    "tiên hiệp",
    "tien hiep",
    "kiếm hiệp",
    "kiem hiep",
    "huyền huyễn",
    "huyen huyen",
    "ngôn tình",
    "ngon tinh",
    "xuyên không",
    "xuyen khong",
    "xuyên việt",
    "xuyên nhanh",
    "trọng sinh",
    "trong sinh",
    "trùng sinh",
    "điền văn",
    "dien van",
    "làm ruộng",
    "cổ đại",
    "co dai",
    "hiện đại",
    "hien dai",
    "đô thị",
    "do thi",
    "tổng tài",
    "tong tai",
    "học đường",
    "hoc duong",
    "truyện teen",
    "truyen teen",
    "nữ phụ",
    "nu phu",
    "thập niên 60 70 80",
    "thập niên",
    "niên đại",
    "hệ thống",
    "he thong",
    "võng du",
    "vong du",
    "lịch sử",
    "lich su",
    "dã sử",
    "da su",
    "quan trường",
    "quan truong",
    "quân sự",
    "quan su",
    "tâm linh",
    "tam linh",
    "kinh doanh",
    "phát triển bản thân",
    "linh dị",
    "linh di",
    "mạt thế",
    "mat the",
    "dị năng",
    "di nang",
    "dị giới",
    "di gioi",
    "khoa huyễn",
    "khoa huyen",
    "kinh dị",
    "trinh thám",
    "trinh tham",
    "thám hiểm",
    "tham hiem",
    "đam mỹ",
    "dam my",
    "bách hợp",
    "bach hop",
    "cung đấu",
    "cung dau",
    "gia đấu",
    "gia dau",
    "hài hước",
    "hai huoc",
    "nữ cường",
    "nu cuong",
    "đoản văn",
    "doan van",
]

# Tham số bộ trích xuất đặc trưng TF-IDF dùng chung cho các mô hình
TFIDF_PARAMS: dict = {
    "ngram_range": (1, 2),
    "max_features": 10000,
    "min_df": 1,
    "sublinear_tf": True,
}
