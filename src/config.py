"""Cấu hình tập trung cho toàn bộ pipeline phân loại thể loại văn học tiếng Việt."""

from pathlib import Path

# Thư mục gốc dự án
BASE_DIR: Path = Path(__file__).resolve().parent.parent

# Đường dẫn dữ liệu
DATA_DIR: Path = BASE_DIR / "data"
DATA_RAW_DIR: Path = DATA_DIR / "raw"
WEBTRUYEN_RAW_CSV: Path = DATA_RAW_DIR / "webtruyen_raw.csv"
ACTIVE_RAW_CSV: Path = WEBTRUYEN_RAW_CSV
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

# Danh sách 22 nhãn thể loại chuẩn
GENRE_LABELS: list[str] = [
    "ngon_tinh",
    "dam_my",
    "bach_hop",
    "tien_hiep",
    "kiem_hiep",
    "huyen_huyen",
    "linh_di",
    "khoa_huyen_mat_the",
    "vong_du",
    "trinh_tham",
    "dien_van",
    "lich_su_quan_su",
    "co_dai",
    "hien_dai",
    "cung_dau_gia_dau",
    "xuyen_khong",
    "trong_sinh",
    "he_thong",
    "nu_cuong",
    "sung",
    "nguoc",
    "hai_huoc",
]

GENRE_DISPLAY_NAMES: dict[str, str] = {
    "ngon_tinh": "Ngôn tình (Nam–Nữ)",
    "dam_my": "Đam mỹ",
    "bach_hop": "Bách hợp",
    "tien_hiep": "Tiên hiệp",
    "kiem_hiep": "Kiếm hiệp",
    "huyen_huyen": "Huyền huyễn / Dị giới",
    "linh_di": "Linh dị / Kinh dị",
    "khoa_huyen_mat_the": "Khoa huyễn / Mạt thế / Dị năng",
    "vong_du": "Võng du / Game",
    "trinh_tham": "Trinh thám",
    "dien_van": "Điền văn",
    "lich_su_quan_su": "Lịch sử / Quân sự",
    "co_dai": "Cổ đại",
    "hien_dai": "Hiện đại / Đô thị",
    "cung_dau_gia_dau": "Cung đấu / Gia đấu",
    "xuyen_khong": "Xuyên không / Xuyên nhanh",
    "trong_sinh": "Trọng sinh",
    "he_thong": "Hệ thống",
    "nu_cuong": "Nữ cường",
    "sung": "Sủng / Ngọt",
    "nguoc": "Ngược",
    "hai_huoc": "Hài hước",
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
    "thập niên 60 70 80",
    "thập niên",
    "niên đại",
    "hệ thống",
    "he thong",
    "võng du",
    "vong du",
    "lịch sử",
    "lich su",
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
]

# Tham số bộ trích xuất đặc trưng TF-IDF dùng chung cho các mô hình
TFIDF_PARAMS: dict = {
    "ngram_range": (1, 2),
    "max_features": 10000,
    "min_df": 1,
    "sublinear_tf": True,
}
