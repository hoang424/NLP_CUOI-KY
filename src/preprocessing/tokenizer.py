"""Tách từ tiếng Việt và loại bỏ stopwords."""

from __future__ import annotations

import re
from pathlib import Path

from src.config import STOPWORDS_PATH
from src.preprocessing.clean import clean_text
from src.preprocessing.leakage import remove_label_leakage

try:
    import ctypes
    import os
    from underthesea import word_tokenize as _underthesea_tokenize

    try:
        import underthesea.models.fast_crf_sequence_tagger as _fast_crf
    except ImportError:  # pragma: no cover
        _fast_crf = None

    if (
        os.name == "nt"
        and _fast_crf is not None
        and not getattr(_fast_crf.FastCRFSequenceTagger.load, "_short_path_patched", False)
    ):
        _orig_crf_load = _fast_crf.FastCRFSequenceTagger.load

        def _win_short_path(path: object) -> str:
            """Chuyển đường dẫn Windows sang dạng 8.3 ASCII cho pycrfsuite C++."""
            p_str = str(path)
            buf = ctypes.create_unicode_buffer(512)
            res = ctypes.windll.kernel32.GetShortPathNameW(p_str, buf, 512)
            return buf.value if res else p_str

        def _patched_crf_load(self, base_path):
            return _orig_crf_load(self, _win_short_path(base_path))

        _patched_crf_load._short_path_patched = True
        _fast_crf.FastCRFSequenceTagger.load = _patched_crf_load
except ImportError:  # pragma: no cover - fallback khi chưa cài underthesea
    _underthesea_tokenize = None

try:
    from pyvi import ViTokenizer as _pyvi_tokenizer
except ImportError:  # pragma: no cover - fallback khi chưa cài pyvi
    _pyvi_tokenizer = None


def load_stopwords(filepath: Path = STOPWORDS_PATH) -> set[str]:
    """Đọc danh sách stopwords tiếng Việt từ file UTF-8."""
    if not filepath.exists():
        return set()
    lines = filepath.read_text(encoding="utf-8").splitlines()
    return {line.strip().lower() for line in lines if line.strip() and not line.startswith("#")}


def tokenize_vietnamese(
    text: str,
    stopwords: set[str] | None = None,
    method: str = "auto",
) -> str:
    """Tách từ tiếng Việt (underthesea/pyvi/fallback) và lọc stopwords."""
    if not text or not text.strip():
        return ""

    segmented: str
    if method in ("auto", "underthesea") and _underthesea_tokenize is not None:
        segmented = _underthesea_tokenize(text, format="text")
    elif method in ("auto", "pyvi") and _pyvi_tokenizer is not None:
        segmented = _pyvi_tokenizer.tokenize(text)
    else:
        segmented = " ".join(re.findall(r"\w+", text, flags=re.UNICODE))

    stop_set = stopwords or set()
    tokens = [
        tok.lower()
        for tok in segmented.split()
        if tok.strip()
        and tok.lower() not in stop_set
        and tok.lower().replace("_", " ") not in stop_set
    ]
    return " ".join(tokens)


def preprocess_text(
    raw_text: object,
    stopwords: set[str] | None = None,
    remove_leakage: bool = True,
) -> tuple[str, str]:
    """Chạy trọn bộ làm sạch + chống rò rỉ + tách từ cho một đoạn văn bản."""
    cleaned = clean_text(raw_text)
    if remove_leakage:
        cleaned = remove_label_leakage(cleaned)
    tokens = tokenize_vietnamese(cleaned, stopwords=stopwords)
    return cleaned, tokens
