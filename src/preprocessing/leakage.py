"""Loại bỏ từ khóa tên thể loại để tránh rò rỉ nhãn (label leakage)."""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable

from src.config import LEAKAGE_KEYWORDS

MULTISPACE_RE = re.compile(r"\s+")
# Cắt cụm khai báo mở đầu dạng "Thể loại: Ngôn tình, ... Văn án:" thường gặp ở webnovel.vn
HEADER_GENRE_BLOCK_RE = re.compile(
    r"(?<!\w)thể\s*loại\s*[:\-]?\s+.{0,450}?(?:văn\s*án|giới\s*thiệu|tóm\s*tắt|nội\s*dung)\s*[:\-]?",
    flags=re.IGNORECASE | re.UNICODE | re.DOTALL,
)
LEADING_LABEL_PREFIX_RE = re.compile(
    r"(?<!\w)(?:thể\s*loại|văn\s*án)\s*[:\-]?",
    flags=re.IGNORECASE | re.UNICODE,
)


def remove_label_leakage(
    text: str,
    leakage_keywords: Iterable[str] | None = None,
) -> str:
    """Xóa phần khai báo thể loại ở đầu văn án và các từ trùng tên thể loại."""
    if not text:
        return ""

    normalized_text = unicodedata.normalize("NFC", text)
    normalized_text = HEADER_GENRE_BLOCK_RE.sub(" ", normalized_text)
    normalized_text = LEADING_LABEL_PREFIX_RE.sub(" ", normalized_text)

    keywords = list(leakage_keywords) if leakage_keywords is not None else list(LEAKAGE_KEYWORDS)

    # Ưu tiên xóa cụm từ dài trước để tránh để lại mảnh từ
    sorted_keywords = sorted(
        {unicodedata.normalize("NFC", kw.strip().lower()) for kw in keywords if kw.strip()},
        key=len,
        reverse=True,
    )
    if not sorted_keywords:
        return MULTISPACE_RE.sub(" ", normalized_text).strip()

    pattern = re.compile(
        r"(?<!\w)(?:" + "|".join(re.escape(kw) for kw in sorted_keywords) + r")(?!\w)",
        flags=re.IGNORECASE | re.UNICODE,
    )
    cleaned = pattern.sub(" ", normalized_text)
    return MULTISPACE_RE.sub(" ", cleaned).strip()
