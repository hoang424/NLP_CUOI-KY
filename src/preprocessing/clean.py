"""Các hàm làm sạch văn bản tiếng Việt cơ bản."""

from __future__ import annotations

import html
import re
import unicodedata

HTML_TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
NON_WORD_RE = re.compile(r"[^\w\s]", re.UNICODE)
MULTISPACE_RE = re.compile(r"\s+")
WORD_RE = re.compile(r"\w+", re.UNICODE)

# Tên riêng / thương hiệu viết CamelCase hợp lệ, không tách ranh giới hoa/thường
CAMELCASE_WHITELIST: frozenset[str] = frozenset(
    {
        "wechat",
        "iphone",
        "ipad",
        "ios",
        "icloud",
        "imac",
        "youtube",
        "tiktok",
        "macbook",
        "playstation",
        "javascript",
        "wordpress",
        "whatsapp",
        "mcdonald",
        "mcdonalds",
        "esports",
        "lol",
        "defi",
        "wifi",
        "latex",
        "powerpoint",
        "onepiece",
        "blackpink",
        "starcraft",
        "minecraft",
        "chatgpt",
        "openai",
        "webtruyen",
        "truyenfull",
        "metruyenchu",
        "linkedin",
        "paypal",
        "github",
        "netease",
        "douyin",
        "bilibili",
        "kpop",
        "cpop",
        "jpop",
        "tangthuvien",
        "wikidich",
        "facebook",
        "ebook",
        "email",
    }
)


def _split_case_token(match: re.Match[str]) -> str:
    """Tách một token tại ranh giới thường->Hoa và VIẾT_TẮT->Hoa+thường."""
    tok = match.group(0)
    if len(tok) < 3 or tok.lower() in CAMELCASE_WHITELIST:
        return tok

    out: list[str] = [tok[0]]
    n = len(tok)
    for i in range(1, n):
        prev, ch = tok[i - 1], tok[i]
        nxt = tok[i + 1] if i + 1 < n else ""
        if prev.islower() and ch.isupper():
            out.append(" ")
        elif prev.isupper() and ch.isupper() and nxt.islower():
            out.append(" ")
        out.append(ch)
    return "".join(out)


def split_case_boundaries(text: str) -> str:
    """Chuẩn hóa NFC và tách từ bị dính tại ranh giới hoa/thường (chạy trước lower())."""
    if not text:
        return ""
    normalized = unicodedata.normalize("NFC", text)
    return WORD_RE.sub(_split_case_token, normalized)


def clean_text(text: object) -> str:
    """Chuẩn hóa Unicode NFC, tách ranh giới hoa/thường, xóa HTML/URL/ký tự đặc biệt và đưa về chữ thường."""
    if text is None:
        return ""
    text_str = str(text)
    if text_str.lower() == "nan":
        return ""

    text_str = unicodedata.normalize("NFC", text_str)
    text_str = html.unescape(text_str)
    text_str = URL_RE.sub(" ", text_str)
    text_str = HTML_TAG_RE.sub(" ", text_str)
    # Tách từ dính tại ranh giới hoa/thường (phải chạy TRƯỚC lower())
    text_str = split_case_boundaries(text_str)
    text_str = text_str.lower()
    text_str = text_str.replace("_", " ")
    text_str = NON_WORD_RE.sub(" ", text_str)
    text_str = MULTISPACE_RE.sub(" ", text_str).strip()
    return text_str
