"""Công cụ crawl dữ liệu truyện/sách từ webnovel.vn, metruyenchu, webtruyen (wetruyen.com) và Wiki."""

from __future__ import annotations

import argparse
import gzip
import json
import re
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urljoin, urlparse

import lxml.html
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.config import DATA_LABELED_DIR, DATA_RAW_DIR

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover
    BeautifulSoup = None

WEBNOVEL_SITEMAP_URL = "https://webnovel.vn/sitemap/stories/sitemap_story_0.xml.gz"
WEBNOVEL_ALL_URL_TEMPLATE = "https://webnovel.vn/all/?page={page}"
DISALLOWED_PREFIXES = ("/pub/", "/app/", "/blog/")

METRUYENCHU_BASE = "https://metruyenchu.com.vn"
METRUYENCHU_CATEGORIES: list[str] = [
    "lich-su",
    "quan-su",
    "quan-truong",
    "cung-dau",
    "gia-dau",
    "dong-phuong",
    "dien-van",
    "trong-sinh",
    "xuyen-khong",
    "tien-hiep",
    "kiem-hiep",
    "huyen-huyen",
    "he-thong",
    "ngon-tinh",
    "do-thi",
    "hien-dai",
]
METRUYENCHU_SKIP_PATHS = (
    "/the-loai",
    "/danh-sach",
    "/tac-gia",
    "/cdn-cgi",
    "/tim-kiem",
    "/lien-he",
    "/gioi-thieu",
)
METRUYENCHU_SEO_PREFIX_RE = re.compile(
    r"^(?:MeTruyenChu vừa cập nhật bộ truyện mới .*? mới này\.|"
    r"Bộ truyện full .*? đăng tải mỗi ngày\.)",
    flags=re.IGNORECASE | re.DOTALL,
)

# Đích chuyển hướng chính thức của webtruyen.vn và dtruyen.com
WEBTRUYEN_BASE = "https://wetruyen.com"
WEBTRUYEN_CATEGORIES: list[tuple[str, str]] = [
    ("truyen-tien-hiep", "Tiên Hiệp"),
    ("truyen-kiem-hiep", "Kiếm Hiệp"),
    ("truyen-ngon-tinh", "Ngôn Tình"),
    ("truyen-quan-truong", "Quan Trường"),
    ("truyen-xuyen-khong", "Xuyên Không"),
    ("truyen-trinh-tham", "Trinh Thám"),
    ("truyen-tham-hiem", "Thám Hiểm"),
    ("truyen-linh-di", "Linh Dị"),
    ("truyen-nguoc", "Ngược"),
    ("truyen-sung", "Sủng"),
    ("truyen-cung-dau", "Cung Đấu"),
    ("truyen-nu-cuong", "Nữ Cường"),
    ("truyen-dong-phuong", "Đông Phương"),
    ("truyen-dam-my", "Đam Mỹ"),
    ("truyen-bach-hop", "Bách Hợp"),
    ("truyen-hai-huoc", "Hài Hước"),
    ("truyen-co-dai", "Cổ Đại"),
    ("truyen-mat-the", "Mạt Thế"),
    ("truyen-tieu-thuyet", "Tiểu Thuyết"),
    ("truyen-khac", "Khác"),
    ("truyen-vong-du", "Võng Du"),
    ("truyen-khoa-huyen", "Khoa Huyễn"),
    ("truyen-di-nang", "Dị Năng"),
    ("truyen-huyen-huyen", "Huyền Huyễn"),
    ("truyen-trong-sinh", "Trọng Sinh"),
    ("truyen-do-thi", "Đô Thị"),
    ("truyen-di-gioi", "Dị Giới"),
    ("truyen-gia-dau", "Gia Đấu"),
    ("truyen-dien-van", "Điền Văn"),
    ("truyen-nu-phu", "Nữ Phụ"),
    ("truyen-quan-su", "Quân Sự"),
    ("truyen-lich-su", "Lịch Sử"),
    ("truyen-teen", "Truyện Teen"),
    ("truyen-light-novel", "Light Novel"),
    ("truyen-doan-van", "Đoản Văn"),
    ("truyen-viet-nam", "Việt Nam"),
    ("truyen-da-su", "Dã Sử"),
    ("truyen-phuong-tay", "Phương Tây"),
    ("truyen-he-thong", "Hệ Thống"),
    ("truyen-xuyen-nhanh", "Xuyên Nhanh"),
    ("truyen-hien-dai", "Hiện Đại"),
    ("truyen-tong-tai", "Tổng tài"),
]
WEBTRUYEN_SYSTEM_SLUGS: set[str] = {
    "truyen-hot",
    "truyen-full",
    "truyen-moi-cap-nhat",
    "truyen-moi-dang",
}

WIKI_LITERATURE_CATEGORIES: list[tuple[str, str, str, str]] = [
    ("https://vi.wikipedia.org/w/api.php", "wikipedia.org", "Thể loại:Tiểu thuyết lịch sử", "Lịch Sử|Cổ Đại"),
    ("https://vi.wikipedia.org/w/api.php", "wikipedia.org", "Thể loại:Tác phẩm văn học Việt Nam", "Lịch Sử"),
    ("https://vi.wikipedia.org/w/api.php", "wikipedia.org", "Thể loại:Sử thi Việt Nam", "Lịch Sử|Cổ Đại"),
    ("https://vi.wikipedia.org/w/api.php", "wikipedia.org", "Thể loại:Truyện thơ Nôm", "Cổ Đại|Ngôn Tình"),
    ("https://vi.wikipedia.org/w/api.php", "wikipedia.org", "Thể loại:Tiểu thuyết Trung Quốc", "Cổ Đại|Lịch Sử"),
    ("https://vi.wikipedia.org/w/api.php", "wikipedia.org", "Thể loại:Tiểu thuyết Kim Dung", "Tiên Hiệp|Kiếm Hiệp|Cổ Đại"),
    ("https://vi.wikipedia.org/w/api.php", "wikipedia.org", "Thể loại:Tiểu thuyết Việt Nam", "Hiện Đại|Lịch Sử"),
    ("https://vi.wikipedia.org/w/api.php", "wikipedia.org", "Thể loại:Truyện ngắn Việt Nam", "Hiện Đại"),
    ("https://vi.wikipedia.org/w/api.php", "wikipedia.org", "Thể loại:Tiểu thuyết lãng mạn", "Ngôn Tình"),
    ("https://vi.wikipedia.org/w/api.php", "wikipedia.org", "Thể loại:Tiểu thuyết khoa học viễn tưởng", "Huyền Huyễn|Khoa Huyễn"),
    ("https://vi.wikipedia.org/w/api.php", "wikipedia.org", "Thể loại:Sách tự lực", "Phát triển bản thân"),
    ("https://vi.wikipedia.org/w/api.php", "wikipedia.org", "Thể loại:Sách kinh doanh", "Kinh doanh"),
    ("https://vi.wikisource.org/w/api.php", "wikisource.org", "Thể loại:Tiểu thuyết", "Lịch Sử|Cổ Đại"),
    ("https://vi.wikisource.org/w/api.php", "wikisource.org", "Thể loại:Truyện ngắn", "Hiện Đại"),
    ("https://vi.wikisource.org/w/api.php", "wikisource.org", "Thể loại:Truyện cổ tích", "Cổ Đại|Huyền Huyễn"),
    ("https://vi.wikisource.org/w/api.php", "wikisource.org", "Thể loại:Truyện kiếm hiệp", "Tiên Hiệp|Kiếm Hiệp|Cổ Đại"),
]

DEFAULT_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
}

WIKI_HEADERS: dict[str, str] = {
    "User-Agent": "NLP-DoAn-De11-VietnameseLiteratureBot/1.0 (educational NLP coursework)",
    "Accept": "application/json",
}

RAW_ORDERED_COLS: list[str] = [
    "id",
    "title",
    "author",
    "summary",
    "genres_raw",
    "status",
    "chapters",
    "rating",
    "url",
    "source",
    "crawl_date",
]


def build_session(pool_maxsize: int = 8) -> requests.Session:
    """Khởi tạo requests.Session có cơ chế retry và connection pool."""
    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)
    retry = Retry(
        total=3,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=pool_maxsize, pool_maxsize=pool_maxsize)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _clean_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


_BLOCK_TAGS: frozenset[str] = frozenset(
    {"br", "p", "div", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "td", "blockquote"}
)


def _block_text(el: lxml.html.HtmlElement) -> str:
    """Lấy text của phần tử HTML, chèn khoảng trắng tại các thẻ khối (<br>, <p>, <div>...)."""
    for node in el.iter():
        if isinstance(node.tag, str) and node.tag.lower() in _BLOCK_TAGS:
            if node is not el:
                node.tail = " " + (node.tail or "")
            if node.tag.lower() != "br":
                node.text = " " + (node.text or "")
    return _clean_whitespace(el.text_content())


def fetch_page(
    url: str,
    delay: float = 0.0,
    timeout: int = 20,
    session: requests.Session | None = None,
) -> str:
    """Tải nội dung HTML từ URL với khoảng nghỉ lịch sự giữa các request."""
    if delay > 0:
        time.sleep(delay)
    client = session or requests
    response = client.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
    response.raise_for_status()
    response.encoding = response.apparent_encoding or "utf-8"
    return response.text


def export_for_label_studio(records: list[dict[str, Any]], output_path: Path) -> Path:
    """Xuất danh sách văn bản sang JSON chuẩn để import/kiểm nhãn trong Label Studio."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "id": idx + 1,
            "data": {
                "title": item.get("title", ""),
                "text": item.get("summary", ""),
                "genres_raw": item.get("genres_raw", ""),
                "source": item.get("source", ""),
                "url": item.get("url", ""),
            },
        }
        for idx, item in enumerate(records)
    ]
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def _save_raw_records(
    records: list[dict[str, Any]],
    id_prefix: str,
    output_csv: Path,
    output_json: Path | None = None,
    label_studio_json: Path | None = None,
    sort_records: bool = True,
) -> pd.DataFrame:
    """Chuẩn hóa cột và lưu danh sách bản ghi crawl thô ra CSV/JSON."""
    if sort_records:
        records.sort(key=lambda item: (item.get("url", ""), item.get("title", "")))
    for idx, rec in enumerate(records, start=1):
        rec["id"] = f"{id_prefix}_{idx:04d}"

    df = pd.DataFrame(records)
    for col in RAW_ORDERED_COLS:
        if col not in df.columns:
            df[col] = ""
    extra_cols = [c for c in df.columns if c not in RAW_ORDERED_COLS]
    df = df[RAW_ORDERED_COLS + extra_cols]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8")
    if output_json is not None:
        output_json.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    if label_studio_json is not None:
        export_for_label_studio(records, label_studio_json)
    return df


def _crawl_urls_concurrently(
    urls: list[str],
    worker_fn: Callable[[str], dict[str, Any] | None],
    max_workers: int = 6,
    label: str = "crawl",
) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    """Tải và phân tích song song danh sách URL với ThreadPoolExecutor."""
    records: list[dict[str, Any]] = []
    errors: list[tuple[str, str]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(worker_fn, url): url for url in urls}
        for done_idx, future in enumerate(as_completed(future_map), start=1):
            url = future_map[future]
            try:
                record = future.result()
                if record is not None:
                    records.append(record)
            except Exception as exc:
                errors.append((url, str(exc)))
            if done_idx % 50 == 0 or done_idx == len(urls):
                print(
                    f"  [{label}] Đã xử lý {done_idx}/{len(urls)} trang "
                    f"(hợp lệ: {len(records)}, lỗi: {len(errors)})"
                )
    return records, errors


# =====================================================================
# 1. WEBNOVEL.VN
# =====================================================================


def _is_valid_webnovel_url(url: str) -> bool:
    """Kiểm tra URL truyện hợp lệ theo đúng quy tắc robots.txt của webnovel.vn."""
    parsed = urlparse(url.strip())
    if parsed.netloc != "webnovel.vn":
        return False
    path = parsed.path or "/"
    if path in ("/", "/all/", "/sach-hay/", "/truyen-full/", "/truyen-moi-dang/"):
        return False
    if path.startswith(DISALLOWED_PREFIXES) or "/chuong-" in path:
        return False
    return True


def discover_webnovel_story_urls(
    session: requests.Session | None = None,
    max_all_pages: int = 5,
) -> list[str]:
    """Thu thập toàn bộ URL trang truyện từ sitemap chính thức và danh sách /all/."""
    client = session or build_session()
    discovered: set[str] = set()

    resp = client.get(WEBNOVEL_SITEMAP_URL, timeout=20)
    resp.raise_for_status()
    xml_bytes = gzip.decompress(resp.content) if resp.content[:2] == b"\x1f\x8b" else resp.content
    root = ET.fromstring(xml_bytes)
    for elem in root.iter():
        if elem.tag.endswith("loc") and elem.text:
            url = elem.text.strip()
            if _is_valid_webnovel_url(url):
                discovered.add(url)

    for page in range(1, max_all_pages + 1):
        list_url = "https://webnovel.vn/all/" if page == 1 else WEBNOVEL_ALL_URL_TEMPLATE.format(page=page)
        try:
            html = fetch_page(list_url, delay=0.2, session=client)
            doc = lxml.html.fromstring(html)
            for card in doc.find_class("catx-card"):
                for href in card.xpath(".//a/@href"):
                    href = href.strip()
                    if _is_valid_webnovel_url(href):
                        discovered.add(href)
        except Exception:
            continue

    return sorted(discovered)


def _extract_jsonld_book(doc: lxml.html.HtmlElement) -> dict[str, Any]:
    """Đọc dữ liệu cấu trúc Schema.org Book/BreadcrumbList làm phương án dự phòng."""
    info: dict[str, Any] = {}
    for script_text in doc.xpath('//script[@type="application/ld+json"]/text()'):
        try:
            payload = json.loads(script_text)
        except json.JSONDecodeError:
            continue
        graph = payload.get("@graph", [payload] if isinstance(payload, dict) else [])
        for node in graph:
            if not isinstance(node, dict):
                continue
            node_type = node.get("@type")
            if node_type == "Book":
                info["title"] = node.get("name", "")
                author = node.get("author")
                if isinstance(author, dict):
                    info["author"] = author.get("name", "")
                elif isinstance(author, str):
                    info["author"] = author
                info["description"] = node.get("description", "")
                rating = node.get("aggregateRating")
                if isinstance(rating, dict):
                    info["rating"] = rating.get("ratingValue", "")
            elif node_type == "BreadcrumbList":
                crumbs = [
                    item.get("name", "").strip()
                    for item in node.get("itemListElement", [])
                    if isinstance(item, dict) and item.get("position") == 2
                ]
                if crumbs:
                    info["breadcrumb_genre"] = crumbs[0]
    return info


def parse_webnovel_story_html(html: str, url: str) -> dict[str, Any]:
    """Trích xuất đầy đủ metadata và phần tóm tắt từ trang truyện webnovel.vn."""
    doc = lxml.html.fromstring(html)
    ld_info = _extract_jsonld_book(doc)

    title_els = doc.find_class("book-detail__title")
    author_els = doc.find_class("book-detail__author")
    genre_els = doc.find_class("book-detail__genres")
    summary_els = doc.find_class("book-detail__summary")
    status_els = doc.find_class("book-detail__status")
    stat_els = doc.find_class("book-detail__stat-text")

    title = _clean_whitespace(title_els[0].text_content()) if title_els else str(ld_info.get("title", ""))
    author = _clean_whitespace(author_els[0].text_content()) if author_els else str(ld_info.get("author", ""))

    genres: list[str] = []
    if genre_els:
        for link in genre_els[0].xpath(".//a"):
            genre_name = _clean_whitespace(link.text_content())
            if genre_name and genre_name not in genres:
                genres.append(genre_name)
    if not genres and ld_info.get("breadcrumb_genre"):
        genres.append(str(ld_info["breadcrumb_genre"]))

    if summary_els:
        summary = _block_text(summary_els[0])
    else:
        panel_els = doc.xpath('//*[@id="panel-summary"]')
        summary = (
            _block_text(panel_els[0])
            if panel_els
            else _clean_whitespace(str(ld_info.get("description", "")))
        )

    return {
        "title": title,
        "author": author,
        "summary": summary,
        "genres_raw": "|".join(genres),
        "status": _clean_whitespace(status_els[0].text_content()) if status_els else "",
        "chapters": _clean_whitespace(stat_els[0].text_content()) if stat_els else "",
        "rating": str(ld_info.get("rating", "")),
        "url": url,
        "source": "webnovel.vn",
        "crawl_date": date.today().isoformat(),
    }


def crawl_all_webnovel(
    output_csv: Path = DATA_RAW_DIR / "webnovel_raw.csv",
    output_json: Path = DATA_RAW_DIR / "webnovel_raw.json",
    label_studio_json: Path = DATA_LABELED_DIR / "webnovel_label_studio.json",
    max_workers: int = 6,
    delay_per_request: float = 0.15,
) -> pd.DataFrame:
    """Crawl toàn bộ truyện từ sitemap webnovel.vn và lưu ra data/raw + data/labeled."""
    session = build_session(pool_maxsize=max_workers + 2)
    urls = discover_webnovel_story_urls(session=session)
    print(f"[webnovel.vn] Phát hiện {len(urls)} URL truyện hợp lệ.")

    def _worker(target_url: str) -> dict[str, Any]:
        html = fetch_page(target_url, delay=delay_per_request, session=session)
        return parse_webnovel_story_html(html, url=target_url)

    records, errors = _crawl_urls_concurrently(urls, _worker, max_workers=max_workers, label="webnovel.vn")
    df = _save_raw_records(records, "WN", output_csv, output_json, label_studio_json)
    print(f"[webnovel.vn] Hoàn tất: {len(df)} truyện -> {output_csv} (lỗi: {len(errors)})")
    return df


# =====================================================================
# 2. METRUYENCHU.COM.VN
# =====================================================================


def parse_metruyenchu_story_html(
    html: str,
    url: str,
    min_summary_words: int = 15,
) -> dict[str, Any] | None:
    """Trích xuất truyện từ metruyenchu.com.vn, bỏ đoạn SEO mẫu nếu không có văn án thật."""
    doc = lxml.html.fromstring(html)
    h1_els = doc.xpath("//h1")
    author_els = doc.xpath('//a[@itemprop="author"]')
    genre_els = doc.xpath('//li[contains(@class, "genres")]//a')
    scroll_els = doc.find_class("scrolltext")

    title = _clean_whitespace(h1_els[0].text_content()) if h1_els else ""
    author = _clean_whitespace(author_els[0].text_content()) if author_els else ""
    genres = []
    for a_el in genre_els:
        g = _clean_whitespace(a_el.text_content())
        if g and g not in genres:
            genres.append(g)

    raw_summary = _block_text(scroll_els[0]) if scroll_els else ""
    summary = METRUYENCHU_SEO_PREFIX_RE.sub("", raw_summary).strip()
    summary = re.sub(r"^Giới thiệu nội dung\s*:?\s*", "", summary, flags=re.IGNORECASE).strip()

    if not title or not genres or len(summary.split()) < min_summary_words:
        return None

    return {
        "title": title,
        "author": author,
        "summary": summary,
        "genres_raw": "|".join(genres),
        "status": "",
        "chapters": "",
        "rating": "",
        "url": url,
        "source": "metruyenchu.com.vn",
        "crawl_date": date.today().isoformat(),
    }


def crawl_metruyenchu(
    output_csv: Path = DATA_RAW_DIR / "metruyenchu_raw.csv",
    max_pages_per_cat: int = 10,
    max_workers: int = 10,
    delay_per_request: float = 0.05,
) -> pd.DataFrame:
    """Crawl truyện đa nhãn từ các chuyên mục thể loại trên metruyenchu.com.vn."""
    session = build_session(pool_maxsize=max_workers + 2)
    story_urls: set[str] = set()

    for cat_slug in METRUYENCHU_CATEGORIES:
        for page in range(1, max_pages_per_cat + 1):
            cat_url = f"{METRUYENCHU_BASE}/the-loai/{cat_slug}" + (f"?page={page}" if page > 1 else "")
            try:
                html = fetch_page(cat_url, delay=0.05, timeout=15, session=session)
                doc = lxml.html.fromstring(html)
                for href in doc.xpath("//a/@href"):
                    href = href.strip()
                    if (
                        href.startswith("/")
                        and href.count("/") == 1
                        and len(href) > 5
                        and not href.startswith(METRUYENCHU_SKIP_PATHS)
                    ):
                        story_urls.add(f"{METRUYENCHU_BASE}{href}")
            except Exception:
                continue

    urls = sorted(story_urls)
    print(f"[metruyenchu.com.vn] Phát hiện {len(urls)} URL truyện từ {len(METRUYENCHU_CATEGORIES)} thể loại.")

    def _worker(target_url: str) -> dict[str, Any] | None:
        html = fetch_page(target_url, delay=delay_per_request, timeout=15, session=session)
        return parse_metruyenchu_story_html(html, url=target_url)

    records, errors = _crawl_urls_concurrently(urls, _worker, max_workers=max_workers, label="metruyenchu")
    df = _save_raw_records(records, "MT", output_csv)
    print(f"[metruyenchu.com.vn] Hoàn tất: {len(df)} truyện có văn án thật -> {output_csv} (lỗi: {len(errors)})")
    return df


# =====================================================================
# 3. WEBTRUYEN (WEBTRUYEN.VN -> WETRUYEN.COM)
# =====================================================================


def discover_webtruyen_categories(
    session: requests.Session | None = None,
    base_url: str = WEBTRUYEN_BASE,
) -> list[tuple[str, str]]:
    """Tự động khám phá toàn bộ danh mục thể loại trên WebTruyen (fallback về 42 thẻ chuẩn)."""
    client = session or build_session()
    try:
        html = fetch_page(f"{base_url.rstrip('/')}/", timeout=20, session=client)
        doc = lxml.html.fromstring(html)
        discovered: list[tuple[str, str]] = []
        seen_slugs: set[str] = set()

        menu_links = doc.xpath(
            '//ul[contains(@class,"dropdown-menu")]//a[@href] | '
            '//div[contains(@class,"list-cat")]//a[@href]'
        )
        for a_el in menu_links:
            href = urljoin(base_url, str(a_el.get("href", "")).strip())
            match = re.match(r"^https?://(?:www\.)?wetruyen\.com/(truyen-[a-z0-9-]+)/full/?$", href)
            if not match:
                continue
            slug = match.group(1)
            if slug in WEBTRUYEN_SYSTEM_SLUGS or slug in seen_slugs:
                continue

            txt = _clean_whitespace(a_el.text_content())
            name = re.sub(r"\s+Full$", "", txt, flags=re.IGNORECASE).strip()
            if slug != "truyen-teen":
                name = re.sub(r"^Truyện\s+", "", name, flags=re.IGNORECASE).strip()
            if not name:
                name = slug.replace("truyen-", "").replace("-", " ").title()

            seen_slugs.add(slug)
            discovered.append((slug, name))

        if discovered:
            return discovered
    except Exception:
        pass
    return list(WEBTRUYEN_CATEGORIES)


def _extract_webtruyen_max_page(doc: lxml.html.HtmlElement) -> int:
    """Trích xuất số trang lớn nhất từ thanh phân trang của trang chuyên mục."""
    max_page = 1
    pag_links = doc.xpath(
        '//ul[@id="pagination"]//a/@href | '
        '//ul[contains(@class,"pagination")]//a/@href | '
        '//div[contains(@class,"pagination")]//a/@href'
    )
    for href in pag_links:
        match = re.search(r"trang-(\d+)", str(href))
        if match:
            max_page = max(max_page, int(match.group(1)))
    return max_page


def _extract_webtruyen_category_story_urls(
    doc: lxml.html.HtmlElement,
    base_url: str = WEBTRUYEN_BASE,
) -> list[str]:
    """Lấy danh sách URL truyện thuộc khối danh sách chính (bỏ qua sidebar Truyện Hot)."""
    base_host = urlparse(base_url).netloc.lower()
    urls: list[str] = []
    seen: set[str] = set()
    for href in doc.xpath('//div[contains(@class, "col-truyen-main")]//h3//a/@href'):
        full_url = urljoin(base_url, str(href).strip())
        parsed = urlparse(full_url)
        if parsed.netloc.lower() == base_host and re.search(r"\.\d+/?$", parsed.path) and full_url not in seen:
            seen.add(full_url)
            urls.append(full_url)
    return urls


def parse_webtruyen_story_html(
    html: str,
    url: str,
    extra_genres: list[str] | None = None,
    category_crawled: str = "",
    min_summary_words: int = 15,
) -> dict[str, Any] | None:
    """Trích xuất metadata và văn án từ trang truyện WebTruyen (wetruyen.com)."""
    doc = lxml.html.fromstring(html)
    h1_els = doc.xpath("//h1")
    author_els = doc.xpath('//div[contains(@class, "info-holder")]//a[@itemprop="author"]')
    genre_els = doc.xpath('//div[contains(@class, "info-holder")]//a[@itemprop="genre"]')
    desc_els = doc.xpath('//div[@itemprop="description"]') or doc.find_class("desc-text")
    rating_els = doc.xpath('//span[@itemprop="ratingValue"]/text()')

    title = _clean_whitespace(h1_els[0].text_content()) if h1_els else ""
    author = _clean_whitespace(author_els[0].text_content()) if author_els else ""

    status = ""
    info_blocks = doc.xpath('//div[contains(@class, "info-holder")]//div[contains(@class, "info")]')
    if info_blocks:
        info_text = _clean_whitespace(info_blocks[0].text_content())
        m_status = re.search(r"Trạng thái\s*:\s*([A-Za-zÀ-ỹ\s]+?)(?:\s+Đánh giá|$)", info_text, flags=re.IGNORECASE)
        if m_status:
            status = m_status.group(1).strip()

    genres: list[str] = []
    for a_el in genre_els:
        g = _clean_whitespace(a_el.text_content())
        if g and g not in genres:
            genres.append(g)
    for g in extra_genres or []:
        if g and g not in genres:
            genres.append(g)

    summary = _block_text(desc_els[0]) if desc_els else ""
    if not title or not genres or len(summary.split()) < min_summary_words:
        return None

    return {
        "title": title,
        "author": author,
        "summary": summary,
        "genres_raw": "|".join(genres),
        "status": status,
        "chapters": "",
        "rating": str(rating_els[0]).strip() if rating_els else "",
        "url": url,
        "source": "webtruyen.vn",
        "crawl_date": date.today().isoformat(),
        "category_crawled": category_crawled,
    }


def crawl_webtruyen(
    output_csv: Path = DATA_RAW_DIR / "webtruyen_raw.csv",
    categories: list[tuple[str, str]] | None = None,
    max_pages_per_cat: int | None = None,
    max_workers: int = 10,
    delay_per_request: float = 0.05,
    checkpoint_every_cat: bool = True,
    resume: bool = False,
) -> pd.DataFrame:
    """Cào lần lượt từng thẻ thể loại và đầy đủ tất cả các trang trên WebTruyen (chưa lọc trùng giữa các thẻ)."""
    session = build_session(pool_maxsize=max_workers + 2)
    cat_list = list(categories) if categories is not None else discover_webtruyen_categories(session=session)
    print(
        f"[webtruyen.vn] Bắt đầu cào {len(cat_list)} chuyên mục thể loại (max_pages_per_cat={max_pages_per_cat}).",
        flush=True,
    )

    cache_path = output_csv.with_suffix(".story_cache.jsonl")
    records: list[dict[str, Any]] = []
    done_categories: set[str] = set()
    story_cache: dict[str, dict[str, Any] | None] = {}

    if resume:
        if output_csv.exists() and output_csv.stat().st_size > 0:
            prev_df = pd.read_csv(output_csv, encoding="utf-8")
            for rec in prev_df.fillna("").to_dict("records"):
                cat_done = str(rec.get("category_crawled", "")).strip()
                if cat_done:
                    done_categories.add(cat_done)
                rec_url = str(rec.get("url", "")).strip()
                if rec_url and rec_url not in story_cache:
                    cached = dict(rec)
                    cached["category_crawled"] = ""
                    story_cache[rec_url] = cached
                records.append(rec)
        if cache_path.exists():
            with cache_path.open("r", encoding="utf-8") as fp:
                for line in fp:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    u = str(item.get("url", "")).strip()
                    if u:
                        story_cache[u] = item.get("rec")
        if records or story_cache:
            print(
                f"[webtruyen.vn] Resume: đã nạp {len(records)} dòng từ {len(done_categories)} thể loại, "
                f"cache {len(story_cache)} truyện.",
                flush=True,
            )
    elif cache_path.exists():
        cache_path.unlink()

    def _fetch_cat_page(cat_slug: str, page_num: int) -> tuple[int, list[str]]:
        page_url = (
            f"{WEBTRUYEN_BASE}/{cat_slug}/full/"
            if page_num == 1
            else f"{WEBTRUYEN_BASE}/{cat_slug}/full/trang-{page_num}/"
        )
        html = fetch_page(page_url, delay=delay_per_request, timeout=20, session=session)
        doc = lxml.html.fromstring(html)
        return page_num, _extract_webtruyen_category_story_urls(doc, base_url=WEBTRUYEN_BASE)

    def _fetch_story(target_url: str) -> tuple[str, dict[str, Any] | None]:
        html = fetch_page(target_url, delay=delay_per_request, timeout=20, session=session)
        parsed = parse_webtruyen_story_html(html, url=target_url)
        return target_url, parsed

    def _flush_cache_batch(batch: list[tuple[str, dict[str, Any] | None]]) -> None:
        if not batch or not checkpoint_every_cat:
            return
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("a", encoding="utf-8") as fp:
            for u_item, rec_item in batch:
                fp.write(json.dumps({"url": u_item, "rec": rec_item}, ensure_ascii=False) + "\n")

    for cat_idx, (slug, cat_name) in enumerate(cat_list, start=1):
        if cat_name in done_categories:
            print(f"  [{cat_idx}/{len(cat_list)}] Bỏ qua {cat_name} (đã cào trước đó).", flush=True)
            continue

        page1_url = f"{WEBTRUYEN_BASE}/{slug}/full/"
        try:
            html1 = fetch_page(page1_url, delay=delay_per_request, timeout=20, session=session)
            doc1 = lxml.html.fromstring(html1)
            detected_max_page = _extract_webtruyen_max_page(doc1)
            target_pages = detected_max_page if max_pages_per_cat is None else min(detected_max_page, max_pages_per_cat)
            page_results: dict[int, list[str]] = {1: _extract_webtruyen_category_story_urls(doc1, base_url=WEBTRUYEN_BASE)}
        except Exception as exc:
            print(f"  [{cat_idx}/{len(cat_list)}] Lỗi đọc trang 1 của {cat_name}: {exc}", flush=True)
            continue

        if target_pages > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_map = {
                    executor.submit(_fetch_cat_page, slug, page_num): page_num
                    for page_num in range(2, target_pages + 1)
                }
                for future in as_completed(future_map):
                    p_num = future_map[future]
                    try:
                        _, urls_on_page = future.result()
                        page_results[p_num] = urls_on_page
                    except Exception:
                        page_results[p_num] = []

        cat_urls: list[str] = []
        seen_in_cat: set[str] = set()
        for p_num in range(1, target_pages + 1):
            for story_url in page_results.get(p_num, []):
                if story_url not in seen_in_cat:
                    seen_in_cat.add(story_url)
                    cat_urls.append(story_url)

        missing_urls = [u for u in cat_urls if u not in story_cache]
        if missing_urls:
            pending_batch: list[tuple[str, dict[str, Any] | None]] = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_map = {executor.submit(_fetch_story, u): u for u in missing_urls}
                for done_cnt, future in enumerate(as_completed(future_map), start=1):
                    target_u = future_map[future]
                    try:
                        _, parsed_rec = future.result()
                    except Exception:
                        parsed_rec = None
                    story_cache[target_u] = parsed_rec
                    pending_batch.append((target_u, parsed_rec))
                    if len(pending_batch) >= 250:
                        _flush_cache_batch(pending_batch)
                        pending_batch.clear()
                        print(
                            f"    [{cat_name}] Đã tải chi tiết {done_cnt}/{len(missing_urls)} truyện...",
                            flush=True,
                        )
            _flush_cache_batch(pending_batch)

        added_in_cat = 0
        for u in cat_urls:
            base_rec = story_cache.get(u)
            if base_rec is None:
                continue
            rec = dict(base_rec)
            existing_genres = [g.strip() for g in str(rec.get("genres_raw", "")).split("|") if g.strip()]
            if cat_name and cat_name not in existing_genres:
                existing_genres.append(cat_name)
            rec["genres_raw"] = "|".join(existing_genres)
            rec["category_crawled"] = cat_name
            records.append(rec)
            added_in_cat += 1

        print(
            f"  [{cat_idx}/{len(cat_list)}] {cat_name}: {target_pages}/{detected_max_page} trang, "
            f"{len(cat_urls)} URL, mới tải {len(missing_urls)} truyện, thêm {added_in_cat} bản ghi "
            f"(tổng thô: {len(records)})",
            flush=True,
        )

        if checkpoint_every_cat and records:
            _save_raw_records(records, "WT", output_csv, sort_records=False)

    if records:
        df = _save_raw_records(records, "WT", output_csv, sort_records=False)
    else:
        cols = RAW_ORDERED_COLS + ["category_crawled"]
        df = pd.DataFrame(columns=cols)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False, encoding="utf-8")

    if cache_path.exists():
        cache_path.unlink()

    print(f"[webtruyen.vn] Hoàn tất: {len(df)} bản ghi thô -> {output_csv}", flush=True)
    return df


# =====================================================================
# 4. WIKIPEDIA & WIKISOURCE (MEDIAWIKI API)
# =====================================================================


def crawl_wiki_literature(
    output_csv: Path = DATA_RAW_DIR / "wiki_raw.csv",
    label_studio_json: Path = DATA_LABELED_DIR / "wiki_label_studio.json",
    max_pages_per_cat: int = 60,
    min_summary_words: int = 15,
) -> pd.DataFrame:
    """Thu thập tóm tắt tác phẩm văn học từ Wikipedia và Wikisource tiếng Việt qua API."""
    session = build_session(pool_maxsize=4)
    session.headers.update(WIKI_HEADERS)

    merged_by_url: dict[str, dict[str, Any]] = {}

    for api_url, source_name, cat_title, genres_str in WIKI_LITERATURE_CATEGORIES:
        fetched_count = 0
        gcmcontinue: str | None = None

        while fetched_count < max_pages_per_cat:
            batch_limit = min(20, max_pages_per_cat - fetched_count)
            params: dict[str, Any] = {
                "action": "query",
                "generator": "categorymembers",
                "gcmtitle": cat_title,
                "gcmtype": "page",
                "gcmlimit": batch_limit,
                "prop": "extracts|info",
                "exintro": 1,
                "explaintext": 1,
                "exlimit": batch_limit,
                "inprop": "url",
                "format": "json",
            }
            if gcmcontinue:
                params["gcmcontinue"] = gcmcontinue

            try:
                time.sleep(0.25)
                resp = session.get(api_url, params=params, timeout=15)
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                print(f"  [wiki] Bỏ qua {cat_title} ({exc})")
                break

            pages = data.get("query", {}).get("pages", {})
            if not pages:
                break

            for page in pages.values():
                title = _clean_whitespace(str(page.get("title", "")))
                summary = _clean_whitespace(str(page.get("extract", "")))
                page_url = str(page.get("fullurl", "")).strip()
                if not title or not page_url or len(summary.split()) < min_summary_words:
                    continue

                if page_url in merged_by_url:
                    existing_genres = merged_by_url[page_url]["genres_raw"].split("|")
                    for g in genres_str.split("|"):
                        if g and g not in existing_genres:
                            existing_genres.append(g)
                    merged_by_url[page_url]["genres_raw"] = "|".join(existing_genres)
                else:
                    merged_by_url[page_url] = {
                        "title": title,
                        "author": "",
                        "summary": summary,
                        "genres_raw": genres_str,
                        "status": "",
                        "chapters": "",
                        "rating": "",
                        "url": page_url,
                        "source": source_name,
                        "crawl_date": date.today().isoformat(),
                    }

            fetched_count += len(pages)
            gcmcontinue = data.get("continue", {}).get("gcmcontinue")
            if not gcmcontinue:
                break

    records = list(merged_by_url.values())
    df = _save_raw_records(records, "WK", output_csv, label_studio_json=label_studio_json)
    print(f"[Wiki] Hoàn tất: {len(df)} tác phẩm văn học -> {output_csv}")
    return df


# =====================================================================
# HÀM TỔNG HỢP & CLI
# =====================================================================


def parse_story_html(
    html: str,
    url: str,
    source_name: str,
    title_selector: str = "h1",
    summary_selector: str = "div.book-detail__summary, div.story-summary, div.mw-parser-output p",
    genre_selector: str = "div.book-detail__genres a, a.genre-tag",
) -> dict[str, Any]:
    """Hàm tổng quát trích xuất tiêu đề, tóm tắt và thể loại từ HTML trang truyện."""
    if "webnovel.vn" in url:
        return parse_webnovel_story_html(html, url)
    if "metruyenchu.com.vn" in url:
        parsed = parse_metruyenchu_story_html(html, url, min_summary_words=1)
        if parsed is not None:
            return parsed
    if "wetruyen.com" in url or "webtruyen.vn" in url or "dtruyen.com" in url:
        parsed = parse_webtruyen_story_html(html, url, min_summary_words=1)
        if parsed is not None:
            return parsed

    if BeautifulSoup is None:
        doc = lxml.html.fromstring(html)
        h1_els = doc.xpath(f"//{title_selector}")
        title = _clean_whitespace(h1_els[0].text_content()) if h1_els else ""
        return {
            "title": title,
            "author": "",
            "summary": "",
            "genres_raw": "",
            "url": url,
            "source": source_name,
            "crawl_date": date.today().isoformat(),
        }

    soup = BeautifulSoup(html, "lxml")
    title_el = soup.select_one(title_selector)
    summary_els = soup.select(summary_selector)
    genre_els = soup.select(genre_selector)

    return {
        "title": title_el.get_text(" ", strip=True) if title_el else "",
        "author": "",
        "summary": " ".join(el.get_text(" ", strip=True) for el in summary_els).strip(),
        "genres_raw": "|".join(
            el.get_text(" ", strip=True) for el in genre_els if el.get_text(strip=True)
        ),
        "url": url,
        "source": source_name,
        "crawl_date": date.today().isoformat(),
    }


def crawl_story_item(
    url: str,
    source_name: str = "webnovel.vn",
    delay: float = 0.2,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """Crawl một trang truyện và trả về bản ghi đúng schema thô."""
    html = fetch_page(url, delay=delay, session=session)
    return parse_story_html(html, url=url, source_name=source_name)


def crawl_all_sources(
    sources: list[str] | None = None,
    max_pages_per_cat: int | None = None,
    resume: bool = False,
) -> dict[str, pd.DataFrame]:
    """Chạy crawl cho các nguồn được chọn (webnovel, metruyenchu, webtruyen, wiki)."""
    selected = set(sources or ["webnovel", "metruyenchu", "webtruyen", "wiki"])
    outputs: dict[str, pd.DataFrame] = {}
    if "webnovel" in selected:
        outputs["webnovel"] = crawl_all_webnovel()
    if "metruyenchu" in selected:
        outputs["metruyenchu"] = crawl_metruyenchu()
    if "webtruyen" in selected:
        outputs["webtruyen"] = crawl_webtruyen(
            max_pages_per_cat=max_pages_per_cat,
            resume=resume,
        )
    if "wiki" in selected:
        outputs["wiki"] = crawl_wiki_literature()
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crawl dữ liệu văn học tiếng Việt")
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["webnovel", "metruyenchu", "webtruyen", "wiki"],
        choices=["webnovel", "metruyenchu", "webtruyen", "wiki"],
        help="Danh sách nguồn cần crawl",
    )
    parser.add_argument(
        "--max-pages-per-cat",
        type=int,
        default=None,
        help="Giới hạn số trang mỗi thể loại khi crawl WebTruyen (mặc định: tất cả các trang)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Tiếp tục crawl WebTruyen từ file checkpoint hiện có",
    )
    args = parser.parse_args()
    crawl_all_sources(
        args.sources,
        max_pages_per_cat=args.max_pages_per_cat,
        resume=args.resume,
    )
