"""Gộp dữ liệu thô + dữ liệu Label Studio, chuẩn hóa nhãn và tạo stories.csv."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.config import (
    ACTIVE_RAW_CSV,
    DATA_LABELED_DIR,
    DATA_RAW_DIR,
    GENRE_LABELS,
    MIN_SUMMARY_WORDS,
    STORIES_CSV,
)
from src.crawl.sources import GENRE_MAPPING, NON_GENRE_TAGS

REQUIRED_RAW_COLUMNS = [
    "id",
    "title",
    "author",
    "summary",
    "genres_raw",
    "url",
    "source",
    "crawl_date",
]


def normalize_genres(
    genres_raw: str | Iterable[str],
    allowed_labels: list[str] | None = None,
) -> list[str]:
    """Chuẩn hóa chuỗi thể loại thô ngăn cách bằng '|' thành danh sách nhãn chuẩn."""
    allowed = set(allowed_labels or GENRE_LABELS)
    if isinstance(genres_raw, str):
        tokens = [part.strip().lower() for part in re.split(r"[|,;/]", genres_raw)]
    else:
        tokens = [str(part).strip().lower() for part in genres_raw]

    normalized: list[str] = []
    for token in tokens:
        if not token or token in NON_GENRE_TAGS:
            continue
        mapped = GENRE_MAPPING.get(token, token.replace(" ", "_"))
        if mapped in allowed and mapped not in normalized:
            normalized.append(mapped)
    return normalized


def normalize_title(title: str) -> str:
    """Chuẩn hóa tên truyện để phát hiện bản ghi trùng lặp giữa các chuyên mục."""
    text = unicodedata.normalize("NFC", str(title or "")).lower().strip()
    return re.sub(r"\s+", " ", text)


def _merge_pipe_values(values: Iterable[str]) -> str:
    """Gộp các thẻ thể loại phân cách bằng '|' không phân biệt hoa thường, giữ thứ tự xuất hiện."""
    merged: list[str] = []
    seen_lower: set[str] = set()
    for val in values:
        for part in re.split(r"[|,;/]", str(val or "")):
            cleaned = part.strip()
            low = cleaned.lower()
            if cleaned and low not in seen_lower:
                seen_lower.add(low)
                merged.append(cleaned)
    return "|".join(merged)


def deduplicate_stories(df: pd.DataFrame) -> pd.DataFrame:
    """Loại truyện trùng tên (title) và trùng văn án (summary), đồng thời gộp các thẻ genres_raw."""
    out = df.copy()
    out["summary"] = out["summary"].fillna("").astype(str).str.strip()
    out["title"] = out["title"].fillna("").astype(str).str.strip()
    if "genres_raw" not in out.columns:
        out["genres_raw"] = ""
    out["genres_raw"] = out["genres_raw"].fillna("").astype(str)

    out = out[out["summary"] != ""].reset_index(drop=True)
    if out.empty:
        return out

    out["_orig_idx"] = range(len(out))
    out["_title_norm"] = out["title"].apply(normalize_title)

    # 1. Gộp các bản ghi trùng tên truyện, giữ văn án dài nhất và gộp toàn bộ thẻ thể loại
    non_empty_title = out[out["_title_norm"] != ""]
    empty_title = out[out["_title_norm"] == ""]
    dedup_title_rows: list[pd.Series] = []
    for _, group in non_empty_title.groupby("_title_norm", sort=False):
        if len(group) == 1:
            dedup_title_rows.append(group.iloc[0].copy())
            continue
        best_pos = max(
            range(len(group)),
            key=lambda pos: (len(str(group.iloc[pos]["summary"])), -int(group.iloc[pos]["_orig_idx"])),
        )
        rep = group.iloc[best_pos].copy()
        rep["genres_raw"] = _merge_pipe_values(group["genres_raw"].tolist())
        rep["_orig_idx"] = int(group["_orig_idx"].min())
        dedup_title_rows.append(rep)

    merged_by_title = pd.DataFrame(dedup_title_rows) if dedup_title_rows else pd.DataFrame(columns=out.columns)
    if not empty_title.empty:
        merged_by_title = pd.concat([merged_by_title, empty_title], ignore_index=True)
    merged_by_title = merged_by_title.sort_values("_orig_idx").reset_index(drop=True)

    # 2. Gộp tiếp các bản ghi trùng chính xác văn án (summary)
    dedup_sum_rows: list[pd.Series] = []
    for _, group in merged_by_title.groupby("summary", sort=False):
        if len(group) == 1:
            dedup_sum_rows.append(group.iloc[0].copy())
            continue
        rep = group.sort_values("_orig_idx").iloc[0].copy()
        rep["genres_raw"] = _merge_pipe_values(group["genres_raw"].tolist())
        dedup_sum_rows.append(rep)

    result = pd.DataFrame(dedup_sum_rows).sort_values("_orig_idx").drop(columns=["_orig_idx", "_title_norm"]).reset_index(drop=True)
    return result


def _load_csv_folder(folder: Path) -> list[pd.DataFrame]:
    """Đọc toàn bộ file CSV hợp lệ trong thư mục nguồn."""
    frames: list[pd.DataFrame] = []
    if not folder.exists():
        return frames
    for csv_path in sorted(folder.glob("*.csv")):
        df = pd.read_csv(csv_path, encoding="utf-8")
        frames.append(df)
    return frames


def merge_raw_and_labeled(
    raw_dir: Path = DATA_RAW_DIR,
    labeled_dir: Path = DATA_LABELED_DIR,
    output_csv: Path = STORIES_CSV,
    min_words: int = MIN_SUMMARY_WORDS,
    input_csvs: Iterable[Path] | None = (ACTIVE_RAW_CSV,),
) -> pd.DataFrame:
    """Gộp dữ liệu nguồn (mặc định webtruyen_raw.csv hoặc data/raw + data/labeled), loại trùng tên/tóm tắt và tạo cột one-hot."""
    if input_csvs is not None:
        csv_paths = [Path(p) for p in input_csvs]
        frames = [pd.read_csv(p, encoding="utf-8") for p in csv_paths if p.exists()]
    else:
        frames = _load_csv_folder(raw_dir) + _load_csv_folder(labeled_dir)
    if not frames:
        raise FileNotFoundError(
            f"Không tìm thấy file CSV nào trong {input_csvs if input_csvs is not None else (raw_dir, labeled_dir)}."
        )

    df = pd.concat(frames, ignore_index=True)
    for col in REQUIRED_RAW_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    # Loại bản ghi trùng tên truyện / trùng văn án (đồng thời gộp nhãn) và bỏ bài quá ngắn
    df = deduplicate_stories(df)
    if "category_crawled" in df.columns:
        df = df.drop(columns=["category_crawled"])
    df["n_words"] = df["summary"].apply(lambda text: len(text.split()))
    df = df[df["n_words"] >= min_words].reset_index(drop=True)

    # Chuẩn hóa nhãn và loại bỏ mẫu không còn nhãn hợp lệ
    parsed_labels = df["genres_raw"].apply(normalize_genres)
    df["genres"] = parsed_labels.apply(lambda items: "|".join(items))
    df["n_labels"] = parsed_labels.apply(len)
    df = df[df["n_labels"] > 0].reset_index(drop=True)

    for label in GENRE_LABELS:
        df[f"label_{label}"] = df["genres"].apply(
            lambda g: int(label in g.split("|"))
        )

    df["id"] = [f"ST_{i + 1:04d}" for i in range(len(df))]

    standard_cols = REQUIRED_RAW_COLUMNS + [
        "genres", "n_labels", "n_words"
    ] + [
        f"label_{label}" for label in GENRE_LABELS
    ]
    extra_cols = [c for c in df.columns if c not in standard_cols]
    df = df[standard_cols + extra_cols]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8")
    return df


if __name__ == "__main__":
    merged_df = merge_raw_and_labeled()
    print(f"Đã gộp và chuẩn hóa {len(merged_df)} truyện -> {STORIES_CSV}")
    for label in GENRE_LABELS:
        print(f"  - label_{label}: {int(merged_df[f'label_{label}'].sum())} mẫu")
