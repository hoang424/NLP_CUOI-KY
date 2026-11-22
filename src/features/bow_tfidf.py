"""Trích xuất đặc trưng Bag-of-Words và TF-IDF dùng chung giữa các mô hình."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from src.config import TFIDF_PARAMS, TFIDF_VECTORIZER_PATH


def build_bow_vectorizer(**kwargs) -> CountVectorizer:
    """Khởi tạo CountVectorizer cho thực nghiệm baseline Bag-of-Words."""
    params = {"ngram_range": (1, 2), "max_features": 10000, "min_df": 1}
    params.update(kwargs)
    return CountVectorizer(**params)


def build_tfidf_vectorizer(**kwargs) -> TfidfVectorizer:
    """Khởi tạo TfidfVectorizer theo cấu hình chuẩn trong src/config.py."""
    params = dict(TFIDF_PARAMS)
    params.update(kwargs)
    return TfidfVectorizer(**params)


def fit_transform_tfidf(
    train_texts: Iterable[str],
    val_texts: Iterable[str] | None = None,
    test_texts: Iterable[str] | None = None,
    save_path: Path | None = TFIDF_VECTORIZER_PATH,
    **kwargs,
):
    """Chỉ fit TF-IDF trên tập train để tránh data leakage, sau đó transform val/test."""
    vectorizer = build_tfidf_vectorizer(**kwargs)
    x_train = vectorizer.fit_transform(list(train_texts))
    x_val = vectorizer.transform(list(val_texts)) if val_texts is not None else None
    x_test = vectorizer.transform(list(test_texts)) if test_texts is not None else None

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(vectorizer, save_path)

    return vectorizer, x_train, x_val, x_test


def load_tfidf_vectorizer(path: Path = TFIDF_VECTORIZER_PATH) -> TfidfVectorizer:
    """Nạp TfidfVectorizer đã lưu từ đĩa."""
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy TF-IDF vectorizer tại {path}")
    return joblib.load(path)
