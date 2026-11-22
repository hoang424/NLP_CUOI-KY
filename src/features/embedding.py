"""Chuẩn bị từ điển, chuỗi chỉ số và ma trận Word Embedding cho mô hình TextCNN."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np

from src.config import RANDOM_STATE

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def build_vocab(
    texts: Iterable[str],
    min_freq: int = 1,
    max_vocab_size: int = 20000,
) -> dict[str, int]:
    """Xây dựng từ điển ánh xạ token -> chỉ số nguyên từ tập văn bản train."""
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(str(text).split())

    vocab: dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for word, freq in counter.most_common(max_vocab_size):
        if freq < min_freq:
            continue
        if word not in vocab:
            vocab[word] = len(vocab)
    return vocab


def texts_to_sequences(
    texts: Iterable[str],
    vocab: dict[str, int],
    max_len: int = 200,
) -> np.ndarray:
    """Chuyển danh sách văn bản đã tách từ sang mảng chỉ số có padding cố định."""
    pad_idx = vocab.get(PAD_TOKEN, 0)
    unk_idx = vocab.get(UNK_TOKEN, 1)
    sequences = []

    for text in texts:
        ids = [vocab.get(tok, unk_idx) for tok in str(text).split()[:max_len]]
        if len(ids) < max_len:
            ids.extend([pad_idx] * (max_len - len(ids)))
        sequences.append(ids)

    return np.asarray(sequences, dtype=np.int64)


def load_pretrained_embeddings(
    embedding_path: Path,
    vocab: dict[str, int],
    embedding_dim: int = 300,
) -> np.ndarray:
    """Nạp vector fastText / PhoW2V định dạng .vec cho các từ có trong từ điển."""
    rng = np.random.default_rng(RANDOM_STATE)
    matrix = rng.normal(scale=0.05, size=(len(vocab), embedding_dim)).astype(np.float32)
    matrix[vocab.get(PAD_TOKEN, 0)] = 0.0

    if not embedding_path.exists():
        return matrix

    with embedding_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.rstrip().split()
            if len(parts) <= 2:
                continue
            word = parts[0]
            if word in vocab and len(parts[1:]) == embedding_dim:
                matrix[vocab[word]] = np.asarray(parts[1:], dtype=np.float32)
    return matrix


def build_corpus_embeddings(
    texts: Iterable[str],
    vocab: dict[str, int],
    embedding_dim: int = 128,
    window: int = 4,
    min_cooccur: int = 2,
    n_iter: int = 3,
    scale: float = 0.5,
) -> np.ndarray:
    """Khởi tạo ma trận Word Embedding bằng PPMI + TruncatedSVD trực tiếp trên tập train."""
    from scipy import sparse
    from sklearn.decomposition import TruncatedSVD

    vocab_size = len(vocab)
    rng = np.random.default_rng(RANDOM_STATE)
    matrix = rng.normal(scale=0.05, size=(vocab_size, embedding_dim)).astype(np.float32)
    pad_idx = vocab.get(PAD_TOKEN, 0)
    matrix[pad_idx] = 0.0

    rows_list: list[np.ndarray] = []
    cols_list: list[np.ndarray] = []
    weights_list: list[np.ndarray] = []
    for text in texts:
        ids = np.array(
            [vocab[tok] for tok in str(text).split() if tok in vocab and vocab[tok] != pad_idx],
            dtype=np.int32,
        )
        n_tok = len(ids)
        if n_tok < 2:
            continue
        for offset in range(1, min(window + 1, n_tok)):
            rows_list.append(ids[:-offset])
            cols_list.append(ids[offset:])
            weights_list.append(np.full(n_tok - offset, 1.0 / offset, dtype=np.float32))

    if not rows_list:
        return matrix

    rows = np.concatenate(rows_list)
    cols = np.concatenate(cols_list)
    weights = np.concatenate(weights_list)
    cooc = sparse.coo_matrix((weights, (rows, cols)), shape=(vocab_size, vocab_size)).tocsr()
    cooc = cooc + cooc.T
    cooc.sum_duplicates()

    coo = cooc.tocoo()
    keep = coo.data >= float(min_cooccur)
    if not np.any(keep):
        return matrix

    row_sum = np.asarray(cooc.sum(axis=1)).ravel().astype(np.float64)
    total = float(row_sum.sum())
    r_idx = coo.row[keep]
    c_idx = coo.col[keep]
    vals = coo.data[keep].astype(np.float64)
    pmi = np.log2((vals * total) / (row_sum[r_idx] * row_sum[c_idx] + 1e-12))
    pos = pmi > 0.0
    if not np.any(pos):
        return matrix

    ppmi = sparse.coo_matrix(
        (pmi[pos].astype(np.float32), (r_idx[pos], c_idx[pos])),
        shape=(vocab_size, vocab_size),
    ).tocsr()

    n_comp = min(embedding_dim, max(1, vocab_size - 1))
    svd = TruncatedSVD(n_components=n_comp, n_iter=n_iter, random_state=RANDOM_STATE)
    emb = svd.fit_transform(ppmi).astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    valid = norms.ravel() > 1e-8
    matrix[valid, :n_comp] = (emb[valid] / norms[valid]) * float(scale)
    matrix[pad_idx] = 0.0
    return matrix
