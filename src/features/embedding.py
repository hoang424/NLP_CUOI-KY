"""Chuẩn bị từ điển, chuỗi chỉ số và ma trận Word Embedding cho mô hình TextCNN."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from src.config import RANDOM_STATE, TEXTCNN_MAX_LEN

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


def texts_to_id_lists(
    texts: Iterable[str],
    vocab: dict[str, int],
    max_len: int = TEXTCNN_MAX_LEN,
) -> list[list[int]]:
    """Chuyển danh sách văn bản đã tách từ sang danh sách chỉ số cắt tối đa max_len (chưa pad cố định)."""
    unk_idx = vocab.get(UNK_TOKEN, 1)
    return [
        [vocab.get(tok, unk_idx) for tok in str(text).split()[:max_len]]
        for text in texts
    ]


def texts_to_sequences(
    texts: Iterable[str],
    vocab: dict[str, int],
    max_len: int = TEXTCNN_MAX_LEN,
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


class LengthBucketBatchSampler:
    """Gom batch theo bucket độ dài (mega-batch sort + shuffle batch) để giảm padding khi Dynamic Padding."""

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int = 256,
        bucket_mult: int = 16,
        seed: int = RANDOM_STATE,
    ) -> None:
        self.lengths = [int(x) for x in lengths]
        self.batch_size = max(1, int(batch_size))
        self.bucket_mult = max(1, int(bucket_mult))
        self.seed = int(seed)
        self._epoch = 0

    def __len__(self) -> int:
        n = len(self.lengths)
        return (n + self.batch_size - 1) // self.batch_size if n > 0 else 0

    def __iter__(self):
        n = len(self.lengths)
        if n == 0:
            return
        rng = np.random.default_rng(self.seed + self._epoch)
        self._epoch += 1
        indices = rng.permutation(n)
        pool_size = max(self.batch_size, self.batch_size * self.bucket_mult)
        batches: list[list[int]] = []
        for start in range(0, n, pool_size):
            pool = indices[start : start + pool_size].tolist()
            pool.sort(key=lambda idx: self.lengths[idx])
            for b_start in range(0, len(pool), self.batch_size):
                batches.append(pool[b_start : b_start + self.batch_size])
        for b_idx in rng.permutation(len(batches)):
            yield batches[int(b_idx)]


def make_dynamic_dataloader(
    id_lists: Sequence[Sequence[int]],
    labels: np.ndarray | None = None,
    batch_size: int = 256,
    shuffle: bool = False,
    min_len: int = 4,
    pad_idx: int = 0,
    pin_memory: bool = False,
    seed: int = RANDOM_STATE,
    bucket_mult: int = 16,
) -> Any:
    """Tạo DataLoader PyTorch với Dynamic Padding theo độ dài lớn nhất trong từng batch."""
    import torch
    from torch.utils.data import DataLoader, Dataset

    class _SequenceDataset(Dataset):
        def __init__(self, seqs: Sequence[Sequence[int]], targets: np.ndarray | None) -> None:
            self.seqs = seqs
            self.targets = targets

        def __len__(self) -> int:
            return len(self.seqs)

        def __getitem__(self, idx: int):
            if self.targets is None:
                return (self.seqs[idx],)
            return self.seqs[idx], self.targets[idx]

    def _collate(batch: list[tuple[Any, ...]]):
        seqs = [item[0] for item in batch]
        batch_max = max(int(min_len), max((len(s) for s in seqs), default=0))
        padded = np.full((len(seqs), batch_max), pad_idx, dtype=np.int64)
        for row_idx, seq in enumerate(seqs):
            if seq:
                padded[row_idx, : len(seq)] = seq
        x_tensor = torch.from_numpy(padded)
        if labels is None:
            return (x_tensor,)
        y_arr = np.stack([item[1] for item in batch]).astype(np.float32, copy=False)
        return x_tensor, torch.from_numpy(y_arr)

    dataset = _SequenceDataset(id_lists, labels)
    if shuffle:
        sampler = LengthBucketBatchSampler(
            lengths=[len(s) for s in id_lists],
            batch_size=batch_size,
            bucket_mult=bucket_mult,
            seed=seed,
        )
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=_collate,
            pin_memory=pin_memory,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate,
        pin_memory=pin_memory,
    )


def estimate_dynamic_pad_ratio(
    id_lists: Sequence[Sequence[int]],
    batch_size: int = 256,
    min_len: int = 4,
    bucketed: bool = True,
    seed: int = RANDOM_STATE,
    bucket_mult: int = 16,
) -> float:
    """Ước lượng tỷ lệ phần tử <PAD> thực tế khi áp dụng Dynamic Padding theo batch."""
    n = len(id_lists)
    if n == 0:
        return 0.0
    lengths = [len(s) for s in id_lists]
    if bucketed:
        batches = list(
            LengthBucketBatchSampler(
                lengths=lengths,
                batch_size=batch_size,
                bucket_mult=bucket_mult,
                seed=seed,
            )
        )
    else:
        batches = [list(range(i, min(i + batch_size, n))) for i in range(0, n, batch_size)]

    total_slots = 0
    total_tokens = 0
    for batch in batches:
        batch_max = max(int(min_len), max((lengths[i] for i in batch), default=0))
        total_slots += len(batch) * batch_max
        total_tokens += sum(lengths[i] for i in batch)
    return float(total_slots - total_tokens) / float(total_slots) if total_slots > 0 else 0.0


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
