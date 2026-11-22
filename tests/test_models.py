"""Unit test cho mô hình TextCNN, Word Embedding và danh sách mô hình ML."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.features.embedding import build_corpus_embeddings, build_vocab, texts_to_sequences
from src.models.ml_models import get_all_tfidf_models
from src.models.textcnn import (
    TextCNNClassifier,
    compute_pos_weights,
    load_textcnn_checkpoint,
    save_textcnn_checkpoint,
)


class TestModels(unittest.TestCase):
    def test_get_all_tfidf_models_keys(self) -> None:
        models = get_all_tfidf_models()
        self.assertEqual(
            set(models.keys()),
            {"linear_svm", "logistic_regression", "multinomial_nb"},
        )

    def test_textcnn_forward_shapes_and_padding_safety(self) -> None:
        texts = [
            "thiếu_niên tu_luyện tiên_kiếm giữa thiên_địa",
            "nàng xuyên_qua cổ_đại vào vương_phủ",
            "tổng_tài sủng ngọt hiện_đại",
            "",
        ]
        vocab = build_vocab(texts, min_freq=1, max_vocab_size=100)
        emb = build_corpus_embeddings(texts, vocab, embedding_dim=16, min_cooccur=1)
        self.assertEqual(emb.shape, (len(vocab), 16))
        self.assertTrue(np.allclose(emb[0], 0.0))

        model = TextCNNClassifier(
            vocab_size=len(vocab),
            num_classes=5,
            embedding_dim=16,
            num_filters=8,
            kernel_sizes=(2, 3, 4, 5),
            dropout=0.2,
            pretrained_embeddings=emb,
            spatial_dropout=0.1,
        )
        model.eval()

        # Trường hợp chuỗi chuẩn và chuỗi toàn padding
        seqs = texts_to_sequences(texts, vocab, max_len=10)
        logits = model(torch.tensor(seqs, dtype=torch.long))
        self.assertEqual(logits.shape, (4, 5))
        self.assertTrue(torch.isfinite(logits).all())

        # Trường hợp chuỗi ngắn hơn kernel_size lớn nhất
        short_seqs = texts_to_sequences(texts[:2], vocab, max_len=2)
        short_logits = model(torch.tensor(short_seqs, dtype=torch.long))
        self.assertEqual(short_logits.shape, (2, 5))
        self.assertTrue(torch.isfinite(short_logits).all())

    def test_textcnn_checkpoint_roundtrip_and_pos_weights(self) -> None:
        y = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        pw_sqrt = compute_pos_weights(y, max_weight=3.5, mode="sqrt")
        pw_lin = compute_pos_weights(y, max_weight=3.5, mode="linear")
        self.assertEqual(pw_sqrt.shape, (3,))
        self.assertGreater(float(pw_lin[1]), float(pw_sqrt[1]))

        vocab = {"<PAD>": 0, "<UNK>": 1, "tu_luyện": 2, "cổ_đại": 3}
        model = TextCNNClassifier(
            vocab_size=len(vocab),
            num_classes=3,
            embedding_dim=12,
            num_filters=4,
            kernel_sizes=(2, 3),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "textcnn.pt"
            save_textcnn_checkpoint(model, vocab=vocab, max_len=16, thresholds=[0.5, 0.4, 0.45], path=ckpt_path)
            bundle = load_textcnn_checkpoint(ckpt_path, device="cpu")
            self.assertEqual(bundle["max_len"], 16)
            self.assertEqual(bundle["vocab"], vocab)
            self.assertEqual(bundle["thresholds"], [0.5, 0.4, 0.45])


if __name__ == "__main__":
    unittest.main()
