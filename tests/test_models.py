"""Unit test cho mô hình TextCNN, Word Embedding và danh sách mô hình ML."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.features.embedding import (
    build_corpus_embeddings,
    build_vocab,
    estimate_dynamic_pad_ratio,
    make_dynamic_dataloader,
    texts_to_id_lists,
    texts_to_sequences,
)
from src.models.ml_models import get_all_tfidf_models
from src.models.textcnn import (
    TextCNNClassifier,
    compute_pos_weights,
    load_textcnn_checkpoint,
    predict_torch_probs,
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

    def test_dynamic_dataloader_bucketing_and_order(self) -> None:
        texts = [
            "a b",
            "a b c d e f g h i j",
            "",
            "c d e",
            "a c e g i",
            "b d f h",
            "a",
        ]
        vocab = build_vocab(texts, min_freq=1, max_vocab_size=50)
        id_lists = texts_to_id_lists(texts, vocab, max_len=6)
        self.assertEqual(max(len(seq) for seq in id_lists), 6)

        y = np.eye(len(texts), 3, dtype=np.float32)
        train_loader = make_dynamic_dataloader(
            id_lists,
            labels=y,
            batch_size=3,
            shuffle=True,
            min_len=4,
            seed=42,
            bucket_mult=2,
        )
        batches_seen = list(train_loader)
        self.assertEqual(len(batches_seen), len(train_loader))
        self.assertEqual(sum(x.size(0) for x, _ in batches_seen), len(texts))
        for x_b, y_b in batches_seen:
            self.assertGreaterEqual(x_b.size(1), 4)
            self.assertLessEqual(x_b.size(1), 6)
            self.assertEqual(x_b.size(0), y_b.size(0))

        # Bảo toàn thứ tự mẫu gốc khi shuffle=False và tương thích predict_torch_probs
        eval_loader = make_dynamic_dataloader(
            id_lists,
            labels=None,
            batch_size=3,
            shuffle=False,
            min_len=4,
        )
        eval_batches = list(eval_loader)
        self.assertEqual(len(eval_batches), len(eval_loader))
        first_x = eval_batches[0][0]
        self.assertEqual(first_x[0, :2].tolist(), id_lists[0])
        self.assertTrue((first_x[2] == 0).all())

        model = TextCNNClassifier(
            vocab_size=len(vocab),
            num_classes=3,
            embedding_dim=8,
            num_filters=4,
            kernel_sizes=(2, 3, 4),
        )
        probs = predict_torch_probs(model, eval_loader, device="cpu")
        self.assertEqual(probs.shape, (len(texts), 3))
        self.assertTrue(np.isfinite(probs).all())

        pad_ratio = estimate_dynamic_pad_ratio(id_lists, batch_size=3, min_len=4, bucketed=True)
        self.assertGreaterEqual(pad_ratio, 0.0)
        self.assertLess(pad_ratio, 1.0)


if __name__ == "__main__":
    unittest.main()
