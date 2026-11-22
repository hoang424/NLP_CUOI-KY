"""Unit test cho module đánh giá đa nhãn."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.evaluation.metrics import evaluate_model, save_experiment_result


class TestMetrics(unittest.TestCase):
    def test_evaluate_model_perfect_predictions(self) -> None:
        y_true = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
        y_pred = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
        metrics = evaluate_model(y_true, y_pred, label_names=["a", "b", "c"])

        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertEqual(metrics["micro_f1"], 1.0)
        self.assertEqual(metrics["hamming_loss"], 0.0)

    def test_evaluate_model_partial_error(self) -> None:
        y_true = np.array([[1, 0], [0, 1]])
        y_pred = np.array([[1, 1], [0, 1]])
        metrics = evaluate_model(y_true, y_pred, label_names=["a", "b"])

        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["hamming_loss"], 0.25)
        self.assertGreater(metrics["micro_f1"], 0.0)

    def test_save_experiment_result_writes_and_updates_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "experiments.csv"
            metrics = {
                "accuracy": 0.75,
                "macro_f1": 0.80,
                "micro_f1": 0.82,
                "hamming_loss": 0.12,
            }
            save_experiment_result("TN1", "Linear SVM", "TF-IDF", metrics, output_path=csv_path)
            df = save_experiment_result("TN1", "Linear SVM", "TF-IDF", metrics, notes="updated", output_path=csv_path)
            self.assertEqual(len(df), 1)
            self.assertEqual(df.iloc[0]["notes"], "updated")


if __name__ == "__main__":
    unittest.main()
