"""Các mô hình Machine Learning truyền thống trên đặc trưng TF-IDF/BoW."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from src.config import MODELS_DIR, RANDOM_STATE


def get_linear_svm(
    c: float = 1.0,
    use_calibration: bool = False,
    class_weight: str | None = None,
):
    """Mô hình ML #1: Linear SVM bọc OneVsRestClassifier cho bài toán đa nhãn."""
    base_svm = LinearSVC(
        C=c,
        class_weight=class_weight,
        random_state=RANDOM_STATE,
        dual="auto",
    )
    if use_calibration:
        base_svm = CalibratedClassifierCV(base_svm, cv=3)
    return OneVsRestClassifier(base_svm)


def get_logistic_regression(
    c: float = 1.0,
    class_weight: str | None = None,
):
    """Mô hình ML #2: Logistic Regression bọc OneVsRestClassifier."""
    return OneVsRestClassifier(
        LogisticRegression(
            C=c,
            max_iter=1000,
            class_weight=class_weight,
            random_state=RANDOM_STATE,
        )
    )


def get_naive_bayes(alpha: float = 1.0):
    """Mô hình ML #3: Multinomial Naive Bayes bọc OneVsRestClassifier."""
    return OneVsRestClassifier(MultinomialNB(alpha=alpha))


def get_random_forest(n_estimators: int = 200):
    """Mô hình bổ sung: Random Forest hỗ trợ trực tiếp đầu ra đa nhãn."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def get_all_tfidf_models() -> dict[str, tuple[str, object]]:
    """Trả về 3 mô hình ML truyền thống sử dụng chung đầu vào TF-IDF."""
    return {
        "linear_svm": ("Linear SVM (TF-IDF)", get_linear_svm(c=0.25, class_weight="balanced")),
        "logistic_regression": (
            "Logistic Regression (TF-IDF)",
            get_logistic_regression(c=1.0, class_weight="balanced"),
        ),
        "multinomial_nb": ("Multinomial NB (TF-IDF)", get_naive_bayes(alpha=0.05)),
    }


def predict_probabilities(model: object, features) -> np.ndarray:
    """Trả về ma trận xác suất [0, 1] cho mỗi nhãn (hỗ trợ cả LinearSVC)."""
    if hasattr(model, "predict_proba"):
        probs = np.asarray(model.predict_proba(features), dtype=np.float32)
        if probs.ndim == 2:
            return probs
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(features), dtype=np.float32)
        if scores.ndim == 1:
            scores = scores[:, None]
        return 1.0 / (1.0 + np.exp(-np.clip(scores, -20.0, 20.0)))
    return np.asarray(model.predict(features), dtype=np.float32)


def tune_per_label_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    grid: np.ndarray | None = None,
) -> np.ndarray:
    """Tìm ngưỡng quyết định tối ưu theo từng nhãn trên tập val để tối đa hóa F1."""
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=np.float32)
    candidates = (
        np.round(np.linspace(0.15, 0.85, 15), 2)
        if grid is None
        else np.asarray(grid, dtype=np.float32)
    )

    n_labels = y_true_arr.shape[1]
    best_thresholds = np.full(n_labels, 0.5, dtype=np.float32)
    for col in range(n_labels):
        best_score = float(
            f1_score(
                y_true_arr[:, col],
                (y_prob_arr[:, col] >= 0.5).astype(int),
                zero_division=0,
            )
        )
        for thresh in candidates:
            score = float(
                f1_score(
                    y_true_arr[:, col],
                    (y_prob_arr[:, col] >= float(thresh)).astype(int),
                    zero_division=0,
                )
            )
            if score > best_score:
                best_score = score
                best_thresholds[col] = float(thresh)
    return best_thresholds


def apply_thresholds(
    y_prob: np.ndarray,
    thresholds: float | np.ndarray = 0.5,
    ensure_non_empty: bool = True,
) -> np.ndarray:
    """Chuyển ma trận xác suất sang dự đoán nhị phân theo ngưỡng và bảo đảm có ít nhất 1 nhãn."""
    y_prob_arr = np.asarray(y_prob, dtype=np.float32)
    thresh_arr = np.asarray(thresholds, dtype=np.float32)
    preds = (y_prob_arr >= thresh_arr).astype(int)
    if ensure_non_empty:
        for row_idx in range(len(preds)):
            if preds[row_idx].sum() == 0:
                preds[row_idx, int(np.argmax(y_prob_arr[row_idx]))] = 1
    return preds


def save_model(model: object, name: str, models_dir: Path = MODELS_DIR) -> Path:
    """Lưu mô hình đã huấn luyện xuống thư mục models/."""
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / f"{name}.pkl"
    joblib.dump(model, path)
    return path


def load_model(name: str, models_dir: Path = MODELS_DIR) -> object:
    """Nạp mô hình đã lưu từ thư mục models/."""
    path = models_dir / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy mô hình tại {path}")
    return joblib.load(path)
