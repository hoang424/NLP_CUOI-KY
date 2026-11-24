"""Độ đo đánh giá đa nhãn, lưu bảng thí nghiệm và vẽ biểu đồ phân tích lỗi."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    hamming_loss,
    multilabel_confusion_matrix,
)

from src.config import EXPERIMENTS_CSV, FIGURES_DIR, GENRE_DISPLAY_NAMES, GENRE_LABELS

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str] | None = None,
) -> dict[str, Any]:
    """Tính tập độ đo chuẩn cho phân loại đa nhãn: Accuracy, Macro-F1, Micro-F1, Hamming Loss."""
    names = label_names or GENRE_LABELS
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    macro_f1 = float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    micro_f1 = float(f1_score(y_true_arr, y_pred_arr, average="micro", zero_division=0))
    h_loss = float(hamming_loss(y_true_arr, y_pred_arr))

    per_label = classification_report(
        y_true_arr,
        y_pred_arr,
        target_names=names,
        output_dict=True,
        zero_division=0,
    )
    mcm = multilabel_confusion_matrix(y_true_arr, y_pred_arr)

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "micro_f1": round(micro_f1, 4),
        "hamming_loss": round(h_loss, 4),
        "per_label": per_label,
        "confusion_matrices": mcm,
    }


def save_experiment_result(
    experiment_id: str,
    model_name: str,
    feature_type: str,
    metrics: dict[str, Any],
    notes: str = "",
    output_path: Path = EXPERIMENTS_CSV,
) -> pd.DataFrame:
    """Lưu hoặc cập nhật một dòng kết quả thí nghiệm vào outputs/tables/experiments.csv."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "experiment_id",
        "model_name",
        "feature_type",
        "accuracy",
        "macro_f1",
        "micro_f1",
        "hamming_loss",
        "notes",
    ]
    new_row = {
        "experiment_id": experiment_id,
        "model_name": model_name,
        "feature_type": feature_type,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "micro_f1": metrics["micro_f1"],
        "hamming_loss": metrics["hamming_loss"],
        "notes": notes,
    }

    if output_path.exists() and output_path.stat().st_size > 0:
        df = pd.read_csv(output_path, encoding="utf-8")
        if "experiment_id" in df.columns:
            df = df[df["experiment_id"] != experiment_id]
        if df.empty:
            df = pd.DataFrame([new_row], columns=columns)
        else:
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df = pd.DataFrame([new_row], columns=columns)

    df = df.sort_values("experiment_id").reset_index(drop=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    return df


def plot_model_comparison(
    experiments_df: pd.DataFrame,
    output_path: Path = FIGURES_DIR / "model_comparison.png",
) -> Path | None:
    """Vẽ biểu đồ cột so sánh Macro-F1, Micro-F1 và Subset Accuracy giữa các mô hình."""
    if plt is None or experiments_df.empty:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ax = experiments_df.plot(
        x="model_name",
        y=["accuracy", "macro_f1", "micro_f1"],
        kind="bar",
        figsize=(9, 5),
        ylim=(0, 1.05),
        rot=15,
    )
    ax.set_title("So sánh hiệu năng các mô hình phân loại thể loại truyện")
    ax.set_ylabel("Điểm số")
    ax.set_xlabel("")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def plot_f1_per_label(
    per_label_dict: dict[str, Any],
    label_names: list[str] | None = None,
    model_name: str = "Best ML Model",
    output_path: Path = FIGURES_DIR / "ml_f1_per_label.png",
) -> Path | None:
    """Vẽ biểu đồ cột ngang Precision, Recall và F1-score trên từng thể loại."""
    if plt is None:
        return None

    names = label_names or GENRE_LABELS
    rows = []
    for name in names:
        metrics_row = per_label_dict.get(name, {})
        rows.append(
            {
                "display": GENRE_DISPLAY_NAMES.get(name, name),
                "precision": float(metrics_row.get("precision", 0.0)),
                "recall": float(metrics_row.get("recall", 0.0)),
                "f1": float(metrics_row.get("f1-score", 0.0)),
                "support": int(metrics_row.get("support", 0)),
            }
        )
    df_plot = pd.DataFrame(rows).sort_values("f1", ascending=True).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    y_pos = np.arange(len(df_plot))
    height = 0.26

    fig, ax = plt.subplots(figsize=(10, max(6, 0.42 * len(df_plot) + 1.8)))
    ax.barh(y_pos - height, df_plot["precision"], height=height, label="Precision", color="#4c72b0")
    ax.barh(y_pos, df_plot["recall"], height=height, label="Recall", color="#dd8452")
    bars_f1 = ax.barh(y_pos + height, df_plot["f1"], height=height, label="F1-score", color="#55a868")

    for bar, sup in zip(bars_f1, df_plot["support"]):
        val = bar.get_width()
        ax.text(min(val + 0.015, 1.02), bar.get_y() + bar.get_height() / 2, f"{val:.2f} (n={sup})", va="center", fontsize=8.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_plot["display"])
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Điểm số")
    ax.set_title(f"Precision / Recall / F1 theo từng thể loại — {model_name}")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_multilabel_confusion_matrices(
    mcm: np.ndarray,
    label_names: list[str] | None = None,
    model_name: str = "Best ML Model",
    output_path: Path = FIGURES_DIR / "ml_confusion_matrices.png",
) -> Path | None:
    """Vẽ lưới ma trận nhầm lẫn nhị phân (tự động chia cột/hàng) cho tất cả nhãn thể loại."""
    if plt is None:
        return None

    names = label_names or GENRE_LABELS
    n_labels = len(names)
    n_cols = 3 if n_labels == 9 else 5
    n_rows = int(np.ceil(n_labels / n_cols))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.4 * n_cols, 3.6 * n_rows))
    axes_flat = np.asarray(axes).ravel()

    for idx in range(len(axes_flat)):
        ax = axes_flat[idx]
        if idx >= n_labels:
            ax.axis("off")
            continue
        cm = mcm[idx]
        ax.imshow(cm, cmap="Blues", aspect="auto")
        thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
        for r in range(2):
            for c in range(2):
                color = "white" if cm[r, c] > thresh else "black"
                ax.text(c, r, f"{int(cm[r, c])}", ha="center", va="center", color=color, fontsize=10, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["0", "1"])
        ax.set_yticklabels(["0", "1"])
        ax.set_xlabel("Dự đoán", fontsize=8.5)
        ax.set_ylabel("Thực tế", fontsize=8.5)
        ax.set_title(GENRE_DISPLAY_NAMES.get(names[idx], names[idx]), fontsize=10)

    fig.suptitle(f"Ma trận nhầm lẫn nhị phân cho {n_labels} thể loại — {model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97 if n_rows >= 6 else 0.94])
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_dl_learning_curves(
    histories: dict[str, dict[str, list[float]]],
    output_path: Path = FIGURES_DIR / "dl_learning_curves.png",
) -> Path | None:
    """Vẽ đường học (Train/Val Loss và Val Macro-F1 theo epoch) cho các mô hình Deep Learning."""
    if plt is None or not histories:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]

    for idx, (model_label, hist) in enumerate(histories.items()):
        color = colors[idx % len(colors)]
        epochs = np.arange(1, len(hist["train_loss"]) + 1)

        axes[0].plot(epochs, hist["train_loss"], color=color, linestyle="-", marker="o", markersize=3, label=f"{model_label} (Train)")
        axes[0].plot(epochs, hist["val_loss"], color=color, linestyle="--", marker="x", markersize=4, label=f"{model_label} (Val)")

        axes[1].plot(epochs, hist["val_macro_f1"], color=color, linestyle="-", marker="o", markersize=4, label=f"{model_label} (Macro-F1)")
        best_ep = int(np.argmax(hist["val_macro_f1"])) + 1
        best_val = max(hist["val_macro_f1"])
        axes[1].scatter([best_ep], [best_val], color=color, s=55, zorder=5)
        axes[1].annotate(f"{best_val:.3f} (ep {best_ep})", (best_ep, best_val), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8.5)

    axes[0].set_title("Đường cong hàm mất mát (BCEWithLogitsLoss)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(fontsize=8.5)

    axes[1].set_title("Validation Macro-F1 qua các Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro-F1")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend(fontsize=8.5)

    fig.suptitle("Quá trình huấn luyện các mô hình Deep Learning", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
