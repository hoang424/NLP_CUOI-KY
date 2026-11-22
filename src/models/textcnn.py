"""Kiến trúc Deep Learning PyTorch: TextCNN đa nhãn trên Word Embedding."""

from __future__ import annotations

import contextlib
import copy
import os
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score

from src.config import DEFAULT_THRESHOLD, MODELS_DIR, RANDOM_STATE
from src.models.ml_models import apply_thresholds, tune_per_label_thresholds

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - chỉ yêu cầu khi chạy nhánh PyTorch
    torch = None
    nn = None
    F = None


def _require_torch() -> None:
    if torch is None or nn is None:
        raise ImportError(
            "Cần cài đặt PyTorch để sử dụng TextCNNClassifier: pip install torch"
        )


if nn is not None:

    class TextCNNClassifier(nn.Module):
        """Mô hình TextCNN đa nhãn (Multi-scale Conv1d + Global Max & Masked Mean Pooling)."""

        def __init__(
            self,
            vocab_size: int,
            num_classes: int,
            embedding_dim: int = 128,
            num_filters: int = 64,
            kernel_sizes: tuple[int, ...] = (2, 3, 4),
            dropout: float = 0.35,
            pretrained_embeddings: np.ndarray | None = None,
            freeze_embeddings: bool = False,
            spatial_dropout: float = 0.15,
        ) -> None:
            super().__init__()
            self.kernel_sizes = tuple(int(k) for k in kernel_sizes)
            self.num_filters = int(num_filters)
            self.dropout_rate = float(dropout)
            self.spatial_dropout_rate = float(spatial_dropout)

            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            if pretrained_embeddings is not None:
                weight = torch.tensor(pretrained_embeddings, dtype=torch.float32)
                self.embedding = nn.Embedding.from_pretrained(
                    weight, freeze=freeze_embeddings, padding_idx=0
                )

            emb_dim = self.embedding.embedding_dim
            self.embedding_dim = emb_dim
            self.spatial_drop = (
                nn.Dropout2d(spatial_dropout) if spatial_dropout > 0 else None
            )
            self.dropout = nn.Dropout(dropout)

            self.convs = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv1d(
                            in_channels=emb_dim,
                            out_channels=num_filters,
                            kernel_size=k,
                            padding=0,
                        ),
                        nn.BatchNorm1d(num_filters),
                        nn.GELU(),
                    )
                    for k in self.kernel_sizes
                ]
            )

            total_filters = len(self.kernel_sizes) * num_filters
            proj_dim = num_filters * 2
            self.classifier = nn.Sequential(
                nn.Linear(total_filters * 2, proj_dim),
                nn.BatchNorm1d(proj_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(proj_dim, num_classes),
            )

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            mask = input_ids != 0
            embedded = self.embedding(input_ids)
            if self.spatial_drop is not None:
                embedded = (
                    self.spatial_drop(embedded.transpose(1, 2).unsqueeze(-1))
                    .squeeze(-1)
                    .transpose(1, 2)
                )
            else:
                embedded = self.dropout(embedded)

            x = embedded.transpose(1, 2)
            pooled_branches: list[torch.Tensor] = []
            seq_len = x.size(-1)

            for k, conv in zip(self.kernel_sizes, self.convs):
                if seq_len < k:
                    pad_amt = k - seq_len
                    x_in = F.pad(x, (0, pad_amt))
                    mask_in = F.pad(mask, (0, pad_amt), value=False)
                else:
                    x_in = x
                    mask_in = mask

                conv_out = conv(x_in)
                win_mask = mask_in[:, : conv_out.size(-1)]
                win_mask_exp = win_mask.unsqueeze(1)

                mask_float = win_mask_exp.float()
                lengths = mask_float.sum(dim=-1).clamp(min=1.0)
                mean_pool = (conv_out * mask_float).sum(dim=-1) / lengths

                mask_val = -1e4 if conv_out.dtype == torch.float16 else -1e9
                masked_for_max = conv_out.masked_fill(~win_mask_exp, mask_val)
                max_pool = masked_for_max.max(dim=-1).values
                all_pad = ~win_mask.any(dim=1, keepdim=True)
                max_pool = torch.where(all_pad, torch.zeros_like(max_pool), max_pool)

                pooled_branches.append(max_pool)
                pooled_branches.append(mean_pool)

            features = torch.cat(pooled_branches, dim=1)
            return self.classifier(self.dropout(features))

else:

    class TextCNNClassifier:  # pragma: no cover
        """Placeholder báo lỗi rõ ràng khi chưa cài PyTorch."""

        def __init__(self, *args, **kwargs) -> None:
            _require_torch()


def compute_pos_weights(
    y_train: np.ndarray,
    max_weight: float = 3.5,
    mode: str = "sqrt",
):
    """Tính vector pos_weight cho BCEWithLogitsLoss để cân bằng lớp đa nhãn."""
    _require_torch()
    y_arr = np.asarray(y_train, dtype=np.float32)
    pos_counts = np.clip(y_arr.sum(axis=0), 1.0, None)
    neg_counts = len(y_arr) - pos_counts
    ratio = neg_counts / pos_counts
    if mode in {"sqrt", "dampened"}:
        ratio = np.sqrt(np.clip(ratio, 1.0, None))
    raw_weights = np.clip(ratio, 1.0, max_weight)
    return torch.tensor(raw_weights, dtype=torch.float32)


def _autocast_ctx(amp_enabled: bool):
    if not amp_enabled:
        return contextlib.nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", enabled=True)
    return torch.cuda.amp.autocast(enabled=True)  # pragma: no cover


def _make_grad_scaler(amp_enabled: bool):
    if not amp_enabled:
        return None
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=True)
    return torch.cuda.amp.GradScaler(enabled=True)  # pragma: no cover


def predict_torch_probs(
    model: Any,
    data_loader: Any,
    device: str = "cpu",
) -> np.ndarray:
    """Dự đoán xác suất sigmoid trên toàn bộ DataLoader."""
    _require_torch()
    model.eval()
    probs_batches = []
    with torch.no_grad():
        for batch in data_loader:
            x_batch = batch[0].to(device)
            logits = model(x_batch)
            probs = torch.sigmoid(logits.float()).cpu().numpy()
            probs_batches.append(probs)
    return np.vstack(probs_batches).astype(np.float32)


def train_torch_multilabel(
    model: Any,
    train_loader: Any,
    val_loader: Any,
    y_val: np.ndarray,
    epochs: int = 10,
    lr: float = 3e-3,
    weight_decay: float = 1e-4,
    pos_weight: Any = None,
    patience: int = 4,
    device: str = "cpu",
    use_scheduler: bool = True,
    scheduler_type: str = "onecycle",
    use_amp: bool = True,
    tune_val_thresholds_each_epoch: bool = False,
) -> tuple[Any, dict[str, list[float]]]:
    """Huấn luyện mô hình PyTorch đa nhãn với Early Stopping, Scheduler và AMP."""
    _require_torch()
    if str(device) == "cpu":
        torch.set_num_threads(max(4, os.cpu_count() or 4))
    torch.manual_seed(RANDOM_STATE)
    model = model.to(device)

    weight_tensor = pos_weight.to(device) if pos_weight is not None else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    amp_enabled = bool(use_amp and str(device).startswith("cuda") and torch.cuda.is_available())
    scaler = _make_grad_scaler(amp_enabled)

    scheduler: Any = None
    step_per_batch = False
    if use_scheduler and len(train_loader) > 0:
        sched_key = scheduler_type.lower()
        if sched_key == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=lr,
                total_steps=max(1, epochs * len(train_loader)),
                pct_start=0.15,
            )
            step_per_batch = True
        elif sched_key == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, epochs), eta_min=lr * 0.05
            )
        elif sched_key == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-5
            )

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_macro_f1": [],
        "val_micro_f1": [],
    }
    best_macro_f1 = -1.0
    best_state = copy.deepcopy(model.state_dict())
    no_improve = 0

    for _ in range(epochs):
        model.train()
        total_loss = 0.0
        total_items = 0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast_ctx(amp_enabled):
                logits = model(x_batch)
                loss = criterion(logits, y_batch)

            if scaler is not None:
                prev_scale = scaler.get_scale()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                scaler.step(optimizer)
                scaler.update()
                step_ok = scaler.get_scale() >= prev_scale
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()
                step_ok = True

            if scheduler is not None and step_per_batch and step_ok:
                scheduler.step()

            total_loss += float(loss.item()) * len(x_batch)
            total_items += len(x_batch)

        train_loss = total_loss / max(1, total_items)

        model.eval()
        val_loss_sum = 0.0
        val_items = 0
        val_probs_list = []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                with _autocast_ctx(amp_enabled):
                    logits = model(x_batch)
                    loss = criterion(logits, y_batch)
                val_loss_sum += float(loss.item()) * len(x_batch)
                val_items += len(x_batch)
                val_probs_list.append(torch.sigmoid(logits.float()).cpu().numpy())

        val_loss = val_loss_sum / max(1, val_items)
        val_probs = np.vstack(val_probs_list)
        if tune_val_thresholds_each_epoch:
            epoch_thresh = tune_per_label_thresholds(y_val, val_probs)
            val_preds = apply_thresholds(
                val_probs, thresholds=epoch_thresh, ensure_non_empty=True
            )
        else:
            val_preds = apply_thresholds(
                val_probs, thresholds=DEFAULT_THRESHOLD, ensure_non_empty=True
            )
        val_macro = float(f1_score(y_val, val_preds, average="macro", zero_division=0))
        val_micro = float(f1_score(y_val, val_preds, average="micro", zero_division=0))

        if scheduler is not None and not step_per_batch:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_macro)
            else:
                scheduler.step()

        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_macro_f1"].append(round(val_macro, 4))
        history["val_micro_f1"].append(round(val_micro, 4))

        if val_macro > best_macro_f1:
            best_macro_f1 = val_macro
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    return model, history


def predict_from_logits(
    logits: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray]:
    """Chuyển logits sang xác suất sigmoid và nhãn nhị phân theo ngưỡng."""
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))
    preds = (probs >= threshold).astype(int)
    return preds, probs


def save_textcnn_checkpoint(
    model: TextCNNClassifier,
    vocab: dict[str, int],
    max_len: int,
    thresholds: list[float] | np.ndarray,
    path: Path = MODELS_DIR / "textcnn.pt",
) -> Path:
    """Lưu trọng số và siêu tham số TextCNN xuống đĩa."""
    _require_torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    thresh_list = (
        thresholds.tolist() if isinstance(thresholds, np.ndarray) else list(thresholds)
    )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "vocab": vocab,
            "max_len": int(max_len),
            "embedding_dim": int(model.embedding_dim),
            "num_filters": int(model.num_filters),
            "kernel_sizes": list(model.kernel_sizes),
            "dropout": float(model.dropout_rate),
            "spatial_dropout": float(model.spatial_dropout_rate),
            "num_classes": int(model.classifier[-1].out_features),
            "thresholds": thresh_list,
        },
        path,
    )
    return path


def load_textcnn_checkpoint(
    path: Path = MODELS_DIR / "textcnn.pt",
    device: str = "cpu",
) -> dict[str, Any]:
    """Nạp mô hình TextCNN đã lưu cùng vocab, max_len và ngưỡng."""
    _require_torch()
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint TextCNN tại {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = TextCNNClassifier(
        vocab_size=len(ckpt["vocab"]),
        num_classes=int(ckpt["num_classes"]),
        embedding_dim=int(ckpt["embedding_dim"]),
        num_filters=int(ckpt.get("num_filters", 64)),
        kernel_sizes=tuple(int(k) for k in ckpt.get("kernel_sizes", (2, 3, 4))),
        dropout=float(ckpt.get("dropout", 0.35)),
        spatial_dropout=float(ckpt.get("spatial_dropout", 0.15)),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return {
        "model": model,
        "vocab": ckpt["vocab"],
        "max_len": int(ckpt["max_len"]),
        "thresholds": ckpt.get("thresholds"),
    }
