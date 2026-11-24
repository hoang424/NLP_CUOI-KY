"""Pipeline tổng chạy từ dữ liệu stories.csv đến huấn luyện và đánh giá các mô hình ML & TextCNN."""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import (
    EXPERIMENTS_CSV,
    FIGURES_DIR,
    GENRE_LABELS,
    MODELS_DIR,
    RANDOM_STATE,
    STORIES_CLEAN_CSV,
    STORIES_CSV,
    TEXTCNN_MAX_LEN,
    TFIDF_VECTORIZER_PATH,
    TRAIN_RATIO,
    VAL_RATIO,
)
from src.preprocessing.tokenizer import load_stopwords, preprocess_text


def assign_splits(df: pd.DataFrame, label_cols: list[str]) -> pd.Series:
    """Chia tập train/val/test theo tỷ lệ cố định, ưu tiên giữ cân bằng đa nhãn."""
    n_samples = len(df)
    if n_samples < 5:
        raise ValueError("Cần ít nhất 5 mẫu dữ liệu để chia train/val/test.")

    try:
        from skmultilearn.model_selection import IterativeStratification

        y = df[label_cols].to_numpy(dtype=int)
        indices = np.arange(n_samples).reshape(-1, 1)
        strat_1 = IterativeStratification(
            n_splits=2,
            order=1,
            sample_distribution_per_fold=[1.0 - TRAIN_RATIO, TRAIN_RATIO],
        )
        train_idx, rest_idx = next(strat_1.split(indices, y))

        rest_ratio_val = VAL_RATIO / (1.0 - TRAIN_RATIO)
        strat_2 = IterativeStratification(
            n_splits=2,
            order=1,
            sample_distribution_per_fold=[1.0 - rest_ratio_val, rest_ratio_val],
        )
        val_rel, test_rel = next(strat_2.split(rest_idx.reshape(-1, 1), y[rest_idx]))
        val_idx = rest_idx[val_rel]
        test_idx = rest_idx[test_rel]
    except Exception:
        rng = np.random.default_rng(RANDOM_STATE)
        shuffled = rng.permutation(n_samples)
        n_train = max(1, int(round(n_samples * TRAIN_RATIO)))
        n_val = max(1, int(round(n_samples * VAL_RATIO)))
        if n_train + n_val >= n_samples:
            n_val = max(1, (n_samples - n_train) // 2)
        train_idx = shuffled[:n_train]
        val_idx = shuffled[n_train : n_train + n_val]
        test_idx = shuffled[n_train + n_val :]

    splits = np.empty(n_samples, dtype=object)
    splits[train_idx] = "train"
    splits[val_idx] = "val"
    splits[test_idx] = "test"
    return pd.Series(splits, index=df.index)


def prepare_clean_dataset(
    stories_csv: pd.DataFrame | str | Path = STORIES_CSV,
    output_clean_csv: Path = STORIES_CLEAN_CSV,
    preserve_existing_split: bool = True,
) -> pd.DataFrame:
    """Loại truyện trùng tên (nếu còn), tiền xử lý toàn bộ summary, loại rò rỉ nhãn, tách từ và chia train/val/test."""
    from src.crawl.merge_data import deduplicate_stories, normalize_title

    df = stories_csv.copy() if isinstance(stories_csv, pd.DataFrame) else pd.read_csv(stories_csv, encoding="utf-8")
    if "title" in df.columns and "summary" in df.columns:
        norm_titles = df["title"].fillna("").astype(str).apply(normalize_title)
        non_empty_titles = norm_titles[norm_titles != ""]
        if non_empty_titles.duplicated().any():
            df = deduplicate_stories(df)
            if "n_words" in df.columns:
                df["n_words"] = df["summary"].apply(lambda text: len(str(text).split()))

    stopwords = load_stopwords()

    cleaned_pairs = [preprocess_text(text, stopwords=stopwords) for text in df["summary"]]
    df["summary_clean"] = [pair[0] for pair in cleaned_pairs]
    df["tokens"] = [pair[1] for pair in cleaned_pairs]

    label_cols = [f"label_{label}" for label in GENRE_LABELS]
    if not (preserve_existing_split and "split" in df.columns and set(df["split"]).issuperset({"train", "val", "test"})):
        df["split"] = assign_splits(df, label_cols)

    output_clean_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_clean_csv, index=False, encoding="utf-8")
    return df


def _load_or_prepare_dataset(reprocess: bool = False) -> pd.DataFrame:
    label_cols = [f"label_{label}" for label in GENRE_LABELS]
    required_cols = {"tokens", "split", *label_cols}
    if not reprocess and STORIES_CLEAN_CSV.exists():
        df_clean = pd.read_csv(STORIES_CLEAN_CSV, encoding="utf-8")
        if required_cols.issubset(df_clean.columns) and set(df_clean["split"]).issuperset({"train", "val", "test"}):
            return df_clean

    if STORIES_CLEAN_CSV.exists():
        raw_df = pd.read_csv(STORIES_CSV, encoding="utf-8")
        clean_existing = pd.read_csv(STORIES_CLEAN_CSV, encoding="utf-8")
        if "split" in clean_existing.columns and len(clean_existing) == len(raw_df):
            raw_df["split"] = clean_existing["split"]
        return prepare_clean_dataset(raw_df, preserve_existing_split=True)

    return prepare_clean_dataset(STORIES_CSV, preserve_existing_split=False)


def run_pipeline(reprocess: bool = False, include_dl: bool = True) -> pd.DataFrame:
    """Chạy toàn bộ pipeline: preprocessing -> BoW/TF-IDF -> 4 ML + TextCNN -> evaluation."""
    from src.evaluation.metrics import (
        evaluate_model,
        plot_dl_learning_curves,
        plot_f1_per_label,
        plot_model_comparison,
        save_experiment_result,
    )
    from src.features.bow_tfidf import build_bow_vectorizer, fit_transform_tfidf
    from src.features.embedding import (
        build_corpus_embeddings,
        build_vocab,
        make_dynamic_dataloader,
        texts_to_id_lists,
    )
    from src.models.ml_models import (
        apply_thresholds,
        get_linear_svm,
        get_logistic_regression,
        get_naive_bayes,
        predict_probabilities,
        save_model,
        tune_per_label_thresholds,
    )

    df = _load_or_prepare_dataset(reprocess=reprocess)

    label_cols = [f"label_{label}" for label in GENRE_LABELS]
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    train_tokens = train_df["tokens"].fillna("")
    val_tokens = val_df["tokens"].fillna("")
    test_tokens = test_df["tokens"].fillna("")

    _, x_train_tfidf, x_val_tfidf, x_test_tfidf = fit_transform_tfidf(
        train_tokens,
        val_tokens,
        test_tokens,
        save_path=TFIDF_VECTORIZER_PATH,
    )

    bow_vec = build_bow_vectorizer()
    x_train_bow = bow_vec.fit_transform(train_tokens)
    x_val_bow = bow_vec.transform(val_tokens)
    x_test_bow = bow_vec.transform(test_tokens)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bow_vec, MODELS_DIR / "bow_vectorizer.pkl")

    y_train = train_df[label_cols].to_numpy(dtype=int)
    y_val = val_df[label_cols].to_numpy(dtype=int)
    y_test = test_df[label_cols].to_numpy(dtype=int)

    use_test = len(test_df) > 0
    eval_split_name = "test" if use_test else "val"
    eval_y = y_test if use_test else y_val

    if EXPERIMENTS_CSV.exists() and EXPERIMENTS_CSV.stat().st_size > 0:
        exp_existing = pd.read_csv(EXPERIMENTS_CSV, encoding="utf-8")
        if "experiment_id" in exp_existing.columns:
            valid_exp_ids = [
                "TN1_nb_bow",
                "TN2_multinomial_nb",
                "TN3_linear_svm",
                "TN4_logistic_regression",
                "TN5_textcnn",
            ]
            exp_existing = exp_existing[exp_existing["experiment_id"].isin(valid_exp_ids)]
            exp_existing.to_csv(EXPERIMENTS_CSV, index=False, encoding="utf-8")

    ml_configs = [
        (
            "TN1_nb_bow",
            "nb_bow",
            "Multinomial NB (BoW)",
            "BoW (1-2 gram)",
            get_naive_bayes(alpha=0.25),
            "alpha=0.25",
            x_train_bow,
            x_val_bow,
            x_test_bow if use_test else x_val_bow,
        ),
        (
            "TN2_multinomial_nb",
            "multinomial_nb",
            "Multinomial NB (TF-IDF)",
            "TF-IDF (1-2 gram)",
            get_naive_bayes(alpha=0.05),
            "alpha=0.05",
            x_train_tfidf,
            x_val_tfidf,
            x_test_tfidf if use_test else x_val_tfidf,
        ),
        (
            "TN3_linear_svm",
            "linear_svm",
            "Linear SVM (TF-IDF)",
            "TF-IDF (1-2 gram)",
            get_linear_svm(c=0.25, class_weight="balanced"),
            "C=0.25, cw=balanced",
            x_train_tfidf,
            x_val_tfidf,
            x_test_tfidf if use_test else x_val_tfidf,
        ),
        (
            "TN4_logistic_regression",
            "logistic_regression",
            "Logistic Regression (TF-IDF)",
            "TF-IDF (1-2 gram)",
            get_logistic_regression(c=2.0, class_weight="balanced"),
            "C=2.0, cw=balanced",
            x_train_tfidf,
            x_val_tfidf,
            x_test_tfidf if use_test else x_val_tfidf,
        ),
    ]

    ml_thresholds_map: dict[str, dict[str, float]] = {}
    results_df = pd.DataFrame()
    for exp_id, model_key, display_name, feat_type, model, params_desc, x_tr, x_va, x_ev in ml_configs:
        model.fit(x_tr, y_train)
        save_model(model, model_key, models_dir=MODELS_DIR)

        va_probs = predict_probabilities(model, x_va)
        va_def = evaluate_model(y_val, apply_thresholds(va_probs, 0.5), label_names=GENRE_LABELS)
        opt_thresh = tune_per_label_thresholds(y_val, va_probs)
        va_tun = evaluate_model(y_val, apply_thresholds(va_probs, opt_thresh), label_names=GENRE_LABELS)
        ml_thresholds_map[model_key] = {
            lbl: round(float(opt_thresh[i]), 2) for i, lbl in enumerate(GENRE_LABELS)
        }

        use_tuned = va_tun["macro_f1"] >= va_def["macro_f1"]
        chosen_thresh = opt_thresh if use_tuned else 0.5
        thresh_mode = "tuned_thresholds" if use_tuned else "threshold=0.5"

        ev_probs = predict_probabilities(model, x_ev)
        ev_preds = apply_thresholds(ev_probs, thresholds=chosen_thresh, ensure_non_empty=True)
        metrics = evaluate_model(eval_y, ev_preds, label_names=GENRE_LABELS)
        results_df = save_experiment_result(
            experiment_id=exp_id,
            model_name=display_name,
            feature_type=feat_type,
            metrics=metrics,
            notes=f"{eval_split_name.capitalize()} set ({len(eval_y)} mẫu) | {params_desc} | {thresh_mode}",
            output_path=EXPERIMENTS_CSV,
        )

    (MODELS_DIR / "ml_thresholds.json").write_text(
        json.dumps(ml_thresholds_map, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if include_dl:
        import torch

        from src.models.textcnn import (
            TextCNNClassifier,
            compute_pos_weights,
            predict_torch_probs,
            save_textcnn_checkpoint,
            train_torch_multilabel,
        )

        np.random.seed(RANDOM_STATE)
        torch.manual_seed(RANDOM_STATE)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(RANDOM_STATE)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        max_len = TEXTCNN_MAX_LEN
        emb_dim = 128
        num_filters = 64
        kernel_sizes = (2, 3, 4)

        vocab = build_vocab(train_tokens, min_freq=2, max_vocab_size=20000)
        x_ids_train = texts_to_id_lists(train_tokens, vocab, max_len=max_len)
        x_ids_val = texts_to_id_lists(val_tokens, vocab, max_len=max_len)
        x_ids_eval = texts_to_id_lists(test_tokens if use_test else val_tokens, vocab, max_len=max_len)

        pretrained_matrix = build_corpus_embeddings(
            train_tokens,
            vocab,
            embedding_dim=emb_dim,
            window=4,
            min_cooccur=2,
            n_iter=3,
            scale=0.5,
        )

        pin_mem = device == "cuda"
        seq_train_loader = make_dynamic_dataloader(
            x_ids_train,
            labels=y_train,
            batch_size=256,
            shuffle=True,
            pin_memory=pin_mem,
            seed=RANDOM_STATE,
        )
        seq_val_loader = make_dynamic_dataloader(
            x_ids_val,
            labels=y_val,
            batch_size=256,
            shuffle=False,
            pin_memory=pin_mem,
        )
        seq_eval_loader = make_dynamic_dataloader(
            x_ids_eval,
            labels=eval_y,
            batch_size=256,
            shuffle=False,
            pin_memory=pin_mem,
        )

        pos_weight = compute_pos_weights(y_train, max_weight=3.5, mode="sqrt")
        textcnn_model = TextCNNClassifier(
            vocab_size=len(vocab),
            num_classes=len(GENRE_LABELS),
            embedding_dim=emb_dim,
            num_filters=num_filters,
            kernel_sizes=kernel_sizes,
            dropout=0.35,
            pretrained_embeddings=pretrained_matrix,
            freeze_embeddings=False,
            spatial_dropout=0.15,
        )
        t0 = time.perf_counter()
        textcnn_model, cnn_history = train_torch_multilabel(
            textcnn_model,
            train_loader=seq_train_loader,
            val_loader=seq_val_loader,
            y_val=y_val,
            epochs=10 if len(train_df) > 5000 else 20,
            lr=3e-3,
            weight_decay=1e-4,
            pos_weight=pos_weight,
            patience=4,
            device=device,
            use_scheduler=True,
            scheduler_type="onecycle",
            use_amp=True,
        )
        _ = time.perf_counter() - t0

        thresh_grid = np.round(np.linspace(0.15, 0.85, 71), 2)
        cnn_val_probs = predict_torch_probs(textcnn_model, seq_val_loader, device=device)
        cnn_val_def = evaluate_model(y_val, apply_thresholds(cnn_val_probs, 0.5), label_names=GENRE_LABELS)
        cnn_opt_thresh = tune_per_label_thresholds(y_val, cnn_val_probs, grid=thresh_grid)
        cnn_val_tun = evaluate_model(y_val, apply_thresholds(cnn_val_probs, cnn_opt_thresh), label_names=GENRE_LABELS)

        use_tuned = cnn_val_tun["macro_f1"] >= cnn_val_def["macro_f1"]
        chosen_thresh = cnn_opt_thresh if use_tuned else np.full(len(GENRE_LABELS), 0.5, dtype=np.float32)
        thresh_mode = "tuned_thresholds" if use_tuned else "threshold=0.5"

        save_textcnn_checkpoint(
            textcnn_model,
            vocab=vocab,
            max_len=max_len,
            thresholds=chosen_thresh,
            path=MODELS_DIR / "textcnn.pt",
        )
        (MODELS_DIR / "dl_thresholds.json").write_text(
            json.dumps(
                {"TN5_textcnn": {lbl: round(float(chosen_thresh[i]), 2) for i, lbl in enumerate(GENRE_LABELS)}},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        ev_probs = predict_torch_probs(textcnn_model, seq_eval_loader, device=device)
        ev_preds = apply_thresholds(ev_probs, thresholds=chosen_thresh, ensure_non_empty=True)
        cnn_metrics = evaluate_model(eval_y, ev_preds, label_names=GENRE_LABELS)
        results_df = save_experiment_result(
            experiment_id="TN5_textcnn",
            model_name="TextCNN (Word Embedding)",
            feature_type=f"PPMI-SVD Emb + Multi-scale CNN (max_len={max_len})",
            metrics=cnn_metrics,
            notes=f"{eval_split_name.capitalize()} set ({len(eval_y)} mẫu) | TextCNN (k=2,3,4; f={num_filters}; emb={emb_dim} PPMI-SVD; dynamic_pad) + sqrt pos_weight | {thresh_mode}",
            output_path=EXPERIMENTS_CSV,
        )

        plot_dl_learning_curves(
            {"TextCNN (PPMI-SVD)": cnn_history},
            output_path=FIGURES_DIR / "dl_learning_curves.png",
        )
        plot_f1_per_label(
            cnn_metrics["per_label"],
            label_names=GENRE_LABELS,
            model_name="TextCNN (Word Embedding)",
            output_path=FIGURES_DIR / "dl_f1_per_label.png",
        )

    plot_model_comparison(results_df, output_path=FIGURES_DIR / "all_models_comparison.png")
    print("\n=== KẾT QUẢ SO SÁNH MÔ HÌNH ===")
    print(results_df[["experiment_id", "model_name", "accuracy", "macro_f1", "micro_f1", "hamming_loss"]])
    return results_df


if __name__ == "__main__":
    run_pipeline()
