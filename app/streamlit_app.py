"""Ứng dụng Streamlit demo phân loại đa nhãn thể loại truyện tiếng Việt (Giai đoạn 6)."""

from __future__ import annotations

import importlib.util
import io
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _is_running_in_streamlit() -> bool:
    """Kiểm tra script có đang chạy bên trong Streamlit runtime hay không."""
    if "streamlit" not in sys.modules:
        return False

    try:
        from streamlit.runtime import exists as st_runtime_exists

        if st_runtime_exists():
            return True
    except Exception:
        pass

    for module_name in (
        "streamlit.runtime.scriptrunner",
        "streamlit.runtime.scriptrunner_utils.script_run_context",
    ):
        try:
            mod = __import__(module_name, fromlist=["get_script_run_ctx"])
            get_ctx = getattr(mod, "get_script_run_ctx", None)
            if callable(get_ctx) and get_ctx() is not None:
                return True
        except Exception:
            continue

    return False


def _resolve_python_executable() -> str:
    """Chọn Python hiện tại nếu đủ thư viện, ngược lại ưu tiên .venv của dự án."""
    required_modules = ("streamlit", "sklearn", "torch", "underthesea", "wordcloud")
    if all(importlib.util.find_spec(mod) is not None for mod in required_modules):
        return sys.executable

    for candidate in (
        ROOT / ".venv" / "Scripts" / "python.exe",
        ROOT / ".venv" / "bin" / "python",
    ):
        if candidate.is_file():
            return str(candidate)

    return sys.executable


def _patch_python_397_streamlit_compat() -> None:
    """Khắc phục lỗi Python 3.9.7 typing.Protocol (bpo-45121) cho Streamlit và Tornado."""
    if sys.version_info[:3] != (3, 9, 7):
        return

    import inspect
    import typing

    if not getattr(typing, "_bpo_45121_patched", False):
        orig_no_init = getattr(typing, "_no_init", None)
        protocol_cls = getattr(typing, "Protocol", None)
        if orig_no_init is not None and protocol_cls is not None:
            def _no_init_or_replace_init(self: object, *args: object, **kwargs: object) -> None:
                cls = type(self)
                if getattr(cls, "_is_protocol", False):
                    raise TypeError("Protocols cannot be instantiated")
                if cls.__init__ is not _no_init_or_replace_init:
                    return
                for base in cls.__mro__:
                    init = base.__dict__.get("__init__", _no_init_or_replace_init)
                    if init is not _no_init_or_replace_init and init is not orig_no_init:
                        cls.__init__ = init  # type: ignore[misc]
                        break
                else:
                    cls.__init__ = object.__init__  # type: ignore[misc]
                cls.__init__(self, *args, **kwargs)  # type: ignore[misc]

            def _walk_subclasses(base: type) -> list[type]:
                found: list[type] = []
                for sub in base.__subclasses__():
                    found.append(sub)
                    found.extend(_walk_subclasses(sub))
                return found

            typing._no_init = _no_init_or_replace_init  # type: ignore[attr-defined]
            for sub in _walk_subclasses(protocol_cls):
                if sub.__dict__.get("__init__") is orig_no_init:
                    sub.__init__ = _no_init_or_replace_init  # type: ignore[misc]

            orig_init_subclass = protocol_cls.__init_subclass__.__func__

            @classmethod  # type: ignore[misc]
            def _patched_init_subclass(cls: type, *args: object, **kwargs: object) -> None:
                had_own_init = "__init__" in cls.__dict__
                orig_init_subclass(cls, *args, **kwargs)
                if getattr(cls, "_is_protocol", False):
                    if cls.__dict__.get("__init__") is orig_no_init:
                        cls.__init__ = _no_init_or_replace_init  # type: ignore[misc]
                elif not had_own_init and "__init__" in cls.__dict__:
                    delattr(cls, "__init__")

            protocol_cls.__init_subclass__ = _patched_init_subclass  # type: ignore[assignment]
            typing._bpo_45121_patched = True  # type: ignore[attr-defined]

    from streamlit.runtime.state.session_state import SessionStateStatProvider

    if len(inspect.signature(SessionStateStatProvider).parameters) == 0:
        def _stat_provider_init(self: object, _session_mgr: object) -> None:
            setattr(self, "_session_mgr", _session_mgr)

        SessionStateStatProvider.__init__ = _stat_provider_init  # type: ignore[method-assign]


def _launch_streamlit() -> int:
    """Tự động khởi chạy `streamlit run` khi thực thi trực tiếp file Python."""
    python_exe = _resolve_python_executable()
    if Path(python_exe).resolve() != Path(sys.executable).resolve():
        if os.environ.get("WEBNOVEL_STREAMLIT_REEXEC") == "1":
            print(
                "Không thể chuyển sang môi trường `.venv` để khởi chạy Streamlit.\n"
                "Vui lòng chạy trực tiếp bằng: .\\.venv\\Scripts\\python.exe app/streamlit_app.py",
                file=sys.stderr,
            )
            return 1

        env = os.environ.copy()
        env["WEBNOVEL_STREAMLIT_REEXEC"] = "1"
        cmd = [python_exe, str(Path(__file__).resolve()), *sys.argv[1:]]
        try:
            proc = subprocess.run(cmd, cwd=str(ROOT), env=env, check=False)
            return proc.returncode
        except KeyboardInterrupt:
            return 0

    if importlib.util.find_spec("streamlit") is None:
        print(
            "Không tìm thấy thư viện `streamlit` trong môi trường Python hiện tại.\n"
            "Vui lòng kích hoạt `.venv` hoặc chạy: pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 1

    _patch_python_397_streamlit_compat()
    os.chdir(str(ROOT))
    sys.argv = ["streamlit", "run", str(Path(__file__).resolve()), *sys.argv[1:]]

    from streamlit.web import cli as stcli

    try:
        stcli.main(prog_name="streamlit")
        return 0
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 0
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__" and not _is_running_in_streamlit():
    raise SystemExit(_launch_streamlit())

import numpy as np
import pandas as pd
import streamlit as st
from wordcloud import WordCloud

from src.config import (
    DEFAULT_THRESHOLD,
    EXPERIMENTS_CSV,
    FIGURES_DIR,
    GENRE_DISPLAY_NAMES,
    GENRE_LABELS,
    MODELS_DIR,
    RANDOM_STATE,
    STORIES_CLEAN_CSV,
    STORIES_CSV,
    TFIDF_VECTORIZER_PATH,
)
from src.features.bow_tfidf import load_tfidf_vectorizer
from src.features.embedding import texts_to_sequences
from src.models.ml_models import apply_thresholds, load_model, predict_probabilities
from src.models.textcnn import load_textcnn_checkpoint
from src.pipeline import run_pipeline
from src.preprocessing.tokenizer import load_stopwords, preprocess_text

MODEL_OPTIONS: dict[str, str] = {
    "logistic_regression": "Logistic Regression (TF-IDF) — Macro-F1 (0.4897) & Micro-F1 cao nhất (0.6035)",
    "linear_svm": "Linear SVM (TF-IDF) — Macro-F1 (0.4767) & Accuracy (0.0618)",
    "textcnn": "TextCNN (Word Embedding) — Macro-F1 (0.4237) & Micro-F1 (0.5657)",
    "multinomial_nb": "Multinomial Naive Bayes (TF-IDF) — Macro-F1 (0.4031) & Micro-F1 (0.5877)",
}

SAMPLE_SUMMARIES: dict[str, str] = {
    "Tự nhập văn bản mới": "",
    "Mẫu 1: Tiên hiệp / Huyền huyễn": (
        "Thiếu niên nghèo tình cờ nhặt được cổ ngọc ẩn chứa truyền thừa tu luyện trên núi cao, "
        "từng bước đột phá cảnh giới, chiến đấu với ma tộc và vượt qua thiên kiếp để bảo vệ tông môn."
    ),
    "Mẫu 2: Ngôn tình / Cổ đại": (
        "Nàng là đích nữ phủ thừa tướng chịu nhiều uất ức, sau khi gả vào vương phủ đã dùng trí tuệ "
        "vượt qua chốn hậu viện hiểm ác và từng bước chinh phục trái tim vị vương gia lạnh lùng."
    ),
    "Mẫu 3: Lịch sử / Quân sự": (
        "Giữa thời loạn thế cuối triều đại cũ, vị tướng trẻ tập hợp nghĩa quân dựng cờ khởi nghĩa, "
        "vận dụng binh pháp bày trận đẩy lùi quân xâm lược và ổn định triều chính."
    ),
    "Mẫu 4: Xuyên không / Điền văn": (
        "Sau một tai nạn, nàng xuyên về vùng nông thôn nghèo khó thời cổ đại, trở thành trưởng tỷ "
        "của mấy đứa em nhỏ. Dựa vào kiến thức nông nghiệp và buôn bán, nàng khai khẩn ruộng đất, "
        "làm giàu nuôi gia đình và gây dựng cuộc sống bình yên nơi thôn dã."
    ),
}

FIGURE_CAPTIONS: dict[str, str] = {
    "eda_overview.png": "Tổng quan EDA dữ liệu webtruyen.vn (Phân bố 22 thể loại & Độ dài văn án)",
    "ml_model_comparison.png": "So sánh 4 cấu hình ML truyền thống & Naive Baseline (22 thể loại)",
    "all_models_comparison.png": "So sánh toàn bộ 5 thí nghiệm (ML vs TextCNN)",
    "ml_f1_per_label.png": "Precision / Recall / F1 theo 22 thể loại (Logistic Regression)",
    "ml_confusion_matrices.png": "Ma trận nhầm lẫn nhị phân 22 thể loại (Logistic Regression)",
    "dl_learning_curves.png": "Đường cong huấn luyện Deep Learning (TextCNN)",
    "dl_f1_per_label.png": "Precision / Recall / F1 theo 22 thể loại (TextCNN)",
    "error_analysis_slices.png": "Phân tích hiệu năng theo cấu trúc văn án, độ dài và số nhãn",
    "error_confusion_pairs.png": "Ma trận cặp nhãn nhầm lẫn (True Label × False Positive)",
    "top_features_per_genre.png": "Top-8 từ khóa TF-IDF trọng số cao nhất",
}


def _find_vietnamese_font() -> str | None:
    for candidate in [
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\segoeui.ttf"),
        Path(r"C:\Windows\Fonts\tahoma.ttf"),
    ]:
        if candidate.exists():
            return str(candidate)
    return None


@st.cache_resource
def get_stopwords() -> set[str]:
    return load_stopwords()


@st.cache_resource
def get_tfidf_vectorizer():
    if not TFIDF_VECTORIZER_PATH.exists():
        run_pipeline()
    return load_tfidf_vectorizer(TFIDF_VECTORIZER_PATH)


@st.cache_resource
def get_sklearn_model(model_key: str):
    if not (MODELS_DIR / f"{model_key}.pkl").exists():
        run_pipeline()
    return load_model(model_key, models_dir=MODELS_DIR)


@st.cache_resource
def get_torch_textcnn_bundle():
    ckpt_path = MODELS_DIR / "textcnn.pt"
    if not ckpt_path.exists():
        return None
    return load_textcnn_checkpoint(ckpt_path, device="cpu")


@st.cache_data
def load_thresholds_map() -> dict[str, dict[str, float]]:
    merged: dict[str, dict[str, float]] = {}
    for fname in ["ml_thresholds.json", "dl_thresholds.json"]:
        path = MODELS_DIR / fname
        if path.exists():
            merged.update(json.loads(path.read_text(encoding="utf-8")))
    return merged


@st.cache_data
def load_dataset_csv(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8")


@st.cache_data
def build_genre_wordcloud_png(genre_key: str) -> bytes | None:
    df_clean = load_dataset_csv(str(STORIES_CLEAN_CSV))
    col = f"label_{genre_key}"
    if df_clean.empty or col not in df_clean.columns or "tokens" not in df_clean.columns:
        return None
    corpus = " ".join(df_clean[df_clean[col] == 1]["tokens"].fillna("").astype(str).tolist()).strip()
    if not corpus:
        return None

    wc = WordCloud(
        width=900,
        height=420,
        background_color="white",
        font_path=_find_vietnamese_font(),
        max_words=100,
        random_state=RANDOM_STATE,
        collocations=False,
    ).generate(corpus)

    buf = io.BytesIO()
    wc.to_image().save(buf, format="PNG")
    return buf.getvalue()


def get_thresholds_for_model(model_key: str, use_tuned: bool, fixed_threshold: float) -> np.ndarray:
    if not use_tuned:
        return np.full(len(GENRE_LABELS), fixed_threshold, dtype=np.float32)

    thresh_map = load_thresholds_map()
    lookup_key = {
        "textcnn": "TN5_textcnn",
    }.get(model_key, model_key)

    if lookup_key in thresh_map:
        return np.array([float(thresh_map[lookup_key][g]) for g in GENRE_LABELS], dtype=np.float32)
    return np.full(len(GENRE_LABELS), fixed_threshold, dtype=np.float32)


def predict_story_probs(model_key: str, tokens: str) -> np.ndarray:
    if model_key == "textcnn":
        import torch

        bundle = get_torch_textcnn_bundle()
        if bundle is not None:
            seq = texts_to_sequences([tokens], bundle["vocab"], max_len=bundle["max_len"])
            with torch.no_grad():
                logits = bundle["model"](torch.tensor(seq, dtype=torch.long))
                return torch.sigmoid(logits).cpu().numpy()[0]
        raise FileNotFoundError(
            "Chưa tìm thấy mô hình TextCNN (models/textcnn.pt). Vui lòng chạy `python -m src.pipeline` hoặc `notebooks/04_deep_learning.ipynb`."
        )

    vectorizer = get_tfidf_vectorizer()
    features = vectorizer.transform([tokens])
    model = get_sklearn_model(model_key)
    return predict_probabilities(model, features)[0]


def render_prediction_tab() -> None:
    """Tab chính: Nhập tóm tắt truyện và dự đoán toàn bộ thể loại đạt ngưỡng xác suất."""
    st.subheader("Dự đoán thể loại từ phần giới thiệu / tóm tắt truyện")

    col_cfg1, col_cfg2 = st.columns([2.4, 1.4])
    with col_cfg1:
        selected_key = st.selectbox(
            "Chọn mô hình",
            options=list(MODEL_OPTIONS.keys()),
            format_func=lambda k: MODEL_OPTIONS[k],
        )
    with col_cfg2:
        use_tuned = st.checkbox("Dùng ngưỡng tối ưu theo từng nhãn", value=True)
        fixed_threshold = float(DEFAULT_THRESHOLD)
        if not use_tuned:
            fixed_threshold = st.slider("Ngưỡng xác suất cố định", 0.1, 0.9, float(DEFAULT_THRESHOLD), 0.05)

    selected_sample = st.selectbox(
        "Chọn văn bản mẫu (mặc định: Tự nhập văn bản mới)",
        options=list(SAMPLE_SUMMARIES.keys()),
        index=0,
    )
    user_text = st.text_area(
        "Nhập hoặc dán đoạn văn bản giới thiệu truyện:",
        value=SAMPLE_SUMMARIES[selected_sample],
        placeholder="Nhập hoặc dán đoạn giới thiệu / tóm tắt truyện tiếng Việt vào đây...",
        height=160,
    )

    if st.button("Phân loại thể loại", type="primary"):
        if not user_text.strip():
            st.warning("Vui lòng nhập nội dung tóm tắt truyện trước khi dự đoán.")
            return

        cleaned, tokens = preprocess_text(user_text, stopwords=get_stopwords())
        probs = predict_story_probs(selected_key, tokens)
        thresholds = get_thresholds_for_model(selected_key, use_tuned=use_tuned, fixed_threshold=fixed_threshold)
        pred_flags = apply_thresholds(probs[None, :], thresholds=thresholds, ensure_non_empty=True)[0]

        results = sorted(
            [
                {
                    "label": label,
                    "display_name": GENRE_DISPLAY_NAMES.get(label, label),
                    "probability": float(probs[idx]),
                    "threshold": float(thresholds[idx]),
                    "meets_threshold": bool(probs[idx] >= thresholds[idx]),
                    "predicted": bool(pred_flags[idx]),
                }
                for idx, label in enumerate(GENRE_LABELS)
            ],
            key=lambda item: item["probability"],
            reverse=True,
        )

        matched_results = [r for r in results if r["meets_threshold"]]
        if matched_results:
            predicted_names = [r["display_name"] for r in matched_results]
            st.success(
                f"**Thể loại dự đoán ({len(matched_results)} thể loại đạt ngưỡng):** {', '.join(predicted_names)}"
            )

            st.markdown(f"**Toàn bộ thể loại đạt ngưỡng ({len(matched_results)} thể loại):**")
            for rank, item in enumerate(matched_results, start=1):
                st.write(
                    f"{rank}. **{item['display_name']}** — {item['probability'] * 100:.1f}% "
                    f"(ngưỡng: {item['threshold'] * 100:.1f}%)"
                )
        else:
            fallback = next((r for r in results if r["predicted"]), results[0])
            st.warning(
                f"**Không có thể loại nào đạt ngưỡng áp dụng.** "
                f"Gợi ý thể loại có xác suất cao nhất: **{fallback['display_name']}** — "
                f"{fallback['probability'] * 100:.1f}% (ngưỡng: {fallback['threshold'] * 100:.1f}%)"
            )

        prob_df = pd.DataFrame(
            {
                "Thể loại": [r["display_name"] for r in results],
                "Xác suất": [round(r["probability"], 4) for r in results],
            }
        ).set_index("Thể loại")
        st.bar_chart(prob_df)

        detail_df = pd.DataFrame(
            {
                "Thể loại": [r["display_name"] for r in results],
                "Xác suất (%)": [round(r["probability"] * 100, 1) for r in results],
                "Ngưỡng áp dụng (%)": [round(r["threshold"] * 100, 1) for r in results],
                "Dự đoán": [
                    "Có" if r["meets_threshold"] else ("Có (dự phòng)" if r["predicted"] else "Không")
                    for r in results
                ],
            }
        )
        with st.expander(f"Xem bảng xác suất chi tiết {len(GENRE_LABELS)} nhãn & văn bản sau tiền xử lý"):
            st.dataframe(detail_df, use_container_width=True, hide_index=True)
            st.write(f"**Làm sạch & chống rò rỉ nhãn:** {cleaned}")
            st.write(f"**Tokens:** {tokens}")


def render_eda_tab() -> None:
    """Tab phụ: Hiển thị thống kê EDA, WordCloud theo thể loại và so sánh mô hình."""
    st.subheader("Khám phá dữ liệu (EDA) & So sánh hiệu năng mô hình")

    df = load_dataset_csv(str(STORIES_CSV))
    if not df.empty:
        label_cols = [f"label_{g}" for g in GENRE_LABELS if f"label_{g}" in df.columns]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Tổng số truyện", f"{len(df):,}")
        m2.metric("Số thể loại", len(label_cols))
        m3.metric("Số nhãn TB / truyện", f"{df['n_labels'].mean():.2f}" if "n_labels" in df.columns else "-")
        m4.metric("Độ dài TB (từ)", f"{df['n_words'].mean():.1f}" if "n_words" in df.columns else "-")

        col_eda1, col_eda2 = st.columns(2)
        with col_eda1:
            if label_cols:
                counts = df[label_cols].sum().sort_values(ascending=False)
                counts.index = [GENRE_DISPLAY_NAMES.get(c.replace("label_", ""), c) for c in counts.index]
                st.markdown("**Phân bố số lượng mẫu theo thể loại**")
                st.bar_chart(counts)
        with col_eda2:
            if "source" in df.columns and df["source"].nunique() > 1:
                src_counts = df["source"].value_counts()
                st.markdown("**Phân bố dữ liệu theo nguồn thu thập**")
                st.bar_chart(src_counts)
            elif "n_labels" in df.columns:
                n_lbl_counts = df["n_labels"].value_counts().sort_index()
                n_lbl_counts.index = [f"{int(k)} nhãn" for k in n_lbl_counts.index]
                st.markdown("**Phân bố số nhãn trên mỗi truyện (webtruyen.vn)**")
                st.bar_chart(n_lbl_counts)

    st.markdown("**WordCloud từ vựng đặc trưng theo thể loại**")
    selected_wc_genre = st.selectbox(
        "Chọn thể loại để xem WordCloud",
        options=GENRE_LABELS,
        format_func=lambda g: GENRE_DISPLAY_NAMES.get(g, g),
    )
    wc_png = build_genre_wordcloud_png(selected_wc_genre)
    if wc_png is not None:
        st.image(
            wc_png,
            caption=f"WordCloud thể loại: {GENRE_DISPLAY_NAMES.get(selected_wc_genre, selected_wc_genre)}",
            use_container_width=True,
        )

    exp_df = load_dataset_csv(str(EXPERIMENTS_CSV))
    if not exp_df.empty:
        clean_df = load_dataset_csv(str(STORIES_CLEAN_CSV))
        n_test = int((clean_df["split"] == "test").sum()) if (not clean_df.empty and "split" in clean_df.columns) else 0
        test_desc = f"trên {n_test} mẫu tập Test — webtruyen.vn" if n_test > 0 else "trên tập Test — webtruyen.vn"
        st.markdown(f"**Bảng tổng hợp kết quả các mô hình ({test_desc})**")
        st.dataframe(exp_df, use_container_width=True, hide_index=True)

    figure_files = [FIGURES_DIR / fname for fname in FIGURE_CAPTIONS if (FIGURES_DIR / fname).exists()]
    if figure_files:
        st.markdown("**Biểu đồ phân tích đã lưu**")
        cols = st.columns(2)
        for idx, fig_path in enumerate(figure_files):
            with cols[idx % 2]:
                st.image(
                    str(fig_path),
                    caption=FIGURE_CAPTIONS.get(fig_path.name, fig_path.name),
                    use_container_width=True,
                )


def main() -> None:
    st.set_page_config(
        page_title="Phân loại thể loại truyện tiếng Việt",
        layout="wide",
    )
    st.title("Đồ án NLP Đề 11: Phân loại thể loại truyện tiếng Việt")
    st.caption(
        "Dữ liệu webtruyen.vn (18.603 truyện, 22 thể loại) | "
        "Pipeline: Tiền xử lý tiếng Việt (underthesea) → TF-IDF / Word Embedding → Logistic Regression / Linear SVM / Multinomial NB / TextCNN"
    )

    tab_predict, tab_eda = st.tabs(["Dự đoán thể loại", "EDA & So sánh mô hình"])
    with tab_predict:
        render_prediction_tab()
    with tab_eda:
        render_eda_tab()


if __name__ == "__main__":
    main()
