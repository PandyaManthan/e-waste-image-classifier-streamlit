from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Needed for loading older TensorFlow/Keras-saved models (common in notebooks).
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import tensorflow as tf
from PIL import Image


st.set_page_config(
    page_title="E-Waste Classifier",
    page_icon="♻️",
    layout="wide",
)


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATHS = [
    PROJECT_ROOT / "artifacts/ewaste_classifier.h5",
    PROJECT_ROOT / "artifacts/ewaste_classifier.keras",
    PROJECT_ROOT / "artifacts/ewaste_classifier",
    PROJECT_ROOT / "artifacts/ewaste_classifier.keras.h5",
    PROJECT_ROOT / "ewaste_classifier.keras",
    PROJECT_ROOT / "ewaste_classifier",
    PROJECT_ROOT / "ewaste_classifier.h5",
]
CLASS_NAMES_PATH = PROJECT_ROOT / "artifacts/class_names.json"
TRAIN_DIR = PROJECT_ROOT / "modified-dataset/train"
IMG_SIZE = (128, 128)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            .main {
                background: linear-gradient(180deg, #f8fbff 0%, #eef7f1 100%);
            }
            .hero {
                border-radius: 18px;
                padding: 1.2rem 1.4rem;
                background: rgba(255,255,255,0.90);
                border: 1px solid #d7e3ef;
                box-shadow: 0 8px 24px rgba(14, 30, 37, 0.08);
                margin-bottom: 1rem;
            }
            .metric-box {
                border-radius: 14px;
                background: #ffffff;
                border: 1px solid #e5edf5;
                padding: 1rem;
                box-shadow: 0 4px 12px rgba(14, 30, 37, 0.06);
            }
            .tip-box {
                border-left: 4px solid #00a878;
                border-radius: 10px;
                background: #f4fffb;
                padding: 0.8rem 1rem;
                margin-top: 0.8rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_available_model_path() -> Path | None:
    for path in MODEL_PATHS:
        if path.exists():
            return path
    return None


def load_class_names() -> list[str]:
    if CLASS_NAMES_PATH.exists():
        try:
            with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list) and all(isinstance(x, str) for x in data):
                return data
        except json.JSONDecodeError:
            # Fall back safely if labels file gets corrupted by merge conflicts.
            pass

    if TRAIN_DIR.exists():
        return sorted([p.name for p in TRAIN_DIR.iterdir() if p.is_dir()])

    return []


@st.cache_resource
def load_model():
    available_paths = [path for path in MODEL_PATHS if path.exists()]
    if not available_paths:
        return None, None, "Model file not found in expected paths."

    errors: list[str] = []
    for model_path in available_paths:
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            return model, model_path, None
        except Exception as exc:
            errors.append(f"{model_path.name}: {exc}")

    return None, available_paths[0], " | ".join(errors)


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(image, dtype=np.float32)
    return np.expand_dims(arr, axis=0)


def model_has_rescaling_layer(model) -> bool:
    def _walk_layers(current_model):
        for layer in current_model.layers:
            yield layer
            if isinstance(layer, tf.keras.Model):
                yield from _walk_layers(layer)

    return any(layer.__class__.__name__ == "Rescaling" for layer in _walk_layers(model))


def get_disposal_tip(label: str) -> str:
    tips = {
        "battery": "Do not throw batteries in regular bins. Drop them at approved battery recycling points.",
        "mobile": "Remove SIM and storage cards before e-waste drop-off.",
        "laptop": "Back up and securely wipe your data before recycling.",
        "monitor": "Handle carefully and use certified e-waste handlers.",
        "keyboard": "Collect with other peripherals to recycle in batches.",
    }
    key = label.strip().lower()
    return tips.get(key, "Use an authorized e-waste collection center for safe disposal.")


def main() -> None:
    inject_styles()

    model, model_path, model_error = load_model()
    class_names = load_class_names()

    st.markdown(
        """
        <div class="hero">
            <h2 style="margin-bottom: 0.4rem;">♻️ E-Waste Image Classifier</h2>
            <p style="margin:0;color:#334e68;">
                AI-powered classification of electronic waste items for smarter segregation and recycling.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.subheader("Deployment Status")
        if model is None:
            st.error("Model file not found")
            st.caption("Expected: `artifacts/ewaste_classifier.keras` or `.h5`")
            if model_path is not None:
                st.caption(f"Found path but load failed: `{model_path.as_posix()}`")
            if model_error:
                st.caption(f"Load error: `{model_error}`")
        else:
            st.success("Model loaded")
            st.caption(f"Using: `{model_path.as_posix()}`")

        st.write("")
        st.subheader("Classes")
        if class_names:
            st.caption(f"Detected {len(class_names)} classes")
            st.code(", ".join(class_names), language="text")
        else:
            st.warning("Class names not found")
            st.caption("Add `artifacts/class_names.json` or `modified-dataset/train`.")

    tab_upload, tab_camera = st.tabs(["Upload Image", "Use Camera"])

    uploaded_file = None
    captured_image = None

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Upload an e-waste image",
            type=["jpg", "jpeg", "png", "webp"],
            help="Use a clear image with one main object for better accuracy.",
        )

    with tab_camera:
        captured_image = st.camera_input("Capture image from webcam")

    image_source = uploaded_file if uploaded_file is not None else captured_image

    if image_source is not None:
        image = Image.open(image_source)
        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.image(image, caption="Input Image", use_container_width=True)

        with col2:
            if model is None:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.error("Prediction unavailable: model file is missing.")
                st.info("Save your trained model and restart the app.")
                st.markdown("</div>", unsafe_allow_html=True)
                return

            batch = preprocess_image(image)
            # Keep inference preprocessing aligned with training graph.
            if not model_has_rescaling_layer(model):
                batch = batch / 255.0
            probs = model.predict(batch, verbose=0)[0]
            pred_idx = int(np.argmax(probs))

            if class_names and len(class_names) == len(probs):
                pred_label = class_names[pred_idx]
            else:
                pred_label = f"class_{pred_idx}"

            confidence = float(probs[pred_idx] * 100)

            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.subheader("Prediction")
            st.write(f"**Detected Item:** `{pred_label}`")
            st.write(f"**Confidence:** `{confidence:.2f}%`")
            st.markdown(
                f'<div class="tip-box"><b>Disposal Tip:</b> {get_disposal_tip(pred_label)}</div>',
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

            if class_names and len(class_names) == len(probs):
                chart_df = pd.DataFrame(
                    {"Class": class_names, "Confidence": probs * 100}
                ).sort_values("Confidence", ascending=False)
            else:
                chart_df = pd.DataFrame(
                    {
                        "Class": [f"class_{i}" for i in range(len(probs))],
                        "Confidence": probs * 100,
                    }
                ).sort_values("Confidence", ascending=False)

            st.write("")
            st.bar_chart(
                chart_df.set_index("Class"),
                use_container_width=True,
            )
    else:
        st.info("Upload or capture an image to get a prediction.")


if __name__ == "__main__":
    main()
