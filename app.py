import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Leaf Disease Classifier",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Source+Sans+3:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Source Sans 3', sans-serif;
}

.main-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    color: #1a3a2a;
    text-align: center;
    margin-bottom: 0.2rem;
    letter-spacing: -0.5px;
}

.subtitle {
    text-align: center;
    color: #5a7a6a;
    font-size: 1.05rem;
    margin-bottom: 2rem;
    font-weight: 300;
}

.result-card {
    background: linear-gradient(135deg, #e8f5e9, #f1f8e9);
    border-left: 5px solid #2e7d32;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin-top: 1.5rem;
}

.result-label {
    font-size: 1.8rem;
    font-family: 'Playfair Display', serif;
    color: #1b5e20;
    font-weight: 700;
}

.confidence-bar-wrap {
    background: #c8e6c9;
    border-radius: 20px;
    height: 14px;
    margin-top: 0.4rem;
    overflow: hidden;
}

.confidence-bar-fill {
    height: 100%;
    border-radius: 20px;
    background: linear-gradient(90deg, #43a047, #66bb6a);
    transition: width 0.6s ease;
}

.error-card {
    background: #fff3e0;
    border-left: 5px solid #e65100;
    border-radius: 10px;
    padding: 1rem 1.5rem;
    color: #bf360c;
    margin-top: 1rem;
}

.top5-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 0;
    border-bottom: 1px solid #e0e0e0;
    font-size: 0.95rem;
}

.pill {
    display: inline-block;
    background: #e8f5e9;
    color: #2e7d32;
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-bottom: 1rem;
}

.info-badge {
    display: inline-block;
    background: #e3f2fd;
    color: #1565c0;
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-left: 8px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  CLASS LABELS
#  Same order as your training folders/classes
# ─────────────────────────────────────────────
CLASS_NAMES = [
    "Boron Deficiency",
    "Healthy",
    "Kalium (K) Deficiency",
    "Magnesium (Mg) Deficiency",
    "Nitrogen (N) Deficiency",
]

# ─────────────────────────────────────────────
#  MODEL PATHS  (must be in same folder as app.py)
# ─────────────────────────────────────────────
MODEL_PATHS = {
    "Simple CNN":   "cnn_model.h5",
    "ResNet50":     "ResNet50.h5",
    "MobileNetV2":  "mobilenetv2.h5",
}

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model(path: str):
    """
    Load Keras .h5 model with compile=False to avoid
    CategoricalCrossentropy 'fn' deserialization error
    (happens when model saved with older Keras version).
    """
    return tf.keras.models.load_model(path, compile=False)


def get_model_input_size(model) -> tuple:
    """
    Auto-detect (Height, Width) from the model's input shape.
    Falls back to (224, 224) if shape is dynamic or unknown.
    """
    try:
        shape = model.input_shape          # e.g. (None, 128, 128, 3)
        if isinstance(shape, list):
            shape = shape[0]
        h, w = shape[1], shape[2]
        if h is None or w is None:
            return (224, 224)
        return (int(h), int(w))
    except Exception:
        return (224, 224)


def preprocess_image(img: Image.Image, target_size: tuple) -> np.ndarray:
    img = img.convert("RGB").resize(target_size)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def predict(model, img_array: np.ndarray):
    preds = model.predict(img_array, verbose=0)[0]
    top_idx = int(np.argmax(preds))
    return top_idx, float(preds[top_idx]), preds


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    model_choice = st.selectbox("Select Model", list(MODEL_PATHS.keys()), index=0)
    st.markdown("---")
    st.markdown("**Class Labels**")
    for i, name in enumerate(CLASS_NAMES):
        st.markdown(f"`{i}` — {name}")
    st.markdown("---")
    st.caption("Leaf Disease Classification · v1.0")

# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown('<div class="main-title">🌿 Leaf Disease Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a leaf image to detect nutrient deficiencies using deep learning</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MODEL LOADING
# ─────────────────────────────────────────────
model_path = MODEL_PATHS[model_choice]

if not os.path.exists(model_path):
    st.markdown(f"""
    <div class="error-card">
        ⚠️ <b>Model file not found:</b> <code>{model_path}</code><br>
        Make sure the <code>.h5</code> file is in the same folder as <code>app.py</code>.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

with st.spinner(f"Loading {model_choice} model…"):
    model = load_model(model_path)

# Auto-detect input size from the loaded model
input_size = get_model_input_size(model)

st.markdown(
    f'<span class="pill">✅ {model_choice} loaded</span>'
    f'<span class="info-badge">Input: {input_size[0]}×{input_size[1]}</span>',
    unsafe_allow_html=True
)

# ─────────────────────────────────────────────
#  FILE UPLOADER
# ─────────────────────────────────────────────
uploaded = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    help="Supported formats: JPG, PNG, BMP, WEBP",
)

if uploaded is not None:
    img = Image.open(uploaded)

    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.image(img, caption="Uploaded Image", use_container_width=True)

    with col2:
        with st.spinner("Analysing…"):
            # input_size auto-detected from model — no hardcoding needed
            img_array = preprocess_image(img, input_size)
            top_idx, confidence, all_preds = predict(model, img_array)

        predicted_label = CLASS_NAMES[top_idx] if top_idx < len(CLASS_NAMES) else f"Class {top_idx}"
        conf_pct = confidence * 100

        st.markdown(f"""
        <div class="result-card">
            <div style="font-size:0.85rem; color:#558b2f; font-weight:600; text-transform:uppercase; letter-spacing:1px;">Prediction</div>
            <div class="result-label">{predicted_label}</div>
            <div style="margin-top:0.8rem; color:#388e3c; font-size:0.95rem;">Confidence: <b>{conf_pct:.1f}%</b></div>
            <div class="confidence-bar-wrap">
                <div class="confidence-bar-fill" style="width:{conf_pct}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── All class probabilities ──────────────────
    st.markdown("#### 📊 All Class Probabilities")
    sorted_indices = np.argsort(all_preds)[::-1]
    for rank, idx in enumerate(sorted_indices):
        label = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"Class {idx}"
        prob = float(all_preds[idx]) * 100
        bar_color = "#43a047" if idx == top_idx else "#90a4ae"
        st.markdown(f"""
        <div class="top5-row">
            <span>{'🥇' if rank==0 else '&nbsp;&nbsp;&nbsp;'} {label}</span>
            <span style="font-weight:600; color:{bar_color}">{prob:.2f}%</span>
        </div>
        """, unsafe_allow_html=True)
        st.progress(float(all_preds[idx]))

else:
    st.info("⬆️ Please upload a leaf image to start classification.")