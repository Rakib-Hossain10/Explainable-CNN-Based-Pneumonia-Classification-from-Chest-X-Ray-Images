# app.py

import hashlib
import io
from html import escape
from typing import Dict, Tuple

import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

try:
    from model_utils import load_model_and_metadata, predict_image
    from gradcam_utils import (
        build_gradcam_model,
        generate_gradcam,
        create_side_by_side_gradcam_image,
    )
except ImportError:
    from .model_utils import load_model_and_metadata, predict_image
    from .gradcam_utils import (
        build_gradcam_model,
        generate_gradcam,
        create_side_by_side_gradcam_image,
    )


# ---------------------------------------------------------
# Streamlit page setup
# ---------------------------------------------------------

st.set_page_config(
    page_title="Explainable Pneumonia Classification",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# Custom UI styling
# ---------------------------------------------------------

APP_CSS = """
<style>
:root {
    --bg-deep: #020617;
    --bg-mid: #07111f;
    --bg-soft: #0b1b33;
    --glass: rgba(255, 255, 255, 0.075);
    --glass-strong: rgba(255, 255, 255, 0.135);
    --border: rgba(178, 221, 255, 0.18);
    --border-blue: rgba(56, 189, 248, 0.34);
    --text-main: #f5fbff;
    --text-muted: #a9c4dd;
    --text-soft: #d7ecff;
    --blue-hot: #38bdf8;
    --blue-neon: #00e5ff;
    --blue-deep: #075985;
    --blue-royal: #2563eb;
    --indigo: #4f46e5;
    --cyan-soft: #67e8f9;
    --success: #22c55e;
    --warning: #facc15;
    --danger: #fb7185;
    --shadow-blue: 0 24px 90px rgba(14, 165, 233, 0.20);
    --shadow-dark: 0 24px 80px rgba(0, 0, 0, 0.42);
}

* {
    scroll-behavior: smooth;
}

.stApp {
    background:
        radial-gradient(circle at 10% 8%, rgba(56, 189, 248, 0.28), transparent 30%),
        radial-gradient(circle at 88% 12%, rgba(37, 99, 235, 0.26), transparent 32%),
        radial-gradient(circle at 52% 88%, rgba(103, 232, 249, 0.11), transparent 34%),
        radial-gradient(circle at 20% 78%, rgba(79, 70, 229, 0.16), transparent 28%),
        linear-gradient(135deg, #020617 0%, #06101f 42%, #071827 100%);
    color: var(--text-main);
}

.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    background-image:
        linear-gradient(rgba(255,255,255,0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.025) 1px, transparent 1px);
    background-size: 44px 44px;
    mask-image: linear-gradient(to bottom, rgba(0,0,0,0.75), transparent 78%);
    z-index: 0;
}

[data-testid="stHeader"] {
    background: rgba(2, 6, 23, 0.46);
    backdrop-filter: blur(22px);
    border-bottom: 1px solid rgba(148, 211, 255, 0.10);
}

.block-container {
    padding-top: 2.1rem;
    padding-bottom: 4rem;
    max-width: 1240px;
    position: relative;
    z-index: 1;
}

section[data-testid="stSidebar"] {
    background:
        radial-gradient(circle at 22% 8%, rgba(56, 189, 248, 0.24), transparent 36%),
        radial-gradient(circle at 84% 55%, rgba(37, 99, 235, 0.16), transparent 35%),
        linear-gradient(180deg, rgba(3, 10, 27, 0.97), rgba(2, 6, 23, 0.99)) !important;
    border-right: 1px solid rgba(148, 211, 255, 0.14);
    box-shadow: 18px 0 70px rgba(0, 0, 0, 0.28);
}

section[data-testid="stSidebar"] > div {
    padding-top: 2rem;
}

h1, h2, h3 {
    color: #f7fcff !important;
    letter-spacing: -0.035em;
}

p, li, label, span {
    color: inherit;
}

div[data-testid="stMarkdownContainer"] p {
    color: var(--text-muted);
    line-height: 1.72;
}

.hero-card {
    position: relative;
    overflow: hidden;
    padding: 2.55rem 2.55rem 2.15rem;
    border-radius: 34px;
    background:
        linear-gradient(135deg, rgba(255,255,255,0.155), rgba(255,255,255,0.055)),
        radial-gradient(circle at top right, rgba(56, 189, 248, 0.34), transparent 34%),
        radial-gradient(circle at 25% 95%, rgba(37, 99, 235, 0.20), transparent 34%),
        radial-gradient(circle at 86% 82%, rgba(103, 232, 249, 0.12), transparent 28%);
    border: 1px solid var(--border);
    box-shadow: var(--shadow-blue), var(--shadow-dark);
    backdrop-filter: blur(26px);
    margin-bottom: 1.35rem;
}

.hero-card::before {
    content: "";
    position: absolute;
    inset: -2px;
    background:
        linear-gradient(120deg, transparent 0%, rgba(255,255,255,0.18) 43%, transparent 60%);
    transform: translateX(-78%);
    pointer-events: none;
    animation: glassShine 8s ease-in-out infinite;
}

.hero-card::after {
    content: "";
    position: absolute;
    width: 260px;
    height: 260px;
    right: -90px;
    top: -90px;
    border-radius: 999px;
    background:
        radial-gradient(circle, rgba(103, 232, 249, 0.32), rgba(56, 189, 248, 0.08) 45%, transparent 70%);
    filter: blur(2px);
    pointer-events: none;
}

@keyframes glassShine {
    0%, 58% {
        transform: translateX(-82%);
        opacity: 0;
    }
    68% {
        opacity: 1;
    }
    100% {
        transform: translateX(82%);
        opacity: 0;
    }
}

.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.44rem 0.78rem;
    border-radius: 999px;
    color: #e8f9ff;
    background: rgba(14, 165, 233, 0.15);
    border: 1px solid rgba(103, 232, 249, 0.32);
    box-shadow:
        inset 0 1px 0 rgba(255,255,255,0.18),
        0 12px 36px rgba(14, 165, 233, 0.15);
    font-size: 0.82rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

.hero-title {
    margin: 0;
    font-size: clamp(2.3rem, 5vw, 4.45rem);
    line-height: 0.96;
    font-weight: 950;
    max-width: 980px;
    background: linear-gradient(90deg, #ffffff 0%, #dff7ff 34%, #67e8f9 68%, #38bdf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    filter: drop-shadow(0 16px 38px rgba(56, 189, 248, 0.14));
}

.hero-subtitle {
    margin-top: 1.15rem;
    max-width: 920px;
    font-size: 1.08rem;
    color: #bdd8ef;
}

.hero-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 0.72rem;
    margin-top: 1.4rem;
}

.hero-badge {
    padding: 0.64rem 0.86rem;
    border-radius: 16px;
    background:
        linear-gradient(135deg, rgba(255,255,255,0.10), rgba(255,255,255,0.035));
    border: 1px solid rgba(148, 211, 255, 0.18);
    box-shadow:
        inset 0 1px 0 rgba(255,255,255,0.12),
        0 12px 26px rgba(0,0,0,0.16);
    color: #eefaff;
    font-weight: 800;
    font-size: 0.92rem;
}

.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 0.9rem;
    padding: 1rem;
    border-radius: 24px;
    background:
        linear-gradient(135deg, rgba(255,255,255,0.105), rgba(255,255,255,0.045));
    border: 1px solid rgba(148, 211, 255, 0.16);
    box-shadow: 0 18px 50px rgba(0,0,0,0.28);
    margin-bottom: 1rem;
}

.sidebar-icon {
    width: 3.05rem;
    height: 3.05rem;
    display: grid;
    place-items: center;
    border-radius: 19px;
    background:
        radial-gradient(circle at 30% 24%, rgba(255,255,255,0.36), transparent 24%),
        linear-gradient(135deg, var(--blue-hot), var(--blue-royal) 54%, #1e1b4b);
    box-shadow:
        0 14px 34px rgba(56, 189, 248, 0.30),
        inset 0 1px 0 rgba(255,255,255,0.28);
    font-size: 1.45rem;
}

.sidebar-title {
    color: #ffffff;
    font-weight: 950;
    font-size: 1.05rem;
    line-height: 1.1;
}

.sidebar-subtitle {
    color: #a9cce8;
    font-size: 0.82rem;
    margin-top: 0.2rem;
}

.sidebar-card {
    padding: 1rem;
    border-radius: 22px;
    background:
        linear-gradient(135deg, rgba(255,255,255,0.075), rgba(255,255,255,0.035));
    border: 1px solid rgba(148, 211, 255, 0.14);
    box-shadow:
        inset 0 1px 0 rgba(255,255,255,0.08),
        0 16px 44px rgba(0,0,0,0.20);
    margin: 0.8rem 0 1rem;
}

.sidebar-card p,
.sidebar-card li {
    color: #bcd7ed;
    font-size: 0.92rem;
}

.sidebar-card strong {
    color: #e9f9ff;
}

.sidebar-card ul {
    margin-bottom: 0;
}

.empty-state {
    padding: 1.45rem;
    border-radius: 26px;
    background:
        linear-gradient(135deg, rgba(255,255,255,0.105), rgba(255,255,255,0.04)),
        radial-gradient(circle at top right, rgba(56, 189, 248, 0.20), transparent 34%),
        radial-gradient(circle at bottom left, rgba(37, 99, 235, 0.14), transparent 36%);
    border: 1px solid rgba(148, 211, 255, 0.17);
    box-shadow: 0 20px 58px rgba(0,0,0,0.24);
    margin-top: 0.8rem;
    backdrop-filter: blur(20px);
}

.empty-state-title {
    color: #ffffff;
    font-weight: 950;
    font-size: 1.25rem;
    margin-bottom: 0.35rem;
}

.empty-state-text {
    color: #b8d2e8;
    margin: 0;
}

.section-heading {
    display: flex;
    align-items: center;
    gap: 0.82rem;
    margin: 1.2rem 0 0.82rem;
}

.section-heading-icon {
    width: 2.62rem;
    height: 2.62rem;
    display: grid;
    place-items: center;
    border-radius: 17px;
    background:
        radial-gradient(circle at 28% 22%, rgba(255,255,255,0.32), transparent 24%),
        linear-gradient(135deg, rgba(56, 189, 248, 0.95), rgba(37, 99, 235, 0.95) 62%, rgba(30, 27, 75, 0.95));
    box-shadow:
        0 14px 30px rgba(56, 189, 248, 0.22),
        inset 0 1px 0 rgba(255,255,255,0.22);
}

.section-heading h2 {
    margin: 0;
    font-size: 1.55rem;
}

.section-heading p {
    margin: 0.1rem 0 0;
    color: #aac8df;
    font-size: 0.95rem;
}

.status-banner {
    display: flex;
    align-items: flex-start;
    gap: 0.92rem;
    padding: 1.05rem 1.08rem;
    border-radius: 24px;
    margin: 0.85rem 0 1.1rem;
    backdrop-filter: blur(20px);
    box-shadow: 0 18px 48px rgba(0,0,0,0.24);
}

.status-banner strong {
    color: #ffffff;
}

.status-danger {
    background:
        linear-gradient(135deg, rgba(251, 113, 133, 0.18), rgba(14, 165, 233, 0.06)),
        radial-gradient(circle at top right, rgba(251, 113, 133, 0.14), transparent 36%);
    border: 1px solid rgba(251, 113, 133, 0.34);
}

.status-success {
    background:
        linear-gradient(135deg, rgba(34, 197, 94, 0.16), rgba(56, 189, 248, 0.055)),
        radial-gradient(circle at top right, rgba(34, 197, 94, 0.12), transparent 34%);
    border: 1px solid rgba(74, 222, 128, 0.30);
}

.status-warning {
    background:
        linear-gradient(135deg, rgba(250, 204, 21, 0.15), rgba(56, 189, 248, 0.05)),
        radial-gradient(circle at top right, rgba(250, 204, 21, 0.12), transparent 34%);
    border: 1px solid rgba(250, 204, 21, 0.30);
}

.status-icon {
    width: 2.45rem;
    height: 2.45rem;
    flex: 0 0 auto;
    display: grid;
    place-items: center;
    border-radius: 16px;
    background: rgba(255,255,255,0.105);
    border: 1px solid rgba(255,255,255,0.11);
    font-size: 1.2rem;
}

.status-text {
    color: #c4dced;
    line-height: 1.55;
}

.metric-card {
    min-height: 124px;
    padding: 1.12rem;
    border-radius: 26px;
    background:
        linear-gradient(135deg, rgba(255,255,255,0.115), rgba(255,255,255,0.045)),
        radial-gradient(circle at top right, rgba(56, 189, 248, 0.19), transparent 43%),
        radial-gradient(circle at bottom left, rgba(37, 99, 235, 0.09), transparent 36%);
    border: 1px solid rgba(148, 211, 255, 0.16);
    box-shadow:
        0 20px 54px rgba(0,0,0,0.24),
        inset 0 1px 0 rgba(255,255,255,0.10);
    backdrop-filter: blur(20px);
    transition: transform 180ms ease, border-color 180ms ease, box-shadow 180ms ease;
}

.metric-card:hover {
    transform: translateY(-3px);
    border-color: rgba(103, 232, 249, 0.34);
    box-shadow:
        0 24px 64px rgba(0,0,0,0.28),
        0 12px 36px rgba(56, 189, 248, 0.12);
}

.metric-label {
    color: #93c5df;
    font-size: 0.82rem;
    font-weight: 850;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.metric-value {
    color: #ffffff;
    font-size: 1.72rem;
    font-weight: 950;
    margin-top: 0.55rem;
    letter-spacing: -0.035em;
}

.metric-caption {
    color: #a9c7df;
    margin-top: 0.35rem;
    font-size: 0.86rem;
}

.probability-card {
    padding: 1rem 1.06rem;
    border-radius: 23px;
    background:
        linear-gradient(135deg, rgba(255,255,255,0.075), rgba(255,255,255,0.035));
    border: 1px solid rgba(148, 211, 255, 0.14);
    box-shadow: 0 14px 38px rgba(0,0,0,0.18);
    margin: 0.68rem 0;
    backdrop-filter: blur(18px);
}

.probability-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    margin-bottom: 0.45rem;
}

.probability-name {
    color: #eefaff;
    font-weight: 850;
}

.probability-value {
    color: #67e8f9;
    font-weight: 950;
}

div[data-testid="stFileUploader"] {
    padding: 1rem;
    border-radius: 28px;
    background:
        linear-gradient(135deg, rgba(255,255,255,0.105), rgba(255,255,255,0.04)),
        radial-gradient(circle at top right, rgba(56, 189, 248, 0.13), transparent 36%);
    border: 1px dashed rgba(103, 232, 249, 0.48);
    box-shadow:
        0 20px 54px rgba(0,0,0,0.22),
        inset 0 1px 0 rgba(255,255,255,0.10);
    backdrop-filter: blur(20px);
}

div[data-testid="stFileUploader"] label {
    color: #f7fcff !important;
    font-weight: 900;
}

div[data-testid="stFileUploader"] small {
    color: #a9c8df !important;
}

div[data-testid="stFileUploader"] button {
    border-radius: 14px !important;
    border: 1px solid rgba(103, 232, 249, 0.28) !important;
    background: rgba(255,255,255,0.08) !important;
    color: #e9f9ff !important;
}

.stButton > button,
.stDownloadButton > button {
    width: 100%;
    border: 0 !important;
    border-radius: 17px !important;
    color: #ffffff !important;
    font-weight: 950 !important;
    letter-spacing: 0.01em;
    background:
        radial-gradient(circle at 25% 15%, rgba(255,255,255,0.28), transparent 24%),
        linear-gradient(135deg, #38bdf8 0%, #2563eb 48%, #1e1b4b 100%) !important;
    box-shadow:
        0 18px 42px rgba(37, 99, 235, 0.32),
        0 8px 24px rgba(56, 189, 248, 0.16),
        inset 0 1px 0 rgba(255,255,255,0.30) !important;
    transition: all 180ms ease !important;
    min-height: 3.05rem;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    transform: translateY(-2px);
    box-shadow:
        0 22px 52px rgba(37, 99, 235, 0.40),
        0 10px 28px rgba(56, 189, 248, 0.22),
        inset 0 1px 0 rgba(255,255,255,0.34) !important;
    filter: brightness(1.07);
}

.stButton > button:active,
.stDownloadButton > button:active {
    transform: translateY(0);
}

div[data-testid="stAlert"] {
    border-radius: 22px;
    border: 1px solid rgba(148, 211, 255, 0.15);
    background:
        linear-gradient(135deg, rgba(255,255,255,0.085), rgba(255,255,255,0.04));
    backdrop-filter: blur(18px);
    box-shadow: 0 16px 46px rgba(0,0,0,0.22);
    color: #eaf8ff;
}

div[data-testid="stImage"] {
    padding: 0.68rem;
    border-radius: 26px;
    background:
        linear-gradient(135deg, rgba(255,255,255,0.105), rgba(255,255,255,0.035));
    border: 1px solid rgba(148, 211, 255, 0.16);
    box-shadow:
        0 20px 56px rgba(0,0,0,0.25),
        inset 0 1px 0 rgba(255,255,255,0.10);
    backdrop-filter: blur(20px);
}

div[data-testid="stImage"] img {
    border-radius: 19px;
}

.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #2563eb, #38bdf8, #67e8f9) !important;
}

.stProgress > div > div > div {
    background-color: rgba(255,255,255,0.105) !important;
}

.streamlit-expanderHeader {
    color: #effaff !important;
    font-weight: 900;
    background: rgba(255,255,255,0.055);
    border-radius: 14px;
}

div[data-testid="stExpander"] {
    border: 1px solid rgba(148, 211, 255, 0.15);
    border-radius: 20px;
    background:
        linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.028));
    overflow: hidden;
    backdrop-filter: blur(18px);
    box-shadow: 0 16px 42px rgba(0,0,0,0.18);
}

hr {
    border: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(103, 232, 249, 0.45), transparent);
    margin: 1.8rem 0;
}

.tiny-label {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    color: #67e8f9;
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-weight: 950;
    margin: 0.8rem 0 0.45rem;
    padding: 0.42rem 0.7rem;
    border-radius: 999px;
    background: rgba(14, 165, 233, 0.10);
    border: 1px solid rgba(103, 232, 249, 0.20);
}

.action-copy {
    padding: 1.04rem;
    border-radius: 23px;
    background:
        linear-gradient(135deg, rgba(255,255,255,0.075), rgba(255,255,255,0.035));
    border: 1px solid rgba(148, 211, 255, 0.14);
    box-shadow: 0 14px 38px rgba(0,0,0,0.18);
    margin-bottom: 0.92rem;
    backdrop-filter: blur(18px);
}

.action-copy p {
    margin: 0;
    color: #b9d4e8;
}

.footer-note {
    margin-top: 1.25rem;
    color: #8fb1ca;
    font-size: 0.86rem;
    text-align: center;
}

code {
    border-radius: 12px !important;
}

[data-testid="stSpinner"] {
    color: #dff7ff !important;
}

::-webkit-scrollbar {
    width: 10px;
}

::-webkit-scrollbar-track {
    background: #020617;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #075985, #38bdf8);
    border-radius: 999px;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(180deg, #2563eb, #67e8f9);
}

@media (max-width: 768px) {
    .hero-card {
        padding: 1.65rem;
        border-radius: 28px;
    }

    .hero-title {
        font-size: 2.28rem;
    }

    .metric-value {
        font-size: 1.45rem;
    }

    .section-heading {
        align-items: flex-start;
    }
}
</style>
"""


# ---------------------------------------------------------
# Styling helpers
# ---------------------------------------------------------

def render_custom_css() -> None:
    """
    Apply custom glassy dark-blue styling to the Streamlit app.
    """
    st.markdown(APP_CSS, unsafe_allow_html=True)


def render_section_heading(icon: str, title: str, subtitle: str = "") -> None:
    """
    Render a styled section heading.
    """
    subtitle_html = f"<p>{escape(subtitle)}</p>" if subtitle else ""

    st.markdown(
        f"""
        <div class="section-heading">
            <div class="section-heading-icon">{escape(icon)}</div>
            <div>
                <h2>{escape(title)}</h2>
                {subtitle_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str, caption: str = "") -> None:
    """
    Render a glass metric card.
    """
    caption_html = f'<div class="metric-caption">{escape(caption)}</div>' if caption else ""

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{escape(label)}</div>
            <div class="metric-value">{escape(value)}</div>
            {caption_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# Cache model resources
# ---------------------------------------------------------

@st.cache_resource(show_spinner="Loading trained ResNet50 three-class pneumonia model...")
def get_model_and_metadata():
    """
    Load model and metadata only once.

    Streamlit reruns the script after every button click,
    so caching is necessary to avoid loading the model again and again.
    """
    model, metadata = load_model_and_metadata(warmup=True)
    return model, metadata


@st.cache_resource(show_spinner="Preparing Grad-CAM model...")
def get_gradcam_model(_model):
    """
    Build Grad-CAM model only once.

    The underscore in _model tells Streamlit not to hash the Keras model object.
    """
    gradcam_model = build_gradcam_model(_model)
    return gradcam_model


# ---------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------

PREDICTION_STATE_KEYS = [
    "active_image_hash",
    "uploaded_image",
    "prediction_result",
    "gradcam_result",
    "gradcam_target_class",
]


def clear_prediction_state() -> None:
    """
    Clear prediction and Grad-CAM results.
    """
    for key in PREDICTION_STATE_KEYS:
        if key in st.session_state:
            del st.session_state[key]


def safe_rerun() -> None:
    """
    Rerun helper for different Streamlit versions.
    """
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()


# ---------------------------------------------------------
# Image helpers
# ---------------------------------------------------------

def read_uploaded_image(uploaded_file) -> Tuple[Image.Image, str]:
    """
    Read uploaded image safely and return:
    - RGB PIL image
    - image hash

    The hash helps detect when the user uploads a new image.
    """
    image_bytes = uploaded_file.getvalue()
    image_hash = hashlib.sha256(image_bytes).hexdigest()

    image = Image.open(io.BytesIO(image_bytes))
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")

    return image, image_hash


def pil_image_to_png_bytes(image: Image.Image) -> bytes:
    """
    Convert PIL image to PNG bytes for Streamlit download button.
    """
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.getvalue()


def show_image(image: Image.Image, caption: str = "") -> None:
    """
    Display image with compatibility for different Streamlit versions.
    """
    try:
        st.image(image, caption=caption, use_container_width=True)
    except TypeError:
        st.image(image, caption=caption, use_column_width=True)


# ---------------------------------------------------------
# UI helpers
# ---------------------------------------------------------

def render_sidebar() -> None:
    """
    Sidebar content.
    """
    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-icon">🫁</div>
            <div>
                <div class="sidebar-title">Pneumonia Classification</div>
                <div class="sidebar-subtitle">ResNet50 • Grad-CAM • 3 classes</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown(
        """
        <div class="sidebar-card">
            <p>This application uses a trained ResNet50 model to classify chest X-ray images into three classes:</p>
            <ul>
                <li><strong>Normal</strong></li>
                <li><strong>Pneumonia-Bacterial</strong></li>
                <li><strong>Pneumonia-Viral</strong></li>
            </ul>
            <p style="margin-top: 0.8rem;">
                Grad-CAM is used to visualize the image regions that influenced the selected prediction class.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.warning(
        "Academic research demo only. "
        "This tool is not a medical diagnosis system."
    )

    st.sidebar.markdown("---")

    if st.sidebar.button("Clear Current Result"):
        clear_prediction_state()
        safe_rerun()


def render_header() -> None:
    """
    Main page header.
    """
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-eyebrow">Explainable Medical AI Demo • Chest X-ray Classifier</div>
            <h1 class="hero-title">Three-Class Pneumonia Classification</h1>
            <p class="hero-subtitle">
                Upload a chest X-ray image and run the ResNet50 classifier to distinguish
                between Normal, Bacterial Pneumonia, and Viral Pneumonia. Grad-CAM
                visualization is provided to highlight the image regions that influenced
                the model's prediction.
            </p>
            <div class="hero-badges">
                <div class="hero-badge">🫁 Normal / Bacterial / Viral</div>
                <div class="hero-badge">🔥 Grad-CAM Explainability</div>
                <div class="hero-badge">⚡ Cached Model Loading</div>
                <div class="hero-badge">🎓 Thesis Demonstration System</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.warning(
        "This application is for academic demonstration only and should not be used "
        "as a substitute for professional medical diagnosis."
    )


def render_upload_instructions() -> None:
    """
    Instructions shown before image upload.
    """
    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-state-title">Upload a chest X-ray to begin</div>
            <p class="empty-state-text">
                Please upload a chest X-ray image in JPG, JPEG, or PNG format.
                After upload, the prediction controls will appear automatically.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Expected workflow"):
        st.markdown(
            """
            1. Upload a chest X-ray image.  
            2. Click **Classify Chest X-ray**.  
            3. View the predicted class and confidence.  
            4. Review the probability distribution for all three classes.  
            5. Generate a Grad-CAM explanation for the predicted class or a selected target class.  
            6. Download the Grad-CAM result if needed.  
            """
        )


def render_probabilities(probabilities_percent: Dict[str, float]) -> None:
    """
    Display class probabilities using progress bars.
    """
    render_section_heading(
        icon="📊",
        title="Class Probabilities",
        subtitle="Model confidence distribution across the three output classes.",
    )

    for class_name, probability in probabilities_percent.items():
        probability_value = float(probability)
        progress_value = int(round(max(0.0, min(100.0, probability_value))))

        st.markdown(
            f"""
            <div class="probability-card">
                <div class="probability-row">
                    <span class="probability-name">{escape(class_name)}</span>
                    <span class="probability-value">{probability_value:.2f}%</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(progress_value)


def render_prediction_status(result: Dict) -> None:
    """
    Display a colored status banner according to prediction.
    """
    diagnosis_group = str(result.get("diagnosis_group", "Unknown"))
    predicted_class = str(result.get("predicted_class", "Unknown"))
    pneumonia_type = str(result.get("pneumonia_type", "Not applicable"))

    if diagnosis_group == "Pneumonia":
        if "Bacterial" in pneumonia_type:
            icon = "🦠"
            title = "Bacterial pneumonia pattern detected"
            message = (
                "The model classified this chest X-ray as "
                f"<strong>{escape(predicted_class)}</strong>. "
                "A Grad-CAM explanation can be generated to review the highlighted attention region."
            )
        elif "Viral" in pneumonia_type:
            icon = "🧬"
            title = "Viral pneumonia pattern detected"
            message = (
                "The model classified this chest X-ray as "
                f"<strong>{escape(predicted_class)}</strong>. "
                "A Grad-CAM explanation can be generated to review the highlighted attention region."
            )
        else:
            icon = "⚠️"
            title = "Pneumonia detected"
            message = (
                "The model classified this chest X-ray as a pneumonia-related class. "
                "A Grad-CAM explanation can be generated for interpretability."
            )

        st.markdown(
            f"""
            <div class="status-banner status-danger">
                <div class="status-icon">{icon}</div>
                <div class="status-text">
                    <strong>{escape(title)}</strong><br>
                    {message}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    elif diagnosis_group == "Normal":
        st.markdown(
            """
            <div class="status-banner status-success">
                <div class="status-icon">✅</div>
                <div class="status-text">
                    <strong>Normal chest X-ray pattern detected</strong><br>
                    The model classified this image as <strong>Normal</strong>.
                    Grad-CAM can still be generated to visualize the regions that influenced
                    the Normal prediction.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:
        st.markdown(
            """
            <div class="status-banner status-warning">
                <div class="status-icon">ℹ️</div>
                <div class="status-text">
                    <strong>Unknown prediction group</strong><br>
                    The model returned a class that could not be assigned to a known diagnosis group.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_prediction_result(result: Dict) -> None:
    """
    Display prediction result.
    """
    predicted_class = result["predicted_class"]
    diagnosis_group = result.get("diagnosis_group", "Unknown")
    pneumonia_type = result.get("pneumonia_type", "Not applicable")
    confidence_percent = float(result["confidence_percent"])
    pneumonia_probability_percent = float(result["pneumonia_probability_percent"])

    st.markdown("---")
    render_section_heading(
        icon="🧠",
        title="Prediction Result",
        subtitle="Three-class ResNet50 output with confidence and pneumonia probability.",
    )

    render_prediction_status(result)

    metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4)

    with metric_col_1:
        render_metric_card(
            label="Predicted Class",
            value=str(predicted_class),
            caption="Top class selected by the model",
        )

    with metric_col_2:
        render_metric_card(
            label="Confidence",
            value=f"{confidence_percent:.2f}%",
            caption="Confidence of the predicted class",
        )

    with metric_col_3:
        render_metric_card(
            label="Diagnosis Group",
            value=str(diagnosis_group),
            caption="Normal or pneumonia-level grouping",
        )

    with metric_col_4:
        render_metric_card(
            label="Pneumonia Probability",
            value=f"{pneumonia_probability_percent:.2f}%",
            caption="Bacterial + viral probability",
        )

    if diagnosis_group == "Pneumonia":
        st.info(f"Predicted pneumonia subtype: **{pneumonia_type}**")

    render_probabilities(result["probabilities_percent"])


def render_gradcam_controls(prediction_result: Dict, metadata: Dict) -> Tuple[str, bool]:
    """
    Render Grad-CAM target selection and button.

    Returns:
    - selected target class name
    - whether button was clicked
    """
    st.markdown("---")
    render_section_heading(
        icon="🔥",
        title="Grad-CAM Explanation",
        subtitle="Generate a heatmap for the predicted class or another selected class.",
    )

    class_names = list(metadata["class_names"])
    predicted_class = prediction_result["predicted_class"]

    try:
        default_index = class_names.index(predicted_class)
    except ValueError:
        default_index = 0

    st.markdown(
        """
        <div class="action-copy">
            <p>
                Grad-CAM highlights image regions that contributed strongly to the selected
                class prediction. For thesis demonstration, using the predicted class as
                the target is usually the most appropriate choice.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected_target_class = st.selectbox(
        "Grad-CAM target class",
        options=class_names,
        index=default_index,
        help="Choose which output class the Grad-CAM heatmap should explain.",
    )

    gradcam_clicked = st.button("Generate Grad-CAM Explanation")

    return selected_target_class, gradcam_clicked


def render_gradcam_result(gradcam_result: Dict) -> None:
    """
    Display Grad-CAM output images.
    """
    st.markdown("---")
    render_section_heading(
        icon="🔥",
        title="Grad-CAM Visualization Result",
        subtitle="Visual explanation of model attention regions.",
    )

    used_class_name = gradcam_result.get("used_class_name", "Unknown")
    used_probability = float(gradcam_result.get("used_class_probability_percent", 0.0))
    target_layer = gradcam_result.get("target_layer_name", "unknown")

    st.info(
        f"Grad-CAM target class: **{used_class_name}** "
        f"({used_probability:.2f}% probability). "
        f"Target convolution layer: **{target_layer}**."
    )

    col_1, col_2 = st.columns(2)

    with col_1:
        show_image(
            gradcam_result["display_img"],
            caption="Original Chest X-ray",
        )

    with col_2:
        show_image(
            gradcam_result["overlay_img"],
            caption=f"Grad-CAM Overlay for {used_class_name}",
        )

    with st.expander("Show heatmap only"):
        show_image(
            gradcam_result["heatmap_img"],
            caption="Grad-CAM Heatmap",
        )

    side_by_side = create_side_by_side_gradcam_image(
        display_img=gradcam_result["display_img"],
        overlay_img=gradcam_result["overlay_img"],
    )

    st.download_button(
        label="Download Grad-CAM Result",
        data=pil_image_to_png_bytes(side_by_side),
        file_name="pneumonia_3class_gradcam_result.png",
        mime="image/png",
    )


def show_exception(title: str, error: Exception) -> None:
    """
    Display clean error message with technical details.
    """
    st.error(title)

    with st.expander("Show technical error details"):
        st.code(str(error))


# ---------------------------------------------------------
# Main app
# ---------------------------------------------------------

def main() -> None:
    render_custom_css()
    render_sidebar()
    render_header()

    # Load model early so metadata is available for Grad-CAM class selection.
    try:
        model, metadata = get_model_and_metadata()
    except Exception as error:
        show_exception("Model or metadata loading failed.", error)
        st.stop()

    st.markdown('<div class="tiny-label">Upload image</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload chest X-ray image",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
    )

    if uploaded_file is None:
        render_upload_instructions()
        return

    try:
        uploaded_image, image_hash = read_uploaded_image(uploaded_file)
    except UnidentifiedImageError:
        st.error(
            "The uploaded file could not be read as an image. "
            "Please upload a valid JPG, JPEG, or PNG image."
        )
        return
    except Exception as error:
        show_exception("Something went wrong while reading the uploaded image.", error)
        return

    # Reset prediction and Grad-CAM result when a new image is uploaded.
    if st.session_state.get("active_image_hash") != image_hash:
        clear_prediction_state()
        st.session_state["active_image_hash"] = image_hash

    st.session_state["uploaded_image"] = uploaded_image

    st.markdown("---")

    image_col, action_col = st.columns([1, 1])

    with image_col:
        render_section_heading(
            icon="🖼️",
            title="Uploaded Image",
            subtitle="Chest X-ray selected for analysis.",
        )
        show_image(uploaded_image, caption="Chest X-ray selected for analysis")

    with action_col:
        render_section_heading(
            icon="⚡",
            title="Model Action",
            subtitle="Run the three-class pneumonia classifier.",
        )

        st.markdown(
            """
            <div class="action-copy">
                <p>
                    Click the button below to classify the chest X-ray as
                    Normal, Bacterial Pneumonia, or Viral Pneumonia.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        check_clicked = st.button("Classify Chest X-ray")

        if check_clicked:
            try:
                with st.spinner("Running three-class prediction..."):
                    prediction_result = predict_image(
                        model=model,
                        image_input=st.session_state["uploaded_image"],
                        metadata=metadata,
                    )

                st.session_state["prediction_result"] = prediction_result

                if "gradcam_result" in st.session_state:
                    del st.session_state["gradcam_result"]

            except Exception as error:
                show_exception("Prediction failed.", error)
                return

    if "prediction_result" not in st.session_state:
        st.info("After uploading the image, click **Classify Chest X-ray** to continue.")
        return

    prediction_result = st.session_state["prediction_result"]
    render_prediction_result(prediction_result)

    # -----------------------------------------------------
    # Grad-CAM section
    # -----------------------------------------------------

    selected_target_class, gradcam_clicked = render_gradcam_controls(
        prediction_result=prediction_result,
        metadata=metadata,
    )

    st.session_state["gradcam_target_class"] = selected_target_class

    if gradcam_clicked:
        try:
            with st.spinner("Generating Grad-CAM explanation..."):
                gradcam_model = get_gradcam_model(model)

                gradcam_result = generate_gradcam(
                    gradcam_model=gradcam_model,
                    image_input=st.session_state["uploaded_image"],
                    metadata=metadata,
                    target_class_name=selected_target_class,
                    alpha=0.70,#60
                    cmap_name="jet",
                    blur_radius=0,
                    min_percentile=80,#98
                    activation_cutoff=0.20,#45
                )

            st.session_state["gradcam_result"] = gradcam_result

        except Exception as error:
            show_exception("Grad-CAM generation failed.", error)
            return

    if "gradcam_result" in st.session_state:
        render_gradcam_result(st.session_state["gradcam_result"])

    st.markdown(
        """
        <div class="footer-note">
            Academic research interface • Predictions should always be reviewed by qualified medical professionals.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()