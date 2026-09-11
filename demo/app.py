"""
app.py  —  Streamlit demo for traffic sign detection.

Run:
    venv_gpu\\Scripts\\streamlit run demo/app.py

Features:
  - Upload an image OR a video, OR click a sample image to load instantly
  - Choose between Classical CV baseline, any Fine-Tuned YOLO variant, or side-by-side comparison
  - Video mode: processes frame by frame and lets you download the annotated video
  - Show confidence scores and class names in a results table
  - Download annotated image
"""

import io
import os
import sys
import tempfile
import warnings
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Suppress all Python and library warnings / verbose logs
warnings.filterwarnings("ignore")
os.environ["YOLO_VERBOSE"] = "False"
os.environ["PYTHONWARNINGS"] = "ignore"

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.classical_detector import ClassicalDetector
from src.models.yolo_wrapper import YOLOWrapper

# ──────────────────────────────────────────────────────────────────────────────
# Page configuration
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Traffic Sign Detector — Group 17",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS — clean, modern light theme
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Google font ─────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

/* ── Global background ───────────────────────── */
.stApp { background-color: #F7F9FC; }

/* ── Sidebar ─────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0A295C 0%, #154E9A 100%);
}
[data-testid="stSidebar"] * { color: #E8F0FE !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label { color: #A8C4E8 !important; font-size: 0.82rem; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #FFFFFF !important; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.15); }
[data-testid="stSidebar"] .sidebar-badge {
    background: rgba(255,255,255,0.12);
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 0.82rem;
}

/* ── Header hero ─────────────────────────────── */
.hero-header {
    background: linear-gradient(135deg, #0A295C 0%, #1565C0 60%, #0097A7 100%);
    border-radius: 16px;
    padding: 28px 36px 22px 36px;
    margin-bottom: 24px;
    box-shadow: 0 4px 24px rgba(10,41,92,0.13);
}
.hero-header h1 { color: #FFFFFF; font-size: 2.1rem; font-weight: 700; margin: 0 0 4px 0; }
.hero-header p  { color: rgba(255,255,255,0.75); font-size: 0.95rem; margin: 0; }
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    border-radius: 20px;
    padding: 3px 14px;
    font-size: 0.82rem;
    color: #A8D8FF;
    margin-top: 10px;
}

/* ── Tabs ────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: #FFFFFF;
    border-radius: 12px 12px 0 0;
    border-bottom: 2px solid #E2E8F0;
    gap: 4px;
    padding: 0 8px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px 10px 0 0;
    font-weight: 500;
    font-size: 0.88rem;
    color: #5E6E87;
    padding: 10px 20px;
    border: none;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #0A295C, #1565C0) !important;
    color: #FFFFFF !important;
    font-weight: 600;
}

/* ── Cards ───────────────────────────────────── */
.detect-card {
    background: #FFFFFF;
    border-radius: 14px;
    padding: 18px;
    box-shadow: 0 2px 12px rgba(10,41,92,0.07);
    border: 1px solid #E2E8F0;
    height: 100%;
}
.detect-card h4 { color: #0A295C; font-size: 0.92rem; font-weight: 600; margin: 0 0 10px 0; }

/* ── Metric tiles ─────────────────────────────── */
.metric-tile {
    background: linear-gradient(135deg, #0A295C, #1565C0);
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
    color: #FFFFFF;
}
.metric-tile .val { font-size: 2rem; font-weight: 700; line-height: 1; }
.metric-tile .lbl { font-size: 0.78rem; opacity: 0.8; margin-top: 4px; }

/* ── Result table ─────────────────────────────── */
.result-row {
    background: #FFFFFF;
    border-radius: 10px;
    padding: 10px 16px;
    margin-bottom: 6px;
    border-left: 4px solid #1565C0;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.result-row .cls  { font-weight: 600; color: #0A295C; font-size: 0.9rem; }
.result-row .conf { 
    background: #E8F0FE; 
    color: #1565C0; 
    border-radius: 20px; 
    padding: 2px 12px; 
    font-size: 0.82rem; 
    font-weight: 600;
}

/* ── Buttons ─────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #0A295C, #1565C0);
    color: white !important;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    font-size: 0.9rem;
    padding: 10px 24px;
    transition: all 0.2s;
    box-shadow: 0 2px 8px rgba(10,41,92,0.2);
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(10,41,92,0.3);
}

/* ── Download button ─────────────────────────── */
[data-testid="stDownloadButton"] > button {
    background: #FFFFFF !important;
    color: #0A295C !important;
    border: 2px solid #0A295C !important;
    border-radius: 10px;
    font-weight: 600;
}

/* ── File uploader ───────────────────────────── */
[data-testid="stFileUploaderDropzone"] {
    background: #EEF2F8;
    border: 2px dashed #90A4C8;
    border-radius: 14px;
}

/* ── Info / warning / success ─────────────────── */
.stAlert { border-radius: 10px; }

/* ── Sample thumbnail buttons ─────────────────── */
.sample-btn button {
    background: #EEF2F8 !important;
    color: #0A295C !important;
    border: 1px solid #C5D3E8 !important;
    border-radius: 8px !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    width: 100%;
}
.sample-btn button:hover {
    background: #D0E0F8 !important;
    border-color: #1565C0 !important;
}

/* ── Progress bar ─────────────────────────────── */
.stProgress > div > div {
    background: linear-gradient(90deg, #0A295C, #0097A7);
    border-radius: 10px;
}

/* ── Footer ───────────────────────────────────── */
.footer {
    background: #FFFFFF;
    border-radius: 12px;
    padding: 14px 24px;
    text-align: center;
    font-size: 0.8rem;
    color: #8A9BB5;
    border: 1px solid #E2E8F0;
    margin-top: 24px;
}

/* ── Divider ─────────────────────────────────── */
.section-divider {
    height: 2px;
    background: linear-gradient(90deg, #0A295C, #0097A7, transparent);
    border-radius: 2px;
    margin: 20px 0;
}

/* ── Suppress Streamlit default elements ─────── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Hero header
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero-header">
    <h1>🚦 Traffic Sign Detection</h1>
    <p>Real-time traffic sign detection using Classical CV and YOLOv8 deep learning</p>
    <span class="hero-badge">CO543/CO5430 &nbsp;|&nbsp; Group 17 &nbsp;|&nbsp; University of Peradeniya</span>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🚦 Detection Settings")
    st.markdown("<hr>", unsafe_allow_html=True)

    MODEL_OPTIONS = {
        "🏆 YOLOv8s — Best (97.1% mAP)":         str(PROJECT_ROOT / "results/checkpoints/gtsdb_yolov8s_v1_best.pt"),
        "🔵 YOLOv8n — Standard (95.5% mAP)":      str(PROJECT_ROOT / "results/checkpoints/gtsdb_yolov8n_v1_best.pt"),
        "⚗️ YOLOv8n — No Augmentation (84.8%)":   str(PROJECT_ROOT / "results/checkpoints/gtsdb_yolov8n_noaug_v1_best.pt"),
        "🔬 Classical CV Baseline":                "classical",
        "🆚 Side-by-Side Comparison":              "compare",
        "🧪 Fine-Grained 43-Class (Stretch Goal)": str(PROJECT_ROOT / "runs/detect/train/weights/best.pt"),
    }

    model_choice = st.selectbox(
        "Detection Model",
        options=list(MODEL_OPTIONS.keys()),
        help="Select the model variant to run detection with.",
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("**Detection Parameters**")

    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.05, max_value=0.90, value=0.20, step=0.05,
        help="Minimum confidence score to show a detection."
    )
    iou_threshold = st.slider(
        "NMS IoU Threshold",
        min_value=0.10, max_value=0.90, value=0.45, step=0.05,
        help="IoU threshold for non-maximum suppression."
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("**Sign Classes**")
    st.markdown("""
<div class="sidebar-badge">🔴 &nbsp;Prohibitory — speed limits, no overtaking</div>
<div class="sidebar-badge">🟡 &nbsp;Danger — bends, construction, hazards</div>
<div class="sidebar-badge">🔵 &nbsp;Mandatory — turn arrows, roundabout</div>
""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("**Model Performance**")
    perf_data = [
        ("YOLOv8s", "97.1%", "#1565C0"),
        ("YOLOv8n", "95.5%", "#1976D2"),
        ("YOLOv8n (no aug)", "84.8%", "#F57F21"),
        ("Classical CV", "—", "#8A9BB5"),
    ]
    for model_name, map_val, color in perf_data:
        st.markdown(f"""
<div style="display:flex;justify-content:space-between;align-items:center;
            background:rgba(255,255,255,0.1);border-radius:8px;
            padding:6px 12px;margin-bottom:5px;font-size:0.8rem;">
    <span>{model_name}</span>
    <span style="font-weight:700;color:{color};">{map_val}</span>
</div>""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Model loading (cached, silent)
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_yolo(weights: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return YOLOWrapper(model=weights)

@st.cache_resource(show_spinner=False)
def load_classical():
    return ClassicalDetector()

# ──────────────────────────────────────────────────────────────────────────────
# Input section
# ──────────────────────────────────────────────────────────────────────────────

SAMPLE_DIR   = Path(__file__).resolve().parent / "sample_media"
sample_files = sorted(SAMPLE_DIR.glob("*.jpg"))

CLASS_NAMES = {0: "Prohibitory", 1: "Danger", 2: "Mandatory"}
CLASS_ICONS = {0: "🔴", 1: "🟡", 2: "🔵"}
CLASS_COLORS = {0: "#D72638", 1: "#F5A623", 2: "#1565C0"}

img_bgr       = None
uploaded_video = None

tab_upload, tab_sample, tab_video = st.tabs([
    "  📤  Upload Image  ",
    "  🖼️  Sample Images  ",
    "  🎬  Upload Video  ",
])

with tab_upload:
    st.markdown("<br>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drag and drop a traffic scene image here, or click to browse",
        type=["jpg", "jpeg", "png", "ppm"],
        label_visibility="visible",
    )
    if uploaded_file is not None:
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

with tab_sample:
    st.markdown("<br>", unsafe_allow_html=True)
    if sample_files:
        cols = st.columns(len(sample_files))
        for i, (col, f) in enumerate(zip(cols, sample_files)):
            with col:
                thumb = Image.open(f).resize((160, 120))
                st.image(thumb, use_column_width=True)
                st.markdown('<div class="sample-btn">', unsafe_allow_html=True)
                if st.button(f.stem, key=f"sample_{i}", use_container_width=True):
                    img_bgr = cv2.imread(str(f))
                st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("No sample images found. Add `.jpg` files to `demo/sample_media/`.")

with tab_video:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
<div style="background:#EEF2F8;border-radius:12px;padding:14px 20px;
            border-left:4px solid #1565C0;margin-bottom:16px;font-size:0.88rem;color:#1A1F36;">
    <strong>How it works:</strong> Upload a short video clip — each frame is processed by the 
    selected YOLO model. An annotated video is generated for playback and download.
    For best performance, use clips under 10 seconds.
</div>
""", unsafe_allow_html=True)

    uploaded_video = st.file_uploader(
        "Drag and drop a video file here, or click to browse",
        type=["mp4", "avi", "mov"],
        label_visibility="visible",
    )

    if uploaded_video is not None:
        selected_vid = MODEL_OPTIONS[model_choice]
        if selected_vid in ("classical", "compare"):
            st.warning("⚠️ Video processing requires a YOLO model. Please select a YOLO model from the sidebar.")
        else:
            ckpt_path = Path(selected_vid)
            if not ckpt_path.exists():
                st.warning(f"⚠️ Model weights not found for **{model_choice}**. Please train the model first.")
            else:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
                    tmp_in.write(uploaded_video.read())
                    tmp_in_path = tmp_in.name

                cap          = cv2.VideoCapture(tmp_in_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
                width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                c1, c2, c3 = st.columns(3)
                c1.metric("Frames", f"{total_frames}")
                c2.metric("Frame Rate", f"{fps:.1f} fps")
                c3.metric("Resolution", f"{width}×{height}")

                if total_frames > 300:
                    st.warning(f"⚠️ Long video ({total_frames} frames ≈ {total_frames/fps:.0f}s). Consider trimming to under 10s for faster processing.")

                if st.button("▶️  Process Video", type="primary", use_container_width=False):
                    with st.spinner("Loading model…"):
                        wrapper = load_yolo(str(ckpt_path))

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_out:
                        tmp_out_path = tmp_out.name

                    fourcc = cv2.VideoWriter_fourcc(*"avc1")
                    writer = cv2.VideoWriter(tmp_out_path, fourcc, fps, (width, height))
                    if not writer.isOpened():
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(tmp_out_path, fourcc, fps, (width, height))

                    progress_bar     = st.progress(0)
                    status_text      = st.empty()
                    frame_idx        = 0
                    sign_count_total = 0
                    preview_frame    = None

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            results   = wrapper.predict(source=frame, conf=conf_threshold,
                                                        iou=iou_threshold, verbose=False)
                        annotated = results[0].plot()
                        writer.write(annotated)

                        n = len(results[0].boxes) if results[0].boxes is not None else 0
                        sign_count_total += n

                        if frame_idx % max(total_frames // 5, 1) == 0:
                            preview_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

                        frame_idx += 1
                        if frame_idx % 5 == 0 or frame_idx == total_frames:
                            pct = min(frame_idx / max(total_frames, 1), 1.0)
                            progress_bar.progress(pct)
                            status_text.markdown(
                                f"⚡ Processing frame **{frame_idx}** of {total_frames}"
                                f"&nbsp;&nbsp;|&nbsp;&nbsp;"
                                f"Signs detected: **{sign_count_total}**"
                            )

                    cap.release()
                    writer.release()
                    progress_bar.empty()
                    status_text.empty()

                    st.markdown(f"""
<div style="background:#E8F5E9;border-radius:12px;padding:14px 20px;
            border-left:4px solid #2E7D32;margin:12px 0;font-size:0.9rem;">
    ✅ &nbsp;<strong>Processing complete!</strong> &nbsp;
    {frame_idx} frames processed &nbsp;|&nbsp; 
    {sign_count_total} total detections across the video
</div>
""", unsafe_allow_html=True)

                    if preview_frame is not None:
                        st.markdown("**Sample annotated frame:**")
                        st.image(preview_frame, use_column_width=True)

                    st.markdown("**Annotated Video Preview:**")
                    with open(tmp_out_path, "rb") as vf:
                        video_bytes = vf.read()
                    st.video(video_bytes)

                    st.download_button(
                        "⬇️  Download Annotated Video",
                        data=video_bytes,
                        file_name="annotated_traffic_video.mp4",
                        mime="video/mp4",
                        use_container_width=False,
                    )

# ──────────────────────────────────────────────────────────────────────────────
# Detection results
# ──────────────────────────────────────────────────────────────────────────────

if img_bgr is not None:
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    selected = MODEL_OPTIONS[model_choice]

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── SIDE-BY-SIDE COMPARISON ────────────────────────────────────
    if selected == "compare":
        st.markdown("### 🆚 Side-by-Side Comparison")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('<div class="detect-card"><h4>📷 Original Image</h4>', unsafe_allow_html=True)
            st.image(img_rgb, use_column_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="detect-card"><h4>🔬 Classical CV Baseline</h4>', unsafe_allow_html=True)
            with st.spinner("Running classical detector…"):
                det        = load_classical()
                detections = det.detect(img_bgr)
                ann_cv     = det.visualize(img_bgr, detections)
                st.image(cv2.cvtColor(ann_cv, cv2.COLOR_BGR2RGB), use_column_width=True)
            st.metric("Signs Found", len(detections))
            st.markdown("</div>", unsafe_allow_html=True)

        with col3:
            ckpt = str(PROJECT_ROOT / "results/checkpoints/gtsdb_yolov8s_v1_best.pt")
            st.markdown('<div class="detect-card"><h4>🏆 Fine-Tuned YOLOv8s</h4>', unsafe_allow_html=True)
            if Path(ckpt).exists():
                with st.spinner("Running YOLO…"):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        wrapper = load_yolo(ckpt)
                        results = wrapper.predict(source=img_bgr, conf=conf_threshold,
                                                  iou=iou_threshold, verbose=False)
                    ann_yolo = results[0].plot()
                    st.image(cv2.cvtColor(ann_yolo, cv2.COLOR_BGR2RGB), use_column_width=True)
                    n = len(results[0].boxes) if results[0].boxes is not None else 0
                    st.metric("Signs Found", n)
            else:
                st.info("YOLOv8s weights not found. Train the model first.")
            st.markdown("</div>", unsafe_allow_html=True)

    # ── CLASSICAL CV MODE ──────────────────────────────────────────
    elif selected == "classical":
        st.markdown("### 🔬 Classical CV Detection")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="detect-card"><h4>📷 Input Image</h4>', unsafe_allow_html=True)
            st.image(img_rgb, use_column_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="detect-card"><h4>🔍 Detection Result</h4>', unsafe_allow_html=True)
            with st.spinner("Running classical detector…"):
                det        = load_classical()
                detections = det.detect(img_bgr)
                ann        = det.visualize(img_bgr, detections)
                st.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB), use_column_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        if detections:
            st.markdown(f"""
<div style="background:#E8F5E9;border-radius:12px;padding:12px 20px;
            border-left:4px solid #2E7D32;margin:12px 0;font-size:0.9rem;">
    ✅ &nbsp;<strong>{len(detections)} region(s) detected</strong>
</div>
""", unsafe_allow_html=True)
            rows = [
                {"#": i + 1, "Bounding Box (x1, y1, x2, y2)": f"({d[0]}, {d[1]}, {d[2]}, {d[3]})"}
                for i, d in enumerate(detections)
            ]
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.info("No regions detected. Try lowering the Confidence Threshold in the sidebar.")

    # ── YOLO IMAGE MODE ────────────────────────────────────────────
    else:
        ckpt_path = Path(selected)

        if not ckpt_path.exists():
            st.markdown(f"""
<div style="background:#FFF3E0;border-radius:12px;padding:14px 20px;
            border-left:4px solid #F57C00;margin:12px 0;font-size:0.9rem;">
    ⚠️ &nbsp;<strong>Model weights not found</strong> for <em>{model_choice}</em>.
    The model needs to be trained first before it can be used.
</div>
""", unsafe_allow_html=True)
            st.image(img_rgb, use_column_width=True, caption="Input image (no detection run)")
        else:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown('<div class="detect-card"><h4>📷 Input Image</h4>', unsafe_allow_html=True)
                st.image(img_rgb, use_column_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="detect-card"><h4>🔍 Detection Result</h4>', unsafe_allow_html=True)
                with st.spinner("Running detection…"):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        wrapper       = load_yolo(str(ckpt_path))
                        results       = wrapper.predict(
                            source=img_bgr, conf=conf_threshold,
                            iou=iou_threshold, verbose=False
                        )
                    annotated_bgr = results[0].plot()
                    st.image(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB), use_column_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                # Summary strip
                st.markdown(f"""
<div style="background:#E8F0FE;border-radius:12px;padding:12px 20px;
            border-left:4px solid #1565C0;margin:12px 0;font-size:0.9rem;">
    🎯 &nbsp;<strong>{len(boxes)} sign(s) detected</strong>
    &nbsp;—&nbsp; using <em>{model_choice}</em>
    &nbsp;|&nbsp; confidence threshold: <strong>{conf_threshold:.0%}</strong>
</div>
""", unsafe_allow_html=True)

                # Individual detection cards
                st.markdown("**Detections:**")
                for i, b in enumerate(boxes):
                    cls_id     = int(b.cls[0])
                    conf_score = float(b.conf[0])
                    x1, y1, x2, y2 = [int(v) for v in b.xyxy[0]]
                    name  = CLASS_NAMES.get(cls_id, f"Class {cls_id}")
                    icon  = CLASS_ICONS.get(cls_id, "⬜")
                    color = CLASS_COLORS.get(cls_id, "#5E6E87")
                    st.markdown(f"""
<div class="result-row">
    <span class="cls">{icon} &nbsp;{name}</span>
    <span style="font-size:0.8rem;color:#8A9BB5;margin:0 auto 0 16px;">
        ({x1}, {y1}) → ({x2}, {y2})
    </span>
    <span class="conf">{conf_score:.1%}</span>
</div>
""", unsafe_allow_html=True)

                # Download button
                annotated_pil = Image.fromarray(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB))
                buf = io.BytesIO()
                annotated_pil.save(buf, format="PNG")
                st.download_button(
                    "⬇️  Download Annotated Image",
                    data=buf.getvalue(),
                    file_name="traffic_sign_detection.png",
                    mime="image/png",
                )
            else:
                st.info("No signs detected in this image. Try lowering the Confidence Threshold in the sidebar.")

elif uploaded_video is None:
    # Landing prompt
    st.markdown("""
<div style="background:#FFFFFF;border-radius:16px;padding:32px;text-align:center;
            border:1px dashed #C5D3E8;margin-top:8px;">
    <div style="font-size:3rem;margin-bottom:12px;">🚦</div>
    <p style="font-size:1.05rem;font-weight:600;color:#0A295C;margin:0 0 6px 0;">
        Ready to detect traffic signs
    </p>
    <p style="font-size:0.88rem;color:#8A9BB5;margin:0;">
        Upload an image, select a sample above, or upload a video to get started.
    </p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="footer">
    CO543/CO5430 — Traffic Sign Detection &nbsp;|&nbsp; Group 17 &nbsp;|&nbsp; 
    University of Peradeniya &nbsp;|&nbsp; 2026 &nbsp;|&nbsp;
    <a href="https://github.com/SajithK203/Traffic-Sign-Detection" 
       style="color:#1565C0;text-decoration:none;">GitHub Repository</a>
</div>
""", unsafe_allow_html=True)
