"""
app.py  —  Streamlit demo for traffic sign detection.

Run:
    streamlit run demo/app.py

Features:
  - Upload an image → detect traffic signs and display annotated result
  - Choose between Classical CV baseline or Fine-Tuned YOLO
  - Show confidence scores and class names
  - Download annotated image
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.classical_detector import ClassicalDetector
from src.models.yolo_wrapper import YOLOWrapper


# ------------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------------

st.set_page_config(
    page_title="🚦 Traffic Sign Detector",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🚦 Traffic Sign Detection")
st.caption("CO543/CO5430 — Computer Vision Project Demo")

# ------------------------------------------------------------------
# Sidebar — model selection & settings
# ------------------------------------------------------------------

st.sidebar.header("⚙️ Settings")

model_choice = st.sidebar.radio(
    "Detection Model",
    options=["Fine-Tuned YOLOv8 (Recommended)", "Classical CV Baseline (HSV + Contour)"],
)

conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.25, 0.05)
iou_threshold  = st.sidebar.slider("NMS IoU Threshold",    0.1, 0.9, 0.45, 0.05)

yolo_weights_path = st.sidebar.text_input(
    "YOLO Checkpoint Path",
    value="results/checkpoints/best.pt",
    help="Path to your fine-tuned YOLO .pt checkpoint.",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Class Labels**\n"
    "- 🔴 Prohibitory (red)\n"
    "- 🔵 Mandatory (blue)\n"
    "- 🟡 Danger/Warning (yellow)"
)

# ------------------------------------------------------------------
# Model loading (cached)
# ------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model...")
def load_yolo(weights: str):
    return YOLOWrapper(model=weights)

@st.cache_resource(show_spinner="Loading classical detector...")
def load_classical():
    return ClassicalDetector()

# ------------------------------------------------------------------
# Image upload & inference
# ------------------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a traffic scene image",
    type=["jpg", "jpeg", "png", "ppm"],
)

if uploaded_file is not None:
    # Decode uploaded image
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Input Image")
        st.image(img_rgb, use_column_width=True)

    with col2:
        st.subheader("🔍 Detection Result")

        with st.spinner("Running detection..."):

            if "Classical" in model_choice:
                detector   = load_classical()
                detections = detector.detect(img_bgr)
                annotated  = detector.visualize(img_bgr, detections)
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                st.image(annotated_rgb, use_column_width=True)
                st.success(f"Found **{len(detections)}** candidate region(s)")

                if detections:
                    class_names = {0: "Prohibitory", 1: "Mandatory", 2: "Danger"}
                    rows = [
                        {"Class": class_names.get(d[4], str(d[4])),
                         "BBox (x1,y1,x2,y2)": f"({d[0]},{d[1]},{d[2]},{d[3]})"}
                        for d in detections
                    ]
                    st.dataframe(rows, use_container_width=True)

            else:  # YOLO
                weights_path = str(PROJECT_ROOT / yolo_weights_path)
                if not Path(weights_path).exists():
                    st.error(
                        f"Checkpoint not found at `{weights_path}`. "
                        "Please train the model first (`python src/train.py`) "
                        "or update the path in the sidebar."
                    )
                else:
                    wrapper = load_yolo(weights_path)
                    results = wrapper.predict(
                        source=img_bgr, conf=conf_threshold, iou=iou_threshold
                    )
                    annotated_bgr = results[0].plot()
                    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                    st.image(annotated_rgb, use_column_width=True)

                    boxes = results[0].boxes
                    n = len(boxes) if boxes is not None else 0
                    st.success(f"Found **{n}** sign(s)")

        # Download button
        annotated_pil = Image.fromarray(annotated_rgb)
        import io
        buf = io.BytesIO()
        annotated_pil.save(buf, format="PNG")
        st.download_button(
            "⬇️ Download Annotated Image",
            data=buf.getvalue(),
            file_name="detection_result.png",
            mime="image/png",
        )

else:
    st.info("👆 Upload an image to get started.")
    st.markdown(
        """
        **How to use:**
        1. Select a detection model in the sidebar
        2. Upload a traffic scene image (JPG, PNG)
        3. View detection results and download the annotated image

        **Note**: The Fine-Tuned YOLOv8 model requires a trained checkpoint.
        Run `python src/train.py --config configs/gtsdb_yolov8n.yaml` first.
        """
    )

# ------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------
st.markdown("---")
st.caption("CO543/CO5430 — Traffic Sign Detection | Computer Vision Project 2026")
