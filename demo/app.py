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
import sys
import tempfile
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
st.caption("CO543/CO5430 — Computer Vision Project Demo | Group 17")

# ------------------------------------------------------------------
# Sidebar — model selection & settings
# ------------------------------------------------------------------

st.sidebar.header("⚙️ Settings")

MODEL_OPTIONS = {
    "🏆 Fine-Tuned YOLOv8s (Best — 97.1% mAP)": str(PROJECT_ROOT / "results/checkpoints/gtsdb_yolov8s_v1_best.pt"),
    "🔵 Fine-Tuned YOLOv8n (95.5% mAP)":        str(PROJECT_ROOT / "results/checkpoints/gtsdb_yolov8n_v1_best.pt"),
    "⚗️ Fine-Tuned YOLOv8n No Aug (84.8% mAP)": str(PROJECT_ROOT / "results/checkpoints/gtsdb_yolov8n_noaug_v1_best.pt"),
    "🔬 Classical CV Baseline (HSV + Contour)":  "classical",
    "🆚 Side-by-Side: Classical vs YOLOv8s":     "compare",
}

model_choice = st.sidebar.selectbox(
    "Detection Model",
    options=list(MODEL_OPTIONS.keys()),
)

conf_threshold = st.sidebar.slider("Confidence Threshold", 0.05, 0.9, 0.20, 0.05)
iou_threshold  = st.sidebar.slider("NMS IoU Threshold",    0.1,  0.9, 0.45, 0.05)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Class Labels**\n"
    "- 🔴 Prohibitory (red circle signs)\n"
    "- 🟡 Danger (red triangle signs)\n"
    "- 🔵 Mandatory (blue circle signs)"
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Model Performance**\n"
    "| Model | mAP@0.5 |\n"
    "|---|---|\n"
    "| Classical CV | — |\n"
    "| YOLOv8n | 95.5% |\n"
    "| YOLOv8s | **97.1%** |"
)

# ------------------------------------------------------------------
# Model loading (cached)
# ------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading YOLO model...")
def load_yolo(weights: str):
    return YOLOWrapper(model=weights)

@st.cache_resource(show_spinner="Loading classical detector...")
def load_classical():
    return ClassicalDetector()

# ------------------------------------------------------------------
# Input — Image or Video tabs
# ------------------------------------------------------------------

SAMPLE_DIR  = Path(__file__).resolve().parent / "sample_media"
sample_files = sorted(SAMPLE_DIR.glob("*.jpg"))

CLASS_NAMES = {0: "Prohibitory 🔴", 1: "Danger 🟡", 2: "Mandatory 🔵"}

st.markdown("### 📂 Load an Image or Video")
tab_upload, tab_sample, tab_video = st.tabs(["📤 Upload Image", "🖼️ Sample Image", "🎬 Upload Video"])

img_bgr    = None
video_path = None

# ── Image upload ──────────────────────────────────────────────────
with tab_upload:
    uploaded_file = st.file_uploader(
        "Upload a traffic scene image",
        type=["jpg", "jpeg", "png", "ppm"],
        label_visibility="collapsed",
    )
    if uploaded_file is not None:
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

# ── Sample image buttons ──────────────────────────────────────────
with tab_sample:
    if sample_files:
        cols = st.columns(len(sample_files))
        for i, (col, f) in enumerate(zip(cols, sample_files)):
            thumb = Image.open(f).resize((160, 120))
            col.image(thumb, caption=f.stem, use_column_width=True)
            if col.button(f"Load {f.stem}", key=f"sample_{i}"):
                img_bgr = cv2.imread(str(f))
    else:
        st.info("No sample images found in `demo/sample_media/`. Add some `.jpg` files there.")

# ── Video upload ──────────────────────────────────────────────────
with tab_video:
    st.info(
        "Upload a short video clip (MP4 / AVI / MOV). "
        "Each frame will be processed by the YOLO model and an annotated video will be generated for download."
    )
    uploaded_video = st.file_uploader(
        "Upload a traffic scene video",
        type=["mp4", "avi", "mov"],
        label_visibility="collapsed",
    )

    if uploaded_video is not None:
        selected = MODEL_OPTIONS[model_choice]
        if selected in ("classical", "compare"):
            st.warning("Video mode only supports YOLO models. Please select a YOLO model from the sidebar.")
        else:
            ckpt_path = Path(selected)
            if not ckpt_path.exists():
                st.error(f"Checkpoint not found at `{ckpt_path}`. Run the training notebook first.")
            else:
                # Save uploaded video to a temp file so OpenCV can read it
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
                    tmp_in.write(uploaded_video.read())
                    tmp_in_path = tmp_in.name

                cap = cv2.VideoCapture(tmp_in_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
                width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                st.markdown(
                    f"**Video info:** {total_frames} frames · {fps:.1f} fps · {width}×{height} px"
                )

                # Warn if video is very long
                if total_frames > 300:
                    st.warning(
                        f"⚠️ This video has {total_frames} frames (~{total_frames/fps:.0f}s). "
                        "Processing may take a while. Consider trimming to under 10 seconds for the demo."
                    )

                if st.button("▶️ Process Video", type="primary"):
                    wrapper = load_yolo(str(ckpt_path))

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_out:
                        tmp_out_path = tmp_out.name

                    # Try H.264 first (best browser compatibility), fall back to mp4v
                    fourcc = cv2.VideoWriter_fourcc(*"avc1")
                    writer = cv2.VideoWriter(tmp_out_path, fourcc, fps, (width, height))
                    if not writer.isOpened():
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(tmp_out_path, fourcc, fps, (width, height))

                    progress_bar     = st.progress(0, text="Processing frames…")
                    status_text      = st.empty()
                    frame_idx        = 0
                    sign_count_total = 0
                    preview_frame    = None

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break

                        results   = wrapper.predict(source=frame, conf=conf_threshold,
                                                    iou=iou_threshold, verbose=False)
                        annotated = results[0].plot()
                        writer.write(annotated)

                        n = len(results[0].boxes) if results[0].boxes is not None else 0
                        sign_count_total += n

                        # Store last annotated frame for preview after processing
                        if frame_idx % max(total_frames // 5, 1) == 0:
                            preview_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

                        frame_idx += 1
                        # Update progress bar only every 5 frames to avoid UI slowdown
                        if frame_idx % 5 == 0 or frame_idx == total_frames:
                            progress_bar.progress(
                                min(frame_idx / max(total_frames, 1), 1.0),
                                text=f"Processing frame {frame_idx}/{total_frames}…"
                            )
                            status_text.markdown(
                                f"⚡ Processed **{frame_idx}** / {total_frames} frames  |  "
                                f"Signs detected so far: **{sign_count_total}**"
                            )

                    cap.release()
                    writer.release()
                    progress_bar.progress(1.0, text="✅ Processing complete!")
                    status_text.empty()


                    st.success(
                        f"✅ Done! Processed **{frame_idx}** frames — "
                        f"**{sign_count_total}** sign detections across the video."
                    )

                    # Show a sample annotated frame
                    if preview_frame is not None:
                        st.markdown("**Sample annotated frame:**")
                        st.image(preview_frame, use_column_width=True)

                    # Play the video inline in the browser
                    st.markdown("**▶️ Annotated Video Preview:**")
                    with open(tmp_out_path, "rb") as vf:
                        video_bytes = vf.read()
                    st.video(video_bytes)

                    # Also offer a download button
                    st.download_button(
                        "⬇️ Download Annotated Video",
                        data=video_bytes,
                        file_name="annotated_traffic_video.mp4",
                        mime="video/mp4",
                    )

# ------------------------------------------------------------------
# Run detection when an image is loaded
# ------------------------------------------------------------------

if img_bgr is not None:
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    selected = MODEL_OPTIONS[model_choice]

    st.markdown("---")

    # ── SIDE-BY-SIDE COMPARISON MODE ──────────────────────────────
    if selected == "compare":
        st.subheader("🆚 Side-by-Side: Classical CV vs Fine-Tuned YOLOv8s")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**📷 Original**")
            st.image(img_rgb, use_column_width=True)

        with col2:
            st.markdown("**🔬 Classical CV Baseline**")
            with st.spinner("Running classical detector..."):
                det        = load_classical()
                detections = det.detect(img_bgr)
                annotated_classical = det.visualize(img_bgr, detections)
                st.image(cv2.cvtColor(annotated_classical, cv2.COLOR_BGR2RGB), use_column_width=True)
                st.metric("Signs Found", len(detections))

        with col3:
            ckpt = str(PROJECT_ROOT / "results/checkpoints/gtsdb_yolov8s_v1_best.pt")
            st.markdown("**🏆 Fine-Tuned YOLOv8s**")
            with st.spinner("Running YOLO..."):
                wrapper = load_yolo(ckpt)
                results = wrapper.predict(source=img_bgr, conf=conf_threshold, iou=iou_threshold)
                annotated_yolo = results[0].plot()
                st.image(cv2.cvtColor(annotated_yolo, cv2.COLOR_BGR2RGB), use_column_width=True)
                n = len(results[0].boxes) if results[0].boxes is not None else 0
                st.metric("Signs Found", n)

    # ── CLASSICAL CV MODE ──────────────────────────────────────────
    elif selected == "classical":
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📷 Input Image")
            st.image(img_rgb, use_column_width=True)
        with col2:
            st.subheader("🔍 Classical CV Detection")
            with st.spinner("Running classical detector..."):
                det        = load_classical()
                detections = det.detect(img_bgr)
                annotated  = det.visualize(img_bgr, detections)
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_column_width=True)

            if detections:
                st.success(f"Found **{len(detections)}** candidate region(s)")
                rows = [
                    {"Region": i + 1, "BBox (x1,y1,x2,y2)": f"({d[0]}, {d[1]}, {d[2]}, {d[3]})"}
                    for i, d in enumerate(detections)
                ]
                st.dataframe(rows, use_container_width=True)
            else:
                st.warning("No signs detected. Try lowering the Confidence Threshold.")

    # ── YOLO IMAGE MODE ───────────────────────────────────────────
    else:
        ckpt_path = Path(selected)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📷 Input Image")
            st.image(img_rgb, use_column_width=True)
        with col2:
            st.subheader("🔍 YOLO Detection Result")
            if not ckpt_path.exists():
                st.error(f"Checkpoint not found at `{ckpt_path}`. Run the training notebook first.")
            else:
                with st.spinner("Running YOLO detection..."):
                    wrapper      = load_yolo(str(ckpt_path))
                    results      = wrapper.predict(source=img_bgr, conf=conf_threshold, iou=iou_threshold)
                    annotated_bgr = results[0].plot()
                    st.image(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB), use_column_width=True)

                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    st.success(f"Found **{len(boxes)}** sign(s)")
                    rows = []
                    for b in boxes:
                        cls_id     = int(b.cls[0])
                        conf_score = float(b.conf[0])
                        x1, y1, x2, y2 = [int(v) for v in b.xyxy[0]]
                        rows.append({
                            "Class": CLASS_NAMES.get(cls_id, str(cls_id)),
                            "Confidence": f"{conf_score:.1%}",
                            "BBox (x1,y1,x2,y2)": f"({x1}, {y1}, {x2}, {y2})",
                        })
                    st.dataframe(rows, use_container_width=True)
                else:
                    st.warning("No signs detected. Try lowering the Confidence Threshold slider.")

                # Download annotated image
                annotated_pil = Image.fromarray(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB))
                buf = io.BytesIO()
                annotated_pil.save(buf, format="PNG")
                st.download_button(
                    "⬇️ Download Annotated Image",
                    data=buf.getvalue(),
                    file_name="detection_result.png",
                    mime="image/png",
                )

elif video_path is None and uploaded_video is None:
    st.info("👆 Select a tab above to upload an image, choose a sample, or upload a video.")

# ------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------
st.markdown("---")
st.caption("CO543/CO5430 — Traffic Sign Detection | Group 17 | University of Peradeniya 2026")
