---
layout: home
permalink: index.html

# Repository configuration
repository-name: e22-co543-traffic-sign-detection
title: Real-Time Traffic Sign Detection and Recognition
---

# Real-Time Traffic Sign Detection and Recognition

## CO543 / CO5430 Computer Vision Project | Group 17 | University of Peradeniya

---

## 📌 Abstract & Overview

Autonomous vehicles and Advanced Driver Assistance Systems (ADAS) depend critically on reliable, real-time traffic sign detection. This project presents an end-to-end computer vision and deep learning solution evaluated on the **German Traffic Sign Detection Benchmark (GTSDB)**.

We compare three paradigms:
1. **Classical Computer Vision Baseline**: Multi-scale HSV color segmentation, morphological filtering, and contour analysis.
2. **Zero-shot Foundation Baseline**: Pretrained YOLOv8 detection.
3. **Fine-tuned Deep Learning (Ours)**: Custom-trained YOLOv8n and YOLOv8s models supporting both 4-macro-category and 43-fine-grained traffic sign classes with specialized small-object anchor strategies and Mosaic augmentation.

---

## 🎯 Key Objectives

- **High Detection Accuracy**: Detect small, obscured, or motion-blurred traffic signs across varied lighting.
- **Real-Time Inference**: Achieve > 80 FPS on GPU and > 15 FPS on CPU for edge/embedded deployment.
- **Interactive Demonstration**: A production-grade Streamlit web interface with real-time confidence tuning, IoU adjustment, and classical vs. deep learning side-by-side analysis.

---

## 🔬 Methodology

### 1. Classical Computer Vision Pipeline
- **Color Space Transformation**: RGB $\to$ HSV color thresholding isolating red (danger/prohibitory), blue (mandatory), and yellow (caution).
- **Morphology**: Opening and closing filters to remove noise and bridge contours.
- **Bounding Box Extraction**: Aspect-ratio filtering, circularity/triangularity matching.

### 2. Deep Learning Pipeline (YOLOv8)
- **Architecture**: YOLOv8 CSPDarknet backbone with Path Aggregation Network (PAN) neck and decoupled anchor-free detection head.
- **Augmentation**: Mosaic, HSV jitter, random scaling, translation, and horizontal flipping.
- **Optimization**: SGD with momentum (.937$), weight decay (.0005$), and cosine learning rate scheduling (=0.01 \to lr_f=0.0001$).

---

## 📊 Experimental Results

| Model | Classes | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | Inference (ms) | FPS |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Classical CV (HSV + Contours)** | 3 | 0.052 | 0.448 | 0.038 | 0.015 | 38.2 | 26.2 |
| **Pretrained YOLOv8 (Zero-Shot)** | COCO | 0.120 | 0.180 | 0.095 | 0.041 | 8.2 | 122.0 |
| **YOLOv8n (4 Macro Classes)** | 4 | **0.865** | **0.812** | **0.871** | **0.684** | **3.8** | **263.2** |
| **YOLOv8s (4 Macro Classes)** | 4 | **0.884** | **0.835** | **0.892** | **0.709** | **7.1** | **140.8** |
| **YOLOv8n (43 Fine-grained Classes)** | 43 | **0.640** | **0.315** | **0.247** | **0.198** | **3.9** | **256.4** |

---

## 💻 Interactive Demo

An interactive Streamlit application is included:
- Single-image and batch inference.
- Video and webcam stream processing.
- Interactive confidence and IoU threshold sliders.
- Side-by-side comparison between Classical CV and YOLOv8 models.

To run locally:
\\\ash
git clone https://github.com/cepdnaclk/e22-co543-traffic-sign-detection.git
cd e22-co543-traffic-sign-detection
pip install -r requirements.txt
streamlit run demo/app.py
\\\

---

## 👥 Team Members

| Name | Registration No. | Role & Contribution |
| :--- | :--- | :--- |
| **R.M.S.S. Kumara** | E/22/203 | Team Lead, YOLOv8 Fine-tuning & Training, App Development |
| **K.I. Sewmini** | E/22/372 | GTSDB Dataset Preprocessing & Augmentation Pipelines |
| **S.I. Gunawardhana** | E/22/127 | Classical Computer Vision Baseline Pipeline & Analysis |
| **A.W.H. Panchani** | E/22/269 | Evaluation Metrics, Qualitative Visualization & Reporting |

---

## 🔗 Project Links

- [Department Project Profile](https://projects.ce.pdn.ac.lk/)
- [Source Code Repository](https://github.com/cepdnaclk/e22-co543-traffic-sign-detection)
- [Department of Computer Engineering, University of Peradeniya](https://www.ce.pdn.ac.lk/)
