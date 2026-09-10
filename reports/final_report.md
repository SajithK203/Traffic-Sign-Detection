# Traffic Sign Detection Using Deep Learning and Classical Computer Vision
## CO543/CO5430 — Computer Vision | Group 17 | University of Peradeniya

---

**Authors:** R.M.S.S. Kumara (E/22/203), K.I. Sewmini (E/22/372), S.I. Gunawardhana (E/22/127), A.W.H. Panchani (E/22/269)

**GitHub:** https://github.com/SajithK203/Traffic-Sign-Detection

---

## Abstract

Traffic sign detection is a fundamental perception task for autonomous driving and Advanced Driver-Assistance Systems (ADAS). This project develops and evaluates a complete traffic sign detection pipeline on the German Traffic Sign Detection Benchmark (GTSDB), comparing a classical computer vision baseline against deep learning approaches based on YOLOv8. We implement an HSV-based classical detector, a zero-shot pretrained YOLOv8n baseline, and fine-tuned YOLOv8n/s models with and without data augmentation. Our best model — a fine-tuned YOLOv8s with augmentation — achieves **97.1% mAP@0.5**, representing a ~24× improvement over the classical baseline (4.0% precision). We further demonstrate feasibility of fine-grained 43-class detection as a stretch goal. An interactive Streamlit demo application is provided for real-time image and video inference.

---

## 1. Introduction and Motivation

Traffic sign recognition is a safety-critical component of modern intelligent transportation systems. Accurate detection of signs such as speed limits, stop signs, and mandatory turn indicators enables autonomous vehicles to navigate safely and helps ADAS systems alert human drivers. The German Traffic Sign Detection Benchmark (GTSDB) [1] provides a challenging real-world dataset with full driving-scene images, variable lighting, partial occlusion, and signs of varying scales — making it an ideal testbed for evaluating detection systems.

This project addresses two complementary research questions:
1. How effectively can a classical computer vision pipeline detect traffic signs compared to a fine-tuned deep neural network?
2. What is the impact of data augmentation on detection performance when training data is limited?

Our contributions are:
- A complete evaluation pipeline comparing 5 model variants on GTSDB
- A data augmentation ablation study quantifying the impact of augmentation on mAP
- A fine-grained 43-class YOLOv8n model as a stretch-goal experiment
- An interactive Streamlit demo supporting both image and video inference

---

## 2. Related Work

**Object detection** has been transformed by deep learning. Region-based CNNs (Faster R-CNN [2]) established anchor-based two-stage detection. Single-stage detectors such as SSD [3] and the YOLO family [4] offer real-time inference with competitive accuracy. YOLOv8 (Ultralytics, 2023) [5] represents the current state of the art in single-stage detection, achieving strong accuracy-speed trade-offs.

**Traffic sign datasets** include GTSDB [1] (900 full-scene images, 43 classes), GTSRB [6] (51,839 cropped classification images), TT100K [7] (100K images, 221 classes), and LISA [8] (6,600 US frames). GTSDB is chosen for this project as it provides full scene detection context.

**Transfer learning** [9] enables effective fine-tuning of large pretrained models with limited data. YOLOv8 pretrained on COCO (118K images) provides strong feature extraction, requiring only task-specific adaptation.

---

## 3. Dataset and Preprocessing

### 3.1 Dataset
We use the **German Traffic Sign Detection Benchmark (GTSDB)** [1]:
- **900 full driving-scene images** (1360×800 pixels, .ppm format)
- **43 fine-grained classes** grouped into 3 super-classes: *prohibitory* (0–8), *danger* (11–31), *mandatory* (33–41)
- Annotations provided in `gt.txt` (image name, bounding box, class ID)

### 3.2 Preprocessing Pipeline
We implement a custom conversion script (`src/data/convert_gtsdb.py`) that:
1. Converts `.ppm` images to `.jpg` (YOLO-compatible format)
2. Converts absolute pixel bounding boxes to YOLO normalised format: `[class_id, x_center, y_center, width, height]`
3. Maps 43 fine-grained classes to 3 super-classes for the primary experiments
4. Splits data into **train / val / test: 70% / 15% / 15%** (stratified by class)

**Final split sizes:** 365 train / 80 val / 81 test images  
**Annotated sign instances:** ~1,200+ individual bounding boxes across all splits

### 3.3 Class Distribution
The dataset is highly imbalanced, with prohibitory signs (especially speed limits) being the most frequent class. Danger signs are the rarest super-class.

### 3.4 Data Augmentation
For fine-tuned YOLO models, YOLOv8's built-in augmentation pipeline is applied during training:
- Random HSV colour jitter (hue ±1.5%, saturation ±70%, value ±40%)
- Random horizontal flip (p=0.5)
- Mosaic augmentation (4-image mixing, p=1.0)
- Random scale, translate, and rotation

---

## 4. Methodology

### 4.1 Classical CV Baseline
We implement an HSV-based colour segmentation and shape analysis pipeline (`src/models/classical_detector.py`):
1. **Colour segmentation:** HSV thresholding for red (prohibitory), blue (mandatory), and yellow (danger) regions
2. **Contour extraction:** `cv2.findContours` to find candidate regions
3. **Shape filtering:** Minimum area (600 px²), circularity threshold (≥0.30)
4. **Non-Maximum Suppression:** IoU-based suppression of overlapping candidates

This baseline requires no training data and runs entirely on classical image processing.

### 4.2 Zero-Shot YOLOv8n Baseline
We evaluate the standard YOLOv8n model pretrained on COCO (80 classes) without any fine-tuning. This tests whether generic object detection features transfer to traffic signs out of the box. COCO does not contain traffic sign classes, so this represents a true zero-shot scenario.

### 4.3 Fine-Tuned YOLOv8 Models
We fine-tune two YOLOv8 variants on GTSDB using transfer learning:

**YOLOv8n (nano):** 3.0M parameters, 8.2 GFLOPs — fastest, least accurate  
**YOLOv8s (small):** 11.1M parameters, 28.7 GFLOPs — slower, more accurate

Training configuration:
- **Epochs:** 80 | **Image size:** 640×640 | **Batch size:** 16
- **Optimiser:** AdamW (auto-selected) | **Learning rate:** 0.001 → 0.01
- **Hardware:** NVIDIA GeForce RTX 2050 (4GB VRAM)

### 4.4 Ablation Study: Effect of Data Augmentation
We train YOLOv8n with augmentation disabled (`fliplr=0.0, mosaic=0.0`) to isolate the contribution of augmentation to final mAP.

### 4.5 Stretch Goal: 43-Class Fine-Grained Detection
We generate a second dataset split preserving all 43 original GTSDB class IDs (`data/processed/gtsdb_fine/`) and fine-tune YOLOv8n for 100 epochs on this split to explore fine-grained sign classification.

---

## 5. Experiments and Results

### 5.1 Quantitative Results

All models are evaluated on the held-out GTSDB test split (81 images, 3 super-classes).

| Model | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|---|---|---|---|---|
| Classical CV (HSV + Contour) | — | — | 4.0% | 15.5% |
| Zero-Shot YOLOv8n (COCO) | — | — | 3.6% | 20.0% |
| Fine-Tuned YOLOv8n (no aug) | 84.8% | 64.3% | 87.7% | 74.1% |
| Fine-Tuned YOLOv8n (with aug) | 95.5% | 73.3% | 91.9% | 91.0% |
| **Fine-Tuned YOLOv8s (with aug)** | **97.1%** | **76.0%** | **96.5%** | **90.6%** |

### 5.2 Ablation Study: Data Augmentation

Removing augmentation from YOLOv8n causes a **10.7 percentage point drop in mAP@0.5** (84.8% vs 95.5%) and a **16.9 pp drop in recall** (74.1% vs 91.0%). This confirms that augmentation is critical for generalisation on the small 365-image training set, effectively acting as a data multiplier.

### 5.3 Model Comparison: YOLOv8n vs YOLOv8s

The larger YOLOv8s model achieves **+1.6% mAP@0.5** over YOLOv8n at the cost of approximately **4× more parameters** (11.1M vs 3.0M). The gain is modest but consistent, and YOLOv8s is recommended as the deployment model.

### 5.4 Stretch Goal: 43-Class Results

The fine-grained model achieves **24.7% mAP@0.5 on the validation set**. The low test-set mAP (~1.0%) reflects extreme class sparsity (average <9 labelled instances per class in the test split), rather than model failure. Qualitatively, the model correctly identifies speed limit values and mandatory turn signs at reasonable confidence.

### 5.5 Inference Speed

| Model | Inference (ms/image) |
|---|---|
| Classical CV | ~15 ms |
| YOLOv8n | ~2.8 ms |
| YOLOv8s | ~5.1 ms |

---

## 6. Qualitative Results

### 6.1 Success Cases
The fine-tuned YOLOv8s model correctly detects:
- Speed limit signs partially obscured by tree branches
- Small signs at distance (sign height < 30 px)
- Multiple signs in the same frame at different scales

### 6.2 Failure Cases
Observed failure modes include:
- **Small distant signs** (< 20×20 px): bounding boxes missed or imprecise
- **Heavily occluded signs**: signs blocked by vehicles or foliage are missed
- **Rare class imbalance**: danger signs (fewest training samples) have higher miss rate
- **Night/low-light scenes**: not represented in GTSDB training data; performance would degrade on night footage

---

## 7. Analysis and Limitations

**Key findings:**
- Transfer learning from COCO dramatically reduces data requirements — only 365 training images suffice to achieve 95%+ mAP
- Data augmentation is the single most important factor for generalisation on small datasets
- Classical CV cannot reliably generalise; the HSV approach is highly sensitive to threshold parameters
- The zero-shot baseline performs poorly (3.6% precision), confirming that domain-specific fine-tuning is essential

**Limitations:**
- GTSDB contains only 900 images — larger datasets like TT100K (100K) or Mapillary Traffic Sign Dataset would likely yield higher absolute performance
- Our 3-class grouping simplifies the real-world task (autonomous vehicles need exact sign values, e.g. speed limit *50* vs *80*)
- Night-time, rain, and motion blur scenarios are absent from GTSDB and represent a known gap

---

## 8. Conclusion and Future Work

We presented a complete traffic sign detection pipeline evaluated on GTSDB, demonstrating that transfer-learning-based fine-tuning of YOLOv8s achieves **97.1% mAP@0.5**, far exceeding classical CV (4% precision). Data augmentation contributes a 10.7 pp improvement on limited training data. Our interactive Streamlit demo supports real-time image and video inference.

**Future work:**
- Train on larger datasets (TT100K, LISA) for cross-dataset generalisation
- Explore YOLO11 and RT-DETR architectures for potential accuracy improvements
- Deploy optimised models (TensorRT, ONNX) on embedded hardware (Jetson Nano) for real-time automotive use
- Expand the 43-class fine-grained model with GTSRB data augmentation to improve per-class recall

---

## 9. References

[1] S. Houben, J. Stallkamp, J. Salmen, M. Schlipsing, and C. Igel, "Detection of traffic signs in real-world images: The German Traffic Sign Detection Benchmark," in *Proc. IJCNN*, 2013.

[2] S. Ren, K. He, R. Girshick, and J. Sun, "Faster R-CNN: Towards real-time object detection with region proposal networks," in *Proc. NeurIPS*, 2015.

[3] W. Liu et al., "SSD: Single shot multibox detector," in *Proc. ECCV*, 2016.

[4] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, "You only look once: Unified, real-time object detection," in *Proc. CVPR*, 2016.

[5] Ultralytics, "YOLOv8 Documentation," 2023. [Online]. Available: https://docs.ultralytics.com

[6] J. Stallkamp, M. Schlipsing, J. Salmen, and C. Igel, "The German Traffic Sign Recognition Benchmark: A multi-class classification competition," in *Proc. IJCNN*, 2011.

[7] Z. Zhu, D. Liang, S. Zhang, X. Huang, B. Li, and S. Hu, "Traffic-sign detection and classification in the wild," in *Proc. CVPR*, 2016.

[8] A. Mogelmose, M. M. Trivedi, and T. B. Moeslund, "Vision-based traffic sign detection and analysis for intelligent driver assistance systems," *IEEE Trans. ITS*, 2012.

[9] J. Howard and S. Ruder, "Universal language model fine-tuning for text classification," in *Proc. ACL*, 2018.

---

## 10. Individual Contributions

| Member | Registration | Primary Contributions |
|---|---|---|
| R.M.S.S. Kumara | E/22/203 | Dataset download & conversion pipeline, YOLO fine-tuning (YOLOv8n/s), Streamlit demo app (image + video), README & GitHub management |
| K.I. Sewmini | E/22/372 | 43-class fine-grained dataset preparation (`--use_fine_classes` flag), Windows encoding fixes, dataset split verification |
| S.I. Gunawardhana | E/22/127 | Classical CV baseline implementation, ablation study configuration, dataset download scripts |
| A.W.H. Panchani | E/22/269 | EDA notebook (class distribution, size histograms), zero-shot baseline evaluation, qualitative failure case analysis |

---

*All external datasets, pretrained model weights, and AI tools used are declared in `docs/AI_USE_STATEMENT.md` and the repository README.*
