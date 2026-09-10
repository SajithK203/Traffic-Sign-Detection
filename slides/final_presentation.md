# Traffic Sign Detection — Final Presentation
## CO543/CO5430 Computer Vision | Group 17

---

## Slide 1 — Title

# 🚦 Traffic Sign Detection
### Using Deep Learning and Classical Computer Vision

**Group 17 | University of Peradeniya**

| Member | Reg. No. |
|---|---|
| R.M.S.S. Kumara | E/22/203 |
| K.I. Sewmini | E/22/372 |
| S.I. Gunawardhana | E/22/127 |
| A.W.H. Panchani | E/22/269 |

**GitHub:** https://github.com/SajithK203/Traffic-Sign-Detection

---

## Slide 2 — Problem Statement & Motivation

### 🎯 Problem
> Given a road image or video frame, detect **where** every traffic sign is and **what type** it is.

### Why does it matter?
- 🚗 **ADAS**: Warn drivers of speed limits, stop signs, and hazards
- 🤖 **Autonomous vehicles**: Must obey road rules in real time
- 🗺️ **Road mapping**: Automated sign inventory for municipalities

### Our approach
- Compare **Classical CV** vs **Deep Learning (YOLOv8)**
- Quantify impact of **data augmentation** (ablation study)
- Build a **live interactive demo** for real-time inference

---

## Slide 3 — Dataset & Preprocessing

### 📦 German Traffic Sign Detection Benchmark (GTSDB)

| Property | Value |
|---|---|
| Total images | 900 full driving-scene images |
| Resolution | 1360 × 800 pixels |
| Classes | 43 fine-grained → 3 super-classes |
| Split | 365 train / 80 val / 81 test (70/15/15%) |

### Super-class grouping
- 🔴 **Prohibitory** — Speed limits, no-overtaking (classes 0–8)
- 🟡 **Danger** — Bends, construction, pedestrians (classes 11–31)
- 🔵 **Mandatory** — Turn arrows, roundabout (classes 33–41)

### Preprocessing pipeline
1. `.ppm` → `.jpg` image conversion
2. Bounding box → YOLO normalised format
3. Stratified train/val/test split
4. Data augmentation (mosaic, HSV jitter, flip)

---

## Slide 4 — System Architecture

```
Input Image / Video Frame
          │
          ▼
  ┌───────────────────┐
  │ Preprocessing &   │
  │ Augmentation      │
  └────────┬──────────┘
           │
  ┌────────┴─────────┐
  ▼                  ▼
Classical CV     Deep Detector
Baseline         (YOLOv8n / YOLOv8s)
(HSV + Shape)    Fine-tuned on GTSDB
  │                  │
  └────────┬─────────┘
           ▼
     Post-processing
     (NMS + Threshold)
           │
           ▼
  Bounding Box Output
  + Sign Class Label
```

### Models compared
| Model | Parameters | Strategy |
|---|---|---|
| Classical CV | — | HSV + Contour |
| Zero-shot YOLOv8n | 3.0M | No fine-tuning |
| Fine-tuned YOLOv8n | 3.0M | Transfer learning |
| Fine-tuned YOLOv8s | 11.1M | Transfer learning |

---

## Slide 5 — Results

### 📊 Quantitative Comparison

| Model | mAP@0.5 | Precision | Recall |
|---|---|---|---|
| Classical CV Baseline | — | 4.0% | 15.5% |
| Zero-Shot YOLOv8n | — | 3.6% | 20.0% |
| YOLOv8n (no aug) | 84.8% | 87.7% | 74.1% |
| YOLOv8n (with aug) | 95.5% | 91.9% | 91.0% |
| **YOLOv8s (with aug)** | **97.1%** | **96.5%** | **90.6%** |

### 🔑 Key Findings
- ✅ Fine-tuned YOLOv8s achieves **97.1% mAP** — ~24× better than classical CV
- ✅ Data augmentation adds **+10.7% mAP** and **+16.9% recall**
- ✅ YOLOv8s outperforms YOLOv8n by **+1.6% mAP** at 4× parameter cost
- ✅ Zero-shot baseline fails — domain-specific fine-tuning is essential

---

## Slide 6 — Ablation Study & Failure Analysis

### ⚗️ Augmentation Ablation

```
Without Augmentation:  mAP@0.5 = 84.8%  |  Recall = 74.1%
With Augmentation:     mAP@0.5 = 95.5%  |  Recall = 91.0%
                       ──────────────────────────────────
Improvement:           +10.7 pp mAP     |  +16.9 pp Recall
```

Augmentation acts as a **5–10× data multiplier** on 365 training images.

### ❌ Failure Cases

| Failure Mode | Cause | Frequency |
|---|---|---|
| Tiny distant signs (< 20×20 px) | Below feature resolution | Moderate |
| Heavily occluded signs | Insufficient training examples | Low |
| Rare class miss (danger) | Class imbalance | Low |
| Night / low-light scenes | Not in training data | Not tested |

---

## Slide 7 — Live Demo

### 🎬 Interactive Streamlit Application

**Live at:** `http://localhost:8501`
**GitHub:** https://github.com/SajithK203/Traffic-Sign-Detection

#### Features:
- 📷 **Upload Image** — drag & drop any road image
- 🖼️ **Sample Images** — 7 pre-loaded GTSDB test images
- 🎬 **Upload Video** — frame-by-frame YOLO detection with download
- 🆚 **Side-by-Side** — Classical CV vs YOLOv8s comparison
- 🧪 **43-Class Mode** — fine-grained sign classification (stretch goal)
- ⚙️ **Adjustable** confidence and NMS IoU thresholds

#### 43-Class Stretch Goal
Fine-grained YOLOv8n trained on all 43 sign categories:
- Val mAP@0.5 = **24.7%** (vs 3 super-class baseline)
- Detects: `speed_limit_50`, `give_way`, `no_entry`, `go_right` etc.

---

## Slide 8 — Conclusion & Future Work

### ✅ Conclusion
1. Deep learning (YOLOv8s) achieves **97.1% mAP** on GTSDB — far superior to classical CV
2. **Transfer learning + augmentation** enables strong performance with only 365 training images
3. Data augmentation is the **single most impactful** design decision for small datasets
4. A fully functional **interactive demo** validates real-world detection capability

### 🔮 Future Work
- **Larger datasets:** TT100K, LISA, Mapillary for cross-domain generalisation
- **Better architectures:** YOLO11, RT-DETR for accuracy/speed improvement
- **Edge deployment:** TensorRT + Jetson Nano for embedded automotive use
- **Night/weather robustness:** Synthetic augmentation for out-of-distribution scenarios
- **Full 43-class classifier:** Combine GTSDB detection with GTSRB (51K crops) classification

---

*Thank you! Questions?*

**GitHub:** https://github.com/SajithK203/Traffic-Sign-Detection
