# 🚦 Traffic Sign Detection — CO543/CO5430

> **Course**: CO543 / CO5430 — Computer Vision  
> **Project Track**: Application (Transport) + Model/Method (Object Detection)  
> **Period**: 1 July 2026 – 7 September 2026

---

## 📋 Group Details

| Field | Details |
|---|---|
| Group Number | 17 |
| Member 1 | R.M.S.S.KUMARA    E/22/203 |
| Member 2 | K.I.SEWMINI       E/22/372 |
| Member 3 | S.I.GUNAWARDHANA  E/22/127 |
| Member 4 | A.W.H.PANCHANI    E/22/269 |
| GitHub Repo | _(this repo)_ |

---

## 🧩 Problem Statement

This project builds a system that takes an image or video frame as input and produces the **location (bounding box)** of every traffic sign in the scene, together with its **class** where a classification stage is added.

Traffic sign detection is a core perception problem for:
- Advanced Driver-Assistance Systems (ADAS)
- Autonomous vehicles
- Road-asset inventory and mapping

---

## 🏗️ System Architecture

```
Input Image/Frame
       │
       ▼
┌─────────────────────────┐
│  Preprocessing &        │
│  Augmentation           │
└────────────┬────────────┘
             │
    ┌────────┴─────────┐
    ▼                  ▼
Classical CV        Deep Detector
Baseline            (YOLOv8/YOLO11)
(HSV + Shape)       Fine-tuned on GTSDB
    │                  │
    └────────┬─────────┘
             ▼
       Post-processing
       (NMS + Thresholding)
             │
             ▼
    Bounding Box Output
    + (Optional) Sign Class
```

---

## 📦 Datasets

| Dataset | Region | Size | License |
|---|---|---|---|
| **GTSDB** | Germany | 900 imgs, 43 classes | CC BY 4.0 |
| **GTSRB** | Germany | 51,800 crops, 43 classes | Free research |
| _(stretch)_ TT100K | China | 100K imgs, 221 classes | CC BY-NC 2.0 |
| _(stretch)_ LISA | USA | 6,600 frames, 47 classes | Academic |

See [`data/README.md`](data/README.md) for download instructions.

---

## ⚙️ Setup

### Prerequisites
- Python 3.10+
- CUDA GPU recommended (or free Colab/Kaggle GPU)

### 1. Clone
```bash
git clone https://github.com/<your-org>/traffic-sign-detection.git
cd traffic-sign-detection
```

### 2. Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Datasets
Follow [`data/README.md`](data/README.md) and place files as instructed.

---

## 🚀 How to Run

### EDA
```bash
jupyter notebook notebooks/01_eda.ipynb
```

### Classical CV Baseline
```bash
python src/evaluate.py --model classical --data data/processed/gtsdb/test
```

### Zero-Shot Pretrained Baseline
```bash
python src/evaluate.py --model zero-shot --weights yolov8n.pt --data data/processed/gtsdb/test
```

### Train Fine-Tuned Detector
```bash
python src/train.py --config configs/gtsdb_yolov8n.yaml
```

### Evaluate
```bash
python src/evaluate.py --model yolov8 --weights results/checkpoints/best.pt --data data/processed/gtsdb/test
```

### Single Image Inference
```bash
python src/inference.py --weights results/checkpoints/best.pt --source path/to/image.jpg
```

### Demo App
```bash
streamlit run demo/app.py
```

---

## 📊 Results Summary

> *(Updated after experiments — see `results/metrics/` for full run logs)*

| Model | mAP@0.5 | Precision | Recall | FPS |
|---|---|---|---|---|
| Classical CV Baseline | — | — | — | — |
| Zero-Shot YOLOv8n (COCO) | — | — | — | — |
| Fine-Tuned YOLOv8n | — | — | — | — |
| Fine-Tuned YOLOv8s | — | — | — | — |

---

## 🗂️ Repository Structure

```
traffic-sign-detection/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── data/
│   ├── raw/                    # NOT committed — see data/README.md
│   ├── processed/              # small permitted samples only
│   └── README.md
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline_classical.ipynb
│   ├── 03_baseline_pretrained.ipynb
│   ├── 04_training.ipynb
│   └── 05_evaluation.ipynb
├── src/
│   ├── data/                   # annotation converters, dataset loaders
│   ├── models/                 # model wrappers
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   └── utils/
├── configs/                    # one YAML per experiment run
├── results/
│   ├── figures/
│   ├── metrics/
│   └── qualitative_examples/
├── demo/
│   ├── app.py
│   └── sample_media/
├── reports/
├── slides/
└── docs/
    ├── AI_USE_STATEMENT.md
    └── CONTRIBUTIONS.md
```

---

## 👥 Contributions

See [`docs/CONTRIBUTIONS.md`](docs/CONTRIBUTIONS.md) — updated weekly.

---

## 🤖 AI Tool Use

See [`docs/AI_USE_STATEMENT.md`](docs/AI_USE_STATEMENT.md).

---

## 📚 References

- Houben et al. (2013). *GTSDB*. IJCNN.
- Stallkamp et al. (2012). *GTSRB*. Neural Networks.
- Zhu et al. (2016). *TT100K*. CVPR.
- Ren et al. (2015). *Faster R-CNN*. NeurIPS.
- Liu et al. (2016). *SSD*. ECCV.
- Ultralytics. *YOLOv8 Docs*. https://docs.ultralytics.com

---

## 📝 License

Source code: MIT — see [`LICENSE`](LICENSE).  
Dataset licenses: see [`data/README.md`](data/README.md).
