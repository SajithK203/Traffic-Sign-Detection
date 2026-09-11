# Citations and Attributions

This document lists all external datasets, pretrained models, external code libraries,
research papers, and AI tools used in this project. Required as part of the CO5430
(Image Processing) submission at the Department of Computer Engineering, University of Peradeniya.

---

## Dataset

### German Traffic Sign Detection Benchmark (GTSDB)

> **This project uses ONLY the GTSDB dataset for training, validation, and testing.**

| Field | Details |
|---|---|
| **Name** | German Traffic Sign Detection Benchmark (GTSDB) |
| **Authors** | Sebastian Houben, Johannes Stallkamp, Jan Salmen, Marc Schlipsing, Christian Igel |
| **Year** | 2013 |
| **License** | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| **Official Source** | https://benchmark.ini.rub.de/gtsdb_news.html |
| **Mirror Download** | https://sid.erda.dk/public/archives/ff17dc924eba88d5d01a807357d6614c/published-archive.html |
| **Description** | Full driving scene images, 900 images, 43 fine-grained classes in 4 macro categories: prohibitory, danger, mandatory, other. |

**Citation:**

S. Houben, J. Stallkamp, J. Salmen, M. Schlipsing, and C. Igel, "Detection of Traffic Signs in Real-World Images: The German Traffic Sign Detection Benchmark," in Proc. IJCNN, 2013.

### Dataset Usage Clarification

The README contains a table listing GTSDB, GTSRB, TT100K, and LISA as benchmark comparisons.
**Only GTSDB was actually downloaded and used** in all training, validation, testing, and evaluation.
GTSRB, TT100K, and LISA are listed in the README solely for contextual comparison of available benchmarks.

---

## Pretrained Model Weights

### YOLOv8n and YOLOv8s (pretrained on MS-COCO)

| Field | Details |
|---|---|
| **Models** | YOLOv8n (nano) and YOLOv8s (small), pretrained on MS-COCO |
| **Provider** | Ultralytics |
| **Files** | yolov8n.pt, yolov8s.pt (auto-downloaded by Ultralytics library on first run) |
| **Source** | https://github.com/ultralytics/ultralytics |
| **License** | GNU Affero General Public License v3 (AGPL-3.0) |
| **Used for** | Zero-shot baseline evaluation AND transfer learning (fine-tuning starting point) |

> Note: Pretrained weights are NOT committed to this repository.
> They are automatically downloaded by the Ultralytics library.
> Our fine-tuned weights are stored in runs/detect/train/weights/best.pt.

**Citation:**

Ultralytics, "YOLOv8: A New State-of-the-Art Model for Object Detection," GitHub, 2023. https://github.com/ultralytics/ultralytics

---

## External Code Libraries

| Library | Version | Purpose | License |
|---|---|---|---|
| ultralytics | >=8.2.0 | YOLOv8 training, inference, evaluation | AGPL-3.0 |
| opencv-python | >=4.9.0 | Classical CV pipeline (HSV, morphology, contours) | Apache 2.0 |
| torch | >=2.2.0 | Deep learning framework | BSD-style |
| torchvision | >=0.17.0 | Vision transforms and utilities | BSD-style |
| streamlit | >=1.33.0 | Interactive demo web application | Apache 2.0 |
| numpy | >=1.26.4 | Numerical operations | BSD |
| pandas | >=2.2.1 | Annotation parsing and statistics | BSD |
| matplotlib | >=3.8.3 | Evaluation plots and figures | PSF |
| seaborn | >=0.13.2 | Statistical visualizations | BSD |
| scikit-learn | >=1.4.1 | Evaluation metrics, confusion matrix | BSD |
| albumentations | >=1.4.3 | Data augmentation pipeline | MIT |
| Pillow | >=10.2.0 | Image I/O and format conversion | HPND |
| pycocotools | >=2.0.7 | COCO annotation format utilities | BSD |

Full dependency list: see requirements.txt

---

## Research Papers Referenced

[1] S. Houben et al., "Detection of Traffic Signs in Real-World Images: The German Traffic Sign Detection Benchmark," Proc. IJCNN, 2013.

[2] J. Stallkamp et al., "The German Traffic Sign Recognition Benchmark," Proc. IJCNN, 2011.

[3] J. Redmon et al., "You Only Look Once: Unified, Real-Time Object Detection," Proc. CVPR, 2016.

[4] S. Ren et al., "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks," Proc. NeurIPS, 2015.

[5] W. Liu et al., "SSD: Single Shot MultiBox Detector," Proc. ECCV, 2016.

[6] Ultralytics, "YOLOv8 Documentation," 2023. Available: https://docs.ultralytics.com

[7] A. Mogelmose et al., "Vision-based traffic sign detection and analysis for intelligent driver assistance systems," IEEE Trans. ITS, 2012.

---

## AI Tools Used

Full declaration in: docs/AI_USE_STATEMENT.md

| Tool | Provider | Purpose |
|---|---|---|
| Antigravity | Google DeepMind | Code assistance, debugging, documentation |
| GitHub Copilot | GitHub / OpenAI | Inline code completion in VS Code |

All AI-assisted code was manually reviewed, tested, and verified by at least two group members before committing.

---

Last updated: 11 September 2026
Group 17 -- CO5430 Image Processing -- Department of Computer Engineering, University of Peradeniya
