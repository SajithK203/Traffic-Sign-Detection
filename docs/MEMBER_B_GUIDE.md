# 🔴 Member B — Classical CV & Zero-Shot Baseline Guide

**CO543/CO5430 — Traffic Sign Detection**
**Your Role: Classical CV Lead**
**Deadline: 26 Jul (baselines done) → 28 Jul (M2 Checkpoint)**

---

## What You Are Responsible For

| Task | Notebook | Output File | Due |
|---|---|---|---|
| Classical CV Baseline (HSV + Contour) | `02_baseline_classical.ipynb` | `results/metrics/classical_baseline.json` | 26 Jul |
| Zero-Shot YOLO Baseline (no training) | `03_baseline_pretrained.ipynb` | `results/metrics/zero_shot_baseline.json` | 26 Jul |
| Qualitative prediction grids | Both notebooks | `results/qualitative_examples/` | 26 Jul |
| Related Work section of proposal | Google Docs / Overleaf | 3+ IEEE references | 14 Jul |
| HSV mask visualisation figure | Notebook 02, Section 1 | `results/figures/classical_hsv_masks.png` | 26 Jul |

---

## Step 0 — First-Time Machine Setup

> **Do this once on your own machine.**

### 0.1 Clone the Repository
```bash
git clone https://github.com/SajithK203/Traffic-Sign-Detection.git
cd traffic-sign-detection
```

### 0.2 Create a Virtual Environment
```bash
python -m venv venv
```

Activate it:
- **Windows:** `venv\Scripts\activate`
- **Mac/Linux:** `source venv/bin/activate`

You should see `(venv)` in your terminal prompt.

### 0.3 Install All Dependencies
```bash
pip install opencv-python-headless numpy matplotlib seaborn pandas jupyter ultralytics
```

> ⚠️ **ultralytics** is needed for Notebook 03 (zero-shot baseline). Install it now to save time later.

### 0.4 Verify the Dataset is Ready
```bash
python -c "
from pathlib import Path
test = Path('data/processed/gtsdb/test/images')
print('Test images:', len(list(test.glob('*.jpg'))))
"
```
You should see: `Test images: 81`

If you see `0` — ask **Member A** to share the `data/processed/gtsdb/` folder with you (it's not committed to Git because datasets are excluded by `.gitignore`).

---

## Step 1 — Understand the Task

### Why a Classical Baseline?

The project requires you to show **three levels of performance**:

```
Classical CV  →  Zero-Shot YOLO  →  Fine-Tuned YOLO
(worst)           (medium)             (best)
```

The classical baseline proves that simple rule-based colour detection is **not sufficient** and that deep learning fine-tuning is required. Low numbers here are **expected and desirable** for the report.

### How the Classical Detector Works

```
Input Image (BGR)
      ↓
Convert to HSV colour space
      ↓
Apply colour masks:
  Red mask   → Prohibitory signs (speed limits, no entry)
  Blue mask  → Mandatory signs   (turn right, go straight)
  Yellow mask→ Danger signs      (warning triangles)
      ↓
Morphological cleanup (remove tiny noise, fill gaps)
      ↓
Find contours (outlines of coloured blobs)
      ↓
Filter by area (min_area < blob_area < max_area)
      ↓
Output bounding boxes
```

### Why HSV Instead of RGB?

| Colour Space | Problem |
|---|---|
| **RGB** | Red colour has different R,G,B values in shadow vs. sunlight. Hard to threshold. |
| **HSV** | Hue = pure colour. Saturation = how vivid. Value = brightness. A red sign is always Hue≈0° or 170° regardless of lighting. |

---

## Step 2 — Open the Classical Baseline Notebook

```bash
# In the traffic-sign-detection folder, with (venv) active:
jupyter notebook
```

Your browser opens. Click: `notebooks` → `02_baseline_classical.ipynb`

---

## Step 3 — Run Notebook 02 Section by Section

### Section 1 — HSV Mask Visualisation

**Run this cell first.** It shows you 4 images side by side:
1. Original traffic scene
2. Red mask (white = detected red pixels)
3. Blue mask (white = detected blue pixels)
4. Yellow mask (white = detected yellow pixels)

**What good output looks like:**
- Red mask has a clear white blob exactly where the red sign border is
- Minimal white noise in the sky, road, or car body areas

**What bad output looks like (and how to fix):**
| Problem | Cause | Fix |
|---|---|---|
| Entire sky is white in red mask | Hue range too wide | Narrow `red_lower1[0]` to `red_upper1[0]` |
| Sign is not detected at all | Saturation threshold too high | Lower `red_lower1[1]` from 120 → 80 |
| Road markings detected as blue | Blue range too wide | Raise `blue_lower[1]` from 100 → 130 |
| Tiny white dots everywhere | `min_area` too small | Raise `min_area` from 500 → 1000 |

---

### Section 2 — Detect on a Single Image

This draws the prediction boxes on one test image. Check:
- ✅ Green box around the sign = correct detection
- ❌ No box on a real sign = false negative (missed)
- ❌ Box on a car/road/sky = false positive

---

### Section 3 — Tune HSV Thresholds Interactively ⭐ Key Section

This is where you improve the detector. Edit and re-run this cell:

```python
custom_cfg = ClassicalConfig(
    # ── RED SIGNS (Prohibitory) ─────────────────────────────────────────
    # Red wraps around the hue wheel: 0–8 degrees AND 165–180 degrees
    red_lower1   = [0,   120, 60],    # [hue_min, sat_min, val_min]
    red_upper1   = [8,   255, 255],   # [hue_max, sat_max, val_max]
    red_lower2   = [165, 120, 60],
    red_upper2   = [180, 255, 255],

    # ── BLUE SIGNS (Mandatory) ──────────────────────────────────────────
    # Blue is hue 100–130 degrees
    blue_lower   = [100, 100, 60],
    blue_upper   = [130, 255, 255],

    # ── YELLOW/ORANGE SIGNS (Danger) ───────────────────────────────────
    # Yellow is hue 18–32 degrees
    yellow_lower = [18,  100, 100],
    yellow_upper = [32,  255, 255],

    # ── SIZE FILTER ─────────────────────────────────────────────────────
    min_area     = 500,     # blobs smaller than 500 px² are noise → ignored
    max_area     = 60_000,  # blobs larger than 60k px² are background → ignored
)
```

**Tuning strategy — do this iteratively:**

1. Start with the defaults above
2. Run the cell and view results on the sample image
3. If too many false positives → raise the `sat_min` value (second number in `lower`)
4. If signs are being missed → lower the `sat_min` value
5. Repeat until you have a clean mask
6. Then run Section 4 to see how the tuned thresholds perform on the full test set

---

### Section 4 — Evaluate on Full Test Set

**Run this cell after tuning.** It will print:

```
Running classical baseline on test set...

── Classical CV Baseline Results (Test Split) ──
  Precision : 0.4123
  Recall    : 0.3156
  F1 Score  : 0.3572
  AP@0.5    : 0.2841
  Detections: 312
  GT Boxes  : 241
```

It also saves → `results/metrics/classical_baseline.json`

**Record these numbers** — Member D needs them for the checkpoint deck.

> ℹ️ **Expected range**: Precision 0.25–0.60, Recall 0.20–0.50.
> Low numbers are fine — the classical method is intentionally limited.

---

### Section 5 — Qualitative Grid

This saves a 3×3 grid of 9 test images showing:
- 🔴 Red dashed boxes = Ground Truth (what the correct answer is)
- 🟢 Green solid boxes = Your detector's predictions

Saves to: `results/qualitative_examples/classical_baseline_predictions.png`

**Use this image in your report and checkpoint deck.**

---

## Step 4 — Run Notebook 03 (Zero-Shot YOLO Baseline)

After notebook 02 is done, open `notebooks/03_baseline_pretrained.ipynb`.

This uses a **COCO-pretrained YOLOv8n model without any fine-tuning** on GTSDB.

### Why is this a baseline?

COCO has 80 classes but only 1 traffic sign class (`stop sign`). So the model has never seen:
- Speed limit signs
- Warning triangles
- Mandatory turn signs

This means performance will be **very low** — which proves fine-tuning on GTSDB is necessary.

### What you need to do

```bash
# Run this in the terminal if not already installed:
pip install ultralytics
```

Then in the notebook:
1. Run all cells top to bottom
2. Section 1: downloads `yolov8n.pt` (~6 MB, one-time)
3. Section 2: shows predictions on 6 sample images with COCO class names (you'll see things like `frisbee`, `clock` for signs!)
4. Section 3: evaluates on the full 81-image test set
5. Section 4: generates a comparison bar chart (classical vs. zero-shot)

Saves to: `results/metrics/zero_shot_baseline.json`

---

## Step 5 — Push Your Results to GitHub

After both notebooks are done:

```bash
# Stage all results
git add results/metrics/classical_baseline.json
git add results/metrics/zero_shot_baseline.json
git add results/figures/classical_hsv_masks.png
git add results/figures/baseline_comparison.png
git add results/qualitative_examples/classical_baseline_predictions.png
git add results/qualitative_examples/zeroshot_baseline_predictions.png
git add notebooks/02_baseline_classical.ipynb
git add notebooks/03_baseline_pretrained.ipynb

# Commit with a descriptive message
git commit -m "feat: classical and zero-shot baselines complete (M2)

Classical CV (HSV+Contour):
- Precision: X.XXXX  ← fill in your numbers
- Recall:    X.XXXX
- F1:        X.XXXX
- AP@0.5:    X.XXXX

Zero-Shot YOLOv8n (COCO, no fine-tuning):
- Precision: X.XXXX
- Recall:    X.XXXX
- F1:        X.XXXX"

# Push to GitHub
git push
```

> ⚠️ **Before pushing**, always pull first to get any changes from teammates:
> ```bash
> git pull origin main
> ```
> If there are conflicts, resolve them, then push.

---

## Step 6 — Update CONTRIBUTIONS.md

Open `docs/CONTRIBUTIONS.md` and mark your tasks as done:

```markdown
| Classical CV baseline — HSV mask visualisation      | B | ✅ Done — dd Mon |
| Classical CV baseline — full test set evaluation    | B | ✅ Done — dd Mon |
| Zero-shot YOLOv8n baseline — full test set eval     | B | ✅ Done — dd Mon |
```

Push this update too:
```bash
git add docs/CONTRIBUTIONS.md
git commit -m "docs: update contributions log — baselines complete"
git push
```

---

## Step 7 — Report Numbers to Member D

After Step 5, message Member D with both sets of numbers:

```
Classical CV Baseline (test split):
  Precision : ___
  Recall    : ___
  F1        : ___
  AP@0.5    : ___

Zero-Shot YOLOv8n (test split):
  Precision : ___
  Recall    : ___
  F1        : ___
  AP@0.5    : ___
```

Member D will put these in the 5-slide M2 checkpoint deck.

---

## Reference — HSV Hue Values for Traffic Signs

```
Hue wheel (0–180 in OpenCV):
   0° / 180° = Red        ← Prohibitory signs
   30°       = Yellow     ← Danger/Warning signs
   60°       = Green
   90°       = Cyan
   120°      = Blue       ← Mandatory signs
   150°      = Magenta
```

```
OpenCV HSV ranges:
  Hue        : 0–180  (NOT 0–360 — OpenCV halves it)
  Saturation : 0–255  (0 = grey, 255 = fully vivid colour)
  Value      : 0–255  (0 = black, 255 = full brightness)
```

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'cv2'` | opencv not installed | `pip install opencv-python-headless` |
| `ModuleNotFoundError: No module named 'ultralytics'` | ultralytics not installed | `pip install ultralytics` |
| `No test images found` | dataset not on your machine | Copy `data/processed/gtsdb/` from Member A |
| `Best weights not found` | training not done yet | Skip Notebook 04 — that's Member C's job |
| Jupyter kernel crashes on large images | Not enough RAM | Close other applications, reduce `n_samples` in the crop cell |
| `UnicodeEncodeError` on Windows | Emoji in print | Already fixed — do `git pull` to get the fix |

---

## File Structure You Will Be Working With

```
traffic-sign-detection/
├── src/models/
│   └── classical_detector.py   ← The detector code (read this!)
├── src/utils/
│   ├── metrics.py              ← IoU, AP, Precision/Recall calculation
│   └── visualization.py       ← PR curve, confusion matrix plotters
├── notebooks/
│   ├── 02_baseline_classical.ipynb   ← YOUR MAIN NOTEBOOK
│   └── 03_baseline_pretrained.ipynb  ← YOUR SECOND NOTEBOOK
├── results/
│   ├── metrics/
│   │   ├── classical_baseline.json   ← You generate this
│   │   └── zero_shot_baseline.json   ← You generate this
│   ├── figures/
│   │   ├── classical_hsv_masks.png   ← You generate this
│   │   └── baseline_comparison.png   ← You generate this
│   └── qualitative_examples/
│       ├── classical_baseline_predictions.png  ← You generate this
│       └── zeroshot_baseline_predictions.png   ← You generate this
└── data/
    └── processed/gtsdb/
        └── test/               ← The 81 test images you evaluate on
```

---

## Member B Checklist

```
Setup:
  [ ] Clone repo and activate venv
  [ ] pip install opencv-python-headless numpy matplotlib seaborn pandas jupyter ultralytics
  [ ] Verify 81 test images are available

Week 2 (before 14 Jul) — Proposal:
  [ ] Write Related Work section (3+ IEEE references on traffic sign detection)
  [ ] Read src/models/classical_detector.py to understand the code
  [ ] Run Notebook 02, Section 1 (just visualise HSV masks)

Week 3 (before 21 Jul):
  [ ] Tune HSV thresholds (Section 3) until masks are clean
  [ ] Run full evaluation (Section 4) — get Precision/Recall numbers
  [ ] Run qualitative grid (Section 5) — save figure
  [ ] Push classical_baseline.json to GitHub

Week 4 (before 26 Jul — M2 prep):
  [ ] Run Notebook 03 (zero-shot YOLO baseline) — all sections
  [ ] Push zero_shot_baseline.json to GitHub
  [ ] Push baseline_comparison.png to GitHub
  [ ] Send both sets of numbers to Member D
  [ ] Update docs/CONTRIBUTIONS.md
```

---

*Generated for CO543/CO5430 Traffic Sign Detection Project — Group ___*
