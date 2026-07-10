# 📂 Dataset Setup — Images + Labels

**CO543/CO5430 — Traffic Sign Detection**

---

## How It Works — Labels on GitHub, Images on Your Machine

| GitHub has | Your machine needs |
|---|---|
| ✅ `data/processed/gtsdb/train/labels/00001.txt` | ❌ `data/processed/gtsdb/train/images/00001.jpg` |
| ✅ `data/processed/gtsdb/val/labels/00006.txt`   | ❌ `data/processed/gtsdb/val/images/00006.jpg`   |
| ✅ `data/processed/gtsdb/test/labels/00003.txt`  | ❌ `data/processed/gtsdb/test/images/00003.jpg`  |

The `.txt` label files tell the model **where the signs are** in each image.  
The `.jpg` images are the **actual photos**.  
Both must be present for the code to work — but images can't go on GitHub (too large), so each member downloads the raw GTSDB once and generates the images locally.

---

## For Each Team Member — One-Time Setup

### Step 1 — Clone the repo *(gets the .txt labels automatically)*

```bash
git clone https://github.com/SajithK203/Traffic-Sign-Detection.git
cd Traffic-Sign-Detection
```

After this, you already have all the label files:

```
data/processed/gtsdb/train/labels/  ← 365 .txt files ✅
data/processed/gtsdb/val/labels/    ←  80 .txt files ✅
data/processed/gtsdb/test/labels/   ←  81 .txt files ✅
data/processed/gtsdb/all/labels/    ← 506 .txt files ✅
```

But the `images/` folders are empty. Do this next:

---

### Step 2 — Download the GTSDB raw images

Go to: <https://benchmark.ini.rub.de/gtsdb_news.html>

Download: **TrainIJCNN2013.zip**

Extract to: `data/raw/gtsdb/TrainIJCNN2013/`

After extraction you should see:

```
data/raw/gtsdb/TrainIJCNN2013/
  00000.ppm
  00001.ppm
  ...
  00599.ppm
  gt.txt
  ReadMe.txt
```

> ⚠️ **Note on zip extraction path:** The zip may extract into a nested subfolder  
> (e.g. `TrainIJCNN2013/TrainIJCNN2013/`). That's fine — the setup script  
> (Step 4) will automatically find your `.ppm` files wherever they are under  
> `data/raw/gtsdb/`.

---

### Step 3 — Install dependencies

```bash
python -m venv venv

# Activate (Windows):
venv\Scripts\activate

# Activate (Mac/Linux):
# source venv/bin/activate

pip install opencv-python-headless numpy matplotlib seaborn pandas jupyter ultralytics
```

---

### Step 4 — Run the one-command setup *(recommended)*

```bash
python src/data/setup_dataset.py
```

This single script:
- **Auto-detects** your `.ppm` folder under `data/raw/gtsdb/` (no path needed)
- Converts the matching `.ppm` → `.jpg` into `data/processed/gtsdb/all/images/`
- Copies images into `train/`, `val/`, `test/` to match the label files already on GitHub

Expected output:

```
[INFO] Found 506 label files in all/labels/
[INFO] Auto-detecting .ppm directory under: data/raw/gtsdb
[INFO] Found 600 .ppm files in: data/raw/gtsdb/TrainIJCNN2013

============================================================
Step B — Converting .ppm → .jpg
============================================================
  ... converted 50/506
  ... converted 100/506
  ...
  Converted : 506
  Skipped   : 0  (already existed)
  Missing   : 0  (no .ppm found)

============================================================
Step C — Copying images into train / val / test splits
============================================================
  train → images/train: copied=365  already=0  missing=0
  val   → images/val:   copied=80   already=0  missing=0
  test  → images/test:  copied=81   already=0  missing=0

============================================================
Done!
============================================================

Verification:
  train -> images: 365  labels: 365  matched: 365  missing_images: 0
  val   -> images:  80  labels:  80  matched:  80  missing_images: 0
  test  -> images:  81  labels:  81  matched:  81  missing_images: 0

✓ All label files have matching images. Dataset is ready!
  You can now run: jupyter notebook
```

> **Optional flags:**
> ```bash
> python src/data/setup_dataset.py --raw_dir path/to/your/ppm/folder
> python src/data/setup_dataset.py --quality 85        # smaller JPEGs
> python src/data/setup_dataset.py --dry_run           # preview without writing
> ```

---

### Step 4 (Alternative) — Run the scripts manually

If you prefer the step-by-step approach or the auto-script fails:

**4a — Run the converter** (generates .jpg images from .ppm):

```bash
python src/data/convert_gtsdb.py \
  --gt_file data/raw/gtsdb/TrainIJCNN2013/gt.txt \
  --img_dir data/raw/gtsdb/TrainIJCNN2013 \
  --out_dir data/processed/gtsdb/all
```

> ⚠️ If your `.ppm` files extracted into a nested subfolder, adjust the paths:
> ```bash
> python src/data/convert_gtsdb.py \
>   --gt_file data/raw/gtsdb/TrainIJCNN2013/TrainIJCNN2013/gt.txt \
>   --img_dir data/raw/gtsdb/TrainIJCNN2013/TrainIJCNN2013 \
>   --out_dir data/processed/gtsdb/all
> ```

This creates `data/processed/gtsdb/all/images/` with 506 .jpg files.

**4b — Run the split script** (puts images into train/val/test):

```bash
python src/data/split_dataset.py \
  --src_dir data/processed/gtsdb/all \
  --out_dir data/processed/gtsdb \
  --seed 42
```

> ⚠️ **Must use `--seed 42`** — this ensures everyone gets the exact same  
> train/val/test split. The `.txt` label files already on GitHub were  
> created with this seed.

---

### Step 5 — Verify everything is aligned

```bash
python -c "
from pathlib import Path
for split in ['train', 'val', 'test']:
    img_dir = Path(f'data/processed/gtsdb/{split}/images')
    lbl_dir = Path(f'data/processed/gtsdb/{split}/labels')

    imgs = set(p.stem for p in img_dir.glob('*.jpg'))
    lbls = set(p.stem for p in lbl_dir.glob('*.txt'))

    matched   = imgs & lbls
    img_only  = imgs - lbls   # images with no label (should be 0)
    lbl_only  = lbls - imgs   # labels with no image (means image missing)

    print(f'{split:5s} -> images:{len(imgs):4d}  labels:{len(lbls):4d}  matched:{len(matched):4d}  missing_images:{len(lbl_only)}')
"
```

**Expected output:**

```
train -> images: 365  labels: 365  matched: 365  missing_images: 0
val   -> images:  80  labels:  80  matched:  80  missing_images: 0
test  -> images:  81  labels:  81  matched:  81  missing_images: 0
```

If `missing_images` is not `0`:
- If you used the **manual approach** (Step 4 alternative): re-run `split_dataset.py` with `--seed 42`
- If you used the **auto script** (Step 4 recommended): re-run `setup_dataset.py` — it will only copy the missing files

---

### Step 6 — Now run the notebooks

```bash
jupyter notebook
```

| Notebook | What it tests |
|---|---|
| `01_eda.ipynb` | Loads images + labels, shows class distribution |
| `02_baseline_classical.ipynb` | Runs HSV detector on test images |
| `03_baseline_pretrained.ipynb` | Runs YOLOv8n on test images |
| `04_training.ipynb` | Trains YOLO using train/val sets |
| `05_evaluation.ipynb` | Evaluates all models on test set |

---

## Summary — What Each Member Needs to Do Once

| Step | Action | Gets you |
|---|---|---|
| 1 | Clone repo | All `.txt` labels automatically |
| 2 | Download GTSDB zip | Raw `.ppm` images (~600 MB) |
| 3 | Install dependencies | Python packages for notebooks |
| 4 | `python src/data/setup_dataset.py` | `.jpg` images in all splits |
| 5 | Run verify script | Confirm 0 missing images |
| 6 | `jupyter notebook` | All notebooks work |

> **The key insight:** The `.txt` files define *which* images are in each split  
> and *where* the signs are. The images are just the raw pixel data. Anyone  
> who downloads the same GTSDB zip and runs the setup script gets **identical  
> train/val/test sets** to everyone else on the team.

---

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `[ERROR] No .ppm files found` | Zip not extracted yet | Download + extract `TrainIJCNN2013.zip` to `data/raw/gtsdb/` |
| `[ERROR] No label files found in all/labels/` | Repo not cloned correctly | `git pull` or re-clone |
| `missing_images` > 0 after verify | Wrong seed or partial download | Re-run `setup_dataset.py` |
| `ModuleNotFoundError: cv2` | opencv not installed | `pip install opencv-python-headless` |
| Script takes a long time | 506 PPM conversions | Normal — ~2-5 min depending on disk speed |
| Jupyter `No test images found` | Setup script not run yet | Run Step 4 first |

---

*Last updated: 2026-07-10 | CO543/CO5430 Traffic Sign Detection Project*
