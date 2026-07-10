# Dataset Information

> ⚠️ **Raw dataset files are NOT committed to this repository.**  
> Follow the download instructions below and place files in the correct directories.

---

## Primary Dataset — GTSDB (German Traffic Sign Detection Benchmark)

**Paper**: Houben et al. (2013). *Detection of Traffic Signs in Real-World Images: The German Traffic Sign Detection Benchmark*. IJCNN.  
**License**: Free for research use (CC BY 4.0 on common mirrors)  
**Size**: 900 images (600 train / 300 test), 1,000+ sign instances, 43 classes  
**Annotation**: Bounding boxes (`.txt` files)

### Download Steps
1. Visit: https://benchmark.ini.rub.de/gtsdb_news.html
2. Download `TrainIJCNN2013.zip` (~600 MB)
3. Extract and place as:
```
data/raw/gtsdb/
└── TrainIJCNN2013/
    ├── 00000.ppm
    ├── ...
    ├── 00599.ppm
    └── gt.txt
```
4. **Run the one-command setup** (auto-detects `.ppm`, converts, and splits):
```bash
python src/data/setup_dataset.py
```

> Full instructions: [`docs/SETUP_IMAGES.md`](./SETUP_IMAGES.md)

---

## Secondary Dataset — GTSRB (German Traffic Sign Recognition Benchmark)
*(Used for the two-stage classifier stretch goal)*

**Paper**: Stallkamp et al. (2012). *Man vs. computer*. Neural Networks.  
**License**: Free for research use  
**Size**: ~51,800 cropped images, 43 classes (classification only — no bounding boxes)

### Download Steps
1. Visit: https://benchmark.ini.rub.de/gtsrb_news.html
2. Download `GTSRB_Final_Training_Images.zip` and `GTSRB_Final_Test_Images.zip`
3. Place as:
```
data/raw/gtsrb/
├── Final_Training/
│   └── Images/
│       ├── 00000/     # class 00000
│       ├── 00001/
│       └── ... (43 folders)
└── Final_Test/
    └── Images/
```

---

## Stretch Dataset — TT100K (Tsinghua-Tencent 100K)

**Paper**: Zhu et al. (2016). *Traffic-Sign Detection and Classification in the Wild*. CVPR.  
**License**: ⚠️ **CC BY-NC 2.0 — NON-COMMERCIAL USE ONLY**  
**Size**: 100,000 street-view images, 30,000+ sign instances, 221 categories (~45 after filtering)  
**Download**: ~18 GB — only attempt if pursuing the large-scale stretch goal

### Download Steps
1. Visit: https://cg.cs.tsinghua.edu.cn/traffic-sign/
2. Register and download the dataset
3. Place as: `data/raw/tt100k/`
4. **IMPORTANT**: Never commit raw TT100K images to GitHub

---

## Stretch Dataset — LISA Traffic Sign Dataset

**Paper**: Møgelmose et al. (2012). *Vision-based Traffic Sign Detection*. IEEE Trans. ITS.  
**License**: Academic license — citation required  
**Size**: ~6,600 annotated frames, ~7,800 sign instances, 47 US sign classes

### Download Steps
1. Visit: https://cvrr-nas.ucsd.edu/LISA/lisa-traffic-sign-dataset.html
2. Request access and download
3. Place as: `data/raw/lisa/`

---

## Annotation Format (After Conversion)

All datasets are converted to **YOLO txt format**:
```
# One .txt file per image, same name as the image
# Each line: class_id  x_center  y_center  width  height  (all normalized 0–1)
0 0.512 0.384 0.064 0.096
```

Class mappings are stored in `configs/classes_gtsdb.yaml`.

---

## Train / Val / Test Splits

| Dataset | Train | Val | Test |
|---|---|---|---|
| GTSDB (this project) | 365 imgs | 80 imgs | 81 imgs |
| GTSRB | 39,209 (official) | 10% from train | 12,630 (official) |

**Rule**: Never touch the test split during training or hyperparameter tuning. Run final evaluation on test exactly once per model variant.

---

## Ethics & Privacy Notes

- All datasets used are public benchmarks with established academic usage terms.
- Do not publish raw dataset images beyond what each dataset's license permits.
- If using any self-collected footage: blur all faces and license plates before committing.
- TT100K is non-commercial — do not use it in any commercial deployment.
