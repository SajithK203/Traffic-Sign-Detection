"""
convert_gtsdb.py
-----------------
Converts GTSDB annotations from the official .txt format to YOLO normalized
bounding-box format.

GTSDB original format (gt.txt):
    Filename.ppm;x1;y1;x2;y2;ClassID

YOLO output format (one .txt per image):
    class_id  x_center  y_center  width  height   (all values normalized 0–1)

Usage:
    python src/data/convert_gtsdb.py \
        --gt_file  data/raw/gtsdb/TrainIJCNN2013/gt.txt \
        --img_dir  data/raw/gtsdb/TrainIJCNN2013 \
        --out_dir  data/processed/gtsdb/train \
        --verify   5

GTSDB class groups (3 super-classes used for detection):
    0  — prohibitory  (classes 0–8)
    1  — danger       (classes 11–31)
    2  — mandatory    (classes 33–41)

For fine-grained 43-class detection, pass --use_fine_classes.
"""

import argparse
import os
import shutil
from pathlib import Path

import cv2


# ------------------------------------------------------------------
# GTSDB class ID → super-class mapping
# ------------------------------------------------------------------
def get_superclass(class_id: int) -> int | None:
    """Map fine-grained GTSDB class ID to 3-class super-class.

    Returns:
        0 (prohibitory), 1 (danger), 2 (mandatory), or None (other/ignore).
    """
    if 0 <= class_id <= 8:
        return 0   # prohibitory
    elif 11 <= class_id <= 31:
        return 1   # danger
    elif 33 <= class_id <= 41:
        return 2   # mandatory
    return None    # ignore class 9,10,32,42


def convert_gtsdb(
    gt_file: Path,
    img_dir: Path,
    out_dir: Path,
    use_fine_classes: bool = False,
    verify_n: int = 0,
) -> None:
    """Convert GTSDB ground-truth file to per-image YOLO txt files.

    Args:
        gt_file: Path to gt.txt (GTSDB annotation file).
        img_dir: Directory containing the .ppm images.
        out_dir: Output directory for YOLO-format .txt files and images.
        use_fine_classes: If True, keep original 43 class IDs.
        verify_n: If > 0, draw bounding boxes on first N images for visual check.
    """
    out_dir = Path(out_dir)
    labels_dir = out_dir / "labels"
    images_dir = out_dir / "images"
    labels_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    if not gt_file.exists():
        raise FileNotFoundError(f"Ground-truth file not found: {gt_file}")

    # Parse annotations grouped by image filename
    annotations: dict[str, list[tuple]] = {}
    with open(gt_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(";")
            if len(parts) != 6:
                print(f"  [WARN] Skipping malformed line: {line}")
                continue

            fname, x1, y1, x2, y2, cls_id = parts
            x1, y1, x2, y2, cls_id = int(x1), int(y1), int(x2), int(y2), int(cls_id)
            annotations.setdefault(fname, []).append((x1, y1, x2, y2, cls_id))

    converted = 0
    skipped = 0

    for img_fname, boxes in annotations.items():
        img_path = img_dir / img_fname
        if not img_path.exists():
            print(f"  [WARN] Image not found, skipping: {img_path}")
            skipped += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [WARN] Could not read image: {img_path}")
            skipped += 1
            continue

        img_h, img_w = img.shape[:2]
        yolo_lines = []

        for (x1, y1, x2, y2, cls_id) in boxes:
            if use_fine_classes:
                yolo_cls = cls_id
            else:
                yolo_cls = get_superclass(cls_id)
                if yolo_cls is None:
                    continue  # skip ignored classes

            # Normalize to [0, 1]
            x_center = ((x1 + x2) / 2.0) / img_w
            y_center = ((y1 + y2) / 2.0) / img_h
            width    = (x2 - x1) / img_w
            height   = (y2 - y1) / img_h

            # Clamp to [0, 1]
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            width    = max(0.0, min(1.0, width))
            height   = max(0.0, min(1.0, height))

            yolo_lines.append(
                f"{yolo_cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            )

        # Write label file
        stem = Path(img_fname).stem
        label_path = labels_dir / f"{stem}.txt"
        with open(label_path, "w") as f:
            f.write("\n".join(yolo_lines))

        # Copy image (convert .ppm → .jpg for compatibility)
        out_img_path = images_dir / f"{stem}.jpg"
        cv2.imwrite(str(out_img_path), img)

        # Optional visual verification
        if verify_n > 0 and converted < verify_n:
            verify_img = img.copy()
            for line in yolo_lines:
                parts = line.split()
                c, xc, yc, w, h = int(parts[0]), *[float(x) for x in parts[1:]]
                px1 = int((xc - w / 2) * img_w)
                py1 = int((yc - h / 2) * img_h)
                px2 = int((xc + w / 2) * img_w)
                py2 = int((yc + h / 2) * img_h)
                cv2.rectangle(verify_img, (px1, py1), (px2, py2), (0, 255, 0), 2)
                cv2.putText(verify_img, str(c), (px1, py1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            verify_path = out_dir / f"verify_{stem}.jpg"
            cv2.imwrite(str(verify_path), verify_img)
            print(f"  [VERIFY] Saved: {verify_path}")

        converted += 1

    print(f"\nDone. Converted: {converted} | Skipped: {skipped}")
    print(f"   Labels → {labels_dir}")
    print(f"   Images → {images_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert GTSDB to YOLO format")
    parser.add_argument("--gt_file",  required=True,
                        help="Path to gt.txt from GTSDB")
    parser.add_argument("--img_dir",  required=True,
                        help="Directory containing .ppm images")
    parser.add_argument("--out_dir",  required=True,
                        help="Output directory for YOLO files")
    parser.add_argument("--use_fine_classes", action="store_true",
                        help="Keep 43 fine-grained classes instead of 3 super-classes")
    parser.add_argument("--verify", type=int, default=0,
                        help="Draw boxes on first N images for visual check")
    args = parser.parse_args()

    convert_gtsdb(
        gt_file=Path(args.gt_file),
        img_dir=Path(args.img_dir),
        out_dir=Path(args.out_dir),
        use_fine_classes=args.use_fine_classes,
        verify_n=args.verify,
    )


if __name__ == "__main__":
    main()
