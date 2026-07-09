"""
split_dataset.py
-----------------
Takes a FULLY converted YOLO dataset (from convert_gtsdb.py) and splits it
into train / val / test splits with a 70 / 15 / 15 ratio, stratified by class.

Usage:
    python src/data/split_dataset.py \
        --src_dir  data/processed/gtsdb/train \
        --out_dir  data/processed/gtsdb \
        --seed     42
"""

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path


def get_classes_in_label(lbl_path: Path) -> set:
    """Return the set of class IDs present in a YOLO label file."""
    classes = set()
    with open(lbl_path) as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                classes.add(int(parts[0]))
    return classes


def split_dataset(src_dir: Path, out_dir: Path, seed: int = 42):
    """
    Split a flat YOLO dataset into train/val/test (70/15/15) stratified by class.

    Args:
        src_dir: Directory with images/ and labels/ sub-folders.
        out_dir: Parent output directory. Will create train/ val/ test/ inside.
        seed:    Random seed for reproducibility.
    """
    random.seed(seed)

    img_dir = src_dir / "images"
    lbl_dir = src_dir / "labels"

    if not img_dir.exists():
        raise FileNotFoundError(f"images/ not found in {src_dir}")

    # Collect all image stems that have a matching label file
    img_stems = sorted(
        p.stem for p in img_dir.glob("*.jpg")
        if (lbl_dir / (p.stem + ".txt")).exists()
    )

    # Images with NO annotations (negative examples) — put all in train
    no_ann_stems = [
        p.stem for p in img_dir.glob("*.jpg")
        if not (lbl_dir / (p.stem + ".txt")).exists()
           or (lbl_dir / (p.stem + ".txt")).stat().st_size == 0
    ]

    # Group annotated images by their dominant class (for stratification)
    class_buckets: dict[int, list[str]] = defaultdict(list)
    for stem in img_stems:
        lbl_path = lbl_dir / f"{stem}.txt"
        if lbl_path.stat().st_size == 0:
            no_ann_stems.append(stem)
            continue
        classes = get_classes_in_label(lbl_path)
        # Use the first class as the stratification key
        dominant = sorted(classes)[0]
        class_buckets[dominant].append(stem)

    train_stems, val_stems, test_stems = [], [], []

    for cls_id, stems in class_buckets.items():
        random.shuffle(stems)
        n = len(stems)
        n_val  = max(1, int(n * 0.15))
        n_test = max(1, int(n * 0.15))
        test_stems  += stems[:n_test]
        val_stems   += stems[n_test: n_test + n_val]
        train_stems += stems[n_test + n_val:]

    # Shuffle negative (no annotation) examples mostly into train
    random.shuffle(no_ann_stems)
    n_neg = len(no_ann_stems)
    train_stems += no_ann_stems[:int(n_neg * 0.70)]
    val_stems   += no_ann_stems[int(n_neg * 0.70): int(n_neg * 0.85)]
    test_stems  += no_ann_stems[int(n_neg * 0.85):]

    print(f"Split summary:")
    print(f"  Train : {len(train_stems)} images")
    print(f"  Val   : {len(val_stems)} images")
    print(f"  Test  : {len(test_stems)} images")
    print(f"  Total : {len(train_stems) + len(val_stems) + len(test_stems)}")

    def copy_split(stems: list, split_name: str):
        split_img_dir = out_dir / split_name / "images"
        split_lbl_dir = out_dir / split_name / "labels"
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_lbl_dir.mkdir(parents=True, exist_ok=True)

        for stem in stems:
            src_img = img_dir / f"{stem}.jpg"
            src_lbl = lbl_dir / f"{stem}.txt"
            if src_img.exists():
                shutil.copy2(src_img, split_img_dir / f"{stem}.jpg")
            if src_lbl.exists():
                shutil.copy2(src_lbl, split_lbl_dir / f"{stem}.txt")
            else:
                # Write an empty label file (YOLO convention for negative images)
                (split_lbl_dir / f"{stem}.txt").touch()

            print(f"  [OK] {split_name:5s} -> {split_img_dir}")

    copy_split(train_stems, "train")
    copy_split(val_stems,   "val")
    copy_split(test_stems,  "test")

    # Write a splits.txt summary for traceability
    summary_path = out_dir / "splits_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"seed={seed}\n")
        f.write(f"train={len(train_stems)}\n")
        f.write(f"val={len(val_stems)}\n")
        f.write(f"test={len(test_stems)}\n")
        f.write("\n--- TRAIN ---\n" + "\n".join(sorted(train_stems)))
        f.write("\n--- VAL ---\n"   + "\n".join(sorted(val_stems)))
        f.write("\n--- TEST ---\n"  + "\n".join(sorted(test_stems)))
    print(f"\n  Split list saved -> {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Split YOLO dataset into train/val/test")
    parser.add_argument("--src_dir", required=True,
                        help="Directory containing images/ and labels/ (full converted set)")
    parser.add_argument("--out_dir", required=True,
                        help="Output parent directory (train/, val/, test/ created here)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    split_dataset(
        src_dir=Path(args.src_dir),
        out_dir=Path(args.out_dir),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
