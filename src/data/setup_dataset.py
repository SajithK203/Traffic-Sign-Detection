"""
setup_dataset.py
-----------------
One-command dataset setup for team members who have:
  1. Cloned the repo (gets all .txt label files automatically)
  2. Downloaded the GTSDB raw zip and extracted it somewhere under data/raw/gtsdb/

What this script does:
  Step A — Auto-detect the folder containing .ppm files and gt.txt
  Step B — For each image stem that has a label in data/processed/gtsdb/all/labels/,
            convert the matching .ppm → .jpg into data/processed/gtsdb/all/images/
  Step C — Copy images from all/images/ into train/val/test/images/ to match
            the existing label files already committed to GitHub

Usage (from repo root, with venv active):
    python src/data/setup_dataset.py

Optional flags:
    --raw_dir  <path>   Path to the folder containing the .ppm files
                        (auto-detected if not supplied)
    --quality  <int>    JPEG quality 1-100 (default: 95)
    --dry_run           Print what would happen without writing files

Requirements:
    pip install opencv-python-headless
"""

import argparse
import shutil
import sys
from pathlib import Path

try:
    import cv2
except ImportError:
    sys.exit(
        "[ERROR] opencv-python-headless is not installed.\n"
        "  Run: pip install opencv-python-headless"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def find_ppm_dir(start: Path) -> Path | None:
    """Search under `start` for a directory that contains .ppm files."""
    for candidate in sorted(start.rglob("*.ppm")):
        return candidate.parent  # return the first directory that has .ppm files
    return None


def convert_ppm_to_jpg(ppm_path: Path, jpg_path: Path, quality: int = 95) -> bool:
    """Read a .ppm and write a .jpg. Returns True on success."""
    img = cv2.imread(str(ppm_path))
    if img is None:
        print(f"  [WARN] Could not read: {ppm_path}")
        return False
    jpg_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(jpg_path), img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Main logic
# ──────────────────────────────────────────────────────────────────────────────

def setup_dataset(raw_dir: Path | None = None, quality: int = 95, dry_run: bool = False):
    repo_root   = Path(__file__).resolve().parents[2]  # .../Traffic-Sign-Detection
    gtsdb_dir   = repo_root / "data" / "processed" / "gtsdb"
    all_lbl_dir = gtsdb_dir / "all" / "labels"
    all_img_dir = gtsdb_dir / "all" / "images"

    # ── Sanity check: label files must exist ──────────────────────────────────
    if not all_lbl_dir.exists() or not any(all_lbl_dir.glob("*.txt")):
        sys.exit(
            f"[ERROR] No label files found in:\n  {all_lbl_dir}\n\n"
            "Did you clone the repo correctly?\n"
            "  git clone https://github.com/SajithK203/Traffic-Sign-Detection.git"
        )

    all_stems = sorted(p.stem for p in all_lbl_dir.glob("*.txt"))
    print(f"[INFO] Found {len(all_stems)} label files in all/labels/")

    # ── Step A — Locate the .ppm directory ───────────────────────────────────
    if raw_dir is None:
        search_root = repo_root / "data" / "raw" / "gtsdb"
        print(f"[INFO] Auto-detecting .ppm directory under: {search_root}")
        raw_dir = find_ppm_dir(search_root)
        if raw_dir is None:
            sys.exit(
                "[ERROR] No .ppm files found under data/raw/gtsdb/\n\n"
                "Please download and extract TrainIJCNN2013.zip first:\n"
                "  https://benchmark.ini.rub.de/gtsdb_news.html\n\n"
                "Then extract to: data/raw/gtsdb/TrainIJCNN2013/\n"
                "Expected path:   data/raw/gtsdb/TrainIJCNN2013/00000.ppm"
            )
    else:
        raw_dir = Path(raw_dir)
        if not raw_dir.exists():
            sys.exit(f"[ERROR] Specified --raw_dir does not exist: {raw_dir}")

    ppm_files = sorted(raw_dir.glob("*.ppm"))
    if not ppm_files:
        sys.exit(
            f"[ERROR] No .ppm files found in: {raw_dir}\n"
            "Make sure you extracted the zip correctly."
        )

    print(f"[INFO] Found {len(ppm_files)} .ppm files in: {raw_dir}")
    print()

    # ── Step B — Convert .ppm → .jpg for all label stems ─────────────────────
    print("=" * 60)
    print("Step B — Converting .ppm → .jpg")
    print("=" * 60)

    ppm_index = {p.stem: p for p in ppm_files}
    converted  = 0
    skipped    = 0
    missing    = 0

    for stem in all_stems:
        jpg_path = all_img_dir / f"{stem}.jpg"

        if jpg_path.exists():
            skipped += 1
            continue  # already converted

        ppm_path = ppm_index.get(stem)
        if ppm_path is None:
            print(f"  [WARN] No .ppm found for label {stem}.txt — skipping")
            missing += 1
            continue

        if dry_run:
            print(f"  [DRY-RUN] Would convert {ppm_path.name} → {jpg_path}")
            converted += 1
        else:
            if convert_ppm_to_jpg(ppm_path, jpg_path, quality):
                converted += 1
                if converted % 50 == 0:
                    print(f"  ... converted {converted}/{len(all_stems)}")
            else:
                missing += 1

    print(f"\n  Converted : {converted}")
    print(f"  Skipped   : {skipped}  (already existed)")
    print(f"  Missing   : {missing}  (no .ppm found)")
    print()

    # ── Step C — Copy images into train / val / test ──────────────────────────
    print("=" * 60)
    print("Step C — Copying images into train / val / test splits")
    print("=" * 60)

    splits = ["train", "val", "test"]
    total_copied  = 0
    total_missing = 0

    for split in splits:
        lbl_dir = gtsdb_dir / split / "labels"
        img_dir = gtsdb_dir / split / "images"

        if not lbl_dir.exists():
            print(f"  [WARN] {split}/labels/ not found — skipping")
            continue

        split_stems = [p.stem for p in lbl_dir.glob("*.txt")]
        img_dir.mkdir(parents=True, exist_ok=True)

        copied  = 0
        already = 0
        absent  = 0

        for stem in split_stems:
            dst = img_dir / f"{stem}.jpg"
            if dst.exists():
                already += 1
                continue
            src = all_img_dir / f"{stem}.jpg"
            if not src.exists():
                absent += 1
                continue
            if not dry_run:
                shutil.copy2(src, dst)
            copied += 1

        total_copied  += copied
        total_missing += absent
        print(
            f"  {split:5s} → images/{split}: "
            f"copied={copied}  already={already}  missing={absent}"
        )

    print()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("=" * 60)
    print("Done!")
    print("=" * 60)

    if dry_run:
        print("[DRY-RUN] No files were actually written.")
        return

    print("\nVerification:")
    for split in splits:
        img_dir = gtsdb_dir / split / "images"
        lbl_dir = gtsdb_dir / split / "labels"
        imgs = set(p.stem for p in img_dir.glob("*.jpg"))
        lbls = set(p.stem for p in lbl_dir.glob("*.txt"))
        matched = imgs & lbls
        lbl_only = lbls - imgs
        print(
            f"  {split:5s} -> images:{len(imgs):4d}  labels:{len(lbls):4d}  "
            f"matched:{len(matched):4d}  missing_images:{len(lbl_only)}"
        )

    if total_missing > 0:
        print(
            f"\n[WARN] {total_missing} label(s) have no matching image.\n"
            "This usually means the .ppm file was not in the downloaded zip.\n"
            "Re-download TrainIJCNN2013.zip and try again."
        )
    else:
        print("\n✓ All label files have matching images. Dataset is ready!")
        print("  You can now run: jupyter notebook")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="One-command GTSDB image setup for team members"
    )
    parser.add_argument(
        "--raw_dir", default=None,
        help="Path to the folder containing .ppm files (auto-detected if omitted)"
    )
    parser.add_argument(
        "--quality", type=int, default=95,
        help="JPEG quality 1-100 (default: 95)"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print what would happen without writing any files"
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir) if args.raw_dir else None
    setup_dataset(raw_dir=raw_dir, quality=args.quality, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
