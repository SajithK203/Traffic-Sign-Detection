"""
dataset_loader.py
------------------
PyTorch Dataset class for loading YOLO-format annotated traffic sign images.
Used by custom training loops or evaluation scripts outside of the Ultralytics CLI.
"""

import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class TrafficSignDataset(Dataset):
    """Loads images and YOLO-format bounding box labels from disk.

    Directory layout expected:
        root/
          images/  *.jpg (or *.png)
          labels/  *.txt  (one per image, YOLO normalized format)

    Args:
        root (str | Path): Root directory containing images/ and labels/.
        img_size (int): Resize images to (img_size, img_size).
        transform: Optional albumentations transform applied to image + bboxes.
    """

    def __init__(self, root: str | Path, img_size: int = 640, transform=None):
        self.root = Path(root)
        self.img_size = img_size
        self.transform = transform

        self.img_dir = self.root / "images"
        self.lbl_dir = self.root / "labels"

        if not self.img_dir.exists():
            raise FileNotFoundError(f"images/ directory not found: {self.img_dir}")
        if not self.lbl_dir.exists():
            raise FileNotFoundError(f"labels/ directory not found: {self.lbl_dir}")

        self.img_paths = sorted(
            p for p in self.img_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".ppm"}
        )

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        img_path = self.img_paths[idx]
        lbl_path = self.lbl_dir / (img_path.stem + ".txt")

        # Load image (BGR → RGB)
        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"Could not read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))

        # Load labels
        boxes = []
        if lbl_path.exists():
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        boxes.append([float(p) for p in parts])

        boxes = torch.tensor(boxes, dtype=torch.float32)  # shape: (N, 5)

        # Optional albumentations transform
        if self.transform:
            bboxes = boxes[:, 1:].tolist() if len(boxes) else []
            class_labels = boxes[:, 0].tolist() if len(boxes) else []
            transformed = self.transform(
                image=img,
                bboxes=bboxes,
                class_labels=class_labels,
            )
            img = transformed["image"]
            new_bboxes = transformed["bboxes"]
            new_labels = transformed["class_labels"]
            if new_bboxes:
                boxes = torch.tensor(
                    [[c, *b] for c, b in zip(new_labels, new_bboxes)],
                    dtype=torch.float32,
                )
            else:
                boxes = torch.zeros((0, 5), dtype=torch.float32)

        # Normalise image to [0, 1] and convert HWC → CHW
        img_tensor = torch.from_numpy(img).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1)

        return img_tensor, boxes, str(img_path)
