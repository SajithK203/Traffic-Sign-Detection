"""
metrics.py
-----------
Evaluation metric helpers:
  - IoU calculation
  - Precision/Recall/AP computation
  - Classical baseline evaluation loop
"""

from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np


def compute_iou(box_a, box_b) -> float:
    """Compute Intersection over Union for two [x1, y1, x2, y2] boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union_area = area_a + area_b - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def compute_ap(precisions: np.ndarray, recalls: np.ndarray) -> float:
    """11-point interpolated Average Precision."""
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        mask = recalls >= t
        ap += (precisions[mask].max() if mask.any() else 0.0) / 11.0
    return ap


def compute_precision_recall(detections, ground_truths, iou_threshold=0.5):
    """Compute precision/recall curve and AP.

    Args:
        detections:    list of {'box', 'conf', 'img_id'}
        ground_truths: list of {'box', 'img_id'}
        iou_threshold: float

    Returns:
        (precisions, recalls, ap)
    """
    detections = sorted(detections, key=lambda d: d["conf"], reverse=True)
    gt_by_img = {}
    for gt in ground_truths:
        gt_by_img.setdefault(gt["img_id"], []).append({"box": gt["box"], "matched": False})

    tp = np.zeros(len(detections))
    fp = np.zeros(len(detections))
    n_gt = len(ground_truths)

    for i, det in enumerate(detections):
        gts = gt_by_img.get(det["img_id"], [])
        best_iou, best_j = 0.0, -1
        for j, gt in enumerate(gts):
            iou = compute_iou(det["box"], gt["box"])
            if iou > best_iou:
                best_iou, best_j = iou, j

        if best_iou >= iou_threshold and best_j >= 0 and not gts[best_j]["matched"]:
            tp[i] = 1
            gts[best_j]["matched"] = True
        else:
            fp[i] = 1

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recalls    = cum_tp / (n_gt + 1e-10)
    precisions = cum_tp / (cum_tp + cum_fp + 1e-10)
    return precisions, recalls, compute_ap(precisions, recalls)


def evaluate_classical_baseline(test_dir) -> dict:
    """Run classical detector on test images and return metrics."""
    from src.models.classical_detector import ClassicalDetector

    test_dir   = Path(test_dir)
    images_dir = test_dir / "images"
    labels_dir = test_dir / "labels"
    detector   = ClassicalDetector()

    all_dets, all_gts = [], []
    img_paths = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".ppm"}
    )
    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        img_id = img_path.stem

        lbl_path = labels_dir / f"{img_id}.txt"
        if lbl_path.exists():
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id, xc, yc, bw, bh = map(float, parts)
                        x1 = int((xc - bw / 2) * w)
                        y1 = int((yc - bh / 2) * h)
                        x2 = int((xc + bw / 2) * w)
                        y2 = int((yc + bh / 2) * h)
                        all_gts.append({"box": [x1, y1, x2, y2], "img_id": img_id})

        for (x1, y1, x2, y2, _) in detector.detect(img):
            all_dets.append({"box": [x1, y1, x2, y2], "conf": 1.0, "img_id": img_id})

    _, _, ap = compute_precision_recall(all_dets, all_gts)
    n_tp = sum(
        1 for d in all_dets
        if any(compute_iou(d["box"], g["box"]) >= 0.5
               for g in all_gts if g["img_id"] == d["img_id"])
    )
    precision = n_tp / max(len(all_dets), 1)
    recall    = n_tp / max(len(all_gts), 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-10)
    return {"precision": precision, "recall": recall, "f1": f1, "ap": ap,
            "n_detections": len(all_dets), "n_ground_truths": len(all_gts)}
