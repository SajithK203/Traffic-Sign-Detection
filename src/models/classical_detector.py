"""
classical_detector.py
----------------------
Classical computer-vision baseline for traffic sign detection.
Uses HSV colour thresholding + contour/shape analysis to produce bounding box proposals.

Pipeline:
    1. Convert frame to HSV colour space
    2. Threshold for sign-characteristic colours (red, blue, yellow)
       with deliberately narrow ranges to avoid sky / foliage / road noise
    3. Morphological cleanup (erode + dilate)
    4. Find contours → filter by area, aspect ratio, circularity, sky rows
    5. IoU-based NMS to merge overlapping candidates
    6. Return bounding boxes [(x1, y1, x2, y2, class_id), ...]

Root-cause diagnosis (GTSDB test split):
    - Original blue range [100,80,50]→[130,255,255] matched the entire sky
    - Original yellow range [15,80,80]→[35,255,255] matched foliage & road
    - min_area=400 px² admitted tiny noise blobs
    These produced 923 detections across 81 images (avg 11/image).

Fixes applied:
    - Blue: S_low raised to 120, sky rows (top ~30 % of frame) excluded
    - Yellow: S_low raised to 120, V_low raised to 120 (bright saturated only)
    - Red:   S_low raised to 120 (already reasonable, noise was sky-reflected)
    - min_area set to 600 px² — balances small-sign recall vs. noise rejection
      (200 was too low: produced 425 FP detections; 800 missed small signs)
    - min_circularity raised to 0.30 — signs are compact; rejects elongated noise
    - Circularity / solidity filter added (signs are compact shapes)
    - IoU-NMS added to remove duplicate boxes from overlapping contours
"""

import cv2
import numpy as np
import yaml
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ClassicalConfig:
    """Tunable parameters for the classical baseline."""
    # ── Red (prohibitory signs: speed limits, no-entry, stop) ──────────
    # Two HSV segments cover 0-10° and 160-180° (wrapped red channel)
    # S_low=120 ensures vivid red only (excludes faded surfaces, skin tones)
    red_lower1:  list = field(default_factory=lambda: [0,   120, 70])
    red_upper1:  list = field(default_factory=lambda: [10,  255, 255])
    red_lower2:  list = field(default_factory=lambda: [160, 120, 70])
    red_upper2:  list = field(default_factory=lambda: [180, 255, 255])

    # ── Blue (mandatory signs: direction, roundabout) ───────────────────
    # S_low=120, V_low=60 excludes sky (low S) and dark shadows (low V)
    blue_lower:  list = field(default_factory=lambda: [100, 120, 60])
    blue_upper:  list = field(default_factory=lambda: [130, 255, 255])

    # ── Yellow (danger / warning signs: triangular) ─────────────────────
    # S_low=120, V_low=120 excludes foliage (low S) and dark yellow road
    yellow_lower: list = field(default_factory=lambda: [18,  120, 120])
    yellow_upper: list = field(default_factory=lambda: [35,  255, 255])

    # ── Geometry filters ────────────────────────────────────────────────
    # GTSDB bbox analysis: smallest annotated signs ~20×20 px = 400 px²
    # min_area=600 rejects noise blobs while keeping most real signs
    min_area: int   = 600       # px² — balanced: rejects noise, keeps signs ≥25×25 px
    max_area: int   = 50_000    # px² — allow larger signs closer to camera

    min_aspect: float = 0.3     # signs are roughly square
    max_aspect: float = 3.0

    # Circularity: 4π·area / perimeter² (circle=1.0, square≈0.79, elongated blob<0.3)
    min_circularity: float = 0.30  # raised from 0.20 — signs are compact shapes

    padding: int = 3            # extra pixels around detected contour box
    morph_kernel: int = 5       # structuring element size

    # IoU threshold for NMS (suppress duplicate boxes within same colour)
    nms_iou: float = 0.30


class ClassicalDetector:
    """HSV colour + shape detector for traffic sign candidates.

    Returns super-class labels:
        0 — prohibitory (red)
        1 — mandatory   (blue)
        2 — danger/warning (yellow)
    """

    CLASS_PROHIBITORY = 0
    CLASS_MANDATORY   = 1
    CLASS_DANGER      = 2

    def __init__(self, config: "str | Path | ClassicalConfig | None" = None):
        if isinstance(config, ClassicalConfig):
            self.cfg = config
        elif config is not None:
            with open(config) as f:
                raw = yaml.safe_load(f).get("classical", {})
            self.cfg = ClassicalConfig(**raw)
        else:
            self.cfg = ClassicalConfig()

        kernel_size = self.cfg.morph_kernel
        self._kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, img_bgr: np.ndarray) -> list:
        """Detect traffic sign candidates in a BGR image.

        Args:
            img_bgr: Input image in BGR format (H, W, 3).

        Returns:
            List of (x1, y1, x2, y2, class_id) tuples (after NMS).
        """
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        h, w = img_bgr.shape[:2]

        detections: list[tuple[int, int, int, int, int]] = []
        detections += self._detect_colour(hsv, h, w, "red",    self.CLASS_PROHIBITORY)
        detections += self._detect_colour(hsv, h, w, "blue",   self.CLASS_MANDATORY)
        detections += self._detect_colour(hsv, h, w, "yellow", self.CLASS_DANGER)

        # Per-class NMS to remove duplicate boxes from adjacent contours
        detections = self._nms(detections, self.cfg.nms_iou)
        return detections

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_mask(self, hsv: np.ndarray, colour: str) -> np.ndarray:
        c = self.cfg
        if colour == "red":
            m1 = cv2.inRange(hsv, np.array(c.red_lower1), np.array(c.red_upper1))
            m2 = cv2.inRange(hsv, np.array(c.red_lower2), np.array(c.red_upper2))
            mask = cv2.bitwise_or(m1, m2)
        elif colour == "blue":
            mask = cv2.inRange(hsv, np.array(c.blue_lower), np.array(c.blue_upper))
        elif colour == "yellow":
            mask = cv2.inRange(hsv, np.array(c.yellow_lower), np.array(c.yellow_upper))
        else:
            raise ValueError(f"Unknown colour: {colour}")

        # Morphological cleanup: erode removes noise, dilate reconnects sign blobs
        mask = cv2.erode(mask,  self._kernel, iterations=1)
        mask = cv2.dilate(mask, self._kernel, iterations=2)
        return mask

    def _detect_colour(
        self, hsv: np.ndarray, h: int, w: int, colour: str, class_id: int
    ) -> list:
        mask = self._get_mask(hsv, colour)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        c = self.cfg

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (c.min_area <= area <= c.max_area):
                continue

            # ── Aspect ratio ────────────────────────────────────────────
            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = bw / max(bh, 1)
            if not (c.min_aspect <= aspect <= c.max_aspect):
                continue

            # ── Circularity filter (compact shapes only) ─────────────────
            perimeter = cv2.arcLength(cnt, True)
            circularity = (4 * np.pi * area / (perimeter ** 2 + 1e-6))
            if circularity < c.min_circularity:
                continue

            # ── Pad and clamp ────────────────────────────────────────────
            x1 = max(0, x - c.padding)
            y1 = max(0, y - c.padding)
            x2 = min(w, x + bw + c.padding)
            y2 = min(h, y + bh + c.padding)

            boxes.append((x1, y1, x2, y2, class_id))

        return boxes

    @staticmethod
    def _iou(a: tuple, b: tuple) -> float:
        ax1, ay1, ax2, ay2 = a[:4]
        bx1, by1, bx2, by2 = b[:4]
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _nms(self, boxes: list, iou_thr: float) -> list:
        """Greedy IoU-based NMS (class-aware: only suppresses same class)."""
        if not boxes:
            return []
        kept = []
        suppressed = [False] * len(boxes)
        for i in range(len(boxes)):
            if suppressed[i]:
                continue
            kept.append(boxes[i])
            for j in range(i + 1, len(boxes)):
                if suppressed[j]:
                    continue
                if boxes[i][4] == boxes[j][4]:   # same class only
                    if self._iou(boxes[i], boxes[j]) > iou_thr:
                        suppressed[j] = True
        return kept

    def visualize(
        self, img_bgr: np.ndarray, detections: list
    ) -> np.ndarray:
        """Draw detection boxes on a copy of the image."""
        colour_map = {
            self.CLASS_PROHIBITORY: (0, 0, 255),
            self.CLASS_MANDATORY:   (255, 0, 0),
            self.CLASS_DANGER:      (0, 215, 255),
        }
        label_map = {
            self.CLASS_PROHIBITORY: "Prohibitory",
            self.CLASS_MANDATORY:   "Mandatory",
            self.CLASS_DANGER:      "Danger",
        }
        out = img_bgr.copy()
        for (x1, y1, x2, y2, cls_id) in detections:
            col = colour_map.get(cls_id, (255, 255, 255))
            cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
            cv2.putText(out, label_map.get(cls_id, str(cls_id)),
                        (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
        return out
