"""
classical_detector.py
----------------------
Classical computer-vision baseline for traffic sign detection.
Uses HSV colour thresholding + contour/shape analysis to produce bounding box proposals.

Pipeline:
    1. Convert frame to HSV colour space
    2. Threshold for sign-characteristic colours (red, blue, yellow)
    3. Morphological cleanup (erode + dilate)
    4. Find contours → filter by area, aspect ratio, shape
    5. Return bounding boxes [(x1, y1, x2, y2, class_id), ...]

Usage:
    detector = ClassicalDetector(config="configs/classical.yaml")
    boxes = detector.detect(image_bgr)
"""

import cv2
import numpy as np
import yaml
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ClassicalConfig:
    """Tunable parameters for the classical baseline."""
    # HSV ranges: (H_low, S_low, V_low), (H_high, S_high, V_high)
    red_lower1:  list = field(default_factory=lambda: [0,   100, 50])
    red_upper1:  list = field(default_factory=lambda: [10,  255, 255])
    red_lower2:  list = field(default_factory=lambda: [160, 100, 50])
    red_upper2:  list = field(default_factory=lambda: [180, 255, 255])
    blue_lower:  list = field(default_factory=lambda: [100, 80,  50])
    blue_upper:  list = field(default_factory=lambda: [130, 255, 255])
    yellow_lower: list = field(default_factory=lambda: [15,  80,  80])
    yellow_upper: list = field(default_factory=lambda: [35,  255, 255])

    min_area: int = 400          # px² — ignore tiny blobs
    max_area: int = 80_000       # px² — ignore full-frame regions
    min_aspect: float = 0.3      # width/height ratio lower bound
    max_aspect: float = 3.0      # width/height ratio upper bound
    padding: int = 4             # extra pixels around detected contour box

    # Morphological kernel size
    morph_kernel: int = 5


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

    def __init__(self, config: str | Path | ClassicalConfig | None = None):
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

    def detect(self, img_bgr: np.ndarray) -> list[tuple[int, int, int, int, int]]:
        """Detect traffic sign candidates in a BGR image.

        Args:
            img_bgr: Input image in BGR format (H, W, 3).

        Returns:
            List of (x1, y1, x2, y2, class_id) tuples.
        """
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        h, w = img_bgr.shape[:2]

        detections = []
        detections += self._detect_colour(hsv, h, w, "red",    self.CLASS_PROHIBITORY)
        detections += self._detect_colour(hsv, h, w, "blue",   self.CLASS_MANDATORY)
        detections += self._detect_colour(hsv, h, w, "yellow", self.CLASS_DANGER)
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

        # Morphological cleanup
        mask = cv2.erode(mask, self._kernel, iterations=1)
        mask = cv2.dilate(mask, self._kernel, iterations=2)
        return mask

    def _detect_colour(
        self, hsv: np.ndarray, h: int, w: int, colour: str, class_id: int
    ) -> list[tuple]:
        mask = self._get_mask(hsv, colour)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        c = self.cfg
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (c.min_area <= area <= c.max_area):
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = bw / max(bh, 1)
            if not (c.min_aspect <= aspect <= c.max_aspect):
                continue

            # Pad and clamp
            x1 = max(0, x - c.padding)
            y1 = max(0, y - c.padding)
            x2 = min(w, x + bw + c.padding)
            y2 = min(h, y + bh + c.padding)

            boxes.append((x1, y1, x2, y2, class_id))

        return boxes

    def visualize(
        self, img_bgr: np.ndarray, detections: list[tuple]
    ) -> np.ndarray:
        """Draw detection boxes on a copy of the image."""
        colour_map = {
            self.CLASS_PROHIBITORY: (0, 0, 255),    # red
            self.CLASS_MANDATORY:   (255, 0, 0),    # blue
            self.CLASS_DANGER:      (0, 215, 255),  # yellow
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
