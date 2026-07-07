"""
inference.py
-------------
Single-image or batch inference script. Draws bounding boxes and saves
annotated output to results/qualitative_examples/.

Usage:
    # Single image
    python src/inference.py --weights results/checkpoints/best.pt --source img.jpg

    # Directory of images
    python src/inference.py --weights results/checkpoints/best.pt --source data/processed/gtsdb/test/images/

    # Live webcam
    python src/inference.py --weights results/checkpoints/best.pt --source 0

    # Classical baseline
    python src/inference.py --model classical --source img.jpg
"""

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.yolo_wrapper import YOLOWrapper
from src.models.classical_detector import ClassicalDetector


CLASS_NAMES = {0: "Prohibitory", 1: "Mandatory", 2: "Danger"}
COLOURS     = {0: (0, 0, 255), 1: (255, 100, 0), 2: (0, 200, 255)}


def parse_args():
    parser = argparse.ArgumentParser(description="Traffic sign inference")
    parser.add_argument("--source",  required=True,
                        help="Image path, directory, video path, or webcam index (0)")
    parser.add_argument("--model",   default="yolov8",
                        choices=["yolov8", "classical"],
                        help="Model type (default: yolov8)")
    parser.add_argument("--weights", default=None,
                        help="Path to YOLO checkpoint (required for --model yolov8)")
    parser.add_argument("--conf",    type=float, default=0.25,
                        help="Confidence threshold")
    parser.add_argument("--iou",     type=float, default=0.45,
                        help="NMS IoU threshold")
    parser.add_argument("--out_dir", default="results/qualitative_examples",
                        help="Directory for annotated output images")
    parser.add_argument("--show",    action="store_true",
                        help="Display results in a window (requires display)")
    return parser.parse_args()


def run_on_image(img_bgr, model_type, wrapper=None, classical=None, conf=0.25, iou=0.45):
    """Run inference on one BGR image and return annotated image."""
    if model_type == "yolov8":
        results = wrapper.predict(source=img_bgr, conf=conf, iou=iou)
        annotated = results[0].plot()  # Ultralytics built-in annotation
        return annotated

    elif model_type == "classical":
        detections = classical.detect(img_bgr)
        annotated = classical.visualize(img_bgr, detections)
        return annotated


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    wrapper   = None
    classical = None
    if args.model == "yolov8":
        if not args.weights:
            print("[ERROR] --weights required for --model yolov8")
            sys.exit(1)
        wrapper = YOLOWrapper(model=args.weights)
    else:
        classical = ClassicalDetector()

    source = args.source

    # Webcam / video source
    if source.isdigit() or source.endswith((".mp4", ".avi", ".mov")):
        cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
        print("[Inference] Streaming — press 'q' to quit")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            annotated = run_on_image(frame, args.model, wrapper, classical,
                                     args.conf, args.iou)
            if args.show:
                cv2.imshow("Traffic Sign Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        cap.release()
        cv2.destroyAllWindows()
        return

    # Directory or single image
    source_path = Path(source)
    if source_path.is_dir():
        img_paths = sorted(
            p for p in source_path.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".ppm"}
        )
    elif source_path.is_file():
        img_paths = [source_path]
    else:
        print(f"[ERROR] Source not found: {source}")
        sys.exit(1)

    print(f"[Inference] Processing {len(img_paths)} image(s)...")
    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [WARN] Could not read: {img_path}")
            continue

        annotated = run_on_image(img, args.model, wrapper, classical,
                                 args.conf, args.iou)
        out_path = out_dir / f"{args.model}_{img_path.name}"
        cv2.imwrite(str(out_path), annotated)

        if args.show:
            cv2.imshow("Traffic Sign Detection", annotated)
            cv2.waitKey(0)

    print(f"\n✅ Done. Annotated images saved → {out_dir}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
