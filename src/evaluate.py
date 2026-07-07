"""
evaluate.py
------------
Evaluation entry point. Runs any model variant on the test split and
writes metrics to results/metrics/<run_name>.json.

Supported models:
    --model classical    : Classical HSV+contour baseline
    --model zero-shot    : Pretrained COCO YOLOv8n (no fine-tuning)
    --model yolov8       : Fine-tuned YOLO checkpoint (requires --weights)

Usage:
    # Classical baseline
    python src/evaluate.py --model classical --data data/processed/gtsdb/test

    # Zero-shot YOLO baseline (COCO pretrained, no fine-tuning)
    python src/evaluate.py --model zero-shot --weights yolov8n.pt --data configs/gtsdb_yolov8n.yaml

    # Fine-tuned YOLO
    python src/evaluate.py --model yolov8 --weights results/checkpoints/best.pt --data configs/gtsdb_yolov8n.yaml
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.yolo_wrapper import YOLOWrapper
from src.utils.metrics import evaluate_classical_baseline


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a traffic sign detection model")
    parser.add_argument("--model", required=True,
                        choices=["classical", "zero-shot", "yolov8"],
                        help="Which model type to evaluate")
    parser.add_argument("--weights", default=None,
                        help="Path to YOLO checkpoint or pretrained weights file")
    parser.add_argument("--data", required=True,
                        help="Dataset path (YOLO YAML config or test image directory)")
    parser.add_argument("--split", default="test",
                        help="Dataset split to evaluate on (default: test)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold for YOLO predictions")
    parser.add_argument("--iou",  type=float, default=0.45,
                        help="IoU threshold for NMS")
    parser.add_argument("--name", default=None,
                        help="Run name for saving results (default: model type)")
    parser.add_argument("--out_dir", default="results/metrics",
                        help="Directory to save metrics JSON")
    return parser.parse_args()


def main():
    args = parse_args()
    run_name = args.name or args.model
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Traffic Sign Detection — Evaluation")
    print(f"  Model : {args.model}")
    print(f"  Data  : {args.data}")
    print(f"  Split : {args.split}")
    print(f"{'='*60}\n")

    metrics = {}

    # ------------------------------------------------------------------
    # Classical baseline
    # ------------------------------------------------------------------
    if args.model == "classical":
        print("[Evaluate] Running Classical CV Baseline...")
        metrics = evaluate_classical_baseline(test_dir=args.data)

    # ------------------------------------------------------------------
    # Zero-shot YOLO (COCO pretrained, no fine-tuning)
    # ------------------------------------------------------------------
    elif args.model == "zero-shot":
        weights = args.weights or "yolov8n.pt"
        print(f"[Evaluate] Running Zero-Shot YOLO baseline ({weights})...")
        wrapper = YOLOWrapper(model=weights)
        metrics = wrapper.evaluate(data=args.data, split=args.split,
                                   conf=args.conf, iou=args.iou)

    # ------------------------------------------------------------------
    # Fine-tuned YOLO checkpoint
    # ------------------------------------------------------------------
    elif args.model == "yolov8":
        if not args.weights:
            print("[ERROR] --weights is required for --model yolov8")
            sys.exit(1)
        print(f"[Evaluate] Running Fine-Tuned YOLO ({args.weights})...")
        wrapper = YOLOWrapper(model=args.weights)
        metrics = wrapper.evaluate(data=args.data, split=args.split,
                                   conf=args.conf, iou=args.iou)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    out_path = out_dir / f"{run_name}.json"
    with open(out_path, "w") as f:
        json.dump({"model": args.model, "weights": args.weights,
                   "data": args.data, "split": args.split, **metrics}, f, indent=2)

    print(f"\n✅ Metrics saved → {out_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
