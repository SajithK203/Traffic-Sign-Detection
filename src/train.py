"""
train.py
---------
Training entry point for fine-tuning a YOLO detector on a traffic sign dataset.

Usage:
    python src/train.py --config configs/gtsdb_yolov8n.yaml

The config YAML (Ultralytics format) controls all hyperparameters, dataset paths,
and model architecture. See configs/gtsdb_yolov8n.yaml for the full schema.
"""

import argparse
import sys
from pathlib import Path

# Ensure src/ is on the Python path when running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.yolo_wrapper import YOLOWrapper
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Train a YOLO detector on traffic sign data")
    parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to Ultralytics YAML training config (e.g. configs/gtsdb_yolov8n.yaml)",
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="Override model weights (e.g. yolov8s.pt). Defaults to value in config.",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of training epochs.",
    )
    parser.add_argument(
        "--imgsz", type=int, default=None,
        help="Override input image size.",
    )
    parser.add_argument(
        "--batch", type=int, default=None,
        help="Override batch size.",
    )
    parser.add_argument(
        "--device", default=None,
        help="Device: 'cuda', 'cpu', '0', '0,1', etc.",
    )
    parser.add_argument(
        "--name", default=None,
        help="Run name (for results folder).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config)

    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}")
        sys.exit(1)

    # Load config to get model path
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model_weights = args.model or cfg.get("model", "yolov8n.pt")
    print(f"\n{'='*60}")
    print(f"  Traffic Sign Detection — Training")
    print(f"  Config : {config_path}")
    print(f"  Model  : {model_weights}")
    print(f"{'='*60}\n")

    wrapper = YOLOWrapper(model=model_weights)

    # Build keyword overrides (only pass non-None values)
    train_kwargs = {}
    if args.epochs is not None:  train_kwargs["epochs"]  = args.epochs
    if args.imgsz  is not None:  train_kwargs["imgsz"]   = args.imgsz
    if args.batch  is not None:  train_kwargs["batch"]   = args.batch
    if args.device is not None:  train_kwargs["device"]  = args.device
    if args.name   is not None:  train_kwargs["name"]    = args.name

    wrapper.train(config=str(config_path), **train_kwargs)
    print("\n✅ Training complete. Check runs/ for outputs.")


if __name__ == "__main__":
    main()
