"""
yolo_wrapper.py
----------------
Thin wrapper around Ultralytics YOLO for training, evaluation, and inference.
Supports YOLOv8 and YOLO11 models with a consistent interface.

Usage:
    # Fine-tune
    wrapper = YOLOWrapper(model="yolov8n.pt")
    wrapper.train(config="configs/gtsdb_yolov8n.yaml")

    # Evaluate
    metrics = wrapper.evaluate(data="configs/gtsdb_yolov8n.yaml", split="test")

    # Inference
    results = wrapper.predict("path/to/image.jpg", conf=0.25)
"""

from pathlib import Path
from ultralytics import YOLO


class YOLOWrapper:
    """Wrapper around Ultralytics YOLO for traffic sign detection."""

    def __init__(self, model: str = "yolov8n.pt"):
        """Initialise from a model name or checkpoint path.

        Args:
            model: Ultralytics model identifier (e.g. 'yolov8n.pt', 'yolo11n.pt')
                   or path to a trained checkpoint ('results/checkpoints/best.pt').
        """
        self.model_path = model
        self.model = YOLO(model)
        print(f"[YOLOWrapper] Loaded model: {model}")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, config: str, **kwargs) -> None:
        """Fine-tune the model using an Ultralytics YAML dataset config.

        Args:
            config: Path to a YAML file containing dataset paths and hyperparams.
            **kwargs: Additional arguments passed to YOLO.train()
                      (e.g. epochs=50, imgsz=640, batch=16, device='cuda').
        """
        print(f"[YOLOWrapper] Starting training with config: {config}")
        self.model.train(data=config, **kwargs)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, data: str, split: str = "test", **kwargs) -> dict:
        """Run evaluation on a dataset split and return metrics dict.

        Args:
            data: Path to Ultralytics YAML dataset config.
            split: One of 'train', 'val', 'test'.
            **kwargs: Additional args passed to YOLO.val().

        Returns:
            Dict with keys: map50, map50_95, precision, recall, etc.
        """
        print(f"[YOLOWrapper] Evaluating on split='{split}', data='{data}'")
        results = self.model.val(data=data, split=split, **kwargs)
        metrics = {
            "map50":      results.box.map50,
            "map50_95":   results.box.map,
            "precision":  results.box.mp,
            "recall":     results.box.mr,
        }
        print(f"[YOLOWrapper] Results: {metrics}")
        return metrics

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        source: str | Path,
        conf: float = 0.25,
        iou: float = 0.45,
        save: bool = False,
        **kwargs,
    ):
        """Run inference on an image, directory, or video.

        Args:
            source: Image/video path, directory, or URL.
            conf: Confidence threshold.
            iou: NMS IoU threshold.
            save: Whether to save annotated output to disk.
            **kwargs: Additional Ultralytics predict() arguments.

        Returns:
            Ultralytics Results list.
        """
        return self.model.predict(
            source=source, conf=conf, iou=iou, save=save, **kwargs
        )

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export(self, format: str = "onnx", **kwargs) -> Path:
        """Export model to another format (ONNX, TFLite, TensorRT, etc.)."""
        return self.model.export(format=format, **kwargs)
