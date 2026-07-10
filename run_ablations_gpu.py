import sys
from pathlib import Path
import torch
from ultralytics import YOLO
import shutil

def main():
    PROJECT_ROOT = Path('.').resolve()
    CONFIG_PATH = PROJECT_ROOT / 'configs' / 'gtsdb_yolov8n.yaml'

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using Device: {DEVICE}')
    if DEVICE == 'cuda':
        print(f'GPU Name: {torch.cuda.get_device_name(0)}')

    MODEL_WEIGHTS = 'yolov8n.pt'
    EPOCHS = 50
    IMG_SIZE = 640
    BATCH_SIZE = 16

    print('\n--- Running Section 5: Ablation (No Augmentation) ---')
    model_no_aug = YOLO(MODEL_WEIGHTS)
    results_no_aug = model_no_aug.train(
        data=str(CONFIG_PATH),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        name='gtsdb_yolov8n_noaug_v1',
        project=str(PROJECT_ROOT / 'runs' / 'train'),
        exist_ok=True,
        # Disable all augmentation
        fliplr=0.0, flipud=0.0, degrees=0.0,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        mosaic=0.0, mixup=0.0, scale=0.0,
        translate=0.0,
    )

    print('\n--- Running Section 6: Ablation (YOLOv8s) ---')
    model_small = YOLO('yolov8s.pt')
    results_small = model_small.train(
        data=str(CONFIG_PATH),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=8, # Smaller batch for larger model
        device=DEVICE,
        name='gtsdb_yolov8s_v1',
        project=str(PROJECT_ROOT / 'runs' / 'train'),
        exist_ok=True,
        fliplr=0.0,
        degrees=15.0,
    )

    print('\n--- Running Section 7: Copying Checkpoints ---')
    checkpoints_dir = PROJECT_ROOT / 'results' / 'checkpoints'
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    for run_name in ['gtsdb_yolov8n_v1', 'gtsdb_yolov8n_noaug_v1', 'gtsdb_yolov8s_v1']:
        best = PROJECT_ROOT / 'runs' / 'train' / run_name / 'weights' / 'best.pt'
        if best.exists():
            dest = checkpoints_dir / f'{run_name}_best.pt'
            shutil.copy2(best, dest)
            print(f'Copied: {dest}')

    print('\n✅ All training complete!')

if __name__ == '__main__':
    # Fix for multiprocessing on Windows
    import multiprocessing
    multiprocessing.freeze_support()
    main()
