import os
import sys
import json
import csv
import numpy as np
from pathlib import Path
from ultralytics import YOLO

def evaluate_yolo_auto():
    weights = 'runs/detect/train7/weights/best.pt'
    data = 'coco_yolo_exact.yaml'
    split = 'val'
    imgsz = 640
    conf = 0.001
    iou = 0.65
    device = '0'
    batch = 16

    if not Path(weights).exists():
        print(f"ERROR: Weight file '{weights}' not found! Exiting.")
        sys.exit(1)
    if not Path(data).exists():
        print(f"ERROR: Data config '{data}' not found! Exiting.")
        sys.exit(1)

    print(f"Running YOLO model evaluation with:")
    print(f"  Weights: {weights}")
    print(f"  Data: {data}")
    print(f"  Split: {split}")
    print(f"  Image size: {imgsz}")
    print(f"  Confidence: {conf}")
    print(f"  IOU: {iou}")
    print(f"  Device: {device}")
    print(f"  Batch size: {batch}")

    model = YOLO(weights)
    metrics = model.val(
        data=data,
        split=split,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        batch=batch,
        plots=True,
        save_json=True,
        verbose=True
    )

    out_dir = Path(metrics.save_dir)
    names = metrics.names
    per_class_ap = metrics.box.maps

    overall = {
        'map50-95': float(metrics.box.map),
        'map50': float(metrics.box.map50),
        'map75': float(metrics.box.map75),
        'precision': float(np.mean(metrics.box.p)),
        'recall': float(np.mean(metrics.box.r))
    }
    (out_dir / 'overall_metrics.json').write_text(json.dumps(overall, indent=2))

    with open(out_dir / 'per_class_ap.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['class_id', 'class_name', 'AP50-95'])
        for cid, ap in enumerate(per_class_ap):
            w.writerow([cid, names.get(cid, str(cid)), 0.0 if ap is None else float(ap)])

    print(f'Done, results saved to: {out_dir}')
    print(f'- mAP50-95: {overall["map50-95"]:.4f}, mAP50: {overall["map50"]:.4f}, Precision: {overall["precision"]:.4f}, Recall: {overall["recall"]:.4f}')
    print(f'- Confusion matrix, per-class AP, and PR/F1 curves are saved in {out_dir}')

if __name__ == "__main__":
    evaluate_yolo_auto()
