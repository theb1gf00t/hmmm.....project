import numpy as np
from ultralytics import YOLO

# Evaluate pretrained YOLOv8s (baseline)
baseline_model = YOLO('yolov8s.pt')  # Pretrained on COCO 80 classes
baseline_metrics = baseline_model.val(
    data='coco_yolo_exact.yaml',
    split='val',
    imgsz=640,
    conf=0.001,
    iou=0.65,
    device='0',
    batch=16,
    plots=False,
    verbose=True
)

print(f"Baseline YOLOv8s (pretrained):")
print(f"  mAP50-95: {baseline_metrics.box.map:.4f}")
print(f"  mAP50: {baseline_metrics.box.map50:.4f}")
print(f"  Precision: {np.mean(baseline_metrics.box.p):.4f}")
print(f"  Recall: {np.mean(baseline_metrics.box.r):.4f}")

# Evaluate your fine-tuned model
finetuned_model = YOLO('runs/detect/train7/weights/best.pt')
finetuned_metrics = finetuned_model.val(
    data='coco_yolo_exact.yaml',
    split='val',
    imgsz=640,
    conf=0.001,
    iou=0.65,
    device='0',
    batch=16,
    plots=False,
    verbose=True
)

print(f"\nFine-tuned YOLOv8s (your model):")
print(f"  mAP50-95: {finetuned_metrics.box.map:.4f}")
print(f"  mAP50: {finetuned_metrics.box.map50:.4f}")
print(f"  Precision: {np.mean(finetuned_metrics.box.p):.4f}")
print(f"  Recall: {np.mean(finetuned_metrics.box.r):.4f}")

# Calculate improvement
improvement_map50 = ((finetuned_metrics.box.map50 - baseline_metrics.box.map50) / baseline_metrics.box.map50) * 100
improvement_map = ((finetuned_metrics.box.map - baseline_metrics.box.map) / baseline_metrics.box.map) * 100

print(f"\nImprovement:")
print(f"  mAP50: +{improvement_map50:.2f}%")
print(f"  mAP50-95: +{improvement_map:.2f}%")
