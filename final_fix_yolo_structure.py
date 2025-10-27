import shutil
from pathlib import Path

def create_exact_yolo_structure():
    base_path = Path('/mnt/34B471F7B471BBC4/CSO_project/datasets')
    yolo_train = base_path / 'images' / 'train'
    yolo_val = base_path / 'images' / 'val'
    yolo_train_labels = base_path / 'labels' / 'train'
    yolo_val_labels = base_path / 'labels' / 'val'
    for dir_path in [yolo_train, yolo_val, yolo_train_labels, yolo_val_labels]:
        if dir_path.exists():
            shutil.rmtree(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
    original_train_images = base_path / 'filtered_train2017'
    original_train_labels = base_path / 'filtered_train2017' / 'labels'
    train_count = 0
    for image_file in original_train_images.glob('*.jpg'):
        shutil.copy2(image_file, yolo_train / image_file.name)
        label_file = original_train_labels / f"{image_file.stem}.txt"
        if label_file.exists():
            shutil.copy2(label_file, yolo_train_labels / f"{image_file.stem}.txt")
            train_count += 1
    original_val_images = base_path / 'filtered_val2017'
    original_val_labels = base_path / 'filtered_val2017' / 'labels'
    val_count = 0
    for image_file in original_val_images.glob('*.jpg'):
        shutil.copy2(image_file, yolo_val / image_file.name)
        label_file = original_val_labels / f"{image_file.stem}.txt"
        if label_file.exists():
            shutil.copy2(label_file, yolo_val_labels / f"{image_file.stem}.txt")
            val_count += 1
    create_final_yolo_yaml(train_count, val_count)
    return train_count, val_count

def create_final_yolo_yaml(train_count, val_count):
    yaml_content = f"""# YOLO dataset configuration
path: /mnt/34B471F7B471BBC4/CSO_project/datasets
train: images/train
val: images/val
nc: 30
names: ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'backpack', 'umbrella', 'handbag', 'tie',
        'skis', 'snowboard', 'sports ball', 'kite',
        'banana', 'apple', 'sandwich']
"""
    with open('coco_yolo_exact.yaml', 'w') as f:
        f.write(yaml_content)

if __name__ == "__main__":
    create_exact_yolo_structure()
