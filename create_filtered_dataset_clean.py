import shutil
from pathlib import Path

def create_clean_filtered_dataset():
    base_path = Path('/mnt/34B471F7B471BBC4/CSO_project/datasets')
    filtered_train_path = base_path / 'filtered_train2017'
    filtered_val_path = base_path / 'filtered_val2017'
    if filtered_train_path.exists():
        shutil.rmtree(filtered_train_path)
    if filtered_val_path.exists():
        shutil.rmtree(filtered_val_path)
    filtered_train_path.mkdir(exist_ok=True)
    filtered_val_path.mkdir(exist_ok=True)
    (filtered_train_path / 'labels').mkdir(exist_ok=True)
    (filtered_val_path / 'labels').mkdir(exist_ok=True)
    original_train_labels = base_path / 'labels' / 'train2017'
    original_train_images = base_path / 'train2017'
    train_count = 0
    for label_file in original_train_labels.glob('*.txt'):
        image_file = original_train_images / f"{label_file.stem}.jpg"
        if image_file.exists():
            shutil.copy2(image_file, filtered_train_path / image_file.name)
            shutil.copy2(label_file, filtered_train_path / 'labels' / label_file.name)
            train_count += 1
    original_val_labels = base_path / 'labels' / 'val2017'
    original_val_images = base_path / 'validation' / 'val2017'
    val_count = 0
    for label_file in original_val_labels.glob('*.txt'):
        image_file = original_val_images / f"{label_file.stem}.jpg"
        if image_file.exists():
            shutil.copy2(image_file, filtered_val_path / image_file.name)
            shutil.copy2(label_file, filtered_val_path / 'labels' / label_file.name)
            val_count += 1
    create_filtered_yaml(train_count, val_count)
    return train_count, val_count

def create_filtered_yaml(train_count, val_count):
    yaml_content = f"""# Filtered COCO 2017 dataset with 30 classes
path: /mnt/34B471F7B471BBC4/CSO_project/datasets
train: filtered_train2017
val: filtered_val2017
nc: 30
names: ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'backpack', 'umbrella', 'handbag', 'tie',
        'skis', 'snowboard', 'sports ball', 'kite',
        'banana', 'apple', 'sandwich']
"""
    with open('coco_filtered.yaml', 'w') as f:
        f.write(yaml_content)

if __name__ == "__main__":
    create_clean_filtered_dataset()
