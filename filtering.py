import json
from pathlib import Path

def filter_coco_dataset(original_annotations_path, output_annotations_path, selected_classes):
    with open(original_annotations_path, 'r') as f:
        coco_data = json.load(f)
    selected_class_ids = []
    class_name_to_id = {}
    class_id_to_name = {}
    for category in coco_data['categories']:
        if category['name'] in selected_classes:
            selected_class_ids.append(category['id'])
            class_name_to_id[category['name']] = category['id']
            class_id_to_name[category['id']] = category['name']
    selected_image_ids = set()
    filtered_annotations = []
    for annotation in coco_data['annotations']:
        if annotation['category_id'] in selected_class_ids:
            filtered_annotations.append(annotation)
            selected_image_ids.add(annotation['image_id'])
    filtered_images = [img for img in coco_data['images'] if img['id'] in selected_image_ids]
    filtered_categories = [cat for cat in coco_data['categories'] if cat['id'] in selected_class_ids]
    filtered_data = {
        'images': filtered_images,
        'annotations': filtered_annotations,
        'categories': filtered_categories
    }
    with open(output_annotations_path, 'w') as f:
        json.dump(filtered_data, f, indent=2)
    return filtered_data

SELECTED_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'backpack', 'umbrella', 'handbag', 'tie',
    'skis', 'snowboard', 'sports ball', 'kite',
    'banana', 'apple', 'sandwich'
]

def filter_both_train_val():
    train_images_path = Path('/mnt/34B471F7B471BBC4/CSO_project/datasets/train2017')
    val_images_path = Path('/mnt/34B471F7B471BBC4/CSO_project/datasets/validation/val2017')
    annotations_path = Path('/mnt/34B471F7B471BBC4/CSO_project/datasets/train_labels/annotations_trainval2017/annotations')
    train_annotations = annotations_path / 'instances_train2017.json'
    val_annotations = annotations_path / 'instances_val2017.json'
    if not train_annotations.exists() or not val_annotations.exists():
        return
    if not train_images_path.exists() or not val_images_path.exists():
        return
    base_path = Path('/mnt/34B471F7B471BBC4/CSO_project/datasets')
    output_annotations_path = base_path / 'filtered_annotations'
    output_annotations_path.mkdir(parents=True, exist_ok=True)
    filter_coco_dataset(
        original_annotations_path=train_annotations,
        output_annotations_path=output_annotations_path / 'instances_train2017_filtered.json',
        selected_classes=SELECTED_CLASSES
    )
    filter_coco_dataset(
        original_annotations_path=val_annotations,
        output_annotations_path=output_annotations_path / 'instances_val2017_filtered.json',
        selected_classes=SELECTED_CLASSES
    )

def create_custom_yaml_config():
    config_content = """# Custom COCO 2017 dataset with 30 classes
path: /mnt/34B471F7B471BBC4/CSO_project/datasets
train: train2017
val: validation/val2017
nc: 30
names: ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'backpack', 'umbrella', 'handbag', 'tie',
        'skis', 'snowboard', 'sports ball', 'kite',
        'banana', 'apple', 'sandwich']
"""
    with open('custom_coco.yaml', 'w') as f:
        f.write(config_content)

if __name__ == "__main__":
    filter_both_train_val()
    create_custom_yaml_config()
