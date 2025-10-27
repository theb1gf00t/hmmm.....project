# fix_yaml_format.py
def create_correct_yaml():
    """Create YOLO-compatible YAML format"""
    
    yaml_content = """# YOLO-formatted COCO 2017 dataset with 30 classes
path: /mnt/34B471F7B471BBC4/CSO_project/datasets

# Training
train: train2017  # images and labels should be in path/train2017 and path/labels/train2017
val: validation/val2017  # images and labels should be in path/validation/val2017 and path/labels/val2017

# Number of classes
nc: 30

# Class names
names: ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'backpack', 'umbrella', 'handbag', 'tie',
        'skis', 'snowboard', 'sports ball', 'kite',
        'banana', 'apple', 'sandwich']
"""
    
    with open('coco_30_classes_correct.yaml', 'w') as f:
        f.write(yaml_content)
    
    print("✓ Created coco_30_classes_correct.yaml with proper format")

if __name__ == "__main__":
    create_correct_yaml()