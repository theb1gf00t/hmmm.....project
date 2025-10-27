from pathlib import Path
import shutil

def debug_label_structure():
    base_path = Path('/mnt/34B471F7B471BBC4/CSO_project/datasets')
    train_images = base_path / 'train2017'
    train_labels = train_images / 'labels'
    image_files = list(train_images.glob('*.jpg'))
    label_files = list(train_labels.glob('*.txt'))
    if len(image_files) > 0 and len(label_files) > 0:
        sample_images = image_files[:5]
        for img in sample_images:
            label_file = train_labels / f"{img.stem}.txt"
            label_file.exists()
    val_images = base_path / 'validation' / 'val2017'
    val_labels = val_images / 'labels'
    val_image_files = list(val_images.glob('*.jpg'))
    val_label_files = list(val_labels.glob('*.txt'))
    with open('coco_final.yaml', 'r') as f:
        yaml_content = f.read()
        for line in yaml_content.split('\n')[:10]:
            pass

def check_symlinks():
    base_path = Path('/mnt/34B471F7B471BBC4/CSO_project/datasets')
    original_train_labels = base_path / 'labels' / 'train2017'
    target_train_labels = base_path / 'train2017' / 'labels'
    if target_train_labels.exists():
        shutil.rmtree(target_train_labels)
    target_train_labels.symlink_to(original_train_labels, target_is_directory=True)
    original_val_labels = base_path / 'labels' / 'val2017'
    target_val_labels = base_path / 'validation' / 'val2017' / 'labels'
    if target_val_labels.exists():
        shutil.rmtree(target_val_labels)
    target_val_labels.symlink_to(original_val_labels, target_is_directory=True)

if __name__ == "__main__":
    debug_label_structure()
    check_symlinks()
