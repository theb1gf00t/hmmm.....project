from ultralytics import YOLO
from pathlib import Path
import random


def test_random_images(num_images=5):
    model = YOLO('runs/detect/train7/weights/best.pt')
    test_images_path = Path('/mnt/34B471F7B471BBC4/CSO_project/datasets/test_dataset/test2017')
    all_images = list(test_images_path.glob('*.jpg'))
    num_images = min(num_images, len(all_images))
    random_images = random.sample(all_images, num_images)
    
    print("\n" + "="*80)
    print(f"Testing on {num_images} random images")
    print("="*80 + "\n")
    
    for i, image_path in enumerate(random_images, 1):
        print(f"\n{'='*80}")
        print(f"Image {i}/{num_images}: {image_path.name}")
        print(f"{'='*80}")
        
        results = model(image_path)
        boxes = results[0].boxes
        
        print(f"Found {len(boxes)} object(s):\n")
        
        for j, box in enumerate(boxes, 1):
            class_id = int(box.cls[0].cpu().numpy())
            confidence = float(box.conf[0].cpu().numpy())
            xyxy = box.xyxy[0].cpu().numpy()
            
            print(f"  Detection {j}:")
            print(f"    Class: {model.names[class_id]}")
            print(f"    Confidence: {confidence:.3f}")
            print(f"    Box: [{xyxy[0]:.1f}, {xyxy[1]:.1f}, {xyxy[2]:.1f}, {xyxy[3]:.1f}]")
        
        output_filename = f'predicted_{i}_{image_path.name}'
        results[0].save(filename=output_filename)
        print(f"\nSaved: {output_filename}")
    
    print("\n" + "="*80)
    print(f"Completed! {num_images} images processed")
    print("="*80 + "\n")


if __name__ == "__main__":
    num = int(input("How many images? (default 5): ") or 5)
    test_random_images(num_images=num)
