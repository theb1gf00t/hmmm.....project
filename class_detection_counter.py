from ultralytics import YOLO
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter, defaultdict


def count_detections_in_test_set():
    model = YOLO('/mnt/34B471F7B471BBC4/CSO_project/runs/detect/train7/weights/best.pt')
    test_images_path = Path('/mnt/34B471F7B471BBC4/CSO_project/datasets/test_dataset/test2017')
    all_images = list(test_images_path.glob('*.jpg'))
    
    if not all_images:
        return None
    
    total_images = len(all_images)
    print(f"\n🔍 Starting detection analysis on {total_images} images...\n")
    
    class_counter = Counter()
    image_detection_count = defaultdict(int)
    confidence_scores = defaultdict(list)
    total_detections = 0
    images_with_detections = 0
    
    for i, img_path in enumerate(all_images, 1):  # Start from 1 for better readability
        results = model(img_path, verbose=False)
        result = results[0]
        boxes = result.boxes
        
        if len(boxes) > 0:
            images_with_detections += 1
            total_detections += len(boxes)
            detected_classes_in_image = set()
            
            for box in boxes:
                class_id = int(box.cls[0].cpu().numpy())
                confidence = float(box.conf[0].cpu().numpy())
                class_name = model.names[class_id]
                class_counter[class_name] += 1
                confidence_scores[class_name].append(confidence)
                detected_classes_in_image.add(class_name)
            
            for class_name in detected_classes_in_image:
                image_detection_count[class_name] += 1
        
        # Progress counter - print every 100 images or at completion
        if i % 100 == 0 or i == total_images:
            percentage = (i / total_images) * 100
            print(f"Progress: {i}/{total_images} images processed ({percentage:.1f}%) | "
                  f"Detections so far: {total_detections}")
    
    print(f"\n✅ Analysis complete!")
    print(f"Total images processed: {total_images}")
    print(f"Images with detections: {images_with_detections}")
    print(f"Total detections: {total_detections}\n")
    
    return {
        'class_counts': class_counter,
        'image_counts': image_detection_count,
        'confidence_scores': confidence_scores,
        'total_images': len(all_images),
        'images_with_detections': images_with_detections,
        'total_detections': total_detections
    }


def print_detection_summary(results):
    print("\n" + "="*80)
    print(f"{'CLASS NAME':<20} {'DETECTIONS':<10} {'AVG CONF':<12} {'IMAGES':<10} {'PERCENTAGE':<10}")
    print("="*80)
    
    class_counts = results['class_counts']
    image_counts = results['image_counts']
    confidence_scores = results['confidence_scores']
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    
    for class_name, count in sorted_classes:
        avg_confidence = np.mean(confidence_scores[class_name]) if confidence_scores[class_name] else 0
        image_count = image_counts.get(class_name, 0)
        percentage = (count / results['total_detections']) * 100
        print(f"{class_name:<20} {count:<10} {avg_confidence:.3f}       {image_count:<10} {percentage:>6.1f}%")
    
    print("="*80)
    print(f"{'TOTAL':<20} {results['total_detections']:<10} {'':<12} {results['images_with_detections']:<10} {'100%':>10}")
    print("="*80 + "\n")


def plot_detection_distribution(results):
    class_counts = results['class_counts']
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    classes = [item[0] for item in sorted_classes]
    counts = [item[1] for item in sorted_classes]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    bars = ax1.bar(classes, counts, color='skyblue', alpha=0.8)
    ax1.set_title('Number of Detections per Class', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Number of Detections')
    ax1.set_xlabel('Class Name')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}', 
                ha='center', va='bottom', fontsize=8)
    
    total = sum(counts)
    percentages = [(count/total)*100 for count in counts]
    
    if len(classes) > 15:
        top_classes = classes[:15]
        top_percentages = percentages[:15]
        other_percentage = sum(percentages[15:])
        pie_labels = top_classes + ['Other']
        pie_sizes = top_percentages + [other_percentage]
    else:
        pie_labels = classes
        pie_sizes = percentages
    
    ax2.pie(pie_sizes, labels=pie_labels, autopct='%1.1f%%', startangle=90, 
            textprops={'fontsize': 8})
    ax2.set_title('Detection Distribution (%)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('detection_distribution.png', dpi=300, bbox_inches='tight')
    print("📊 Saved: detection_distribution.png")
    plt.show()


def plot_confidence_distribution(results):
    confidence_scores = results['confidence_scores']
    classes_with_detections = [cls for cls, scores in confidence_scores.items() if scores]
    
    if not classes_with_detections:
        return
    
    avg_confidences = {cls: np.mean(scores) for cls, scores in confidence_scores.items() if scores}
    sorted_classes = sorted(avg_confidences.items(), key=lambda x: x[1], reverse=True)
    classes = [item[0] for item in sorted_classes]
    avg_conf = [item[1] for item in sorted_classes]
    
    plt.figure(figsize=(12, 8))
    bars = plt.bar(classes, avg_conf, color='lightcoral', alpha=0.8)
    plt.title('Average Confidence Score per Class', fontsize=14, fontweight='bold')
    plt.ylabel('Average Confidence Score')
    plt.xlabel('Class Name')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.3f}', 
                ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('confidence_distribution.png', dpi=300, bbox_inches='tight')
    print("📊 Saved: confidence_distribution.png")
    plt.show()


def save_detection_results(results):
    class_counts = results['class_counts']
    image_counts = results['image_counts']
    confidence_scores = results['confidence_scores']
    
    data = []
    for class_name in class_counts.keys():
        data.append({
            'class_name': class_name,
            'detection_count': class_counts[class_name],
            'image_count': image_counts.get(class_name, 0),
            'average_confidence': np.mean(confidence_scores[class_name]) if confidence_scores[class_name] else 0,
            'min_confidence': np.min(confidence_scores[class_name]) if confidence_scores[class_name] else 0,
            'max_confidence': np.max(confidence_scores[class_name]) if confidence_scores[class_name] else 0,
            'std_confidence': np.std(confidence_scores[class_name]) if confidence_scores[class_name] else 0
        })
    
    df = pd.DataFrame(data)
    df = df.sort_values('detection_count', ascending=False)
    df.to_csv('detection_summary.csv', index=False)
    print("💾 Saved: detection_summary.csv")
    
    confidence_data = []
    for class_name, scores in confidence_scores.items():
        for score in scores:
            confidence_data.append({
                'class_name': class_name,
                'confidence_score': score
            })
    
    confidence_df = pd.DataFrame(confidence_data)
    confidence_df.to_csv('confidence_scores.csv', index=False)
    print("💾 Saved: confidence_scores.csv")


if __name__ == "__main__":
    print("\n" + "="*80)
    print(" "*20 + "CLASS DETECTION COUNTER")
    print("="*80)
    
    results = count_detections_in_test_set()
    
    if results:
        print_detection_summary(results)
        plot_detection_distribution(results)
        plot_confidence_distribution(results)
        save_detection_results(results)
        print("\n✅ All done! Check the output files and plots.\n")
    else:
        print("❌ No test images found!")
