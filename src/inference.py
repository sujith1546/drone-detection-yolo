"""
Inference script for drone detection using trained YOLOv11 model
"""

import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image as PILImage


def detect_drones(
    model_path,
    image_path,
    conf_threshold=0.25,
    iou_threshold=0.45,
    save_result=True,
    show_result=False,
    output_dir='runs/detect/inference'
):
    """
    Detect drones in an image using trained YOLO model
    
    Args:
        model_path (str): Path to trained model weights (.pt file)
        image_path (str): Path to input image
        conf_threshold (float): Confidence threshold (0.0 to 1.0)
        iou_threshold (float): IoU threshold for NMS
        save_result (bool): Whether to save annotated image
        show_result (bool): Whether to display result
        output_dir (str): Directory to save results
    
    Returns:
        results: YOLO detection results object
    """
    
    # Load model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = YOLO(model_path)
    
    # Run inference
    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        iou=iou_threshold,
        save=save_result,
        show=show_result,
        project=Path(output_dir).parent,
        name=Path(output_dir).name,
        exist_ok=True,
        verbose=False
    )
    
    # Display detection information
    for r in results:
        boxes = r.boxes
        num_detections = len(boxes)
        
        image_name = os.path.basename(image_path)
        print(f"\n🎯 Detected {num_detections} drone(s) in {image_name}")
        
        if num_detections > 0:
            for i, box in enumerate(boxes):
                conf = box.conf[0].item()
                coords = box.xyxy[0].tolist()
                print(f"   Drone {i+1}: Confidence={conf:.2f}, "
                      f"Box=[{int(coords[0])}, {int(coords[1])}, "
                      f"{int(coords[2])}, {int(coords[3])}]")
        else:
            print("   No drones detected in this image")
    
    return results


def batch_inference(
    model_path,
    image_dir,
    conf_threshold=0.25,
    output_dir='runs/detect/batch_inference'
):
    """
    Run inference on multiple images in a directory
    
    Args:
        model_path (str): Path to trained model weights
        image_dir (str): Directory containing images
        conf_threshold (float): Confidence threshold
        output_dir (str): Directory to save results
    
    Returns:
        dict: Summary of detections per image
    """
    
    # Load model
    model = YOLO(model_path)
    
    # Get image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [
        f for f in os.listdir(image_dir)
        if os.path.splitext(f.lower())[1] in image_extensions
    ]
    
    if not image_files:
        print(f"⚠️ No images found in {image_dir}")
        return {}
    
    print(f"\n🔍 Running batch inference on {len(image_files)} images...")
    print("-" * 60)
    
    results_summary = {}
    
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        
        # Run detection
        results = model.predict(
            source=img_path,
            conf=conf_threshold,
            save=True,
            project=Path(output_dir).parent,
            name=Path(output_dir).name,
            exist_ok=True,
            verbose=False
        )
        
        # Store results
        num_detections = len(results[0].boxes)
        results_summary[img_file] = num_detections
        print(f"✓ {img_file}: {num_detections} drone(s)")
    
    print("-" * 60)
    print(f"\n📊 Summary:")
    print(f"   Total images: {len(image_files)}")
    print(f"   Images with drones: {sum(1 for n in results_summary.values() if n > 0)}")
    print(f"   Total drones detected: {sum(results_summary.values())}")
    print(f"\n✓ Results saved to: {output_dir}/")
    
    return results_summary


def visualize_results(original_image_path, result_image_path):
    """
    Display original and detection result side by side
    
    Args:
        original_image_path (str): Path to original image
        result_image_path (str): Path to result image with detections
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Original image
    if os.path.exists(original_image_path):
        original_img = PILImage.open(original_image_path)
        axes[0].imshow(original_img)
        axes[0].axis('off')
        axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    else:
        axes[0].text(0.5, 0.5, 'Original not found',
                    ha='center', va='center', fontsize=12)
        axes[0].axis('off')
    
    # Detection result
    if os.path.exists(result_image_path):
        result_img = PILImage.open(result_image_path)
        axes[1].imshow(result_img)
        axes[1].axis('off')
        axes[1].set_title('Detection Result', fontsize=14, fontweight='bold')
    else:
        axes[1].text(0.5, 0.5, 'Result not found',
                    ha='center', va='center', fontsize=12)
        axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()


def get_detection_stats(results):
    """
    Calculate statistics from detection results
    
    Args:
        results: YOLO results object
    
    Returns:
        dict: Detection statistics
    """
    
    stats = {
        'total_detections': 0,
        'avg_confidence': 0.0,
        'max_confidence': 0.0,
        'min_confidence': 0.0,
        'confidences': []
    }
    
    for r in results:
        boxes = r.boxes
        if len(boxes) > 0:
            confidences = [box.conf[0].item() for box in boxes]
            stats['total_detections'] = len(boxes)
            stats['confidences'] = confidences
            stats['avg_confidence'] = sum(confidences) / len(confidences)
            stats['max_confidence'] = max(confidences)
            stats['min_confidence'] = min(confidences)
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run drone detection inference')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (.pt file)')
    parser.add_argument('--source', type=str, required=True,
                       help='Path to image or directory')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--output', type=str, default='runs/detect/inference',
                       help='Output directory')
    parser.add_argument('--batch', action='store_true',
                       help='Run batch inference on directory')
    
    args = parser.parse_args()
    
    if args.batch:
        # Batch inference
        batch_inference(
            model_path=args.model,
            image_dir=args.source,
            conf_threshold=args.conf,
            output_dir=args.output
        )
    else:
        # Single image inference
        results = detect_drones(
            model_path=args.model,
            image_path=args.source,
            conf_threshold=args.conf,
            output_dir=args.output
        )
        
        # Print statistics
        stats = get_detection_stats(results)
        if stats['total_detections'] > 0:
            print(f"\n📈 Detection Statistics:")
            print(f"   Total detections: {stats['total_detections']}")
            print(f"   Avg confidence: {stats['avg_confidence']:.2f}")
            print(f"   Max confidence: {stats['max_confidence']:.2f}")
            print(f"   Min confidence: {stats['min_confidence']:.2f}")
