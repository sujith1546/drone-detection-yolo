"""
Training script for YOLOv11 drone detection model
"""

import os
import torch
from ultralytics import YOLO
from pathlib import Path


def check_gpu():
    """Check GPU availability and print device information"""
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        return True
    else:
        print("⚠️ WARNING: GPU not available! Training will be slower on CPU")
        return False


def train_model(
    data_yaml_path,
    model_name='yolo11n.pt',
    epochs=50,
    imgsz=640,
    batch=16,
    project='runs/detect',
    name='drone_detection',
    patience=10,
    device=None,
    **kwargs
):
    """
    Train YOLOv11 model for drone detection
    
    Args:
        data_yaml_path (str): Path to data.yaml configuration file
        model_name (str): Pre-trained model to use (yolo11n.pt, yolo11s.pt, etc.)
        epochs (int): Number of training epochs
        imgsz (int): Input image size
        batch (int): Batch size
        project (str): Project directory for results
        name (str): Experiment name
        patience (int): Early stopping patience
        device (int/str): Device to use (0 for GPU, 'cpu' for CPU, None for auto)
        **kwargs: Additional training arguments
    
    Returns:
        results: Training results object
    """
    
    # Check GPU
    has_gpu = check_gpu()
    
    # Set device automatically if not specified
    if device is None:
        device = 0 if has_gpu else 'cpu'
    
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    # Load pre-trained model
    print(f"\nLoading {model_name} model...")
    model = YOLO(model_name)
    
    # Training configuration
    print("\n📋 Training Configuration:")
    print(f"  - Model: {model_name}")
    print(f"  - Epochs: {epochs}")
    print(f"  - Image Size: {imgsz}x{imgsz}")
    print(f"  - Batch Size: {batch}")
    print(f"  - Device: {device}")
    print(f"  - Data Config: {data_yaml_path}")
    print("\n⏱️ Estimated Time: 30-60 minutes (depending on hardware)")
    print("="*60 + "\n")
    
    # Default training parameters
    train_params = {
        'data': data_yaml_path,
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch,
        'name': name,
        'patience': patience,
        'save': True,
        'plots': True,
        'device': device,
        'workers': 2,
        'project': project,
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'AdamW',
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'close_mosaic': 10,
        'cache': False,
        'amp': True,
        'verbose': True
    }
    
    # Update with any additional kwargs
    train_params.update(kwargs)
    
    # Train the model
    results = model.train(**train_params)
    
    print("\n" + "="*60)
    print("✓ TRAINING COMPLETED!")
    print("="*60)
    
    # Print model location
    best_model = Path(project) / name / 'weights' / 'best.pt'
    print(f"\n📁 Best model saved to: {best_model}")
    
    return results


def evaluate_model(model_path):
    """
    Evaluate trained model on validation set
    
    Args:
        model_path (str): Path to trained model weights
    
    Returns:
        metrics: Validation metrics
    """
    print("\n=== MODEL EVALUATION ===")
    
    # Load model
    model = YOLO(model_path)
    
    # Validate
    metrics = model.val()
    
    # Print metrics
    print("\n" + "="*60)
    print("📊 VALIDATION METRICS")
    print("="*60)
    print(f"mAP@50:        {metrics.box.map50:.4f}  (Target: >0.85)")
    print(f"mAP@50-95:     {metrics.box.map:.4f}   (Target: >0.60)")
    print(f"Precision:     {metrics.box.mp:.4f}   (Target: >0.80)")
    print(f"Recall:        {metrics.box.mr:.4f}   (Target: >0.80)")
    
    # Calculate F1 score
    if metrics.box.mp > 0 and metrics.box.mr > 0:
        f1 = 2 * (metrics.box.mp * metrics.box.mr) / (metrics.box.mp + metrics.box.mr)
        print(f"F1-Score:      {f1:.4f}")
    
    print("="*60)
    
    # Performance interpretation
    if metrics.box.map50 > 0.85:
        print("✓ EXCELLENT performance!")
    elif metrics.box.map50 > 0.70:
        print("✓ GOOD performance!")
    else:
        print("⚠️ Consider training for more epochs or using a larger model")
    
    return metrics


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Train YOLOv11 drone detection model')
    parser.add_argument('--data', type=str, required=True, help='Path to data.yaml')
    parser.add_argument('--model', type=str, default='yolo11n.pt', help='Model to use')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--device', default=None, help='Device (0 for GPU, cpu for CPU)')
    
    args = parser.parse_args()
    
    # Train model
    results = train_model(
        data_yaml_path=args.data,
        model_name=args.model,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device
    )
    
    # Evaluate model
    best_model_path = 'runs/detect/drone_detection/weights/best.pt'
    if os.path.exists(best_model_path):
        evaluate_model(best_model_path)
