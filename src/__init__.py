"""
Drone Detection using YOLOv11
A complete pipeline for training and inference of drone detection models
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from .train import train_model
from .inference import detect_drones
from .utils import setup_kaggle, download_dataset, verify_dataset_structure

__all__ = [
    'train_model',
    'detect_drones',
    'setup_kaggle',
    'download_dataset',
    'verify_dataset_structure'
]
