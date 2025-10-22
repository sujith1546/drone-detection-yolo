"""
Utility functions for drone detection project
"""

import os
import json
import yaml
import shutil
import subprocess
from pathlib import Path


def setup_kaggle(kaggle_json_path=None):
    """
    Setup Kaggle API credentials
    
    Args:
        kaggle_json_path (str): Path to kaggle.json file.
                               If None, looks for it in ~/.kaggle/
    
    Returns:
        bool: True if setup successful, False otherwise
    """
    
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_file = kaggle_dir / 'kaggle.json'
    
    # Create .kaggle directory if it doesn't exist
    kaggle_dir.mkdir(exist_ok=True)
    
    # If path provided, copy the file
    if kaggle_json_path:
        if not os.path.exists(kaggle_json_path):
            print(f"❌ ERROR: File not found: {kaggle_json_path}")
            return False
        
        shutil.copy(kaggle_json_path, kaggle_file)
        print(f"✓ Copied kaggle.json to {kaggle_file}")
    
    # Check if kaggle.json exists
    if not kaggle_file.exists():
        print("❌ ERROR: kaggle.json not found!")
        print("\nHow to get your kaggle.json:")
        print("1. Go to https://www.kaggle.com")
        print("2. Click on your profile picture (top right)")
        print("3. Go to 'Settings'")
        print("4. Scroll to 'API' section")
        print("5. Click 'Create New Token'")
        print("6. Download kaggle.json file")
        print(f"7. Place it at: {kaggle_file}")
        return False
    
    # Set proper permissions (read/write for owner only)
    kaggle_file.chmod(0o600)
    
    # Verify content
    try:
        with open(kaggle_file, 'r') as f:
            config = json.load(f)
            if 'username' in config and 'key' in config:
                print(f"✓ Kaggle API configured successfully!")
                print(f"✓ Username: {config['username']}")
                return True
            else:
                print("❌ ERROR: kaggle.json is missing 'username' or 'key'")
                return False
    except json.JSONDecodeError:
        print("❌ ERROR: kaggle.json is not valid JSON")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def download_dataset(dataset_name='muki2003/yolo-drone-detection-dataset',
                    output_dir='data/drone_dataset'):
    """
    Download dataset from Kaggle
    
    Args:
        dataset_name (str): Kaggle dataset identifier
        output_dir (str): Directory to extract dataset
    
    Returns:
        bool: True if download successful, False otherwise
    """
    
    print(f"\n=== DOWNLOADING DATASET FROM KAGGLE ===")
    print(f"Dataset: {dataset_name}")
    print("="*60 + "\n")
    
    # Test Kaggle API
    print("Testing Kaggle API connection...")
    try:
        result = subprocess.run(
            ['kaggle', 'datasets', 'list', '--page-size', '1'],
            capture_output=True,
            text=True,
            check=True
        )
        print("✓ Kaggle API working!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Kaggle API test failed")
        print("Please run setup_kaggle() first")
        return False
    except FileNotFoundError:
        print("❌ Kaggle command not found. Install with: pip install kaggle")
        return False
    
    # Download dataset
    print(f"\nDownloading {dataset_name}...")
    try:
        subprocess.run(
            ['kaggle', 'datasets', 'download', '-d', dataset_name],
            check=True
        )
    except subprocess.CalledProcessError:
        print("❌ Dataset download failed!")
        return False
    
    # Find downloaded zip file
    zip_name = dataset_name.split('/')[-1] + '.zip'
    if not os.path.exists(zip_name):
        print(f"❌ Downloaded file not found: {zip_name}")
        return False
    
    print(f"✓ Dataset downloaded: {zip_name}")
    
    # Extract dataset
    print(f"\nExtracting to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        import zipfile
        with zipfile.ZipFile(zip_name, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        print(f"✓ Dataset extracted to {output_dir}")
        
        # Remove zip file
        os.remove(zip_name)
        print("✓ Cleaned up zip file")
        
        return True
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return False


def verify_dataset_structure(dataset_root):
    """
    Verify and fix dataset structure for YOLO training
    
    Args:
        dataset_root (str): Root directory of dataset
    
    Returns:
        dict: Information about dataset structure
    """
    
    print("\n=== VERIFYING DATASET STRUCTURE ===")
    
    dataset_root = Path(dataset_root)
    
    # Check for nested structure
    print("\nChecking for nested folders...")
    nested_path = dataset_root / 'drone_dataset'
    
    if nested_path.exists():
        print("✓ Detected nested 'drone_dataset' folder")
        print("Fixing structure...")
        
        # Move all contents up one level
        for item in nested_path.iterdir():
            dst = dataset_root / item.name
            
            # Remove destination if exists
            if dst.exists():
                if dst.is_dir():
                    shutil.rmtree(dst)
                else:
                    dst.unlink()
            
            # Move item
            shutil.move(str(item), str(dst))
        
        # Remove empty nested folder
        nested_path.rmdir()
        print("✓ Structure fixed!")
    else:
        print("✓ No nested structure detected")
    
    # Verify expected folders
    required_splits = ['train', 'valid']
    optional_splits = ['test']
    
    info = {
        'train': {'images': 0, 'labels': 0, 'exists': False},
        'valid': {'images': 0, 'labels': 0, 'exists': False},
        'test': {'images': 0, 'labels': 0, 'exists': False}
    }
    
    print("\n=== FILE COUNTS ===")
    
    for split in required_splits + optional_splits:
        img_dir = dataset_root / split / 'images'
        lbl_dir = dataset_root / split / 'labels'
        
        if img_dir.exists() and lbl_dir.exists():
            img_files = [f for f in img_dir.iterdir()
                        if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
            lbl_files = [f for f in lbl_dir.iterdir()
                        if f.suffix == '.txt']
            
            info[split]['exists'] = True
            info[split]['images'] = len(img_files)
            info[split]['labels'] = len(lbl_files)
            
            print(f"✓ {split.capitalize()} Images: {len(img_files)}")
            print(f"✓ {split.capitalize()} Labels: {len(lbl_files)}")
        else:
            print(f"✗ {split.capitalize()}: NOT FOUND")
    
    # Verify image-label mapping
    print("\n=== VERIFYING IMAGE-LABEL MAPPING ===")
    
    for split in required_splits + optional_splits:
        if not info[split]['exists']:
            continue
        
        img_dir = dataset_root / split / 'images'
        lbl_dir = dataset_root / split / 'labels'
        
        img_names = {f.stem for f in img_dir.iterdir()
                    if f.suffix.lower() in ['.jpg', '.jpeg', '.png']}
        lbl_names = {f.stem for f in lbl_dir.iterdir()
                    if f.suffix == '.txt'}
        
        matched = len(img_names & lbl_names)
        total = len(img_names)
        
        if matched == total and total > 0:
            print(f"✓ {split.capitalize()}: {matched}/{total} perfectly matched")
            info[split]['matched'] = matched
        else:
            print(f"⚠️ {split.capitalize()}: {matched}/{total} matched")
            info[split]['matched'] = matched
    
    # Check if we can proceed
    if not (info['train']['exists'] and info['valid']['exists']):
        print("\n❌ ERROR: Missing required folders (train/valid)")
        raise ValueError("Invalid dataset structure")
    
    print("\n✓ Dataset structure verified successfully!")
    
    return info


def create_data_yaml(dataset_root, output_path=None, class_names=None):
    """
    Create data.yaml configuration file for YOLO training
    
    Args:
        dataset_root (str): Root directory of dataset
        output_path (str): Path to save yaml file. If None, saves in dataset_root
        class_names (dict): Dictionary of class IDs and names. 
                           If None, uses {0: 'drone'}
    
    Returns:
        str: Path to created yaml file
    """
    
    print("\n=== CREATING YAML CONFIGURATION ===")
    
    dataset_root = Path(dataset_root)
    
    if class_names is None:
        class_names = {0: 'drone'}
    
    # Create data.yaml content
    data_yaml = {
        'path': str(dataset_root.absolute()),
        'train': 'train/images',
        'val': 'valid/images',
        'names': class_names,
        'nc': len(class_names)
    }
    
    # Add test path if exists
    if (dataset_root / 'test' / 'images').exists():
        data_yaml['test'] = 'test/images'
    
    # Determine output path
    if output_path is None:
        output_path = dataset_root / 'data.yaml'
    
    # Save YAML file
    with open(output_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)
    
    print(f"✓ Configuration saved: {output_path}")
    print("\nYAML Contents:")
    print("-" * 40)
    with open(output_path, 'r') as f:
        print(f.read())
    print("-" * 40)
    
    return str(output_path)


def count_dataset_files(dataset_root):
    """
    Count total images and labels in dataset
    
    Args:
        dataset_root (str): Root directory of dataset
    
    Returns:
        dict: Count of images and labels per split
    """
    
    dataset_root = Path(dataset_root)
    counts = {}
    
    for split in ['train', 'valid', 'test']:
        img_dir = dataset_root / split / 'images'
        lbl_dir = dataset_root / split / 'labels'
        
        if img_dir.exists():
            img_count = len([f for f in img_dir.iterdir()
                           if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
            lbl_count = len([f for f in lbl_dir.iterdir()
                           if f.suffix == '.txt']) if lbl_dir.exists() else 0
            
            counts[split] = {'images': img_count, 'labels': lbl_count}
    
    return counts


if __name__ == "__main__":
    # Example usage
    print("Drone Detection Utils")
    print("\nAvailable functions:")
    print("- setup_kaggle(kaggle_json_path)")
    print("- download_dataset(dataset_name, output_dir)")
    print("- verify_dataset_structure(dataset_root)")
    print("- create_data_yaml(dataset_root, output_path, class_names)")
    print("- count_dataset_files(dataset_root)")
