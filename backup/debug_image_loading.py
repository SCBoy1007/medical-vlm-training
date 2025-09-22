#!/usr/bin/env python3
"""
Debug script to verify image loading and path resolution
"""
import os
import json
from PIL import Image

def resolve_image_path(img_path, base_folder):
    """Same function as in data_qwen.py"""
    if img_path.startswith("images/"):
        # Remove "images/" prefix and find actual file location
        filename = img_path[7:]  # Remove "images/" prefix

        # Try different possible locations
        possible_paths = [
            os.path.join(base_folder, "train", "high_quality", filename),
            os.path.join(base_folder, "train", "low_quality", filename),
            os.path.join(base_folder, "val", "high_quality", filename),
            os.path.join(base_folder, "val", "low_quality", filename),
            os.path.join(base_folder, "test", "high_quality", filename),
            os.path.join(base_folder, "test", "low_quality", filename),
        ]

        # Find the first existing file
        for path in possible_paths:
            if os.path.exists(path):
                return path

        # Fallback to original path if none found
        return os.path.join(base_folder, img_path)
    else:
        return os.path.join(base_folder, img_path)

def test_dataset_loading():
    """Test loading images from one of the datasets"""
    print("=== Testing Image Loading ===")

    # Test one JSON file
    json_path = "data/datasets_grounding/curve_detection/train_high_quality/curve_detection_train_high_quality_grounding.json"
    data_path = "data/images"

    if not os.path.exists(json_path):
        print(f"❌ JSON file not found: {json_path}")
        return False

    with open(json_path, 'r') as f:
        data = json.load(f)

    print(f"📊 Dataset size: {len(data)} samples")

    # Test first 5 samples
    success_count = 0
    for i, sample in enumerate(data[:5]):
        image_path = sample.get('image', '')
        resolved_path = resolve_image_path(image_path, data_path)

        print(f"\n--- Sample {i+1} ---")
        print(f"Original path: {image_path}")
        print(f"Resolved path: {resolved_path}")
        print(f"File exists: {os.path.exists(resolved_path)}")

        if os.path.exists(resolved_path):
            try:
                img = Image.open(resolved_path)
                print(f"✅ Image loaded successfully: {img.size}")
                success_count += 1
            except Exception as e:
                print(f"❌ Failed to load image: {e}")
        else:
            print(f"❌ File not found")

    print(f"\n📈 Success rate: {success_count}/5")
    return success_count == 5

def check_all_datasets():
    """Check all dataset JSON files"""
    print("\n=== Checking All Datasets ===")

    datasets = [
        "data/datasets_text_grounding/curve_detection/train_high_quality/curve_detection_train_high_quality_text_grounding.json",
        "data/datasets_text_grounding/curve_detection/train_low_quality/curve_detection_train_low_quality_text_grounding.json",
        "data/datasets_text_grounding/apex_vertebrae/train_high_quality/apex_vertebrae_train_high_quality_text_grounding.json",
        "data/datasets_text_grounding/apex_vertebrae/train_low_quality/apex_vertebrae_train_low_quality_text_grounding.json",
        "data/datasets_text_grounding/end_vertebrae/train_high_quality/end_vertebrae_train_high_quality_text_grounding.json",
        "data/datasets_text_grounding/end_vertebrae/train_low_quality/end_vertebrae_train_low_quality_text_grounding.json",
    ]

    total_samples = 0
    for dataset in datasets:
        if os.path.exists(dataset):
            with open(dataset, 'r') as f:
                data = json.load(f)
                total_samples += len(data)
                print(f"✅ {os.path.basename(dataset)}: {len(data)} samples")
        else:
            print(f"❌ Missing: {dataset}")

    print(f"\n📊 Total samples: {total_samples}")

if __name__ == "__main__":
    print("🔍 Medical VLM Data Loading Debug Script")
    print("=" * 50)

    # Change to script directory
    os.chdir('/mnt/c/Users/HKUBS/OneDrive/桌面/LLM_Study/medical_vlm_training')

    success = test_dataset_loading()
    check_all_datasets()

    if success:
        print("\n✅ Image loading appears to be working correctly")
    else:
        print("\n❌ Image loading has issues - this could explain loss convergence problems")