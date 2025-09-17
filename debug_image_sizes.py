#!/usr/bin/env python3
"""
Check image size variations in the medical dataset
"""

import json
import os
from PIL import Image
from transformers import AutoProcessor

def check_image_sizes():
    """Check actual image sizes in the dataset"""
    print("=" * 60)
    print(" IMAGE SIZE ANALYSIS")
    print("=" * 60)

    # Load dataset
    dataset_path = "./data/datasets_grounding/curve_detection/train_high_quality.json"

    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        return

    with open(dataset_path, 'r') as f:
        data = json.load(f)

    print(f"Total samples: {len(data)}")

    # Check first 10 images
    image_info = []

    for i, sample in enumerate(data[:10]):
        if 'image' in sample:
            image_path = os.path.join("./data/images", sample['image'])

            if os.path.exists(image_path):
                try:
                    with Image.open(image_path) as img:
                        width, height = img.size
                        aspect_ratio = width / height

                        image_info.append({
                            'index': i,
                            'path': sample['image'],
                            'size': (width, height),
                            'aspect_ratio': aspect_ratio,
                            'pixels': width * height
                        })

                        print(f"Sample {i}: {width}x{height} (AR: {aspect_ratio:.2f}) - {sample['image']}")

                except Exception as e:
                    print(f"Sample {i}: Error loading {image_path} - {e}")
            else:
                print(f"Sample {i}: Image not found - {image_path}")

    if image_info:
        # Statistics
        widths = [info['size'][0] for info in image_info]
        heights = [info['size'][1] for info in image_info]
        ratios = [info['aspect_ratio'] for info in image_info]

        print(f"\nSize Statistics:")
        print(f"  Width range: {min(widths)} - {max(widths)}")
        print(f"  Height range: {min(heights)} - {max(heights)}")
        print(f"  Aspect ratio range: {min(ratios):.2f} - {max(ratios):.2f}")

        # Check if sizes are very different
        width_var = max(widths) / min(widths) if min(widths) > 0 else 0
        height_var = max(heights) / min(heights) if min(heights) > 0 else 0

        print(f"  Width variation: {width_var:.2f}x")
        print(f"  Height variation: {height_var:.2f}x")

        if width_var > 2 or height_var > 2:
            print(f"  ⚠️  HIGH SIZE VARIATION DETECTED!")

def check_processor_output_sizes():
    """Check how processor handles different image sizes"""
    print("\n" + "=" * 60)
    print(" PROCESSOR OUTPUT SIZE ANALYSIS")
    print("=" * 60)

    try:
        processor = AutoProcessor.from_pretrained('./models/Qwen2.5-VL-7B-Instruct')

        # Test different image sizes
        test_sizes = [
            (512, 512),   # Square
            (1024, 512),  # Wide
            (512, 1024),  # Tall
            (800, 600),   # Different aspect ratio
            (1200, 800),  # Larger
        ]

        for width, height in test_sizes:
            # Create test image
            test_image = Image.new('RGB', (width, height), color='red')

            # Process image
            result = processor.image_processor(images=test_image, return_tensors='pt')

            pixel_values = result['pixel_values']
            image_grid_thw = result.get('image_grid_thw', None)

            if image_grid_thw is not None:
                thw = image_grid_thw[0]
                total_patches = thw.prod().item()

                print(f"Image {width}x{height}:")
                print(f"  pixel_values shape: {pixel_values.shape}")
                print(f"  grid_thw: {thw.tolist()}")
                print(f"  total_patches: {total_patches}")
                print(f"  total_patches % 4: {total_patches % 4}")
                print(f"  total_patches % 16: {total_patches % 16}")
            else:
                print(f"Image {width}x{height}: No grid_thw returned")

    except Exception as e:
        print(f"Processor test failed: {e}")

def check_actual_dataset_patches():
    """Check patch counts for actual dataset images"""
    print("\n" + "=" * 60)
    print(" ACTUAL DATASET PATCH ANALYSIS")
    print("=" * 60)

    try:
        processor = AutoProcessor.from_pretrained('./models/Qwen2.5-VL-7B-Instruct')

        dataset_path = "./data/datasets_grounding/curve_detection/train_high_quality.json"

        if not os.path.exists(dataset_path):
            print(f"Dataset not found: {dataset_path}")
            return

        with open(dataset_path, 'r') as f:
            data = json.load(f)

        patch_counts = []

        for i, sample in enumerate(data[:5]):  # Check first 5
            if 'image' in sample:
                image_path = os.path.join("./data/images", sample['image'])

                if os.path.exists(image_path):
                    try:
                        with Image.open(image_path) as img:
                            # Process with actual processor
                            result = processor.image_processor(images=img, return_tensors='pt')

                            image_grid_thw = result.get('image_grid_thw', None)
                            if image_grid_thw is not None:
                                thw = image_grid_thw[0]
                                total_patches = thw.prod().item()
                                patch_counts.append(total_patches)

                                print(f"Sample {i} ({sample['image']}):")
                                print(f"  Image size: {img.size}")
                                print(f"  Grid THW: {thw.tolist()}")
                                print(f"  Total patches: {total_patches}")
                                print(f"  Patches % 4: {total_patches % 4}")
                                print(f"  Patches % 16: {total_patches % 16}")

                    except Exception as e:
                        print(f"Error processing {image_path}: {e}")

        if patch_counts:
            print(f"\nPatch Count Summary:")
            print(f"  Range: {min(patch_counts)} - {max(patch_counts)}")
            print(f"  Unique counts: {sorted(set(patch_counts))}")
            print(f"  All divisible by 4: {all(count % 4 == 0 for count in patch_counts)}")
            print(f"  All divisible by 16: {all(count % 16 == 0 for count in patch_counts)}")

    except Exception as e:
        print(f"Dataset patch analysis failed: {e}")

if __name__ == "__main__":
    check_image_sizes()
    check_processor_output_sizes()
    check_actual_dataset_patches()