#!/usr/bin/env python3
"""
Debug token calculation issues for Qwen2.5-VL training
Focus on grid_thw_merged calculation and sequence length consistency
"""

import os
import json
import math
import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer

def print_section(title):
    print("=" * 60)
    print(f" {title}")
    print("=" * 60)

def test_token_calculation_methods():
    """Test different token calculation methods"""
    print_section("TOKEN CALCULATION COMPARISON")

    # Test with actual patch counts from previous debug
    test_patches = [
        1296,   # 512x512 (square, should be perfect)
        2664,   # 1024x512 (rectangular)
        4284,   # Real dataset sample 0
        4880,   # Real dataset sample 1
        4896,   # Real dataset sample 2
    ]

    merge_size = 2

    print("Patch count -> Token count comparison:")
    print("Patches    | Floor  | Ceil   | Remainder | Equal?")
    print("-" * 50)

    for patches in test_patches:
        floor_tokens = patches // (merge_size**2)
        ceil_tokens = math.ceil(patches / (merge_size**2))
        remainder = patches % (merge_size**2)
        is_equal = floor_tokens == ceil_tokens

        print(f"{patches:8d} | {floor_tokens:6d} | {ceil_tokens:6d} | {remainder:9d} | {is_equal}")

def test_actual_processor_behavior():
    """Test how the actual processor handles images"""
    print_section("ACTUAL PROCESSOR TOKEN CALCULATION")

    try:
        processor = AutoProcessor.from_pretrained('./models/Qwen2.5-VL-7B-Instruct')
        tokenizer = AutoTokenizer.from_pretrained('./models/Qwen2.5-VL-7B-Instruct')

        # Test different sized images
        test_sizes = [
            (512, 512),   # Perfect square
            (1024, 512),  # Rectangular
            (800, 600),   # Different ratio
        ]

        for width, height in test_sizes:
            print(f"\nTesting {width}x{height} image:")

            # Create test image
            test_image = Image.new('RGB', (width, height), color='red')

            # Process with image processor
            img_result = processor.image_processor(images=test_image, return_tensors='pt')
            pixel_values = img_result['pixel_values']
            image_grid_thw = img_result['image_grid_thw']

            # Calculate tokens different ways
            thw = image_grid_thw[0]
            total_patches = thw.prod().item()
            merge_size = processor.image_processor.merge_size

            floor_tokens = total_patches // (merge_size**2)
            ceil_tokens = math.ceil(total_patches / (merge_size**2))

            print(f"  Pixel values shape: {pixel_values.shape}")
            print(f"  Grid THW: {thw.tolist()}")
            print(f"  Total patches: {total_patches}")
            print(f"  Merge size: {merge_size}")
            print(f"  Floor tokens: {floor_tokens}")
            print(f"  Ceil tokens: {ceil_tokens}")
            print(f"  Patches % merge_size²: {total_patches % (merge_size**2)}")

            # Test actual tokenization with image
            text_with_image = "<image>What is in this image?"

            # Simulate our preprocessing
            image_pad_count = ceil_tokens  # What we're currently using
            simulated_text = f"<|vision_start|>{'<|image_pad|>' * image_pad_count}<|vision_end|>What is in this image?"

            try:
                tokens = tokenizer(simulated_text, return_tensors='pt')
                print(f"  Simulated text length with {image_pad_count} pads: {tokens['input_ids'].shape[1]}")
            except Exception as e:
                print(f"  Tokenization error: {e}")

    except Exception as e:
        print(f"Processor test failed: {e}")

def test_sequence_length_consistency():
    """Test if our token calculation creates consistent sequence lengths"""
    print_section("SEQUENCE LENGTH CONSISTENCY TEST")

    try:
        from qwenvl.data.data_qwen import make_supervised_data_module
        from qwenvl.train.argument import DataArguments
        from transformers import AutoTokenizer

        # Create minimal data args
        data_args = DataArguments()
        data_args.dataset_use = 'datasets_grounding'
        data_args.lazy_preprocess = True
        data_args.is_multimodal = True
        data_args.image_aspect_ratio = 'anyres_max_9'

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained('./models/Qwen2.5-VL-7B-Instruct')

        print("Creating data module...")
        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

        train_dataset = data_module['train_dataset']
        print(f"Dataset length: {len(train_dataset)}")

        # Test first few samples
        print("\nTesting first 3 samples:")
        for i in range(min(3, len(train_dataset))):
            try:
                sample = train_dataset[i]
                print(f"\nSample {i}:")
                print(f"  input_ids length: {len(sample['input_ids'])}")
                print(f"  pixel_values shape: {sample['pixel_values'].shape}")

                if 'image_grid_thw' in sample:
                    grid_thw = sample['image_grid_thw']
                    print(f"  image_grid_thw: {grid_thw}")
                    if torch.is_tensor(grid_thw):
                        total_patches = grid_thw.prod().item()
                        print(f"  total patches: {total_patches}")

                        # Check what tokens were actually generated
                        input_ids = sample['input_ids']
                        image_pad_count = input_ids.count(151655)  # IMAGE_TOKEN_INDEX
                        print(f"  actual image_pad tokens in sequence: {image_pad_count}")

                        # Compare with our calculation
                        expected_floor = total_patches // 4
                        expected_ceil = math.ceil(total_patches / 4)
                        print(f"  expected floor tokens: {expected_floor}")
                        print(f"  expected ceil tokens: {expected_ceil}")
                        print(f"  matches floor: {image_pad_count == expected_floor}")
                        print(f"  matches ceil: {image_pad_count == expected_ceil}")

            except Exception as e:
                print(f"  Error processing sample {i}: {e}")

    except Exception as e:
        print(f"Data module test failed: {e}")

def suggest_solutions():
    """Suggest potential solutions based on findings"""
    print_section("POTENTIAL SOLUTIONS")

    print("Based on the analysis above, potential solutions:")
    print()
    print("1. REVERT TO FLOOR DIVISION:")
    print("   - Use: patches // (merge_size**2)")
    print("   - Safer, matches most examples")
    print("   - May lose some tokens for non-divisible cases")
    print()
    print("2. BATCH SIZE = 1:")
    print("   - Avoid mixing different image sizes")
    print("   - Simpler debugging")
    print("   - Slower training but more stable")
    print()
    print("3. FIXED IMAGE SIZE:")
    print("   - Resize all images to same dimensions")
    print("   - Eliminates variability")
    print("   - May affect model performance")
    print()
    print("4. INVESTIGATE TOKEN ID MAPPING:")
    print("   - Check if image_pad token ID is correct")
    print("   - Verify tokenizer vocabulary")
    print("   - Ensure consistent token usage")

def main():
    print("QWEN2.5-VL TOKEN CALCULATION DIAGNOSTIC")
    print("=" * 60)

    test_token_calculation_methods()
    test_actual_processor_behavior()
    test_sequence_length_consistency()
    suggest_solutions()

    print_section("DIAGNOSTIC COMPLETE")
    print("Review the output above to identify the root cause.")
    print("Focus on any mismatches between expected and actual token counts.")

if __name__ == "__main__":
    main()