#!/usr/bin/env python3
"""
Minimal test to diagnose the exact tensor dimension mismatch
Test a single sample to understand the reshape error
"""

import os
import sys
import torch
from pathlib import Path

def test_single_sample():
    """Test loading and processing a single sample"""
    print("=" * 60)
    print(" SINGLE SAMPLE TENSOR DIMENSION TEST")
    print("=" * 60)

    # Set up paths
    project_root = Path(__file__).parent
    sys.path.append(str(project_root))

    try:
        # Import after adding to path
        from qwenvl.train.argument import ModelArguments, DataArguments, TrainingArguments
        from qwenvl.data.data_qwen import make_supervised_data_module
        from transformers import (
            Qwen2_5_VLForConditionalGeneration,
            AutoTokenizer,
            AutoProcessor,
        )

        print("1. Loading tokenizer and processor...")
        tokenizer = AutoTokenizer.from_pretrained('./models/Qwen2.5-VL-7B-Instruct')
        processor = AutoProcessor.from_pretrained('./models/Qwen2.5-VL-7B-Instruct')

        print("2. Creating data arguments...")
        data_args = DataArguments()
        data_args.dataset_use = 'datasets_grounding'
        data_args.lazy_preprocess = True
        data_args.is_multimodal = True
        data_args.image_aspect_ratio = 'anyres_max_9'
        data_args.image_processor = processor.image_processor

        print("3. Creating data module...")
        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
        train_dataset = data_module['train_dataset']

        print(f"4. Dataset length: {len(train_dataset)}")

        print("\n5. Testing first sample...")
        sample = train_dataset[0]

        print("Sample keys:", list(sample.keys()))
        print(f"input_ids length: {len(sample['input_ids'])}")
        print(f"input_ids type: {type(sample['input_ids'])}")

        # Convert to tensor for analysis
        if isinstance(sample['input_ids'], list):
            input_ids_tensor = torch.tensor(sample['input_ids'])
        else:
            input_ids_tensor = sample['input_ids']

        print(f"input_ids tensor shape: {input_ids_tensor.shape}")

        # Check pixel values
        pixel_values = sample['pixel_values']
        print(f"pixel_values shape: {pixel_values.shape}")
        print(f"pixel_values dtype: {pixel_values.dtype}")

        # Check grid_thw
        if 'image_grid_thw' in sample:
            grid_thw = sample['image_grid_thw']
            print(f"image_grid_thw: {grid_thw}")
            print(f"image_grid_thw type: {type(grid_thw)}")

            if torch.is_tensor(grid_thw):
                print(f"image_grid_thw shape: {grid_thw.shape}")
                total_patches = grid_thw.prod().item()
                print(f"total patches (T*H*W): {total_patches}")

                # Check our token calculation
                merge_size = data_args.image_processor.merge_size
                calculated_tokens = total_patches // (merge_size ** 2)
                print(f"merge_size: {merge_size}")
                print(f"calculated tokens: {calculated_tokens}")

                # Count actual image pad tokens in sequence
                IMAGE_TOKEN_INDEX = 151655
                actual_image_tokens = input_ids_tensor.tolist().count(IMAGE_TOKEN_INDEX)
                print(f"actual image_pad tokens in sequence: {actual_image_tokens}")

                print(f"✓ Token calculation matches: {calculated_tokens == actual_image_tokens}")

        print("\n6. Testing model forward pass with this sample...")

        # Load model
        print("Loading model...")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            './models/Qwen2.5-VL-7B-Instruct',
            torch_dtype=torch.float16,
            device_map='cpu'  # Use CPU to avoid CUDA issues
        )

        # Prepare model inputs
        model_inputs = {
            'input_ids': input_ids_tensor.unsqueeze(0),  # Add batch dimension
            'pixel_values': pixel_values.unsqueeze(0),   # Add batch dimension
        }

        if 'image_grid_thw' in sample:
            model_inputs['image_grid_thw'] = sample['image_grid_thw'].unsqueeze(0)

        print("Model input shapes:")
        for key, value in model_inputs.items():
            if torch.is_tensor(value):
                print(f"  {key}: {value.shape}")

        print("\n7. Testing vision encoder separately...")
        try:
            with torch.no_grad():
                # Test just the vision part
                pixel_values = model_inputs['pixel_values']
                image_grid_thw = model_inputs.get('image_grid_thw', None)

                print(f"Testing with pixel_values shape: {pixel_values.shape}")
                if image_grid_thw is not None:
                    print(f"Testing with image_grid_thw: {image_grid_thw}")

                # This should fail at the problematic line
                image_embeds = model.visual(pixel_values, grid_thw=image_grid_thw)
                print(f"✓ Vision encoder succeeded! Output shape: {image_embeds.shape}")

        except Exception as vision_error:
            print(f"✗ Vision encoder failed: {vision_error}")

            # Let's analyze the exact tensors involved
            print("\nDebugging vision encoder internals...")

            # Get the exact values that cause the error
            print(f"pixel_values.shape: {pixel_values.shape}")
            seq_len = pixel_values.shape[1]  # Should be the sequence length
            print(f"seq_len from pixel_values: {seq_len}")

            # Check spatial_merge_unit from model
            spatial_merge_unit = getattr(model.visual, 'spatial_merge_unit', 'Not found')
            print(f"Model spatial_merge_unit: {spatial_merge_unit}")

            if spatial_merge_unit != 'Not found':
                print(f"seq_len // spatial_merge_unit = {seq_len // spatial_merge_unit}")
                print(f"seq_len % spatial_merge_unit = {seq_len % spatial_merge_unit}")

                # The problematic calculation
                expected_first_dim = seq_len // spatial_merge_unit
                hidden_size = pixel_values.shape[-1]  # Should be 1280 or similar
                expected_total_size = expected_first_dim * spatial_merge_unit * hidden_size
                actual_total_size = pixel_values.numel()

                print(f"Expected reshape: [{expected_first_dim}, {spatial_merge_unit}, {hidden_size}]")
                print(f"Expected total elements: {expected_total_size}")
                print(f"Actual total elements: {actual_total_size}")
                print(f"Difference: {actual_total_size - expected_total_size}")

    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_single_sample()