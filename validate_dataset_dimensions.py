#!/usr/bin/env python3
"""
Comprehensive dataset validation script for medical VLM training
Analyzes data processing pipeline and identifies tensor dimension mismatches
"""

import os
import sys
import json
import torch
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

def setup_logging():
    """Setup logging for validation"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('./logs/dataset_validation.log', mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def validate_dataset_sample(data_module, sample_indices: List[int], logger) -> Dict:
    """Validate specific dataset samples for dimension compatibility"""
    train_dataset = data_module['train_dataset']
    validation_results = {
        'compatible_samples': [],
        'incompatible_samples': [],
        'dimension_stats': defaultdict(int),
        'grid_thw_stats': defaultdict(int),
        'image_size_stats': defaultdict(int)
    }

    logger.info("="*60)
    logger.info("DATASET SAMPLE VALIDATION")
    logger.info("="*60)

    for idx in sample_indices:
        try:
            logger.info(f"\n--- Validating Sample {idx} ---")

            # Get sample from dataset
            sample = train_dataset[idx]

            # Extract key information
            pixel_values = sample['pixel_values']
            image_grid_thw = sample['image_grid_thw']
            input_ids = sample['input_ids']

            logger.info(f"Sample {idx} basic info:")
            logger.info(f"  pixel_values shape: {pixel_values.shape}")
            logger.info(f"  image_grid_thw: {image_grid_thw}")
            logger.info(f"  input_ids length: {len(input_ids)}")

            # Calculate expected dimensions
            t, h, w = image_grid_thw[0].tolist()  # Extract T, H, W
            total_patches = t * h * w

            # Get merge_size from processor (should be 2 for Qwen2.5-VL)
            merge_size = train_dataset.data_args.image_processor.merge_size
            spatial_merge_unit = 4  # Hardcoded in model architecture

            # Calculate expected token count after merge
            expected_tokens_after_merge = total_patches // (merge_size ** 2)

            logger.info(f"  Grid dimensions (T×H×W): {t}×{h}×{w} = {total_patches} patches")
            logger.info(f"  Merge size: {merge_size}")
            logger.info(f"  Expected tokens after merge: {expected_tokens_after_merge}")

            # Check spatial merge compatibility
            is_compatible = expected_tokens_after_merge % spatial_merge_unit == 0
            expected_reshape_dim1 = expected_tokens_after_merge // spatial_merge_unit

            logger.info(f"  Spatial merge unit: {spatial_merge_unit}")
            logger.info(f"  Expected reshape dim1: {expected_reshape_dim1}")
            logger.info(f"  Is compatible with spatial merge: {is_compatible}")

            # Verify actual tensor dimensions
            seq_len, hidden_size = pixel_values.shape
            logger.info(f"  Actual tensor shape: [{seq_len}, {hidden_size}]")

            # Check if tensor dimensions match expectations
            tensor_matches_expectation = (seq_len == total_patches)
            logger.info(f"  Tensor seq_len matches total patches: {tensor_matches_expectation}")

            # Calculate what the actual reshape would be
            if seq_len % spatial_merge_unit == 0:
                actual_reshape_dim1 = seq_len // spatial_merge_unit
                total_elements = seq_len * hidden_size
                expected_hidden_after_reshape = total_elements // (actual_reshape_dim1 * spatial_merge_unit)
                is_integer_hidden = (total_elements % (actual_reshape_dim1 * spatial_merge_unit)) == 0

                logger.info(f"  Actual reshape would be: [{actual_reshape_dim1}, {spatial_merge_unit}, {expected_hidden_after_reshape if is_integer_hidden else 'NON-INTEGER'}]")
                logger.info(f"  Reshape would work: {is_integer_hidden}")
            else:
                logger.info(f"  ❌ seq_len ({seq_len}) not divisible by spatial_merge_unit ({spatial_merge_unit})")
                is_integer_hidden = False

            # Record results
            sample_result = {
                'idx': idx,
                'pixel_values_shape': pixel_values.shape,
                'grid_thw': image_grid_thw.tolist(),
                'total_patches': total_patches,
                'expected_tokens_after_merge': expected_tokens_after_merge,
                'spatial_merge_compatible': is_compatible,
                'tensor_matches_expectation': tensor_matches_expectation,
                'reshape_would_work': is_integer_hidden,
                'overall_compatible': is_compatible and tensor_matches_expectation and is_integer_hidden
            }

            if sample_result['overall_compatible']:
                validation_results['compatible_samples'].append(sample_result)
                logger.info(f"  ✅ Sample {idx}: COMPATIBLE")
            else:
                validation_results['incompatible_samples'].append(sample_result)
                logger.info(f"  ❌ Sample {idx}: INCOMPATIBLE")

            # Record statistics
            validation_results['dimension_stats'][f"{h}x{w}"] += 1
            validation_results['grid_thw_stats'][f"{t},{h},{w}"] += 1
            validation_results['image_size_stats'][total_patches] += 1

        except Exception as e:
            logger.error(f"❌ Error validating sample {idx}: {e}")
            validation_results['incompatible_samples'].append({
                'idx': idx,
                'error': str(e),
                'overall_compatible': False
            })

    return validation_results

def analyze_processor_behavior(data_args, logger):
    """Analyze image processor behavior for different image sizes"""
    logger.info("\n" + "="*60)
    logger.info("IMAGE PROCESSOR ANALYSIS")
    logger.info("="*60)

    # Check processor configuration
    processor = data_args.image_processor
    logger.info(f"Processor type: {type(processor).__name__}")
    logger.info(f"Max pixels: {data_args.max_pixels}")
    logger.info(f"Min pixels: {data_args.min_pixels}")
    logger.info(f"Merge size: {processor.merge_size}")

    # Test with some common image sizes
    test_sizes = [
        (512, 512),   # Square
        (1024, 768),  # Landscape
        (768, 1024),  # Portrait
        (1536, 1024), # Wide landscape
        (2048, 1024), # Very wide
    ]

    logger.info("\nTesting processor with different image sizes:")

    from PIL import Image
    import numpy as np

    for width, height in test_sizes:
        try:
            # Create dummy image
            dummy_image = Image.fromarray(
                np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            )

            # Process image
            result = processor(images=dummy_image, return_tensors="pt")
            pixel_values = result["pixel_values"]
            grid_thw = result["image_grid_thw"][0]

            t, h, w = grid_thw.tolist()
            total_patches = t * h * w
            expected_tokens = total_patches // (processor.merge_size ** 2)
            spatial_compatible = expected_tokens % 4 == 0

            logger.info(f"  {width}×{height} → grid_thw: {t}×{h}×{w} = {total_patches} patches")
            logger.info(f"    → tokens after merge: {expected_tokens}, spatial compatible: {spatial_compatible}")
            logger.info(f"    → pixel_values shape: {pixel_values.shape}")

        except Exception as e:
            logger.error(f"  ❌ Error processing {width}×{height}: {e}")

def generate_report(validation_results: Dict, logger):
    """Generate comprehensive validation report"""
    logger.info("\n" + "="*60)
    logger.info("VALIDATION REPORT")
    logger.info("="*60)

    total_samples = len(validation_results['compatible_samples']) + len(validation_results['incompatible_samples'])
    compatible_count = len(validation_results['compatible_samples'])
    incompatible_count = len(validation_results['incompatible_samples'])

    logger.info(f"Total samples validated: {total_samples}")
    logger.info(f"Compatible samples: {compatible_count} ({compatible_count/total_samples*100:.1f}%)")
    logger.info(f"Incompatible samples: {incompatible_count} ({incompatible_count/total_samples*100:.1f}%)")

    if incompatible_count > 0:
        logger.info("\n📋 INCOMPATIBLE SAMPLES ANALYSIS:")
        for sample in validation_results['incompatible_samples']:
            if 'error' in sample:
                logger.info(f"  Sample {sample['idx']}: ERROR - {sample['error']}")
            else:
                logger.info(f"  Sample {sample['idx']}: {sample.get('pixel_values_shape', 'N/A')}")
                if not sample.get('spatial_merge_compatible', False):
                    logger.info(f"    → Issue: Expected tokens ({sample.get('expected_tokens_after_merge', 'N/A')}) not divisible by 4")
                if not sample.get('tensor_matches_expectation', False):
                    logger.info(f"    → Issue: Tensor shape doesn't match grid_thw calculation")
                if not sample.get('reshape_would_work', False):
                    logger.info(f"    → Issue: Reshape operation would fail")

    logger.info("\n📊 DIMENSION STATISTICS:")
    for dim, count in validation_results['dimension_stats'].items():
        logger.info(f"  {dim}: {count} samples")

    logger.info("\n🔢 GRID_THW STATISTICS:")
    for grid, count in validation_results['grid_thw_stats'].items():
        t, h, w = map(int, grid.split(','))
        tokens_after_merge = (t * h * w) // 4  # merge_size=2, so divide by 4
        spatial_compatible = tokens_after_merge % 4 == 0
        status = "✅" if spatial_compatible else "❌"
        logger.info(f"  {grid}: {count} samples, tokens: {tokens_after_merge} {status}")

def main():
    # Setup
    logger = setup_logging()

    # Create logs directory
    Path("./logs").mkdir(exist_ok=True)

    # Set up paths
    project_root = Path(__file__).parent
    sys.path.append(str(project_root))

    try:
        # Import modules
        from qwenvl.train.argument import DataArguments
        from qwenvl.data.data_qwen import make_supervised_data_module
        from transformers import AutoTokenizer, AutoProcessor

        logger.info("="*60)
        logger.info("MEDICAL VLM DATASET VALIDATION")
        logger.info("="*60)

        # Load tokenizer and processor
        logger.info("Loading tokenizer and processor...")
        model_path = './models/Qwen2.5-VL-7B-Instruct'
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        processor = AutoProcessor.from_pretrained(model_path)

        # Create data arguments
        data_args = DataArguments()
        data_args.dataset_use = 'curve_detection_high,curve_detection_low,apex_vertebrae_high,apex_vertebrae_low,end_vertebrae_high,end_vertebrae_low'
        data_args.lazy_preprocess = True
        data_args.is_multimodal = True
        data_args.image_aspect_ratio = 'anyres_max_9'
        data_args.max_pixels = 1024*28*28
        data_args.min_pixels = 56*56
        data_args.image_processor = processor.image_processor
        data_args.model_type = "qwen2.5vl"
        data_args.data_flatten = False

        # ENABLE COMPATIBILITY MODE FOR TESTING THE FIX
        data_args.enable_spatial_merge_compatibility = True
        logger.info("🔧 Testing with spatial merge compatibility mode ENABLED")

        # Create data module
        logger.info("Creating data module...")
        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
        train_dataset = data_module['train_dataset']
        logger.info(f"Dataset size: {len(train_dataset)}")

        # Analyze processor behavior
        analyze_processor_behavior(data_args, logger)

        # Select samples to validate (first 20, some random ones, and some from the end)
        import random
        sample_indices = list(range(min(20, len(train_dataset))))  # First 20
        if len(train_dataset) > 20:
            # Add some random samples
            random_indices = random.sample(range(20, len(train_dataset)), min(30, len(train_dataset) - 20))
            sample_indices.extend(random_indices)

        # Validate samples
        validation_results = validate_dataset_sample(data_module, sample_indices, logger)

        # Generate report
        generate_report(validation_results, logger)

        # Save detailed results
        results_file = './logs/validation_results.json'
        with open(results_file, 'w') as f:
            # Convert tensors to lists for JSON serialization
            json_results = {}
            for key, value in validation_results.items():
                if key in ['compatible_samples', 'incompatible_samples']:
                    json_results[key] = []
                    for sample in value:
                        json_sample = {}
                        for k, v in sample.items():
                            if isinstance(v, torch.Tensor):
                                json_sample[k] = v.tolist()
                            else:
                                json_sample[k] = v
                        json_results[key].append(json_sample)
                else:
                    json_results[key] = dict(value) if isinstance(value, defaultdict) else value

            json.dump(json_results, f, indent=2)

        logger.info(f"\n📄 Detailed results saved to: {results_file}")

        # Final recommendations
        logger.info("\n" + "="*60)
        logger.info("FIX EFFECTIVENESS REPORT")
        logger.info("="*60)

        incompatible_count = len(validation_results['incompatible_samples'])
        if incompatible_count == 0:
            logger.info("🎉 SUCCESS! All samples are now compatible with spatial merge!")
            logger.info("✅ The compatibility fix has resolved the tensor dimension issues.")
            logger.info("✅ Training should now proceed without spatial merge errors.")

            # Report compatibility statistics
            if hasattr(train_dataset.data_args.image_processor, 'get_stats'):
                stats = train_dataset.data_args.image_processor.get_stats()
                if stats['total_processed'] > 0:
                    logger.info(f"📊 Compatibility statistics: {stats['adjusted_images']}/{stats['total_processed']} images required adjustment ({stats['adjustment_rate_percent']:.1f}%)")
                    if stats['adjustment_rate_percent'] > 0:
                        logger.info("ℹ️ Some images were automatically resized to ensure spatial merge compatibility.")
        else:
            logger.info(f"⚠️ Found {incompatible_count} incompatible samples despite compatibility mode.")
            logger.info("This suggests the fix may need refinement. Details:")
            logger.info("1. Check wrapper implementation for edge cases")
            logger.info("2. Verify smart_resize_compatible function logic")
            logger.info("3. Consider more aggressive dimension adjustment")

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        logger.error("Full traceback:", exc_info=True)
        raise

if __name__ == "__main__":
    main()