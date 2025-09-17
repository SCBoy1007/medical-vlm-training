#!/usr/bin/env python3
"""
Comprehensive debug script for medical VLM training environment
Collects system info, tests data pipeline, and identifies potential issues
"""

import sys
import os
import json
import traceback
from pathlib import Path

def print_section(title):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def check_python_environment():
    print_section("PYTHON ENVIRONMENT")
    try:
        import torch
        import transformers
        import accelerate
        import deepspeed

        print(f"Python version: {sys.version}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"Transformers version: {transformers.__version__}")
        print(f"Accelerate version: {accelerate.__version__}")
        print(f"DeepSpeed version: {deepspeed.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            print(f"Current GPU: {torch.cuda.current_device()}")
            print(f"GPU name: {torch.cuda.get_device_name()}")

            # Memory info
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU memory: {total_memory:.1f} GB")

    except Exception as e:
        print(f"Environment check failed: {e}")
        traceback.print_exc()

def check_data_files():
    print_section("DATA FILES CHECK")
    try:
        data_base = "./data/datasets_grounding"
        if os.path.exists(data_base):
            print(f"Data directory exists: {data_base}")

            # Check each dataset
            datasets = [
                "curve_detection/train_high_quality",
                "curve_detection/train_low_quality",
                "apex_vertebrae/train_high_quality",
                "apex_vertebrae/train_low_quality",
                "end_vertebrae/train_high_quality",
                "end_vertebrae/train_low_quality"
            ]

            for dataset in datasets:
                json_file = f"{data_base}/{dataset}/{dataset.replace('/', '_')}_grounding.json"
                if os.path.exists(json_file):
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    print(f"✓ {dataset}: {len(data)} samples")

                    # Sample analysis
                    if len(data) > 0:
                        sample = data[0]
                        conv_lengths = [len(c.get('value', '')) for c in sample.get('conversations', [])]
                        print(f"  - Sample conv lengths: {conv_lengths}")
                        print(f"  - Has image: {'image' in sample}")

                else:
                    print(f"✗ {dataset}: JSON file not found")
        else:
            print(f"Data directory not found: {data_base}")

    except Exception as e:
        print(f"Data files check failed: {e}")
        traceback.print_exc()

def test_tokenizer():
    print_section("TOKENIZER TEST")
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            './models/Qwen2.5-VL-7B-Instruct',
            use_fast=False
        )

        print(f"Tokenizer loaded successfully")
        print(f"Vocab size: {tokenizer.vocab_size}")
        print(f"Model max length: {tokenizer.model_max_length}")
        print(f"Pad token: '{tokenizer.pad_token}' (id: {tokenizer.pad_token_id})")
        print(f"EOS token: '{tokenizer.eos_token}' (id: {tokenizer.eos_token_id})")

        # Test tokenization
        test_texts = [
            "Hello world",
            "What do you see in this image? <image>",
            "The patient shows signs of scoliosis with a Cobb angle of 25 degrees."
        ]

        for text in test_texts:
            tokens = tokenizer.encode(text)
            print(f"'{text[:30]}...' -> {len(tokens)} tokens")

    except Exception as e:
        print(f"Tokenizer test failed: {e}")
        traceback.print_exc()

def test_image_processor():
    print_section("IMAGE PROCESSOR TEST")
    try:
        from transformers import AutoProcessor
        from PIL import Image

        processor = AutoProcessor.from_pretrained('./models/Qwen2.5-VL-7B-Instruct')

        print(f"Processor loaded successfully")
        print(f"Components: {list(processor.__dict__.keys())}")

        # Image processor info
        img_proc = processor.image_processor
        print(f"Max pixels: {getattr(img_proc, 'max_pixels', 'Not found')}")
        print(f"Min pixels: {getattr(img_proc, 'min_pixels', 'Not found')}")
        print(f"Merge size: {getattr(img_proc, 'merge_size', 'Not found')}")

        # Test with dummy image
        test_image = Image.new('RGB', (224, 224), color='red')
        result = img_proc(images=test_image, return_tensors='pt')

        print(f"Image processing result keys: {list(result.keys())}")
        if 'pixel_values' in result:
            print(f"Pixel values shape: {result['pixel_values'].shape}")
        if 'image_grid_thw' in result:
            print(f"Image grid thw shape: {result['image_grid_thw'].shape}")
            print(f"Image grid thw value: {result['image_grid_thw']}")

    except Exception as e:
        print(f"Image processor test failed: {e}")
        traceback.print_exc()

def test_data_preprocessing():
    print_section("DATA PREPROCESSING TEST")
    try:
        sys.path.append('.')
        from qwenvl.data.data_qwen import preprocess_qwen_2_visual
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            './models/Qwen2.5-VL-7B-Instruct',
            use_fast=False
        )

        # Test with simple conversation
        test_sources = [[{
            'role': 'user',
            'content': 'What do you see in this image? <image>'
        }, {
            'role': 'assistant',
            'content': 'I can see medical imaging data showing spinal curvature.'
        }]]

        print("Testing preprocess_qwen_2_visual...")
        result = preprocess_qwen_2_visual(test_sources, tokenizer, [256])

        print(f"✓ Preprocessing successful")
        print(f"Result keys: {list(result.keys())}")

        input_ids = result['input_ids']
        labels = result['labels']

        print(f"Input IDs type: {type(input_ids)}")
        print(f"Labels type: {type(labels)}")

        if hasattr(input_ids, 'shape'):
            print(f"Input IDs shape: {input_ids.shape}")
        elif isinstance(input_ids, list):
            print(f"Input IDs list length: {len(input_ids)}")
            if len(input_ids) > 0:
                print(f"First sample type: {type(input_ids[0])}")
                print(f"First sample length: {len(input_ids[0])}")

        if hasattr(labels, 'shape'):
            print(f"Labels shape: {labels.shape}")
        elif isinstance(labels, list):
            print(f"Labels list length: {len(labels)}")

    except Exception as e:
        print(f"Data preprocessing test failed: {e}")
        traceback.print_exc()

def test_dataset_loading():
    print_section("DATASET LOADING TEST")
    try:
        sys.path.append('.')
        from qwenvl.data.data_qwen import LazySupervisedDataset
        from transformers import AutoTokenizer, AutoProcessor
        from types import SimpleNamespace

        # Setup
        tokenizer = AutoTokenizer.from_pretrained(
            './models/Qwen2.5-VL-7B-Instruct',
            model_max_length=8192,
            padding_side='right',
            use_fast=False
        )
        processor = AutoProcessor.from_pretrained('./models/Qwen2.5-VL-7B-Instruct')

        # Create data_args
        data_args = SimpleNamespace()
        data_args.dataset_use = 'curve_detection_high'
        data_args.lazy_preprocess = True
        data_args.is_multimodal = True
        data_args.sep_image_conv_front = False
        data_args.image_token_len = 256
        data_args.image_folder = './data/images'
        data_args.image_aspect_ratio = 'anyres_max_9'
        data_args.max_pixels = 1003520
        data_args.min_pixels = 3136
        data_args.data_flatten = False
        data_args.image_processor = processor.image_processor
        data_args.model_type = 'qwen2.5vl'

        print("Creating dataset...")
        dataset = LazySupervisedDataset(tokenizer=tokenizer, data_args=data_args)
        print(f"✓ Dataset created successfully")
        print(f"Dataset length: {len(dataset)}")

        # Test loading samples
        print("Testing sample loading...")
        for i in range(min(3, len(dataset))):
            try:
                sample = dataset[i]
                print(f"Sample {i}:")
                print(f"  Keys: {list(sample.keys())}")

                input_ids = sample['input_ids']
                print(f"  input_ids type: {type(input_ids)}")

                if hasattr(input_ids, 'shape'):
                    print(f"  input_ids shape: {input_ids.shape}")
                elif isinstance(input_ids, list):
                    print(f"  input_ids list length: {len(input_ids)}")
                    if len(input_ids) > 0:
                        print(f"  first element type: {type(input_ids[0])}")
                        if hasattr(input_ids[0], '__len__'):
                            print(f"  first element length: {len(input_ids[0])}")

                if 'pixel_values' in sample:
                    print(f"  pixel_values shape: {sample['pixel_values'].shape}")

            except Exception as e:
                print(f"Sample {i} failed: {e}")
                if i == 0:  # Print full traceback for first failure
                    traceback.print_exc()
                break

    except Exception as e:
        print(f"Dataset loading test failed: {e}")
        traceback.print_exc()

def test_data_collator():
    print_section("DATA COLLATOR TEST")
    try:
        sys.path.append('.')
        from qwenvl.data.data_qwen import LazySupervisedDataset, DataCollatorForSupervisedDataset
        from transformers import AutoTokenizer, AutoProcessor
        from types import SimpleNamespace

        # Setup
        tokenizer = AutoTokenizer.from_pretrained(
            './models/Qwen2.5-VL-7B-Instruct',
            model_max_length=8192,
            padding_side='right',
            use_fast=False
        )
        processor = AutoProcessor.from_pretrained('./models/Qwen2.5-VL-7B-Instruct')

        # Create data_args
        data_args = SimpleNamespace()
        data_args.dataset_use = 'curve_detection_high'
        data_args.lazy_preprocess = True
        data_args.is_multimodal = True
        data_args.sep_image_conv_front = False
        data_args.image_token_len = 256
        data_args.image_folder = './data/images'
        data_args.image_aspect_ratio = 'anyres_max_9'
        data_args.max_pixels = 1003520
        data_args.min_pixels = 3136
        data_args.data_flatten = False
        data_args.image_processor = processor.image_processor
        data_args.model_type = 'qwen2.5vl'

        print("Creating dataset and collator...")
        dataset = LazySupervisedDataset(tokenizer=tokenizer, data_args=data_args)
        collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

        print("Testing batch creation...")
        # Get a few samples
        samples = [dataset[i] for i in range(min(2, len(dataset)))]

        print(f"Raw samples info:")
        for i, sample in enumerate(samples):
            input_ids = sample['input_ids']
            print(f"  Sample {i} input_ids type: {type(input_ids)}")
            if isinstance(input_ids, list):
                print(f"  Sample {i} input_ids length: {len(input_ids)}")
            elif hasattr(input_ids, 'shape'):
                print(f"  Sample {i} input_ids shape: {input_ids.shape}")

        # Test collation
        print("Attempting batch collation...")
        batch = collator(samples)

        print(f"✓ Batch creation successful!")
        print(f"Batch keys: {list(batch.keys())}")
        print(f"input_ids shape: {batch['input_ids'].shape}")
        print(f"labels shape: {batch['labels'].shape}")
        print(f"attention_mask shape: {batch['attention_mask'].shape}")

        # Verify tensor properties
        input_ids = batch['input_ids']
        print(f"input_ids dtype: {input_ids.dtype}")
        print(f"input_ids device: {input_ids.device}")
        print(f"input_ids min/max: {input_ids.min().item()}/{input_ids.max().item()}")

    except Exception as e:
        print(f"Data collator test failed: {e}")
        traceback.print_exc()

def main():
    print("MEDICAL VLM TRAINING DEBUG SCRIPT")
    print("=" * 60)

    check_python_environment()
    check_data_files()
    test_tokenizer()
    test_image_processor()
    test_data_preprocessing()
    test_dataset_loading()
    test_data_collator()  # Add the new test

    print_section("DEBUG COMPLETE")
    print("Please review the output above for any errors or issues.")
    print("If you see any failures, please share the output with the development team.")

if __name__ == "__main__":
    main()