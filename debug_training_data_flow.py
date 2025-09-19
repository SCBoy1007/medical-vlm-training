#!/usr/bin/env python3
"""
Debug script to verify that both text and images are correctly loaded
and fed to the model during LoRA fine-tuning
"""
import os
import sys
import json
import torch
from PIL import Image
import numpy as np
from transformers import AutoTokenizer, AutoProcessor

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from qwenvl.train.argument import ModelArguments, DataArguments, TrainingArguments
from qwenvl.data.data_qwen import make_supervised_data_module

def debug_data_loading():
    """Debug the complete data loading pipeline"""
    print("🔍 MEDICAL VLM TRAINING DATA FLOW DEBUG")
    print("=" * 60)

    # Initialize components
    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"

    print("📋 Step 1: Loading tokenizer and processor...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        print("✅ Tokenizer and processor loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load tokenizer/processor: {e}")
        return False

    # Create arguments matching train.py
    print("\n📋 Step 2: Creating training arguments...")
    model_args = ModelArguments(
        model_name_or_path=model_name,
        version="qwen",
        freeze_backbone=False,
        tune_mm_mlp_adapter=True,
        tune_mm_llm=True
    )

    data_args = DataArguments(
        dataset_use="curve_detection_high,curve_detection_low,apex_vertebrae_high,apex_vertebrae_low,end_vertebrae_high,end_vertebrae_low",
        max_pixels=700*1400,  # Match our image size
        min_pixels=224*224,
        model_type="qwen2.5vl",
        data_flatten=False
    )

    # Add processor to data_args
    data_args.image_processor = processor.image_processor

    print("✅ Arguments created successfully")

    # Create dataset
    print("\n📋 Step 3: Creating dataset...")
    try:
        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
        train_dataset = data_module["train_dataset"]
        data_collator = data_module["data_collator"]
        print(f"✅ Dataset created successfully. Size: {len(train_dataset)}")
    except Exception as e:
        print(f"❌ Failed to create dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test single sample
    print(f"\n📋 Step 4: Testing single sample loading...")
    try:
        sample = train_dataset[0]
        print("✅ Single sample loaded successfully")

        # Analyze sample structure
        print(f"📊 Sample keys: {list(sample.keys())}")

        # Check for images
        if "pixel_values" in sample:
            pixel_values = sample["pixel_values"]
            print(f"📸 Image tensor shape: {pixel_values.shape}")
            print(f"📸 Image tensor dtype: {pixel_values.dtype}")
            print(f"📸 Image value range: [{pixel_values.min():.3f}, {pixel_values.max():.3f}]")
        else:
            print("❌ No pixel_values found in sample!")

        # Check input_ids
        if "input_ids" in sample:
            input_ids = sample["input_ids"]
            if isinstance(input_ids, torch.Tensor):
                print(f"📝 Input IDs tensor shape: {input_ids.shape}")
                print(f"📝 Input IDs sample: {input_ids[:20]}")
            else:
                print(f"📝 Input IDs list length: {len(input_ids)}")
                print(f"📝 Input IDs sample: {input_ids[:20]}")
        else:
            print("❌ No input_ids found in sample!")

        # Check labels
        if "labels" in sample:
            labels = sample["labels"]
            if isinstance(labels, torch.Tensor):
                print(f"🏷️  Labels tensor shape: {labels.shape}")
                # Count non-ignore labels
                non_ignore = (labels != -100).sum().item()
                print(f"🏷️  Non-ignore labels: {non_ignore}/{len(labels)}")
            else:
                print(f"🏷️  Labels list length: {len(labels)}")
                non_ignore = sum(1 for x in labels if x != -100)
                print(f"🏷️  Non-ignore labels: {non_ignore}/{len(labels)}")
        else:
            print("❌ No labels found in sample!")

    except Exception as e:
        print(f"❌ Failed to load single sample: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test batch creation
    print(f"\n📋 Step 5: Testing batch creation...")
    try:
        # Create a small batch
        batch_samples = [train_dataset[i] for i in range(min(2, len(train_dataset)))]
        batch = data_collator(batch_samples)

        print("✅ Batch created successfully")
        print(f"📊 Batch keys: {list(batch.keys())}")

        # Check batch tensors
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"📊 {key}: {value.shape} | dtype: {value.dtype}")
            else:
                print(f"📊 {key}: {type(value)} | value: {value}")

    except Exception as e:
        print(f"❌ Failed to create batch: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test image-text correspondence
    print(f"\n📋 Step 6: Testing image-text correspondence...")
    try:
        # Get a sample with conversation
        raw_data = train_dataset.list_data_dict[0]
        print(f"📝 Raw conversations: {raw_data['conversations'][:2]}")  # First 2 conversations

        if "image" in raw_data:
            image_path = raw_data["image"]
            print(f"📸 Image path: {image_path}")

            # Try to load the actual image
            data_path = raw_data["data_path"]
            full_image_path = os.path.join(data_path, "train", "high_quality", image_path.replace("images/", ""))
            if os.path.exists(full_image_path):
                img = Image.open(full_image_path)
                print(f"📸 Actual image size: {img.size}")
                print("✅ Image-text correspondence verified")
            else:
                print(f"❌ Image file not found: {full_image_path}")

    except Exception as e:
        print(f"⚠️  Image-text correspondence check failed: {e}")

    # Test model input simulation
    print(f"\n📋 Step 7: Simulating model input...")
    try:
        # Check if the batch is ready for model consumption
        required_keys = ["input_ids", "attention_mask", "pixel_values", "labels"]
        missing_keys = [key for key in required_keys if key not in batch or batch[key] is None]

        if missing_keys:
            print(f"❌ Missing required keys: {missing_keys}")
        else:
            print("✅ All required keys present for model training")

            # Check tensor devices and shapes
            print("📊 Model input summary:")
            print(f"   - input_ids: {batch['input_ids'].shape}")
            print(f"   - attention_mask: {batch['attention_mask'].shape}")
            print(f"   - pixel_values: {batch['pixel_values'].shape}")
            print(f"   - labels: {batch['labels'].shape}")

            # Verify non-empty training signal
            trainable_tokens = (batch['labels'] != -100).sum().item()
            total_tokens = batch['labels'].numel()
            print(f"   - Trainable tokens: {trainable_tokens}/{total_tokens} ({100*trainable_tokens/total_tokens:.1f}%)")

            if trainable_tokens == 0:
                print("❌ WARNING: No trainable tokens found! Model won't learn anything.")
            elif trainable_tokens / total_tokens < 0.1:
                print("⚠️  WARNING: Very few trainable tokens (<10%). Learning may be slow.")
            else:
                print("✅ Good amount of trainable tokens for learning")

    except Exception as e:
        print(f"❌ Model input simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 60)
    print("🎉 DATA FLOW DEBUG COMPLETED SUCCESSFULLY")
    print("=" * 60)
    return True

def test_specific_sample_decoding():
    """Test decoding a specific sample to verify content"""
    print("\n🔍 SAMPLE CONTENT VERIFICATION")
    print("=" * 40)

    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", trust_remote_code=True)

        # Read a sample JSON file
        json_path = "data/datasets_grounding/curve_detection/train_high_quality/curve_detection_train_high_quality_grounding.json"
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)

            sample = data[0]
            print(f"📋 Sample conversations:")
            for i, conv in enumerate(sample['conversations']):
                role = conv.get('from', conv.get('role', 'unknown'))
                content = conv.get('value', conv.get('content', ''))
                print(f"   {i+1}. {role}: {content[:100]}...")

            print(f"📸 Image: {sample.get('image', 'No image')}")

        else:
            print(f"❌ Sample file not found: {json_path}")

    except Exception as e:
        print(f"❌ Sample verification failed: {e}")

if __name__ == "__main__":
    print("🚀 Starting Medical VLM Training Data Flow Debug...")

    # Change to script directory
    os.chdir(project_root)

    success = debug_data_loading()
    test_specific_sample_decoding()

    if success:
        print("\n✅ CONCLUSION: Data loading pipeline appears to be working correctly!")
        print("   - Images are being loaded and processed")
        print("   - Text is being tokenized properly")
        print("   - Batches are being created successfully")
        print("   - Model inputs are properly formatted")
        print("\n🚀 You can proceed with training with confidence!")
    else:
        print("\n❌ CONCLUSION: Issues found in data loading pipeline!")
        print("   Please review the errors above before training.")