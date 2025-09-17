#!/usr/bin/env python3
"""
Deep model parameter diagnosis script
Check the actual spatial_merge parameters in the model
"""

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoConfig

def check_model_spatial_params():
    """Check actual spatial merge parameters in the model"""
    print("=" * 60)
    print(" MODEL SPATIAL PARAMETERS CHECK")
    print("=" * 60)

    try:
        # Load config
        config = AutoConfig.from_pretrained('./models/Qwen2.5-VL-7B-Instruct')
        print(f"Model config type: {type(config)}")

        # Check vision config
        if hasattr(config, 'vision_config'):
            vision_config = config.vision_config
            print("\nVision Config Parameters:")
            for attr in dir(vision_config):
                if 'merge' in attr.lower() or 'spatial' in attr.lower():
                    value = getattr(vision_config, attr)
                    print(f"  {attr}: {value}")

        # Load actual model to check runtime parameters
        print("\nLoading model to check runtime parameters...")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            './models/Qwen2.5-VL-7B-Instruct',
            torch_dtype=torch.float16,
            device_map='cpu'
        )

        # Check vision model parameters
        if hasattr(model, 'visual'):
            visual_model = model.visual
            print(f"\nVisual model type: {type(visual_model)}")

            # Look for spatial merge related attributes
            for attr in dir(visual_model):
                if 'merge' in attr.lower() or 'spatial' in attr.lower():
                    try:
                        value = getattr(visual_model, attr)
                        print(f"  visual.{attr}: {value}")
                    except:
                        print(f"  visual.{attr}: <property/method>")

            # Check if there are any modules with spatial merge
            for name, module in visual_model.named_modules():
                if hasattr(module, 'spatial_merge_unit'):
                    print(f"  {name}.spatial_merge_unit: {module.spatial_merge_unit}")
                if hasattr(module, 'spatial_merge_size'):
                    print(f"  {name}.spatial_merge_size: {module.spatial_merge_size}")

        print(f"\n✓ Model parameter check completed")

    except Exception as e:
        print(f"Model parameter check failed: {e}")
        import traceback
        traceback.print_exc()

def check_processor_vs_model_mismatch():
    """Check if there's a mismatch between processor and model expectations"""
    print("\n" + "=" * 60)
    print(" PROCESSOR VS MODEL MISMATCH CHECK")
    print("=" * 60)

    try:
        from transformers import AutoProcessor

        # Load processor
        processor = AutoProcessor.from_pretrained('./models/Qwen2.5-VL-7B-Instruct')
        img_proc = processor.image_processor

        # Load model config
        config = AutoConfig.from_pretrained('./models/Qwen2.5-VL-7B-Instruct')

        print("Processor parameters:")
        print(f"  merge_size: {getattr(img_proc, 'merge_size', 'Not found')}")
        print(f"  patch_size: {getattr(img_proc, 'patch_size', 'Not found')}")

        print("\nModel vision config parameters:")
        if hasattr(config, 'vision_config'):
            vision_config = config.vision_config
            print(f"  spatial_merge_size: {getattr(vision_config, 'spatial_merge_size', 'Not found')}")
            print(f"  patch_size: {getattr(vision_config, 'patch_size', 'Not found')}")

        # The potential mismatch
        processor_merge = getattr(img_proc, 'merge_size', None)
        model_merge = getattr(config.vision_config, 'spatial_merge_size', None) if hasattr(config, 'vision_config') else None

        print(f"\nComparison:")
        print(f"  Processor merge_size: {processor_merge}")
        print(f"  Model spatial_merge_size: {model_merge}")

        if processor_merge != model_merge:
            print(f"  ⚠️  MISMATCH DETECTED!")
        else:
            print(f"  ✓ Values match")

    except Exception as e:
        print(f"Processor vs model check failed: {e}")
        import traceback
        traceback.print_exc()

def investigate_spatial_merge_unit():
    """Try to find where spatial_merge_unit=4 comes from"""
    print("\n" + "=" * 60)
    print(" SPATIAL_MERGE_UNIT INVESTIGATION")
    print("=" * 60)

    try:
        # Load the model and search for spatial_merge_unit
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            './models/Qwen2.5-VL-7B-Instruct',
            torch_dtype=torch.float16,
            device_map='cpu'
        )

        def search_for_spatial_merge_unit(obj, prefix=""):
            """Recursively search for spatial_merge_unit"""
            found_params = []

            if hasattr(obj, 'spatial_merge_unit'):
                found_params.append(f"{prefix}.spatial_merge_unit = {obj.spatial_merge_unit}")

            # Search in all modules
            if hasattr(obj, 'named_modules'):
                for name, module in obj.named_modules():
                    if hasattr(module, 'spatial_merge_unit'):
                        found_params.append(f"{prefix}.{name}.spatial_merge_unit = {module.spatial_merge_unit}")

            # Search in all children
            if hasattr(obj, 'named_children'):
                for name, child in obj.named_children():
                    child_results = search_for_spatial_merge_unit(child, f"{prefix}.{name}")
                    found_params.extend(child_results)

            return found_params

        params = search_for_spatial_merge_unit(model, "model")

        if params:
            print("Found spatial_merge_unit parameters:")
            for param in params:
                print(f"  {param}")
        else:
            print("No spatial_merge_unit parameters found in model")
            print("This suggests spatial_merge_unit might be hardcoded somewhere")

    except Exception as e:
        print(f"Spatial merge unit investigation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_model_spatial_params()
    check_processor_vs_model_mismatch()
    investigate_spatial_merge_unit()