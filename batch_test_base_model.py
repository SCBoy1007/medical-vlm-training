#!/usr/bin/env python3
"""
批量测试基模型在三个任务上的表现
测试 curve_detection, apex_vertebrae, end_vertebrae 的所有 test 数据
"""

import os
import json
import time
import torch
import numpy as np
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from pathlib import Path
import re
import gc

# Configuration
MODEL_NAME = "./models/Qwen2.5-VL-7B-Instruct"
DATA_BASE_PATH = "./data/datasets_grounding"
IMAGE_BASE_PATH = "./data/images/test"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "./batch_test_results"

# Task configurations
TASKS = ["curve_detection", "apex_vertebrae", "end_vertebrae"]
QUALITIES = ["high_quality", "low_quality"]

def smart_resize(height, width, factor=28, min_pixels=56*56, max_pixels=14*14*4*1280):
    """Qwen2.5-VL's image resize function"""
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}")

    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor

    if h_bar * w_bar > max_pixels:
        beta = np.sqrt((height * width) / max_pixels)
        h_bar = int(np.floor(height / beta / factor) * factor)
        w_bar = int(np.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = np.sqrt(min_pixels / (height * width))
        h_bar = int(np.ceil(height * beta / factor) * factor)
        w_bar = int(np.ceil(width * beta / factor) * factor)

    return h_bar, w_bar

def get_task_prompt(task_name):
    """Get appropriate prompt for each task"""
    prompts = {
        "curve_detection": "Identify ALL scoliotic curves in this spine X-ray image and provide their bounding box coordinates. There may be 1, 2, or more curves. Output in JSON format: {\"curves\": [{\"bbox_2d\": [x1, y1, x2, y2]}]}",

        "apex_vertebrae": "Identify ALL apex vertebrae (the most deviated vertebrae) in the scoliotic curves of this spine X-ray image. There may be 1, 2, or more apex vertebrae. Output in JSON format: {\"apex_vertebrae\": [{\"bbox_2d\": [x1, y1, x2, y2]}]}",

        "end_vertebrae": "Identify ALL end vertebrae (the boundary vertebrae of scoliotic curves) in this spine X-ray image. For each curve, identify the upper and lower end vertebrae. Output in JSON format: {\"end_vertebrae\": [{\"curve_type\": \"primary/secondary\", \"lower\": {\"name\": \"vertebra_name\", \"bbox_2d\": [x1, y1, x2, y2]}, \"upper\": {\"name\": \"vertebra_name\", \"bbox_2d\": [x1, y1, x2, y2]}}]}"
    }
    return prompts[task_name]

def parse_model_response(response_text, task_name):
    """Parse model response to extract bbox coordinates based on task"""
    try:
        # Remove markdown code blocks if present
        cleaned_text = re.sub(r'```json\s*|\s*```', '', response_text.strip())

        # Try to parse as complete JSON
        try:
            data = json.loads(cleaned_text)

            # Different parsing logic for different tasks
            if task_name == "end_vertebrae":
                # Special handling for end_vertebrae nested structure
                bboxes = []
                if "end_vertebrae" in data:
                    for curve in data["end_vertebrae"]:
                        # Extract upper and lower end vertebrae bboxes
                        if "lower" in curve and "bbox_2d" in curve["lower"]:
                            bboxes.append(curve["lower"]["bbox_2d"])
                        if "upper" in curve and "bbox_2d" in curve["upper"]:
                            bboxes.append(curve["upper"]["bbox_2d"])
                return bboxes
            else:
                # Standard parsing for curve_detection and apex_vertebrae
                key_mappings = {
                    "curve_detection": ["curves"],
                    "apex_vertebrae": ["apex_vertebrae", "curves"]
                }

                for key in key_mappings[task_name]:
                    if key in data:
                        bboxes = []
                        for item in data[key]:
                            if 'bbox_2d' in item:
                                bboxes.append(item['bbox_2d'])
                            elif 'bbox' in item:
                                bboxes.append(item['bbox'])
                        return bboxes
        except json.JSONDecodeError:
            pass

        # Extract bbox patterns directly (fallback)
        bbox_patterns = [
            r'"bbox_2d":\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]',
            r'"bbox":\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]',
            r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
        ]

        for pattern in bbox_patterns:
            matches = re.findall(pattern, response_text)
            if matches:
                bboxes = [[int(x) for x in match] for match in matches]
                return bboxes

        return []
    except Exception as e:
        print(f"Error parsing response for {task_name}: {e}")
        return []

def calculate_iou(pred_bboxes, gt_bboxes):
    """Calculate IOU between predicted and ground truth bboxes"""
    if not pred_bboxes or not gt_bboxes:
        return 0.0

    # Create masks for both sets
    all_bboxes = pred_bboxes + gt_bboxes
    min_x = min(bbox[0] for bbox in all_bboxes)
    min_y = min(bbox[1] for bbox in all_bboxes)
    max_x = max(bbox[2] for bbox in all_bboxes)
    max_y = max(bbox[3] for bbox in all_bboxes)

    width = int(max_x - min_x + 1)
    height = int(max_y - min_y + 1)
    pred_mask = np.zeros((height, width), dtype=bool)
    gt_mask = np.zeros((height, width), dtype=bool)

    # Fill prediction mask
    for bbox in pred_bboxes:
        x1, y1, x2, y2 = bbox
        x1_rel = int(x1 - min_x)
        y1_rel = int(y1 - min_y)
        x2_rel = int(x2 - min_x)
        y2_rel = int(y2 - min_y)
        pred_mask[y1_rel:y2_rel, x1_rel:x2_rel] = True

    # Fill ground truth mask
    for bbox in gt_bboxes:
        x1, y1, x2, y2 = bbox
        x1_rel = int(x1 - min_x)
        y1_rel = int(y1 - min_y)
        x2_rel = int(x2 - min_x)
        y2_rel = int(y2 - min_y)
        gt_mask[y1_rel:y2_rel, x1_rel:x2_rel] = True

    # Calculate intersection and union
    intersection_area = np.sum(pred_mask & gt_mask)
    union_area = np.sum(pred_mask | gt_mask)

    if union_area == 0:
        return 0.0

    return intersection_area / union_area

def get_gpu_memory_info():
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        return f"Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB"
    return "CUDA not available"

def test_single_sample(model, processor, sample_data, image_path, task_name):
    """Test a single sample with enhanced memory management"""
    try:
        # Clear cache before starting
        torch.cuda.empty_cache()
        gc.collect()

        # Get ground truth bboxes
        gt_response = sample_data['conversations'][1]['value']
        gt_bboxes_training = parse_model_response(gt_response, task_name)

        if not gt_bboxes_training:
            return None

        # Load and check image size
        image = Image.open(image_path)
        orig_size = f"{image.width}x{image.height}"
        print(f"Processing {sample_data['image']} (original: {orig_size}) - {get_gpu_memory_info()}")

        # Apply aggressive resize for memory management
        max_edge = 1000  # More aggressive than 1200
        if max(image.width, image.height) > max_edge:
            if image.height > image.width:
                new_height = max_edge
                new_width = int(image.width * max_edge / image.height)
            else:
                new_width = max_edge
                new_height = int(image.height * max_edge / image.width)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            print(f"Resized for memory: {orig_size} -> {image.width}x{image.height}")

        # Get task-specific prompt
        prompt = get_task_prompt(task_name)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": image_path},
                ],
            }
        ]

        # Process input
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = processor(
            text=[text], images=[image], padding=True, return_tensors="pt"
        )

        # Clear image from memory after processor use
        image.close()
        del image
        inputs = inputs.to(model.device)

        print(f"After preprocessing - {get_gpu_memory_info()}")

        # Generate prediction with memory management
        with torch.no_grad():
            try:
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False
                )
            except torch.cuda.OutOfMemoryError as e:
                print(f"OOM for {sample_data['image']} ({img_size}): {e}")
                # Aggressive cleanup
                del inputs
                torch.cuda.empty_cache()
                gc.collect()
                return None

        # Decode output
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        response = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]

        # Aggressive memory cleanup
        del inputs, output_ids, generated_ids, text
        torch.cuda.empty_cache()
        gc.collect()

        # Parse predicted bboxes
        pred_bboxes_training = parse_model_response(response, task_name)

        # Calculate IOU
        iou = calculate_iou(pred_bboxes_training, gt_bboxes_training)

        result = {
            'image': sample_data['image'],
            'gt_count': len(gt_bboxes_training),
            'pred_count': len(pred_bboxes_training),
            'iou': iou,
            'response': response[:100] + "..." if len(response) > 100 else response
        }

        print(f"Completed {sample_data['image']} - IOU: {iou:.4f} - {get_gpu_memory_info()}")
        return result

    except Exception as e:
        print(f"Error processing {sample_data['image']}: {e}")
        # Aggressive cleanup on error
        torch.cuda.empty_cache()
        gc.collect()
        return None

def test_task(model, processor, task_name, quality):
    """Test all samples for a specific task and quality"""
    data_file = f"{DATA_BASE_PATH}/{task_name}/test_{quality}/{task_name}_test_{quality}_grounding.json"
    image_dir = f"{IMAGE_BASE_PATH}/{quality}"

    if not os.path.exists(data_file):
        print(f"Data file not found: {data_file}")
        return []

    with open(data_file, 'r') as f:
        test_data = json.load(f)

    results = []
    total_samples = len(test_data)

    print(f"Testing {task_name} - {quality}: {total_samples} samples")

    for i, sample in enumerate(test_data):
        if i % 10 == 0:
            print(f"  Progress: {i}/{total_samples}")

        image_filename = sample['image'].replace('images/', '')
        image_path = os.path.join(image_dir, image_filename)

        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue

        result = test_single_sample(model, processor, sample, image_path, task_name)
        if result:
            results.append(result)

    return results

def calculate_metrics(results):
    """Calculate overall metrics"""
    if not results:
        return {}

    total_samples = len(results)
    total_gt_count = sum(r['gt_count'] for r in results)
    total_pred_count = sum(r['pred_count'] for r in results)
    avg_iou = np.mean([r['iou'] for r in results])

    # Count samples with predictions
    samples_with_pred = sum(1 for r in results if r['pred_count'] > 0)

    return {
        'total_samples': total_samples,
        'avg_gt_count': total_gt_count / total_samples,
        'avg_pred_count': total_pred_count / total_samples,
        'avg_iou': avg_iou,
        'detection_rate': samples_with_pred / total_samples,
        'samples_with_predictions': samples_with_pred
    }

def main():
    """Main function - single GPU sequential processing"""
    print("=" * 60)
    print("Base Model Performance Testing - Single GPU")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load model once
    print("Loading model...")
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(MODEL_NAME, use_fast=False)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Test all tasks and qualities
    all_results = {}
    start_time = time.time()

    for task_name in TASKS:
        print(f"\n{'='*40}")
        print(f"Testing Task: {task_name}")
        print(f"{'='*40}")

        task_results = {}

        for quality in QUALITIES:
            task_start = time.time()

            results = test_task(model, processor, task_name, quality)
            metrics = calculate_metrics(results)

            task_end = time.time()
            task_duration = task_end - task_start

            task_results[quality] = {
                'results': results,
                'metrics': metrics,
                'duration': task_duration
            }

            print(f"\n{task_name} - {quality} Results:")
            print(f"  Samples: {metrics.get('total_samples', 0)}")
            print(f"  Avg GT Count: {metrics.get('avg_gt_count', 0):.2f}")
            print(f"  Avg Pred Count: {metrics.get('avg_pred_count', 0):.2f}")
            print(f"  Avg IOU: {metrics.get('avg_iou', 0):.4f}")
            print(f"  Detection Rate: {metrics.get('detection_rate', 0):.2%}")
            print(f"  Duration: {task_duration:.1f}s")

        all_results[task_name] = task_results

    total_time = time.time() - start_time

    # Save results
    output_file = os.path.join(OUTPUT_DIR, "base_model_test_results.json")
    final_results = {
        'results': all_results,
        'config': {
            'device': DEVICE,
            'total_duration': total_time,
            'model_name': MODEL_NAME
        }
    }

    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")
    print(f"Total testing time: {total_time:.1f}s ({total_time/60:.1f}m)")

    for task_name in TASKS:
        print(f"\n{task_name.upper()}:")
        for quality in QUALITIES:
            metrics = all_results[task_name][quality]['metrics']
            duration = all_results[task_name][quality]['duration']
            print(f"  {quality}: IOU={metrics.get('avg_iou', 0):.4f}, Detection={metrics.get('detection_rate', 0):.2%}, Time={duration:.1f}s")

    print(f"\nResults saved to: {output_file}")
    print("Testing completed!")

if __name__ == "__main__":
    main()