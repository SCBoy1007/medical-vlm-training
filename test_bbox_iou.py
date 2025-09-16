#!/usr/bin/env python3
"""
Bbox IOU Test Script for Qwen2.5-VL
Test the bbox localization capability and calculate IOU between predicted and ground truth bboxes
"""

import os
import json
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# Configuration
MODEL_NAME = "./models/Qwen2.5-VL-7B-Instruct"
TEST_DATA_PATH = "./data/datasets_grounding/curve_detection/test_high_quality/curve_detection_test_high_quality_grounding.json"
IMAGE_BASE_PATH = "./data/images/test/high_quality"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def smart_resize(height, width, factor=28, min_pixels=56*56, max_pixels=14*14*4*1280):
    """
    Qwen2.5-VL's image resize function
    Rescales the image so that:
    1. Both dimensions are divisible by 'factor'
    2. Total pixels within [min_pixels, max_pixels]
    3. Aspect ratio maintained
    """
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}")

    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    return h_bar, w_bar

def calculate_bbox_area(bbox):
    """Calculate area of a single bbox [x1, y1, x2, y2]"""
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)

def calculate_intersection(bbox1, bbox2):
    """Calculate intersection area between two bboxes"""
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2

    # Calculate intersection coordinates
    xi1 = max(x1, x3)
    yi1 = max(y1, y3)
    xi2 = min(x2, x4)
    yi2 = min(y2, y4)

    # If no intersection
    if xi2 <= xi1 or yi2 <= yi1:
        return 0

    return (xi2 - xi1) * (yi2 - yi1)

def calculate_union_area(bboxes):
    """Calculate total area covered by multiple bboxes (considering overlaps)"""
    if not bboxes:
        return 0

    # For simplicity, we'll use a pixel-based approach for accurate union calculation
    # Find the bounding box that contains all bboxes
    min_x = min(bbox[0] for bbox in bboxes)
    min_y = min(bbox[1] for bbox in bboxes)
    max_x = max(bbox[2] for bbox in bboxes)
    max_y = max(bbox[3] for bbox in bboxes)

    # Create a mask for the union
    width = int(max_x - min_x + 1)
    height = int(max_y - min_y + 1)
    mask = np.zeros((height, width), dtype=bool)

    # Mark pixels covered by any bbox
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        x1_rel = int(x1 - min_x)
        y1_rel = int(y1 - min_y)
        x2_rel = int(x2 - min_x)
        y2_rel = int(y2 - min_y)
        mask[y1_rel:y2_rel, x1_rel:x2_rel] = True

    return np.sum(mask)

def calculate_iou(pred_bboxes, gt_bboxes):
    """
    Calculate IOU between predicted and ground truth bboxes
    Handles multiple bboxes by calculating union areas
    """
    if not pred_bboxes or not gt_bboxes:
        return 0.0

    # Calculate union area for predictions and ground truth
    pred_union_area = calculate_union_area(pred_bboxes)
    gt_union_area = calculate_union_area(gt_bboxes)

    # Calculate intersection between the two unions
    # For simplicity, we'll calculate pairwise intersections and take the max
    max_intersection = 0

    # Find overall bounding box for intersection calculation
    all_bboxes = pred_bboxes + gt_bboxes
    min_x = min(bbox[0] for bbox in all_bboxes)
    min_y = min(bbox[1] for bbox in all_bboxes)
    max_x = max(bbox[2] for bbox in all_bboxes)
    max_y = max(bbox[3] for bbox in all_bboxes)

    # Create masks for both sets
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

def parse_model_response(response_text):
    """Parse model response to extract bbox coordinates"""
    try:
        import re

        # Remove markdown code blocks if present
        cleaned_text = re.sub(r'```json\s*|\s*```', '', response_text.strip())

        # Try to parse as complete JSON
        try:
            data = json.loads(cleaned_text)
            if 'curves' in data:
                bboxes = []
                for curve in data['curves']:
                    if 'bbox_2d' in curve:
                        bboxes.append(curve['bbox_2d'])
                    elif 'bbox' in curve:  # fallback for different naming
                        bboxes.append(curve['bbox'])
                return bboxes
        except json.JSONDecodeError:
            pass

        # Try to find JSON-like structures in the text
        json_pattern = r'\{[^{}]*"curves"[^{}]*\[.*?\][^{}]*\}'
        json_matches = re.findall(json_pattern, response_text, re.DOTALL)

        for match in json_matches:
            try:
                data = json.loads(match)
                if 'curves' in data:
                    bboxes = []
                    for curve in data['curves']:
                        if 'bbox_2d' in curve:
                            bboxes.append(curve['bbox_2d'])
                        elif 'bbox' in curve:
                            bboxes.append(curve['bbox'])
                    return bboxes
            except:
                continue

        # Extract bbox patterns directly
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
        print(f"Error parsing response: {e}")
        return []

def visualize_bboxes(image_path, gt_bboxes, pred_bboxes, save_path=None):
    """Visualize ground truth and predicted bboxes on the image"""
    # Load image
    image = Image.open(image_path)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)

    # Draw ground truth bboxes (green)
    for i, bbox in enumerate(gt_bboxes):
        x1, y1, x2, y2 = bbox
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                               linewidth=2, edgecolor='green', facecolor='none', alpha=0.7)
        ax.add_patch(rect)
        ax.text(x1, y1-10, f'GT-{i+1}', color='green', fontsize=10, weight='bold')

    # Draw predicted bboxes (red)
    for i, bbox in enumerate(pred_bboxes):
        x1, y1, x2, y2 = bbox
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                               linewidth=2, edgecolor='red', facecolor='none', alpha=0.7)
        ax.add_patch(rect)
        ax.text(x1, y1-30, f'Pred-{i+1}', color='red', fontsize=10, weight='bold')

    ax.set_title(f'Ground Truth (Green) vs Predicted (Red) Bboxes\nImage: {os.path.basename(image_path)}')
    ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")

    plt.show()

def load_model():
    """Load Qwen2.5-VL model and processor"""
    print(f"Loading model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")

    # Load processor
    processor = AutoProcessor.from_pretrained(MODEL_NAME, use_fast=False)
    print("Processor loaded successfully")

    # Load model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    print("Model loaded successfully!")
    return model, processor

def test_single_sample(model, processor, sample_data, image_base_path):
    """Test a single sample and calculate IOU"""
    # Handle image path - remove 'images/' prefix from dataset path
    image_filename = sample_data['image'].replace('images/', '')
    image_path = os.path.join(image_base_path, image_filename)

    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return None

    # Load image to get original dimensions
    image = Image.open(image_path)
    orig_width, orig_height = image.size
    print(f"Original image size: {orig_width} x {orig_height}")

    # Apply aggressive resize for memory management FIRST
    max_edge = 1000
    pre_resize_width, pre_resize_height = orig_width, orig_height
    if max(orig_width, orig_height) > max_edge:
        if orig_height > orig_width:
            pre_resize_height = max_edge
            pre_resize_width = int(orig_width * max_edge / orig_height)
        else:
            pre_resize_width = max_edge
            pre_resize_height = int(orig_height * max_edge / orig_width)
        print(f"Pre-resize for memory: {orig_width}x{orig_height} -> {pre_resize_width}x{pre_resize_height}")

    # Calculate resized dimensions (Qwen2.5-VL format) based on pre-resized dimensions
    new_height, new_width = smart_resize(pre_resize_height, pre_resize_width)
    print(f"Final smart_resize dimensions: {new_width} x {new_height}")

    # Actually resize the image if needed
    if max(orig_width, orig_height) > max_edge:
        image = image.resize((pre_resize_width, pre_resize_height), Image.Resampling.LANCZOS)

    # Get ground truth bboxes (these are in training/resized coordinates)
    gt_response = sample_data['conversations'][1]['value']  # GPT response
    gt_bboxes_training = parse_model_response(gt_response)
    print(f"Ground truth bboxes (training coords): {gt_bboxes_training}")

    # Convert GT bboxes from training coords to original coords for visualization
    # GT coordinates are based on training smart_resize (orig -> smart_resize)
    train_height, train_width = smart_resize(orig_height, orig_width)
    gt_bboxes_original = []
    if gt_bboxes_training:
        scale_w = orig_width / train_width
        scale_h = orig_height / train_height
        for bbox in gt_bboxes_training:
            x1, y1, x2, y2 = bbox
            x1_orig = int(x1 * scale_w)
            y1_orig = int(y1 * scale_h)
            x2_orig = int(x2 * scale_w)
            y2_orig = int(y2 * scale_h)
            gt_bboxes_original.append([x1_orig, y1_orig, x2_orig, y2_orig])
        print(f"Training dimensions for GT: {train_width} x {train_height}")
        print(f"Ground truth bboxes (original coords): {gt_bboxes_original}")

    if not gt_bboxes_training:
        print("No ground truth bboxes found!")
        return None

    # Add debugging info about the image
    print(f"DEBUG: Testing with resized image {new_width}x{new_height}")
    print(f"DEBUG: GT bboxes in training coords: {gt_bboxes_training}")

    # Simple and direct prompt with more guidance
    bbox_prompt = """Look at this spine X-ray image carefully. You should see curved sections of the spine (scoliotic curves).

Please identify ALL scoliotic curves and provide their bounding box coordinates. There may be 1, 2, or more curves - detect what you actually see.

Output in JSON format: {"curves": [{"bbox_2d": [x1, y1, x2, y2]}]}

If you see any curved sections of the spine, you MUST include them in your response."""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": bbox_prompt},
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
    inputs = inputs.to(model.device)

    # Generate prediction
    print("Generating model prediction...")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=0.1,
        )

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

    print(f"Model response: {response}")
    print(f"DEBUG: Response length: {len(response)} characters")
    print(f"DEBUG: Response type: {type(response)}")

    # Parse predicted bboxes (these should be in training coordinates since we're using the model)
    pred_bboxes_training = parse_model_response(response)
    print(f"Predicted bboxes (training coords): {pred_bboxes_training}")
    print(f"DEBUG: Found {len(pred_bboxes_training)} predicted bboxes")

    if not pred_bboxes_training:
        print("No predicted bboxes found!")
        return {
            'image_path': image_path,
            'gt_bboxes': gt_bboxes_original,
            'pred_bboxes': [],
            'iou': 0.0,
            'model_response': response
        }

    # Convert predicted bboxes from test coords to original coords for visualization
    # Prediction coordinates are based on test smart_resize (orig -> 1000px -> smart_resize)
    pred_bboxes_original = []
    scale_w = orig_width / new_width  # new_width/height are from test smart_resize
    scale_h = orig_height / new_height
    for bbox in pred_bboxes_training:
        x1, y1, x2, y2 = bbox
        x1_orig = int(x1 * scale_w)
        y1_orig = int(y1 * scale_h)
        x2_orig = int(x2 * scale_w)
        y2_orig = int(y2 * scale_h)
        pred_bboxes_original.append([x1_orig, y1_orig, x2_orig, y2_orig])
    print(f"Test dimensions for predictions: {new_width} x {new_height}")
    print(f"Predicted bboxes (original coords): {pred_bboxes_original}")

    # Calculate IOU: Convert both to same coordinate system (use original coordinates)
    # GT: training space -> original space (already calculated above)
    # Pred: test space -> original space (already calculated above)
    iou = calculate_iou(pred_bboxes_original, gt_bboxes_original)
    print(f"IOU: {iou:.4f}")

    # Visualize results using original coordinates
    save_path = f"./bbox_test_result_{os.path.basename(image_path).replace('.jpg', '.png')}"
    visualize_bboxes(image_path, gt_bboxes_original, pred_bboxes_original, save_path)

    return {
        'image_path': image_path,
        'gt_bboxes': gt_bboxes_original,
        'pred_bboxes': pred_bboxes_original,
        'iou': iou,
        'model_response': response
    }

def main():
    """Main function to run bbox IOU test"""
    print("=" * 60)
    print("Qwen2.5-VL Bbox IOU Test")
    print("=" * 60)

    # Load test data
    print(f"Loading test data from: {TEST_DATA_PATH}")
    with open(TEST_DATA_PATH, 'r') as f:
        test_data = json.load(f)

    print(f"Found {len(test_data)} test samples")

    # Load model
    try:
        model, processor = load_model()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Test second sample (the one that causes OOM in batch)
    print(f"\nTesting second sample...")
    sample = test_data[1]
    print(f"Sample image: {sample['image']}")

    result = test_single_sample(model, processor, sample, IMAGE_BASE_PATH)

    if result:
        print("\n" + "=" * 60)
        print("TEST RESULTS")
        print("=" * 60)
        print(f"Image: {os.path.basename(result['image_path'])}")
        print(f"Ground Truth Bboxes: {len(result['gt_bboxes'])}")
        print(f"Predicted Bboxes: {len(result['pred_bboxes'])}")
        print(f"IOU: {result['iou']:.4f}")
        print(f"Model Response: {result['model_response'][:200]}...")

    print("\nTest completed!")

if __name__ == "__main__":
    main()