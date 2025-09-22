#!/usr/bin/env python3
"""
IoU性能评估脚本
测试基础模型或LoRA微调后的模型在三个bbox任务上的IoU表现
测试 curve_detection, apex_vertebrae, end_vertebrae 的所有 test 数据

使用方法：
1. 测试基础模型：设置 EVALUATION_MODE = "base"
2. 测试LoRA模型：设置 EVALUATION_MODE = "lora"，并指定 LORA_PATH
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
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

# ====== 配置区域 - 直接修改这里的配置 ======
#
# 基础模型路径 - 使用HuggingFace 3B模型
BASE_MODEL_PATH = "Qwen/Qwen2.5-VL-3B-Instruct"

# 评估模式配置 - 选择要评估的模型类型
EVALUATION_MODE = "full_finetuned"  # Options: "base", "lora", "full_finetuned"

# 全参数微调模型路径 - 服务器上的训练输出路径
FULL_FINETUNED_PATH = "./output_grounding_full_r0_alpha0_lr2e-6_ep2p0_bs16"
# 这个路径应该和train.py中的OUTPUT_DIR一致

# LoRA模型路径 (保留用于以后的LoRA实验)
# LORA_PATH = "./output_grounding_lora_r32_alpha16_lr1e-5_ep3p0_bs16"

# 数据和输出路径
DATA_BASE_PATH = "./data/datasets_grounding"
IMAGE_BASE_PATH = "./data/images/test"

# 动态输出目录 - 根据评估模型自动命名
if EVALUATION_MODE == "base":
    OUTPUT_DIR = "./iou_evaluation_results/base_model_3b"
elif EVALUATION_MODE == "full_finetuned":
    # 从全参数微调路径提取模型名称
    model_name = FULL_FINETUNED_PATH.split("/")[-1] if "/" in FULL_FINETUNED_PATH else FULL_FINETUNED_PATH
    OUTPUT_DIR = f"./iou_evaluation_results/{model_name}"
else:  # LoRA mode
    # 从LoRA路径提取模型名称
    model_name = LORA_PATH.replace("./output_", "").replace("/", "_")
    OUTPUT_DIR = f"./iou_evaluation_results/{model_name}"

# 系统配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 多GPU并行配置
USE_MULTI_GPU = True  # 是否使用多GPU并行评估
BATCH_SIZE = 4 if torch.cuda.device_count() >= 4 else 2  # 根据GPU数量调整批次大小

# 快速切换示例：
# 1. 测试基础模型： EVALUATION_MODE = "base"
# 2. 测试当前LoRA： EVALUATION_MODE = "lora", LORA_PATH = "./output_grounding_lora_r32_alpha16_lr2e-7_ep0p5_bs4"
# 3. 测试其他LoRA： 修改LORA_PATH为对应的输出文件夹
# 4. 对比结果查看： ./iou_evaluation_results/{model_name}/ 目录下的json文件
# ========================================

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
    """Get appropriate prompt for each task - Experiment 004: Mixed Strategy Based on Task Success Patterns"""
    prompts = {
        "curve_detection": """Look at this spine X-ray and identify ALL curved sections of the spine (scoliotic curves).

There are usually multiple curves in scoliosis cases. Find each distinct curve and provide its bounding box.

Output in JSON format: {"curves": [{"bbox_2d": [x1, y1, x2, y2]}]}

Make sure to detect every curve you can see - don't combine multiple curves into one box.""",

        "apex_vertebrae": """Find the apex vertebrae in this spine X-ray - these are the most deviated vertebrae at the peak of each spinal curve.

Typically there are 1-2 apex vertebrae per image. Look for the most tilted vertebrae at curve peaks.

Output in JSON format: {"apex_vertebrae": [{"bbox_2d": [x1, y1, x2, y2]}]}

Focus on the most obvious apex points - avoid over-detection.""",

        "end_vertebrae": """Identify ALL end vertebrae (the boundary vertebrae of scoliotic curves) in this spine X-ray image. For each curve, identify the upper and lower end vertebrae.

Output in JSON format: {"end_vertebrae": [{"curve_type": "primary", "lower": {"name": "vertebra_name", "bbox_2d": [x1, y1, x2, y2]}, "upper": {"name": "vertebra_name", "bbox_2d": [x1, y1, x2, y2]}}]}

For each curve, provide both upper and lower boundary vertebrae."""
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

            # Different parsing logic for different tasks (restored from Experiment 001)
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

def visualize_detection_result(image_path, gt_bboxes, pred_bboxes, task_name, sample_info, output_dir):
    """可视化检测结果"""
    try:
        # 加载原始图像
        orig_image = Image.open(image_path)
        orig_width, orig_height = orig_image.size

        # 计算训练时的图像尺寸（用于GT坐标）
        train_height, train_width = smart_resize(orig_height, orig_width)

        # 测试时不进行resize，直接使用原始尺寸进行smart_resize
        test_height, test_width = smart_resize(orig_height, orig_width)

        # 转换GT坐标到原始图像坐标
        gt_bboxes_original = []
        if gt_bboxes:
            scale_w = orig_width / train_width
            scale_h = orig_height / train_height
            for bbox in gt_bboxes:
                x1, y1, x2, y2 = bbox
                gt_bboxes_original.append([
                    int(x1 * scale_w), int(y1 * scale_h),
                    int(x2 * scale_w), int(y2 * scale_h)
                ])

        # 转换预测坐标到原始图像坐标
        pred_bboxes_original = []
        if pred_bboxes:
            scale_w = orig_width / test_width
            scale_h = orig_height / test_height
            for bbox in pred_bboxes:
                x1, y1, x2, y2 = bbox
                pred_bboxes_original.append([
                    int(x1 * scale_w), int(y1 * scale_h),
                    int(x2 * scale_w), int(y2 * scale_h)
                ])

        # 创建可视化
        fig, ax = plt.subplots(1, 1, figsize=(12, 16))
        ax.imshow(orig_image)

        # 绘制GT框（绿色）
        for i, bbox in enumerate(gt_bboxes_original):
            x1, y1, x2, y2 = bbox
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                   linewidth=3, edgecolor='green', facecolor='none', alpha=0.8)
            ax.add_patch(rect)
            ax.text(x1, y1-15, f'GT-{i+1}', color='green', fontsize=12, weight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

        # 绘制预测框（红色）
        for i, bbox in enumerate(pred_bboxes_original):
            x1, y1, x2, y2 = bbox
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                   linewidth=3, edgecolor='red', facecolor='none', alpha=0.8)
            ax.add_patch(rect)
            ax.text(x1, y1-40, f'Pred-{i+1}', color='red', fontsize=12, weight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

        # 添加标题和信息
        task_display = {
            "curve_detection": "Curve Detection",
            "apex_vertebrae": "Apex Vertebrae Detection",
            "end_vertebrae": "End Vertebrae Detection"
        }

        title = f'{task_display.get(task_name, task_name)}\n'
        title += f'Image: {os.path.basename(image_path)}\n'
        title += f'IoU: {sample_info.get("iou", 0):.3f} | '
        title += f'GT: {len(gt_bboxes_original)} | Pred: {len(pred_bboxes_original)}'

        ax.set_title(title, fontsize=14, weight='bold', pad=20)
        ax.axis('off')

        plt.tight_layout()

        # 保存图像
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"{task_name}_visualization_{os.path.basename(image_path).replace('.jpg', '.png')}")
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"Visualization saved: {save_path}")
        return save_path

    except Exception as e:
        print(f"Error creating visualization: {e}")
        return None

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

        # 不进行resize，直接使用原始图像尺寸（所有图像都是700x1400）

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

        # Calculate IOU: Convert both to same coordinate system (original image coordinates)
        # Both GT and prediction coordinates are based on same smart_resize (orig -> smart_resize)

        # Get original image dimensions
        orig_image = Image.open(image_path)
        orig_width, orig_height = orig_image.size

        # Calculate dimensions for both GT and prediction coordinates (same process)
        train_height, train_width = smart_resize(orig_height, orig_width)
        test_height, test_width = train_height, train_width  # 现在训练和测试使用相同的尺寸

        # 由于训练和测试使用相同的smart_resize处理，可以直接在训练坐标系下计算IoU
        # 这样避免了不必要的坐标转换，提高精度
        iou = calculate_iou(pred_bboxes_training, gt_bboxes_training)

        # 为了可视化，仍然需要转换到原始坐标系
        scale_w = orig_width / train_width
        scale_h = orig_height / train_height

        gt_bboxes_original = []
        if gt_bboxes_training:
            for bbox in gt_bboxes_training:
                x1, y1, x2, y2 = bbox
                gt_bboxes_original.append([
                    int(x1 * scale_w), int(y1 * scale_h),
                    int(x2 * scale_w), int(y2 * scale_h)
                ])

        pred_bboxes_original = []
        if pred_bboxes_training:
            for bbox in pred_bboxes_training:
                x1, y1, x2, y2 = bbox
                pred_bboxes_original.append([
                    int(x1 * scale_w), int(y1 * scale_h),
                    int(x2 * scale_w), int(y2 * scale_h)
                ])

        result = {
            'image': sample_data['image'],
            'gt_count': len(gt_bboxes_training),
            'pred_count': len(pred_bboxes_training),
            'gt_bboxes': gt_bboxes_training,
            'pred_bboxes': pred_bboxes_training,
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
    """Test all samples for a specific task and quality with optional batching"""
    data_file = f"{DATA_BASE_PATH}/{task_name}/test_{quality}/{task_name}_test_{quality}_grounding.json"
    image_dir = f"{IMAGE_BASE_PATH}/{quality}"

    if not os.path.exists(data_file):
        print(f"Data file not found: {data_file}")
        return []

    with open(data_file, 'r') as f:
        test_data = json.load(f)

    results = []
    total_samples = len(test_data)
    gpu_count = torch.cuda.device_count()

    print(f"Testing {task_name} - {quality}: {total_samples} samples")
    print(f"Using {'multi-GPU batching' if USE_MULTI_GPU and gpu_count > 1 else 'single sample processing'}")
    print(f"Batch size: {BATCH_SIZE if USE_MULTI_GPU and gpu_count > 1 else 1}")

    if USE_MULTI_GPU and gpu_count > 1 and total_samples > BATCH_SIZE:
        # 批处理模式
        for i in range(0, total_samples, BATCH_SIZE):
            batch_end = min(i + BATCH_SIZE, total_samples)
            batch_samples = test_data[i:batch_end]

            print(f"  Processing batch {i//BATCH_SIZE + 1}/{(total_samples + BATCH_SIZE - 1)//BATCH_SIZE} (samples {i+1}-{batch_end})")

            batch_results = test_batch_samples(model, processor, batch_samples, image_dir, task_name)
            results.extend(batch_results)

            # 清理内存
            torch.cuda.empty_cache()
    else:
        # 单样本模式
        for i, sample in enumerate(test_data):
            if i % 10 == 0:
                print(f"  Progress: {i+1}/{total_samples}")

            image_filename = sample['image'].replace('images/', '')
            image_path = os.path.join(image_dir, image_filename)

            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}")
                continue

            result = test_single_sample(model, processor, sample, image_path, task_name)
            if result:
                results.append(result)

    return results

def test_batch_samples(model, processor, batch_samples, image_dir, task_name):
    """批处理多个样本以提高GPU利用率"""
    batch_results = []

    for sample in batch_samples:
        image_filename = sample['image'].replace('images/', '')
        image_path = os.path.join(image_dir, image_filename)

        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue

        # 仍然使用单样本处理，但可以后续优化为真正的批处理
        result = test_single_sample(model, processor, sample, image_path, task_name)
        if result:
            batch_results.append(result)

    return batch_results

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

def calculate_overall_performance(all_results):
    """Calculate overall performance across all tasks and qualities"""
    all_samples = []
    task_performances = {}
    quality_performances = {'high_quality': [], 'low_quality': []}

    # Collect all samples and organize by task/quality
    for task_name in all_results:
        task_samples = []
        task_performances[task_name] = {}

        for quality in all_results[task_name]:
            results = all_results[task_name][quality]['results']
            metrics = all_results[task_name][quality]['metrics']

            # Add to overall collection
            all_samples.extend(results)

            # Add to quality collection
            quality_performances[quality].extend(results)

            # Add to task collection
            task_samples.extend(results)
            task_performances[task_name][quality] = metrics.get('avg_iou', 0)

        # Calculate task overall performance
        if task_samples:
            task_performances[task_name]['overall'] = np.mean([r['iou'] for r in task_samples])

    # Calculate overall statistics
    if not all_samples:
        return {}

    total_samples = len(all_samples)
    overall_avg_iou = np.mean([r['iou'] for r in all_samples])
    overall_detection_rate = sum(1 for r in all_samples if r['pred_count'] > 0) / total_samples

    # Calculate quality comparison
    quality_comparison = {}
    for quality in quality_performances:
        if quality_performances[quality]:
            quality_comparison[quality] = {
                'avg_iou': np.mean([r['iou'] for r in quality_performances[quality]]),
                'detection_rate': sum(1 for r in quality_performances[quality] if r['pred_count'] > 0) / len(quality_performances[quality]),
                'samples': len(quality_performances[quality])
            }

    # Rank tasks by performance
    task_ranking = []
    for task_name in task_performances:
        if 'overall' in task_performances[task_name]:
            task_ranking.append({
                'task': task_name,
                'avg_iou': task_performances[task_name]['overall']
            })
    task_ranking.sort(key=lambda x: x['avg_iou'], reverse=True)

    return {
        'overall_avg_iou': overall_avg_iou,
        'overall_detection_rate': overall_detection_rate,
        'total_samples': total_samples,
        'task_performances': task_performances,
        'task_ranking': task_ranking,
        'quality_comparison': quality_comparison,
        'best_performing_task': task_ranking[0]['task'] if task_ranking else None,
        'worst_performing_task': task_ranking[-1]['task'] if task_ranking else None
    }

def load_model(model_path, lora_path=None):
    """Load model - supports base model, full fine-tuned model, or LoRA adapter"""
    print(f"Loading model from: {model_path}")
    try:
        # 判断是否为本地微调后的模型还是HuggingFace模型
        if os.path.exists(model_path) and os.path.isdir(model_path):
            print("检测到本地模型路径，加载全参数微调模型...")
            model_type = "Full Fine-tuned Model"
        elif model_path.startswith("Qwen/"):
            print("检测到HuggingFace模型路径，从HuggingFace下载...")
            model_type = "Base Model (HuggingFace)"
        else:
            print("尝试加载模型...")
            model_type = "Unknown Model Type"

        # 加载模型
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True  # 添加这个参数以支持自定义模型
        )

        # 尝试从同一路径加载processor，如果失败则使用基础路径
        try:
            processor = AutoProcessor.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
        except:
            print("从模型路径加载processor失败，使用基础模型processor...")
            base_model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
            processor = AutoProcessor.from_pretrained(base_model_name, use_fast=False, trust_remote_code=True)

        # Apply LoRA adapter if provided
        if lora_path and os.path.exists(lora_path):
            print(f"Loading LoRA adapter from: {lora_path}")
            try:
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, lora_path)
                print("✓ LoRA adapter loaded successfully!")
                model_type = "LoRA Fine-tuned Model"
            except Exception as e:
                print(f"Warning: Failed to load LoRA adapter: {e}")
                print("Continuing with base model...")

        print(f"✓ {model_type} loaded successfully!")
        return model, processor, model_type
    except Exception as e:
        print(f"Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def main():
    """Main function - IoU performance evaluation for medical VLM tasks"""
    # 根据配置确定模型路径
    if EVALUATION_MODE == "base":
        model_path = BASE_MODEL_PATH
        lora_path = None
        print(f"评估模式: 基础模型")
    elif EVALUATION_MODE == "full_finetuned":
        model_path = FULL_FINETUNED_PATH
        lora_path = None
        print(f"评估模式: 全参数微调模型")
    elif EVALUATION_MODE == "lora":
        model_path = BASE_MODEL_PATH
        lora_path = LORA_PATH
        print(f"评估模式: LoRA微调模型")
    else:
        raise ValueError(f"未知的评估模式: {EVALUATION_MODE}")

    print("=" * 60)
    print("IoU性能评估")
    print("=" * 60)
    print(f"模型路径: {model_path}")
    print(f"评估模式: {EVALUATION_MODE}")
    if EVALUATION_MODE == "lora":
        print(f"LoRA路径: {lora_path}")
    elif EVALUATION_MODE == "full_finetuned":
        print(f"全参数微调模型路径: {model_path}")
    print(f"Output Dir: {OUTPUT_DIR}")
    print(f"Device: {DEVICE}")

    # GPU信息
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"GPU Count: {gpu_count}")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
        print(f"Multi-GPU Evaluation: {'Enabled' if USE_MULTI_GPU and gpu_count > 1 else 'Disabled'}")
        print(f"Batch Size: {BATCH_SIZE if USE_MULTI_GPU and gpu_count > 1 else 1}")

    print("=" * 60)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load model
    model, processor, model_type = load_model(model_path, lora_path)
    if model is None:
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

    # Generate visualizations for each task (one sample per task)
    print(f"\n{'='*40}")
    print("Generating Visualizations...")
    print(f"{'='*40}")

    visualization_dir = os.path.join(OUTPUT_DIR, "visualizations")
    for task_name in TASKS:
        print(f"\nCreating visualization for {task_name}...")

        # Use high_quality data for visualization if available, otherwise use low_quality
        quality_to_use = "high_quality" if "high_quality" in all_results[task_name] else "low_quality"
        task_results = all_results[task_name][quality_to_use]['results']

        if task_results:
            # Find a sample with both GT and predictions for better visualization
            best_sample = None
            for sample in task_results:
                if (sample.get('gt_bboxes') and
                    sample.get('pred_bboxes') and
                    sample.get('iou', 0) > 0):
                    best_sample = sample
                    break

            # If no sample with both GT and pred, use the first sample with GT
            if not best_sample:
                for sample in task_results:
                    if sample.get('gt_bboxes'):
                        best_sample = sample
                        break

            # If still no sample, use the first one
            if not best_sample and task_results:
                best_sample = task_results[0]

            if best_sample:
                # Build image path
                image_filename = best_sample['image'].replace('images/', '')
                image_path = os.path.join(IMAGE_BASE_PATH, quality_to_use, image_filename)

                if os.path.exists(image_path):
                    try:
                        gt_bboxes = best_sample.get('gt_bboxes', [])
                        pred_bboxes = best_sample.get('pred_bboxes', [])

                        visualize_detection_result(
                            image_path=image_path,
                            gt_bboxes=gt_bboxes,
                            pred_bboxes=pred_bboxes,
                            task_name=task_name,
                            sample_info=best_sample,
                            output_dir=visualization_dir
                        )
                    except Exception as e:
                        print(f"Failed to create visualization for {task_name}: {e}")
                else:
                    print(f"Image not found for {task_name}: {image_path}")
            else:
                print(f"No suitable sample found for {task_name} visualization")

    total_time = time.time() - start_time

    # Calculate overall performance summary
    overall_performance = calculate_overall_performance(all_results)

    # Generate timestamp for file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Determine model prefix based on evaluation mode
    if EVALUATION_MODE == "base":
        model_prefix = "base"
    elif EVALUATION_MODE == "lora":
        model_prefix = "lora"
    elif EVALUATION_MODE == "full_finetuned":
        model_prefix = "full_finetuned"
    else:
        model_prefix = "unknown"

    # Save results with model type and timestamp
    output_file = os.path.join(OUTPUT_DIR, f"{model_prefix}_model_iou_results_{timestamp}.json")
    final_results = {
        'overall_performance': overall_performance,
        'results': all_results,
        'config': {
            'device': DEVICE,
            'total_duration': total_time,
            'base_model': BASE_MODEL_PATH,
            'evaluation_mode': EVALUATION_MODE,
            'model_path': model_path,
            'lora_path': lora_path if EVALUATION_MODE == "lora" else None,
            'full_finetuned_path': FULL_FINETUNED_PATH if EVALUATION_MODE == "full_finetuned" else None,
            'model_type': model_type,
            'visualization_dir': visualization_dir,
            'timestamp': timestamp
        }
    }

    # 添加GPU配置信息到结果中
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        final_results['config']['gpu_info'] = {
            'gpu_count': gpu_count,
            'multi_gpu_enabled': USE_MULTI_GPU and gpu_count > 1,
            'batch_size': BATCH_SIZE if USE_MULTI_GPU and gpu_count > 1 else 1,
            'gpu_names': [torch.cuda.get_device_name(i) for i in range(gpu_count)]
        }

    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print("OVERALL PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    print(f"Model Type: {model_type}")
    print(f"Total Samples: {overall_performance.get('total_samples', 0)}")
    print(f"Overall Average IoU: {overall_performance.get('overall_avg_iou', 0):.4f}")
    print(f"Overall Detection Rate: {overall_performance.get('overall_detection_rate', 0):.2%}")
    print(f"Total Testing Time: {total_time:.1f}s ({total_time/60:.1f}m)")

    # 显示加速效果
    if torch.cuda.is_available() and USE_MULTI_GPU and torch.cuda.device_count() > 1:
        estimated_single_gpu_time = total_time * torch.cuda.device_count()
        speedup = estimated_single_gpu_time / total_time
        print(f"Multi-GPU Speedup: ~{speedup:.1f}x (estimated)")

    # Task ranking
    task_ranking = overall_performance.get('task_ranking', [])
    if task_ranking:
        print(f"\nTask Performance Ranking:")
        for i, task_info in enumerate(task_ranking, 1):
            print(f"  {i}. {task_info['task']}: IoU={task_info['avg_iou']:.4f}")

    # Quality comparison
    quality_comparison = overall_performance.get('quality_comparison', {})
    if quality_comparison:
        print(f"\nQuality Comparison:")
        for quality, metrics in quality_comparison.items():
            print(f"  {quality}: IoU={metrics['avg_iou']:.4f}, Detection={metrics['detection_rate']:.2%}, Samples={metrics['samples']}")

    print(f"\nDetailed Results by Task:")
    for task_name in TASKS:
        print(f"\n{task_name.upper()}:")
        for quality in QUALITIES:
            metrics = all_results[task_name][quality]['metrics']
            duration = all_results[task_name][quality]['duration']
            print(f"  {quality}: IOU={metrics.get('avg_iou', 0):.4f}, Detection={metrics.get('detection_rate', 0):.2%}, Time={duration:.1f}s")

    print(f"\nResults saved to: {output_file}")
    print(f"Visualizations saved to: {visualization_dir}")
    print("Testing completed!")

if __name__ == "__main__":
    main()