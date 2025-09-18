#!/usr/bin/env python3
"""
批量处理脚本：将所有图片resize到700x1400并更新JSON中的bbox坐标
"""

import json
import re
import os
from pathlib import Path
from PIL import Image
from collections import defaultdict
import traceback

# 目标尺寸
TARGET_SIZE = (700, 1400)  # width, height

# 路径配置
DATA_ROOT = "/mnt/c/Users/HKUBS/OneDrive/桌面/LLM_Study/medical_vlm_training/data"
REFERENCE_JSON_DIR = "/mnt/c/Users/HKUBS/OneDrive/桌面/LLM_Study/aasce/02_data_analysis/quality_stratification/json"

def get_all_bbox_json_files():
    """获取所有包含bbox的JSON文件路径"""
    files = []

    # 3个备份文件
    tasks = ["apex_vertebrae", "curve_detection", "end_vertebrae"]
    for task in tasks:
        files.append(f"datasets_grounding/{task}/{task}_dataset_grounding_backup.json")

    # grounding和text_grounding分割文件
    splits = ["train", "test", "val"]
    qualities = ["high_quality", "low_quality"]

    for task in tasks:
        for split in splits:
            for quality in qualities:
                files.append(f"datasets_grounding/{task}/{split}_{quality}/{task}_{split}_{quality}_grounding.json")
                files.append(f"datasets_text_grounding/{task}/{split}_{quality}/{task}_{split}_{quality}_text_grounding.json")

    return files

def get_reference_image_info():
    """从参考JSON目录获取所有图片的尺寸信息"""
    reference_images = {}

    json_dir = Path(REFERENCE_JSON_DIR)

    for json_file in json_dir.glob("*_curves.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if 'curves' in data and len(data['curves']) > 0:
                # 从文件名推断图片名
                base_name = json_file.stem.replace('_curves', '')
                image_name = base_name + '.jpg'

                # 获取resize信息
                resize_info = data['curves'][0].get('resize_info')
                if resize_info:
                    reference_images[image_name] = {
                        'original_size': resize_info['original_size'],
                        'smart_resize_size': resize_info['resized_size']
                    }

        except Exception as e:
            print(f"Warning: Could not parse {json_file}: {e}")

    return reference_images

def calculate_transform_scale(original_size, smart_resize_size, target_size):
    """计算从训练坐标到目标坐标的变换比例"""
    orig_w, orig_h = original_size
    smart_w, smart_h = smart_resize_size
    target_w, target_h = target_size

    # 直接从训练坐标到目标坐标的比例
    scale_x = target_w / smart_w
    scale_y = target_h / smart_h

    return scale_x, scale_y

def transform_bbox(bbox, scale_x, scale_y):
    """变换bbox坐标"""
    x1, y1, x2, y2 = bbox
    return [
        round(x1 * scale_x),
        round(y1 * scale_y),
        round(x2 * scale_x),
        round(y2 * scale_y)
    ]

def update_bbox_in_text(text, scale_x, scale_y):
    """在文本中查找并更新所有bbox_2d坐标"""
    # 查找所有bbox_2d模式
    bbox_patterns = [
        r'\"bbox_2d\":\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]',  # 标准格式
        r'\\\"bbox_2d\\\":\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]',  # 转义格式
    ]

    updated_text = text
    total_replacements = 0

    for pattern in bbox_patterns:
        def replace_bbox(match):
            nonlocal total_replacements
            x1, y1, x2, y2 = map(int, match.groups())
            new_bbox = transform_bbox([x1, y1, x2, y2], scale_x, scale_y)
            total_replacements += 1

            # 保持原格式（是否转义）
            if '\\\"' in match.group(0):
                return f'\\\"bbox_2d\\\": [{new_bbox[0]}, {new_bbox[1]}, {new_bbox[2]}, {new_bbox[3]}]'
            else:
                return f'\"bbox_2d\": [{new_bbox[0]}, {new_bbox[1]}, {new_bbox[2]}, {new_bbox[3]}]'

        updated_text = re.sub(pattern, replace_bbox, updated_text)

    return updated_text, total_replacements

def process_json_file(file_path, image_transforms):
    """处理单个JSON文件，更新其中的bbox坐标"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content
        total_replacements = 0
        processed_images = set()

        # 按图片处理
        lines = content.split('\n')
        current_image = None

        for i, line in enumerate(lines):
            # 检查是否是图片行
            image_match = re.search(r'"image":\s*"([^"]*([^/\\]+\.jpg))"', line)
            if image_match:
                image_path = image_match.group(1)
                current_image = Path(image_path).name
                continue

            # 如果当前有图片上下文且该图片需要变换
            if current_image and current_image in image_transforms:
                if 'bbox_2d' in line:
                    scale_x, scale_y = image_transforms[current_image]
                    updated_line, replacements = update_bbox_in_text(line, scale_x, scale_y)

                    if replacements > 0:
                        lines[i] = updated_line
                        total_replacements += replacements
                        processed_images.add(current_image)

        # 重新组合内容
        updated_content = '\n'.join(lines)

        # 只有在有变化时才写入文件
        if updated_content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(updated_content)

            return True, total_replacements, len(processed_images)
        else:
            return False, 0, 0

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False, 0, 0

def resize_image(image_path, target_size):
    """将图片resize到目标尺寸"""
    try:
        with Image.open(image_path) as img:
            # 直接resize到目标尺寸
            resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
            resized_img.save(image_path)
            return True
    except Exception as e:
        print(f"Error resizing {image_path}: {e}")
        return False

def main():
    print("=" * 80)
    print("批量图片resize和坐标更新")
    print("=" * 80)

    # 1. 获取参考图片信息
    print("📊 Step 1: 获取参考图片尺寸信息...")
    reference_images = get_reference_image_info()
    print(f"   找到 {len(reference_images)} 张图片的尺寸信息")

    # 2. 计算变换矩阵
    print("\n📊 Step 2: 计算变换矩阵...")
    image_transforms = {}

    for image_name, info in reference_images.items():
        original_size = info['original_size']
        smart_resize_size = info['smart_resize_size']
        scale_x, scale_y = calculate_transform_scale(original_size, smart_resize_size, TARGET_SIZE)
        image_transforms[image_name] = (scale_x, scale_y)

    print(f"   计算了 {len(image_transforms)} 张图片的变换矩阵")

    # 3. 更新JSON文件
    print("\n📊 Step 3: 更新JSON文件中的bbox坐标...")
    bbox_json_files = get_all_bbox_json_files()

    total_files_updated = 0
    total_bbox_updated = 0
    total_images_processed = 0

    for json_file in bbox_json_files:
        file_path = Path(DATA_ROOT) / json_file
        if file_path.exists():
            updated, bbox_count, image_count = process_json_file(str(file_path), image_transforms)
            if updated:
                total_files_updated += 1
                total_bbox_updated += bbox_count
                total_images_processed += image_count
                print(f"   ✓ {json_file}: {bbox_count} bbox, {image_count} 图片")
            else:
                print(f"   - {json_file}: 无需更新")
        else:
            print(f"   ⚠️  文件不存在: {json_file}")

    print(f"\n   总计: {total_files_updated} 文件更新, {total_bbox_updated} bbox变换, {total_images_processed} 图片处理")

    # 4. Resize图片
    print(f"\n📊 Step 4: Resize图片到 {TARGET_SIZE[0]}x{TARGET_SIZE[1]}...")

    # 收集所有图片路径
    image_dirs = [
        "images/train/high_quality",
        "images/train/low_quality",
        "images/test/high_quality",
        "images/test/low_quality",
        "images/val/high_quality",
        "images/val/low_quality"
    ]

    total_images_resized = 0
    total_images_found = 0

    for img_dir in image_dirs:
        dir_path = Path(DATA_ROOT) / img_dir
        if dir_path.exists():
            for img_file in dir_path.glob("*.jpg"):
                total_images_found += 1
                if resize_image(str(img_file), TARGET_SIZE):
                    total_images_resized += 1
                    if total_images_resized % 50 == 0:
                        print(f"   已处理 {total_images_resized} 张图片...")

    print(f"   总计: {total_images_resized}/{total_images_found} 张图片resize成功")

    # 5. 总结
    print("\n" + "=" * 80)
    print("📊 处理完成总结:")
    print("=" * 80)
    print(f"✅ JSON文件更新: {total_files_updated} 个文件")
    print(f"✅ 坐标变换: {total_bbox_updated} 个bbox")
    print(f"✅ 图片resize: {total_images_resized} 张图片")
    print(f"✅ 目标尺寸: {TARGET_SIZE[0]}×{TARGET_SIZE[1]}")
    print("=" * 80)

if __name__ == "__main__":
    main()