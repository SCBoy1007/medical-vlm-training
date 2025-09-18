#!/usr/bin/env python3
"""
验证脚本：检查每张图片在JSON文件中的分布模式
预期：要么出现在所有9个相关JSON文件中，要么不出现（正常脊椎）
"""

import json
import re
from pathlib import Path
from collections import defaultdict

# 路径配置
DATA_ROOT = "/mnt/c/Users/HKUBS/OneDrive/桌面/LLM_Study/medical_vlm_training/data"
REFERENCE_JSON_DIR = "/mnt/c/Users/HKUBS/OneDrive/桌面/LLM_Study/aasce/02_data_analysis/quality_stratification/json"

# 生成所有相关的JSON文件列表（包含bbox的grounding和text_grounding数据集）
def get_all_bbox_json_files():
    """动态生成所有包含bbox的JSON文件路径"""
    files = []

    # 3个备份文件
    tasks = ["apex_vertebrae", "curve_detection", "end_vertebrae"]
    for task in tasks:
        files.append(f"datasets_grounding/{task}/{task}_dataset_grounding_backup.json")

    # grounding 分割文件 (train/test/val × high/low quality)
    splits = ["train", "test", "val"]
    qualities = ["high_quality", "low_quality"]

    for task in tasks:
        for split in splits:
            for quality in qualities:
                files.append(f"datasets_grounding/{task}/{split}_{quality}/{task}_{split}_{quality}_grounding.json")

    # text_grounding 分割文件 (train/test/val × high/low quality)
    for task in tasks:
        for split in splits:
            for quality in qualities:
                files.append(f"datasets_text_grounding/{task}/{split}_{quality}/{task}_{split}_{quality}_text_grounding.json")

    return files

BBOX_JSON_FILES = get_all_bbox_json_files()

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
                        'smart_resize_size': resize_info['resized_size'],
                        'curve_count': len(data['curves'])
                    }

        except Exception as e:
            print(f"Warning: Could not parse {json_file}: {e}")

    return reference_images

def extract_images_from_json(file_path):
    """从JSON文件中提取所有图片名称"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        images = set()

        # 处理不同的JSON格式
        if isinstance(data, list):
            # 列表格式
            for item in data:
                if 'image' in item:
                    image_path = item['image']
                    # 提取文件名
                    image_name = Path(image_path).name
                    images.add(image_name)
        elif isinstance(data, dict):
            # 字典格式，可能有annotations字段
            if 'annotations' in data:
                for image_name, item in data['annotations'].items():
                    images.add(image_name)

        return images

    except Exception as e:
        print(f"Warning: Could not parse {file_path}: {e}")
        return set()

def count_bbox_in_file(file_path, image_name):
    """计算指定图片在文件中的bbox数量"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 简单计数：在图片上下文中的bbox_2d出现次数
        lines = content.split('\n')
        in_target_context = False
        bbox_count = 0

        for line in lines:
            if image_name in line:
                in_target_context = True
                continue

            if in_target_context:
                # 检查是否到了新图片
                if '"image"' in line and image_name not in line:
                    break

                # 计算这行的bbox_2d数量
                bbox_count += len(re.findall(r'bbox_2d.*?\[\d+,\s*\d+,\s*\d+,\s*\d+\]', line))

        return bbox_count

    except Exception as e:
        print(f"Warning: Could not count bboxes in {file_path}: {e}")
        return 0

def main():
    print("=" * 80)
    print("图片分布模式验证 (完整版)")
    print("=" * 80)
    print(f"总共检查 {len(BBOX_JSON_FILES)} 个JSON文件:")
    for i, file in enumerate(BBOX_JSON_FILES, 1):
        print(f"  {i:2d}. {file}")
    print()

    # 1. 获取参考图片信息
    print("📊 Step 1: 获取参考图片信息...")
    reference_images = get_reference_image_info()
    print(f"   从参考JSON找到 {len(reference_images)} 张有弯曲标注的图片")

    # 2. 收集每个JSON文件中的图片
    print("\n📊 Step 2: 收集各JSON文件中的图片...")
    image_distribution = defaultdict(list)

    for json_file in BBOX_JSON_FILES:
        file_path = Path(DATA_ROOT) / json_file
        if file_path.exists():
            images = extract_images_from_json(str(file_path))
            print(f"   {json_file}: {len(images)} 张图片")

            for image in images:
                image_distribution[image].append(json_file)
        else:
            print(f"   ⚠️  文件不存在: {json_file}")

    # 3. 分析分布模式
    print(f"\n📊 Step 3: 分析分布模式...")

    pattern_0 = []  # 不出现在任何文件中
    pattern_9 = []  # 出现在所有9个文件中
    pattern_other = []  # 其他异常模式

    total_expected_files = len(BBOX_JSON_FILES)

    for image, files in image_distribution.items():
        file_count = len(files)

        if file_count == 0:
            pattern_0.append(image)
        elif file_count == total_expected_files:
            pattern_9.append(image)
        else:
            pattern_other.append((image, file_count, files))

    # 4. 检查参考图片是否都遵循pattern_9
    print(f"\n📊 Step 4: 验证参考图片分布...")
    reference_not_in_pattern_9 = []

    for ref_image in reference_images.keys():
        if ref_image not in pattern_9:
            current_count = len(image_distribution.get(ref_image, []))
            reference_not_in_pattern_9.append((ref_image, current_count))

    # 5. 输出结果
    print("\n" + "=" * 80)
    print("📊 分布模式分析结果:")
    print("=" * 80)

    print(f"✅ Pattern 9 (出现在所有{total_expected_files}个文件): {len(pattern_9)} 张图片")
    if len(pattern_9) <= 10:  # 如果不多就全部显示
        for img in pattern_9:
            print(f"     {img}")
    else:
        for img in pattern_9[:5]:
            print(f"     {img}")
        print(f"     ... 还有 {len(pattern_9)-5} 张图片")

    print(f"\n✅ Pattern 0 (不出现在任何文件): {len(pattern_0)} 张图片")

    print(f"\n❌ 异常模式 (部分出现): {len(pattern_other)} 张图片")
    for img, count, files in pattern_other:
        print(f"     {img}: 出现在 {count} 个文件中")
        for file in files[:3]:  # 只显示前3个文件
            print(f"       - {file}")
        if len(files) > 3:
            print(f"       - ... 还有 {len(files)-3} 个文件")

    print(f"\n🔍 参考图片验证:")
    print(f"   参考JSON中有弯曲的图片: {len(reference_images)}")
    print(f"   遵循Pattern 9的参考图片: {len(reference_images) - len(reference_not_in_pattern_9)}")

    if reference_not_in_pattern_9:
        print(f"   ❌ 不遵循Pattern 9的参考图片: {len(reference_not_in_pattern_9)}")
        for img, count in reference_not_in_pattern_9:
            print(f"      {img}: 只出现在 {count} 个文件中")

    # 6. 总结
    print("\n" + "=" * 80)
    print("📊 总结:")
    all_good = len(pattern_other) == 0 and len(reference_not_in_pattern_9) == 0

    if all_good:
        print("✅ 分布模式完全符合预期！")
        print(f"   - {len(pattern_9)} 张有病例图片出现在所有 {total_expected_files} 个文件中")
        print(f"   - {len(pattern_0)} 张正常图片不出现在任何bbox文件中")
    else:
        print("❌ 发现异常分布模式，需要检查数据一致性")

    print("=" * 80)

if __name__ == "__main__":
    main()