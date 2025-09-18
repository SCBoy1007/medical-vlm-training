#!/usr/bin/env python3
"""
测试脚本：分析单张图片的坐标变换需求
目标图片：sunhl-1th-01-Mar-2017-310 a ap.jpg
"""

import json
import re
from pathlib import Path

# 目标图片信息
TARGET_IMAGE = "sunhl-1th-01-Mar-2017-310 a ap.jpg"
TARGET_SIZE = (700, 1400)  # width, height

# 路径配置
REFERENCE_JSON = "/mnt/c/Users/HKUBS/OneDrive/桌面/LLM_Study/aasce/02_data_analysis/quality_stratification/json/sunhl-1th-01-Mar-2017-310 a ap_curves.json"
DATA_ROOT = "/mnt/c/Users/HKUBS/OneDrive/桌面/LLM_Study/medical_vlm_training/data"

def get_image_size_info():
    """从参考JSON获取图片尺寸信息"""
    with open(REFERENCE_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 从第一个curve获取resize_info
    resize_info = data['curves'][0]['resize_info']
    original_size = resize_info['original_size']  # [width, height]
    smart_resize_size = resize_info['resized_size']  # [width, height]

    return original_size, smart_resize_size

def calculate_transform_matrices(original_size, smart_resize_size, target_size):
    """计算坐标变换矩阵"""
    orig_w, orig_h = original_size
    smart_w, smart_h = smart_resize_size
    target_w, target_h = target_size

    # 从训练坐标回到原始坐标的比例
    training_to_original_scale_x = orig_w / smart_w
    training_to_original_scale_y = orig_h / smart_h

    # 从原始坐标到目标坐标的比例
    original_to_target_scale_x = target_w / orig_w
    original_to_target_scale_y = target_h / orig_h

    # 直接从训练坐标到目标坐标的比例
    direct_scale_x = (orig_w / smart_w) * (target_w / orig_w)  # = target_w / smart_w
    direct_scale_y = (orig_h / smart_h) * (target_h / orig_h)  # = target_h / smart_h

    return {
        'training_to_original': (training_to_original_scale_x, training_to_original_scale_y),
        'original_to_target': (original_to_target_scale_x, original_to_target_scale_y),
        'direct_transform': (direct_scale_x, direct_scale_y)
    }

def transform_bbox(bbox, scale_x, scale_y):
    """变换单个bbox坐标"""
    x1, y1, x2, y2 = bbox
    return [
        round(x1 * scale_x),
        round(y1 * scale_y),
        round(x2 * scale_x),
        round(y2 * scale_y)
    ]

def find_bbox_in_json_lines(file_path):
    """在JSON文件中找到包含bbox的行号（改进版，处理复杂嵌套，去重）"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    bbox_lines = []
    in_target_image_context = False

    for i, line in enumerate(lines, 1):
        line_stripped = line.strip()

        # 检查是否找到目标图片
        if TARGET_IMAGE in line:
            in_target_image_context = True
            continue

        # 如果在目标图片上下文中
        if in_target_image_context:
            # 检查是否开始新的对话条目（遇到新的image字段且不是目标图片）
            if '"image"' in line and TARGET_IMAGE not in line:
                in_target_image_context = False
                continue

            # 寻找所有可能的bbox_2d模式（统一处理，避免重复）
            if 'bbox_2d' in line:
                found_bboxes = extract_all_bboxes_from_line(line)

                # 去重：检查是否已经在当前行找到过相同的bbox
                line_existing_bboxes = [item['bbox'] for item in bbox_lines if item['line_number'] == i]

                for bbox in found_bboxes:
                    if bbox not in line_existing_bboxes:  # 去重
                        bbox_lines.append({
                            'line_number': i,
                            'bbox': bbox,
                            'line_content': line_stripped,
                            'context': 'auto_detected'
                        })

    return bbox_lines

def extract_all_bboxes_from_line(line):
    """从行中提取所有bbox坐标（统一处理，去重）"""
    bboxes = []
    seen_bboxes = set()

    # 寻找所有可能的bbox_2d模式
    bbox_patterns = [
        r'\"bbox_2d\":\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]',  # 标准格式
        r'\\\"bbox_2d\\\":\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]',  # 转义格式
        r'bbox_2d.*?\[(\d+)[,\s]+(\d+)[,\s]+(\d+)[,\s]+(\d+)\]',  # 宽松格式
    ]

    for pattern in bbox_patterns:
        matches = re.findall(pattern, line)
        for match in matches:
            bbox = [int(x) for x in match]
            bbox_tuple = tuple(bbox)  # 转换为tuple用于set去重

            if bbox_tuple not in seen_bboxes:
                seen_bboxes.add(bbox_tuple)
                bboxes.append(bbox)

    return bboxes

def analyze_json_files():
    """分析所有包含目标图片的JSON文件"""

    # 获取所有包含目标图片的文件
    json_files = [
        "cobb_angle_ground_truth.json",
        "datasets_grounding/apex_vertebrae/apex_vertebrae_dataset_grounding_backup.json",
        "datasets_grounding/apex_vertebrae/train_high_quality/apex_vertebrae_train_high_quality_grounding.json",
        "datasets_grounding/curve_detection/curve_detection_dataset_grounding_backup.json",
        "datasets_grounding/curve_detection/train_high_quality/curve_detection_train_high_quality_grounding.json",
        "datasets_grounding/end_vertebrae/end_vertebrae_dataset_grounding_backup.json",
        "datasets_grounding/end_vertebrae/train_high_quality/end_vertebrae_train_high_quality_grounding.json",
        "datasets_text_grounding/apex_vertebrae/train_high_quality/apex_vertebrae_train_high_quality_text_grounding.json",
        "datasets_text_grounding/curve_detection/train_high_quality/curve_detection_train_high_quality_text_grounding.json",
        "datasets_text_grounding/end_vertebrae/train_high_quality/end_vertebrae_train_high_quality_text_grounding.json",
        "datasets_text/apex_vertebrae/train_high_quality/apex_vertebrae_train_high_quality_text.json",
        "datasets_text/curve_detection/train_high_quality/curve_detection_train_high_quality_text.json",
        "datasets_text/end_vertebrae/train_high_quality/end_vertebrae_train_high_quality_text.json"
    ]

    results = []

    for json_file in json_files:
        file_path = Path(DATA_ROOT) / json_file
        if file_path.exists():
            bbox_lines = find_bbox_in_json_lines(str(file_path))
            if bbox_lines:
                results.append({
                    'file': json_file,
                    'bbox_lines': bbox_lines
                })

    return results

def main():
    print("=" * 80)
    print("单张图片坐标变换分析")
    print("=" * 80)
    print(f"目标图片: {TARGET_IMAGE}")
    print()

    # 1. 获取尺寸信息
    original_size, smart_resize_size = get_image_size_info()
    print("📐 尺寸信息:")
    print(f"   原始尺寸: {original_size[0]}x{original_size[1]} (width x height)")
    print(f"   Smart Resize尺寸: {smart_resize_size[0]}x{smart_resize_size[1]}")
    print(f"   目标尺寸: {TARGET_SIZE[0]}x{TARGET_SIZE[1]}")
    print()

    # 2. 计算变换矩阵
    transforms = calculate_transform_matrices(original_size, smart_resize_size, TARGET_SIZE)
    print("🔄 变换比例:")
    print(f"   训练坐标→原始坐标: x×{transforms['training_to_original'][0]:.4f}, y×{transforms['training_to_original'][1]:.4f}")
    print(f"   原始坐标→目标坐标: x×{transforms['original_to_target'][0]:.4f}, y×{transforms['original_to_target'][1]:.4f}")
    print(f"   直接变换(训练→目标): x×{transforms['direct_transform'][0]:.4f}, y×{transforms['direct_transform'][1]:.4f}")
    print()

    # 3. 分析JSON文件
    print("📝 需要修改的JSON文件和坐标:")
    print()

    json_results = analyze_json_files()
    scale_x, scale_y = transforms['direct_transform']

    total_bboxes = 0
    for result in json_results:
        print(f"文件: {result['file']}")
        print(f"   包含 {len(result['bbox_lines'])} 个bbox需要修改:")

        for bbox_info in result['bbox_lines']:
            old_bbox = bbox_info['bbox']
            new_bbox = transform_bbox(old_bbox, scale_x, scale_y)
            context = bbox_info.get('context', 'unknown')

            print(f"   第{bbox_info['line_number']:3d}行 ({context}): {old_bbox} → {new_bbox}")
            total_bboxes += 1
        print()

    print("=" * 80)
    print("📊 总结:")
    print(f"   需要修改 {len(json_results)} 个JSON文件")
    print(f"   需要修改 {total_bboxes} 个bbox坐标")
    print(f"   图片需要从 {original_size[0]}x{original_size[1]} resize到 {TARGET_SIZE[0]}x{TARGET_SIZE[1]}")
    print("=" * 80)

if __name__ == "__main__":
    main()