#!/usr/bin/env python3
"""
测试脚本：验证processor是否破坏padding结果
目标：确认988 seq_len的来源，验证processor前后的维度变化
"""

import os
import sys
import torch
import logging
from PIL import Image
import math

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from qwenvl.data import data_list
from qwenvl.train.argument import DataArguments, ModelArguments
from transformers import Qwen2VLProcessor, AutoTokenizer

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_padding_vs_processor():
    """
    核心测试：比较padding前后，以及processor前后的维度变化
    """
    logger.info("=" * 60)
    logger.info("🔍 HYPOTHESIS VERIFICATION TEST")
    logger.info("=" * 60)

    # 使用训练时的相同配置
    model_path = "./models/Qwen2.5-VL-7B-Instruct"

    try:
        # 加载processor
        processor = Qwen2VLProcessor.from_pretrained(model_path)
        logger.info("✅ Processor loaded successfully")

        # 配置processor参数（与训练保持一致）
        processor.image_processor.max_pixels = 802816  # 1024*28*28
        processor.image_processor.min_pixels = 3136    # 56*56

        # 调试processor配置
        logger.info(f"🔧 Processor max_pixels: {processor.image_processor.max_pixels}")
        logger.info(f"🔧 Processor min_pixels: {processor.image_processor.min_pixels}")
        logger.info(f"🔧 Processor type: {type(processor.image_processor)}")

        # 简单测试processor是否工作
        logger.info("🧪 Quick processor test...")
        try:
            # 创建一个小的测试图像
            test_img = Image.new('RGB', (224, 224), (128, 128, 128))
            # Qwen2VL processor需要text参数
            quick_result = processor(text="<image>What is this?", images=[test_img], return_tensors="pt")
            logger.info(f"✅ Quick test passed. Keys: {list(quick_result.keys())}")
            if "pixel_values" in quick_result:
                logger.info(f"✅ pixel_values shape: {quick_result['pixel_values'].shape}")
            if "image_grid_thw" in quick_result:
                logger.info(f"✅ image_grid_thw: {quick_result['image_grid_thw']}")
        except Exception as e:
            logger.error(f"❌ Quick test failed: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return

        # 获取一个训练样本图像
        test_image_path = "./data/images/train/high_quality/sunhl-1th-10-Jan-2017-230 B AP.jpg"

        if not os.path.exists(test_image_path):
            logger.error(f"❌ Test image not found: {test_image_path}")
            # 尝试找第一个可用图像
            data_dir = "./data/images/train/high_quality/"
            if os.path.exists(data_dir):
                images = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]
                if images:
                    test_image_path = os.path.join(data_dir, images[0])
                    logger.info(f"🔄 Using alternative image: {test_image_path}")

        if not os.path.exists(test_image_path):
            logger.error("❌ No test images found")
            return

        logger.info(f"📷 Testing with image: {test_image_path}")

        # 加载原始图像
        original_image = Image.open(test_image_path).convert("RGB")
        orig_width, orig_height = original_image.size

        logger.info(f"📊 Original image: {orig_width}x{orig_height}")

        # === STEP 1: 应用我们的padding算法 ===
        logger.info("\n🔧 STEP 1: Applying our padding algorithm...")
        padded_image = apply_spatial_padding(original_image)
        pad_width, pad_height = padded_image.size

        # 计算padding后的patches
        h_patches_pad = pad_height // 28
        w_patches_pad = pad_width // 28
        total_patches_pad = h_patches_pad * w_patches_pad

        logger.info(f"📊 After padding: {pad_width}x{pad_height}")
        logger.info(f"📦 Patches: {h_patches_pad}x{w_patches_pad} = {total_patches_pad}")
        logger.info(f"✅ Spatial merge compatible: {total_patches_pad % 16 == 0} (remainder: {total_patches_pad % 16})")

        # === STEP 2: 应用processor到原始图像 ===
        logger.info("\n🏭 STEP 2: Processor on ORIGINAL image...")
        try:
            # 使用正确的processor调用方式 - 需要text参数
            result_orig = processor(text="<image>Analyze this medical image", images=[original_image], return_tensors="pt")
            pixel_values_orig = result_orig["pixel_values"]
            grid_thw_orig = result_orig["image_grid_thw"][0]

            logger.info(f"📊 Processor result (original): pixel_values shape = {pixel_values_orig.shape}")
            logger.info(f"📊 Grid THW (original): {grid_thw_orig}")

            # 计算实际的seq_len
            _, channels, proc_height_orig, proc_width_orig = pixel_values_orig.shape
            proc_patches_h_orig = proc_height_orig // 28
            proc_patches_w_orig = proc_width_orig // 28
            total_patches_proc_orig = proc_patches_h_orig * proc_patches_w_orig
            seq_len_orig = total_patches_proc_orig // 4  # spatial_merge_unit = 4

            logger.info(f"📦 Processor patches (original): {proc_patches_h_orig}x{proc_patches_w_orig} = {total_patches_proc_orig}")
            logger.info(f"🎯 Calculated seq_len (original): {seq_len_orig}")
            logger.info(f"🔍 Is this the problematic 988? {seq_len_orig == 988}")

        except Exception as e:
            logger.error(f"❌ Processor failed on original image: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")

        # === STEP 3: 应用processor到padded图像 ===
        logger.info("\n🏭 STEP 3: Processor on PADDED image...")
        try:
            # 使用正确的processor调用方式 - 需要text参数
            result_pad = processor(text="<image>Analyze this medical image", images=[padded_image], return_tensors="pt")
            pixel_values_pad = result_pad["pixel_values"]
            grid_thw_pad = result_pad["image_grid_thw"][0]

            logger.info(f"📊 Processor result (padded): pixel_values shape = {pixel_values_pad.shape}")
            logger.info(f"📊 Grid THW (padded): {grid_thw_pad}")

            # 计算实际的seq_len
            _, channels, proc_height_pad, proc_width_pad = pixel_values_pad.shape
            proc_patches_h_pad = proc_height_pad // 28
            proc_patches_w_pad = proc_width_pad // 28
            total_patches_proc_pad = proc_patches_h_pad * proc_patches_w_pad
            seq_len_pad = total_patches_proc_pad // 4  # spatial_merge_unit = 4

            logger.info(f"📦 Processor patches (padded): {proc_patches_h_pad}x{proc_patches_w_pad} = {total_patches_proc_pad}")
            logger.info(f"🎯 Calculated seq_len (padded): {seq_len_pad}")
            logger.info(f"✅ Spatial merge compatible: {total_patches_proc_pad % 16 == 0} (remainder: {total_patches_proc_pad % 16})")

        except Exception as e:
            logger.error(f"❌ Processor failed on padded image: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")

        # === STEP 4: 验证维度匹配性 ===
        logger.info("\n🔬 STEP 4: Dimension compatibility analysis...")

        if 'pixel_values_orig' in locals():
            # 模拟spatial merge reshape
            try:
                batch, channels, height, width = pixel_values_orig.shape
                total_elements = batch * channels * height * width
                spatial_merge_unit = 4

                logger.info(f"📊 Original tensor total elements: {total_elements}")
                logger.info(f"🎯 Attempting reshape: [{seq_len_orig}, {spatial_merge_unit}, -1]")

                if total_elements % (seq_len_orig * spatial_merge_unit) == 0:
                    hidden_dim = total_elements // (seq_len_orig * spatial_merge_unit)
                    logger.info(f"✅ Reshape would succeed: [{seq_len_orig}, {spatial_merge_unit}, {hidden_dim}]")
                else:
                    logger.error(f"❌ Reshape would FAIL - not evenly divisible")
                    logger.error(f"   Expected: {seq_len_orig * spatial_merge_unit} divisor")
                    logger.error(f"   Actual: {total_elements} elements")
                    logger.error(f"   Remainder: {total_elements % (seq_len_orig * spatial_merge_unit)}")

            except Exception as e:
                logger.error(f"❌ Reshape simulation failed: {e}")

        # === STEP 5: 关键发现总结 ===
        logger.info("\n" + "=" * 60)
        logger.info("🎯 KEY FINDINGS SUMMARY")
        logger.info("=" * 60)

        if 'seq_len_orig' in locals():
            if seq_len_orig == 988:
                logger.info("🔍 CONFIRMED: Processor generates the problematic seq_len=988!")
                logger.info("🔧 This proves our hypothesis - processor modifies image dimensions")
            else:
                logger.info(f"🤔 Processor seq_len = {seq_len_orig}, not 988. Need more investigation.")

        if 'total_patches_proc_pad' in locals() and 'total_patches_pad' in locals():
            if total_patches_proc_pad != total_patches_pad:
                logger.info("🚨 CRITICAL: Processor changed padded image dimensions!")
                logger.info(f"   Our padding: {total_patches_pad} patches")
                logger.info(f"   After processor: {total_patches_proc_pad} patches")
            else:
                logger.info("✅ Processor preserved our padding")

    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

def apply_spatial_padding(image):
    """
    应用与training脚本相同的padding算法
    """
    width, height = image.size

    # 确保28的倍数
    new_width = ((width + 27) // 28) * 28
    new_height = ((height + 27) // 28) * 28

    # 计算patch数量
    h_patches = new_height // 28
    w_patches = new_width // 28
    total_patches = h_patches * w_patches

    # 检查spatial merge compatibility
    if total_patches % 16 != 0:
        # 使用搜索算法找到满足条件的最小padding方案
        best_solution = None
        min_extra_patches = float('inf')

        for extra_h in range(7):
            for extra_w in range(7):
                candidate_h = h_patches + extra_h
                candidate_w = w_patches + extra_w
                candidate_total = candidate_h * candidate_w

                if candidate_total % 16 == 0:
                    extra_patches = candidate_total - total_patches
                    if extra_patches < min_extra_patches:
                        min_extra_patches = extra_patches
                        best_solution = (candidate_h, candidate_w)

        if best_solution is not None:
            new_height = best_solution[0] * 28
            new_width = best_solution[1] * 28

    if new_width == width and new_height == height:
        return image

    # 创建黑色背景的新图像
    padded_image = Image.new('RGB', (new_width, new_height), (0, 0, 0))
    padded_image.paste(image, (0, 0))

    return padded_image

if __name__ == "__main__":
    test_padding_vs_processor()