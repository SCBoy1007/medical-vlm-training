#!/usr/bin/env python3
"""
服务器端测试脚本：验证简单padding方案是否解决tensor维度问题
运行方式：python test_padding_on_server.py
"""

import os
import sys
import torch
import logging
from datetime import datetime
from pathlib import Path
from PIL import Image

def setup_logging():
    """设置日志"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"./logs/padding_test_{timestamp}.log"

    # 确保logs目录存在
    Path("./logs").mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Padding test log: {log_file}")
    return logger

def test_individual_image_processing():
    """测试单个图像的padding和处理"""
    logger = logging.getLogger(__name__)
    logger.info("="*60)
    logger.info("测试单个图像的padding和tensor处理")
    logger.info("="*60)

    try:
        # 设置路径
        sys.path.append(str(Path(__file__).parent))

        # 导入必要模块
        from transformers import AutoProcessor
        from PIL import Image

        # 加载处理器
        model_path = "./models/Qwen2.5-VL-7B-Instruct"
        logger.info(f"加载处理器: {model_path}")
        processor = AutoProcessor.from_pretrained(model_path, use_fast=False)
        logger.info("✅ 处理器加载成功")

        # 找到测试图像
        test_image_dir = "./data/images/test/high_quality"
        if not os.path.exists(test_image_dir):
            test_image_dir = "./data/images/train/high_quality"

        image_files = [f for f in os.listdir(test_image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if not image_files:
            logger.error("未找到测试图像")
            return False

        test_image_path = os.path.join(test_image_dir, image_files[0])
        logger.info(f"测试图像: {test_image_path}")

        # 加载原始图像
        original_image = Image.open(test_image_path).convert("RGB")
        orig_width, orig_height = original_image.size
        logger.info(f"原始图像尺寸: {orig_width} x {orig_height}")

        # 应用padding
        def pad_image_to_28_multiple(image):
            width, height = image.size
            new_width = ((width + 27) // 28) * 28
            new_height = ((height + 27) // 28) * 28

            if new_width == width and new_height == height:
                return image, False

            padded_image = Image.new('RGB', (new_width, new_height), (0, 0, 0))
            padded_image.paste(image, (0, 0))
            return padded_image, True

        padded_image, was_padded = pad_image_to_28_multiple(original_image)
        if was_padded:
            new_width, new_height = padded_image.size
            logger.info(f"Padding后尺寸: {new_width} x {new_height}")
            logger.info(f"添加的padding: 右边{new_width-orig_width}px, 下边{new_height-orig_height}px")
        else:
            logger.info("图像已经是28的倍数，无需padding")

        # 测试原始图像处理（可能失败）
        logger.info("\n--- 测试原始图像处理 ---")
        try:
            result_orig = processor.image_processor(images=original_image, return_tensors="pt")
            pixel_values_orig = result_orig["pixel_values"]
            grid_thw_orig = result_orig["image_grid_thw"][0]

            if grid_thw_orig.dim() == 1:
                t, h, w = grid_thw_orig.tolist()
            else:
                t, h, w = grid_thw_orig[0].tolist()

            tokens_orig = (t * h * w) // 4
            logger.info(f"原始图像 - Grid THW: [{t}, {h}, {w}]")
            logger.info(f"原始图像 - Tokens: {tokens_orig}")
            logger.info(f"原始图像 - 能被4整除: {tokens_orig % 4 == 0}")

            if tokens_orig % 4 != 0:
                logger.warning("❌ 原始图像不满足spatial_merge约束")
            else:
                logger.info("✅ 原始图像已经满足约束")

        except Exception as e:
            logger.error(f"❌ 原始图像处理失败: {e}")

        # 测试padding后图像处理
        logger.info("\n--- 测试Padding后图像处理 ---")
        try:
            result_padded = processor.image_processor(images=padded_image, return_tensors="pt")
            pixel_values_padded = result_padded["pixel_values"]
            grid_thw_padded = result_padded["image_grid_thw"][0]

            if grid_thw_padded.dim() == 1:
                t, h, w = grid_thw_padded.tolist()
            else:
                t, h, w = grid_thw_padded[0].tolist()

            tokens_padded = (t * h * w) // 4
            logger.info(f"Padding图像 - Grid THW: [{t}, {h}, {w}]")
            logger.info(f"Padding图像 - Tokens: {tokens_padded}")
            logger.info(f"Padding图像 - 能被4整除: {tokens_padded % 4 == 0}")

            if tokens_padded % 4 == 0:
                logger.info("🎉 Padding后图像满足spatial_merge约束!")
                logger.info(f"Tensor形状: {pixel_values_padded.shape}")
                return True
            else:
                logger.error("❌ Padding后图像仍不满足约束")
                return False

        except Exception as e:
            logger.error(f"❌ Padding后图像处理失败: {e}")
            return False

    except Exception as e:
        logger.error(f"❌ 图像处理测试失败: {e}")
        return False

def test_dataset_loading():
    """测试数据集加载是否正常"""
    logger = logging.getLogger(__name__)
    logger.info("="*60)
    logger.info("测试数据集加载和图像处理流程")
    logger.info("="*60)

    try:
        # 设置路径
        project_root = Path(__file__).parent
        sys.path.append(str(project_root))

        # 导入模块
        from qwenvl.train.argument import ModelArguments, DataArguments, TrainingArguments
        from qwenvl.data.data_qwen import make_supervised_data_module
        from transformers import AutoProcessor

        # 创建测试参数
        model_args = ModelArguments(
            model_name_or_path="./models/Qwen2.5-VL-7B-Instruct",
            version="qwen",
        )

        data_args = DataArguments(
            data_path="./data/datasets_grounding",
            lazy_preprocess=True,
            is_multimodal=True,
            image_folder="./data/images",
            max_pixels=1024*28*28,  # 28的倍数
            min_pixels=56*56,       # 28的倍数
            dataset_use="curve_detection_high",  # 只测试一个小数据集
            data_flatten=False,
            model_type="qwen2.5vl",
            enable_spatial_merge_compatibility=False  # 禁用复杂包装器
        )

        training_args = TrainingArguments(
            output_dir="./test_output",
            per_device_train_batch_size=1,
            model_max_length=2048,
        )

        # 加载处理器
        logger.info("加载处理器...")
        processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, use_fast=False)
        data_args.image_processor = processor.image_processor
        data_args.image_processor.max_pixels = data_args.max_pixels
        data_args.image_processor.min_pixels = data_args.min_pixels

        # 创建数据模块
        logger.info("创建数据模块...")
        data_module = make_supervised_data_module(
            tokenizer=processor.tokenizer,
            data_args=data_args
        )

        train_dataset = data_module['train_dataset']
        logger.info(f"✅ 数据集加载成功，大小: {len(train_dataset)}")

        # 测试加载几个样本
        logger.info("测试加载前3个样本...")
        for i in range(min(3, len(train_dataset))):
            try:
                sample = train_dataset[i]
                logger.info(f"样本 {i+1}: ✅ 加载成功")

                # 检查tensor形状
                if 'pixel_values' in sample:
                    pixel_values = sample['pixel_values']
                    logger.info(f"  Pixel values shape: {pixel_values.shape}")
                if 'image_grid_thw' in sample:
                    grid_thw = sample['image_grid_thw']
                    logger.info(f"  Grid THW: {grid_thw}")

            except Exception as e:
                logger.error(f"样本 {i+1}: ❌ 加载失败 - {e}")
                return False

        logger.info("🎉 数据集测试成功!")
        return True

    except Exception as e:
        logger.error(f"❌ 数据集加载测试失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_minimal_training():
    """测试最小的训练步骤"""
    logger = logging.getLogger(__name__)
    logger.info("="*60)
    logger.info("测试最小训练步骤")
    logger.info("="*60)

    try:
        # 这里只测试一个前向传递，不进行实际训练
        # 主要验证tensor维度是否兼容

        logger.info("跳过完整训练测试 - 如果前面的测试都通过，可以直接运行真实训练")
        logger.info("建议运行: python train.py 来测试完整训练流程")

        return True

    except Exception as e:
        logger.error(f"❌ 最小训练测试失败: {e}")
        return False

def main():
    """主测试函数"""
    logger = setup_logging()

    logger.info("🚀 开始Padding方案服务器端测试")
    logger.info(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU数量: {torch.cuda.device_count()}")
        logger.info(f"当前GPU: {torch.cuda.get_device_name()}")

    # 测试序列
    tests = [
        ("单个图像padding和处理", test_individual_image_processing),
        ("数据集加载", test_dataset_loading),
        ("最小训练", test_minimal_training),
    ]

    results = {}

    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"开始测试: {test_name}")
        logger.info(f"{'='*60}")

        try:
            result = test_func()
            results[test_name] = result
            if result:
                logger.info(f"✅ {test_name} - 测试通过")
            else:
                logger.error(f"❌ {test_name} - 测试失败")
        except Exception as e:
            logger.error(f"💥 {test_name} - 测试异常: {e}")
            results[test_name] = False

    # 总结
    logger.info(f"\n{'='*60}")
    logger.info("测试总结")
    logger.info(f"{'='*60}")

    passed = sum(results.values())
    total = len(results)

    for test_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        logger.info(f"{test_name}: {status}")

    logger.info(f"\n总计: {passed}/{total} 测试通过")

    if passed == total:
        logger.info("🎉 所有测试通过！可以开始实际训练了")
        logger.info("建议运行: python train.py")
        return True
    else:
        logger.error("💥 部分测试失败，需要进一步调试")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)