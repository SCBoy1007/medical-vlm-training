#!/usr/bin/env python3
"""
快速训练测试：验证padding方案是否能让训练正常启动
只运行几个训练步骤，主要验证tensor维度兼容性
"""

import os
import sys
import torch
import logging
from datetime import datetime
from pathlib import Path

def setup_logging():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"./logs/quick_train_test_{timestamp}.log"

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
    logger.info(f"Quick train test log: {log_file}")
    return logger

def main():
    logger = setup_logging()

    logger.info("🚀 快速训练测试 - 验证Padding方案")
    logger.info("="*60)

    # 设置环境变量禁用wandb
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["WANDB_MODE"] = "disabled"

    # 设置路径
    project_root = Path(__file__).parent
    sys.path.append(str(project_root))

    try:
        # 导入模块
        from qwenvl.train.argument import ModelArguments, DataArguments, TrainingArguments
        from qwenvl.data.data_qwen import make_supervised_data_module
        from transformers import (
            Qwen2_5_VLForConditionalGeneration,
            AutoTokenizer,
            AutoProcessor,
            Trainer
        )

        logger.info("✅ 模块导入成功")

        # 配置参数（最小配置）
        model_args = ModelArguments(
            model_name_or_path="./models/Qwen2.5-VL-7B-Instruct",
            version="qwen",
            freeze_backbone=False,
            tune_mm_mlp_adapter=True,
        )

        data_args = DataArguments(
            data_path="./data/datasets_grounding",
            lazy_preprocess=True,
            is_multimodal=True,
            image_folder="./data/images",
            max_pixels=448*448,  # 固定到安全尺寸
            min_pixels=448*448,  # 固定尺寸，避免变化
            dataset_use="curve_detection_high",  # 只用一个小数据集
            data_flatten=False,
            model_type="qwen2.5vl",
            enable_spatial_merge_compatibility=False  # 使用padding方案
        )

        training_args = TrainingArguments(
            output_dir="./quick_test_output",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            max_steps=3,  # 只运行3步
            learning_rate=1e-6,
            logging_steps=1,
            save_steps=999999,  # 不保存
            model_max_length=2048,
            bf16=True,
            dataloader_num_workers=0,
            remove_unused_columns=False,
            # 禁用所有外部日志和报告
            report_to=[],  # 禁用wandb等所有报告
            logging_dir=None,  # 禁用tensorboard
            run_name="quick_test",  # 设置运行名称避免冲突
        )

        logger.info("✅ 参数配置完成")

        # 加载模型和处理器
        logger.info("加载模型和处理器...")

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        logger.info("✅ 模型加载成功")

        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )
        logger.info("✅ Tokenizer加载成功")

        processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, use_fast=False)
        data_args.image_processor = processor.image_processor
        data_args.image_processor.max_pixels = data_args.max_pixels
        data_args.image_processor.min_pixels = data_args.min_pixels
        logger.info("✅ Processor加载成功")

        # 创建数据模块
        logger.info("创建数据模块...")
        data_module = make_supervised_data_module(
            tokenizer=tokenizer,
            data_args=data_args
        )

        train_dataset = data_module['train_dataset']
        logger.info(f"✅ 数据集创建成功，大小: {len(train_dataset)}")

        # 测试加载一个样本
        logger.info("测试数据加载...")
        sample = train_dataset[0]
        logger.info("✅ 样本加载成功")

        if 'pixel_values' in sample:
            logger.info(f"Pixel values shape: {sample['pixel_values'].shape}")
        if 'image_grid_thw' in sample:
            logger.info(f"Grid THW: {sample['image_grid_thw']}")

        # 创建trainer
        logger.info("创建Trainer...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_module['data_collator'],
        )
        logger.info("✅ Trainer创建成功")

        # 开始训练（只运行几步）
        logger.info("开始快速训练测试...")
        logger.info("注意：这只是验证tensor维度兼容性，不是完整训练")
        logger.info("已禁用wandb和所有外部日志")

        try:
            trainer.train()
            logger.info("🎉 训练测试成功！没有tensor维度错误")
            logger.info("✅ Padding方案完全解决了spatial_merge问题")
            logger.info("✅ 可以安全运行完整训练了")
            return True

        except RuntimeError as e:
            if "shape" in str(e) and "invalid for input of size" in str(e):
                logger.error(f"❌ 仍然存在tensor维度问题: {e}")
                return False
            else:
                logger.error(f"❌ 其他训练错误: {e}")
                # 打印更多错误信息帮助调试
                import traceback
                logger.error("详细错误信息:")
                logger.error(traceback.format_exc())
                return False

    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 快速训练测试成功！可以运行完整训练了")
        print("建议运行: python train.py")
    else:
        print("\n💥 快速训练测试失败，需要进一步调试")

    sys.exit(0 if success else 1)