#!/usr/bin/env python3
"""
LoRA Configuration Debug Script
调试LoRA配置为什么不生效的专用脚本
"""

import os
import sys
import torch
from pathlib import Path

# 设置单GPU环境
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def print_section(title):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def check_lora_support():
    """检查LoRA相关库和支持情况"""
    print_section("LoRA SUPPORT CHECK")

    # 检查PEFT库
    try:
        import peft
        print(f"✅ PEFT library available: {peft.__version__}")

        # 检查支持的LoRA配置
        from peft import LoraConfig, get_peft_model
        print("✅ LoRA classes imported successfully")

        # 检查可用的任务类型
        from peft import TaskType
        print(f"✅ Available task types: {list(TaskType)}")

    except ImportError as e:
        print(f"❌ PEFT library not available: {e}")
        print("💡 Install with: pip install peft")
        return False

    return True

def check_transformers_lora():
    """检查transformers内置的LoRA支持"""
    print_section("TRANSFORMERS LORA CHECK")

    try:
        from transformers import TrainingArguments

        # 创建训练参数看是否支持LoRA
        dummy_args = TrainingArguments(
            output_dir="./temp",
            lora_enable=True,
            lora_r=32,
            lora_alpha=16,
        )

        print("✅ TrainingArguments accepts LoRA parameters")
        print(f"   lora_enable: {dummy_args.lora_enable}")
        print(f"   lora_r: {dummy_args.lora_r}")
        print(f"   lora_alpha: {dummy_args.lora_alpha}")

        return True

    except Exception as e:
        print(f"❌ TrainingArguments LoRA support issue: {e}")
        return False

def test_model_loading():
    """测试模型加载和LoRA配置"""
    print_section("MODEL LOADING TEST")

    MODEL_NAME = "./models/Qwen2.5-VL-7B-Instruct"

    if not os.path.exists(MODEL_NAME):
        print(f"❌ Model not found at: {MODEL_NAME}")
        return None

    try:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer

        print("🔄 Loading model (this may take a moment)...")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="cpu",  # 加载到CPU避免显存问题
        )

        print("✅ Model loaded successfully")

        # 检查模型结构
        total_params = sum(p.numel() for p in model.parameters())
        print(f"📊 Total parameters: {total_params:,}")

        # 检查模型是否支持LoRA
        print("\n🔍 Checking model structure for LoRA compatibility...")

        # 查找线性层（LoRA通常应用于线性层）
        linear_layers = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                linear_layers.append(name)

        print(f"📋 Found {len(linear_layers)} linear layers")
        if len(linear_layers) <= 10:
            for layer in linear_layers:
                print(f"   - {layer}")
        else:
            print(f"   - {linear_layers[0]} ... {linear_layers[-1]} (showing first and last)")

        return model

    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return None

def test_peft_lora(model):
    """测试PEFT库的LoRA配置"""
    print_section("PEFT LORA TEST")

    if model is None:
        print("❌ No model available for PEFT testing")
        return

    try:
        from peft import LoraConfig, get_peft_model, TaskType

        # 创建LoRA配置
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,  # 或者其他适合的任务类型
            r=32,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # 常见的注意力层
        )

        print("✅ LoRA config created")
        print(f"   rank: {lora_config.r}")
        print(f"   alpha: {lora_config.lora_alpha}")
        print(f"   target_modules: {lora_config.target_modules}")

        # 尝试应用LoRA
        try:
            peft_model = get_peft_model(model, lora_config)

            # 检查参数统计
            trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in peft_model.parameters())

            print("✅ PEFT LoRA model created successfully!")
            print(f"📊 Parameter Statistics:")
            print(f"   Trainable: {trainable_params:,}")
            print(f"   Total: {total_params:,}")
            print(f"   Ratio: {trainable_params/total_params*100:.2f}%")

            if trainable_params < total_params * 0.1:  # 少于10%说明LoRA生效
                print("✅ LoRA working correctly - only small fraction trainable!")
            else:
                print("⚠️ LoRA might not be working - too many trainable parameters")

        except Exception as e:
            print(f"❌ PEFT model creation failed: {e}")

    except ImportError:
        print("❌ PEFT library not available")

def test_transformers_integration():
    """测试transformers和LoRA的集成"""
    print_section("TRANSFORMERS INTEGRATION TEST")

    # 检查我们的训练参数配置
    sys.path.append(str(Path(__file__).parent))

    try:
        from qwenvl.train.argument import TrainingArguments

        print("🔄 Testing our TrainingArguments class...")

        args = TrainingArguments(
            output_dir="./temp",
            lora_enable=True,
            lora_r=32,
            lora_alpha=16,
            lora_dropout=0.05,
        )

        print("✅ Our TrainingArguments works with LoRA")
        print(f"   lora_enable: {args.lora_enable}")
        print(f"   lora_r: {args.lora_r}")

        # 检查是否有自定义的LoRA应用逻辑
        from qwenvl.train import trainer
        print("✅ Trainer module imported")

        # 检查是否有LoRA相关的函数
        trainer_functions = [name for name in dir(trainer) if 'lora' in name.lower()]
        if trainer_functions:
            print(f"🔍 Found LoRA-related functions: {trainer_functions}")
        else:
            print("⚠️ No LoRA-related functions found in trainer module")

    except Exception as e:
        print(f"❌ Integration test failed: {e}")

def main():
    print_section("LORA DEBUG SCRIPT STARTED")
    print("This script will diagnose why LoRA is not working properly")

    # 基础环境检查
    print(f"🐍 Python version: {sys.version}")
    print(f"🔥 PyTorch version: {torch.__version__}")
    print(f"🤗 Transformers version: {__import__('transformers').__version__}")

    # 1. 检查LoRA支持
    peft_available = check_lora_support()

    # 2. 检查transformers LoRA支持
    transformers_lora_ok = check_transformers_lora()

    # 3. 测试模型加载
    model = test_model_loading()

    # 4. 测试PEFT LoRA（如果可用）
    if peft_available:
        test_peft_lora(model)

    # 5. 测试集成
    test_transformers_integration()

    print_section("DEBUG SUMMARY")
    print("Key Findings:")
    print(f"  PEFT Available: {'✅' if peft_available else '❌'}")
    print(f"  Transformers LoRA: {'✅' if transformers_lora_ok else '❌'}")
    print(f"  Model Loaded: {'✅' if model is not None else '❌'}")

    if not peft_available:
        print("\n💡 Recommendation: Install PEFT library")
        print("   pip install peft")

    print("\n🔚 Debug script completed!")

if __name__ == "__main__":
    main()