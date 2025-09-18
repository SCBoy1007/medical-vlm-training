#!/usr/bin/env python3
"""
Simplified training script for medical VLM fine-tuning
Modify the DATASET_TYPE variable to switch between experiments:
- "grounding": Pure bbox grounding training
- "text": Pure text training
- "text_grounding": Combined text + grounding training
"""

import os
import sys
import torch
import logging
from datetime import datetime
from pathlib import Path

# ====== MULTI-GPU CONFIGURATION ======
# Fix for tensor dimension issues: Use single GPU to avoid DataParallel problems
# The previous error was caused by DataParallel incompatibility with Qwen2.5-VL attention
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use single GPU for stable training
# For multi-GPU in future: need proper DDP setup with torchrun
# =====================================

# ====== CONFIGURATION SECTION ======
# Change this to switch between datasets:
DATASET_TYPE = "grounding"  # Options: "grounding", "text", "text_grounding"

# Model configuration
MODEL_NAME = "./models/Qwen2.5-VL-7B-Instruct"
OUTPUT_DIR = f"./output_{DATASET_TYPE}"
RUN_NAME = f"qwen2vl-medical-{DATASET_TYPE}"

# Training hyperparameters
LEARNING_RATE = 2e-7
BATCH_SIZE = 1  # Single GPU batch size
GRAD_ACCUM_STEPS = 4  # Restore original gradient accumulation
NUM_EPOCHS = 0.5
MAX_PIXELS = 256*28*28     # 200,704 pixels (reduced from 451,584 for memory efficiency)
MIN_PIXELS = 16*28*28      # 12,544 pixels (keep same)

# Hardware configuration
USE_DEEPSPEED = False  # Temporarily disabled to debug tensor dimension issues
DEEPSPEED_CONFIG = "./scripts/zero3.json"

# Simplified: No longer needed with 700x1400 resized images
# ===================================

def print_gpu_memory_usage(logger, stage=""):
    """打印详细的GPU显存使用情况"""
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device).total_memory / (1024**3)  # GB
        allocated = torch.cuda.memory_allocated(device) / (1024**3)  # GB
        reserved = torch.cuda.memory_reserved(device) / (1024**3)  # GB
        free = total_memory - allocated

        logger.info(f"=== GPU Memory Usage - {stage} ===")
        logger.info(f"Total GPU Memory: {total_memory:.2f} GB")
        logger.info(f"Allocated Memory: {allocated:.2f} GB ({allocated/total_memory*100:.1f}%)")
        logger.info(f"Reserved Memory: {reserved:.2f} GB ({reserved/total_memory*100:.1f}%)")
        logger.info(f"Free Memory: {free:.2f} GB ({free/total_memory*100:.1f}%)")
        efficiency = (allocated/reserved*100) if reserved > 0 else 0.0
        logger.info(f"Memory Efficiency: {efficiency:.1f}% (allocated/reserved)")
        logger.info("=" * 50)

def setup_logging():
    """Setup comprehensive logging to file and console"""
    # Create logs directory
    log_dir = Path("./logs")
    log_dir.mkdir(exist_ok=True)

    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_{DATASET_TYPE}_{timestamp}.log"

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger

def main():
    # Setup logging first
    logger = setup_logging()

    # 设置环境变量禁用wandb
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["WANDB_MODE"] = "disabled"

    # Set up paths
    project_root = Path(__file__).parent
    sys.path.append(str(project_root))

    # Import after adding to path
    from qwenvl.train.argument import ModelArguments, DataArguments, TrainingArguments
    from qwenvl.data.data_qwen import make_supervised_data_module
    from qwenvl.train.trainer import replace_qwen2_vl_attention_class
    from transformers import (
        Qwen2VLForConditionalGeneration,
        Qwen2_5_VLForConditionalGeneration,
        AutoTokenizer,
        AutoProcessor,
        Trainer
    )

    logger.info("="*60)
    logger.info("MEDICAL VLM TRAINING STARTED")
    logger.info("="*60)
    logger.info(f"Training dataset type: {DATASET_TYPE}")
    logger.info(f"Model path: {MODEL_NAME}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"Learning rate: {LEARNING_RATE}")
    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info(f"Gradient accumulation steps: {GRAD_ACCUM_STEPS}")
    logger.info(f"Number of epochs: {NUM_EPOCHS}")
    logger.info(f"Max pixels: {MAX_PIXELS}")
    logger.info(f"Min pixels: {MIN_PIXELS}")
    logger.info(f"DeepSpeed enabled: {USE_DEEPSPEED}")
    logger.info("Using simplified image processing (no custom padding)")
    logger.info("TRAINING MODE: LoRA Fine-tuning")
    logger.info(f"LoRA rank: 32")
    logger.info(f"LoRA alpha: 16")
    logger.info(f"Expected memory usage: ~12GB (LoRA r=32 + reduced resolution)")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        logger.info(f"Current GPU: {torch.cuda.current_device()}")
        logger.info(f"GPU name: {torch.cuda.get_device_name()}")
    print_gpu_memory_usage(logger, "Initial State")
    logger.info("="*60)

    # Dataset mapping
    dataset_mapping = {
        "grounding": "datasets_grounding",
        "text": "datasets_text",
        "text_grounding": "datasets_text_grounding"
    }

    if DATASET_TYPE not in dataset_mapping:
        error_msg = f"Invalid DATASET_TYPE: {DATASET_TYPE}. Must be one of {list(dataset_mapping.keys())}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    dataset_name = dataset_mapping[DATASET_TYPE]
    logger.info(f"Selected dataset: {dataset_name}")

    # Create arguments
    model_args = ModelArguments(
        model_name_or_path=MODEL_NAME,
        version="qwen",
        freeze_backbone=False,  # LoRA模式不需要冻结backbone
        tune_mm_mlp_adapter=True,
        tune_mm_llm=True,       # 启用LLM训练 (LoRA关键参数)
        tune_mm_vision=False,   # 医学图像分析通常冻结vision encoder
        tune_mm_mlp=True,       # 启用multimodal projector
        vision_tower=None,
        mm_vision_select_layer=-2,
        pretrain_mm_mlp_adapter=None,
        mm_projector_type='mlp2x_gelu',
        mm_use_im_start_end=False,
        mm_use_im_patch_token=False,
        mm_patch_merge_type='flat',
        mm_vision_select_feature="patch"
    )

    data_args = DataArguments(
        data_path=f"./data/{dataset_name}",
        lazy_preprocess=True,
        is_multimodal=True,
        sep_image_conv_front=False,
        image_token_len=256,
        image_folder="./data/images",
        image_aspect_ratio='anyres_max_9',
        max_pixels=MAX_PIXELS,
        min_pixels=MIN_PIXELS,
        dataset_use="curve_detection_high,curve_detection_low,apex_vertebrae_high,apex_vertebrae_low,end_vertebrae_high,end_vertebrae_low",
        data_flatten=False
    )

    # Simplified: No need for spatial merge compatibility with 700x1400 resized images

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        cache_dir=None,
        optim="adamw_torch",
        remove_unused_columns=False,
        freeze_mm_mlp_adapter=False,
        mpt_attn_impl="triton",
        model_max_length=8192,
        double_quant=True,
        quant_type="nf4",
        bits=16,
        lora_enable=True,   # 启用LoRA微调
        lora_r=32,          # LoRA rank=32 (平衡效果和显存)
        lora_alpha=16,      # LoRA alpha参数
        lora_dropout=0.05,
        lora_weight_path="",
        lora_bias="none",
        mm_projector_lr=None,
        group_by_modality_length=True,

        # Training parameters
        bf16=True,
        fp16=False,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        evaluation_strategy="no",
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=1,
        learning_rate=LEARNING_RATE,
        weight_decay=0.,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",

        # 禁用外部日志避免wandb配置问题
        report_to=[],  # 禁用wandb等所有报告
        logging_dir=None,  # 禁用tensorboard
        run_name=RUN_NAME,  # 设置运行名称
        logging_steps=1,
        tf32=False,  # Disabled for V100 compatibility (TF32 requires Ampere+)
        dataloader_num_workers=4,
        gradient_checkpointing=True,
        max_grad_norm=1.0,

        # DeepSpeed
        deepspeed=DEEPSPEED_CONFIG if USE_DEEPSPEED else None,

        # Multi-GPU configuration - force DDP over DataParallel
        ddp_find_unused_parameters=False,  # Improves performance for VLM
        dataloader_drop_last=True,        # Ensures consistent batch sizes across GPUs

        # Logging (already configured above to disable wandb)
    )

    # Replace attention class for better memory efficiency (disabled due to GLIBC issues)
    replace_qwen2_vl_attention_class()

    # Load model and tokenizer
    logger.info("Loading model and tokenizer...")
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            cache_dir=training_args.cache_dir,
            attn_implementation="sdpa",  # 使用PyTorch原生注意力替代Flash Attention
            torch_dtype=torch.bfloat16,
        )
        logger.info("✓ Model loaded successfully")
        print_gpu_memory_usage(logger, "After Model Loading")

        # Configure model components (CRITICAL: must be called before LoRA application)
        logger.info("🔧 Configuring model component training states...")
        logger.info(f"   tune_mm_llm: {model_args.tune_mm_llm}")
        logger.info(f"   tune_mm_vision: {model_args.tune_mm_vision}")
        logger.info(f"   tune_mm_mlp: {model_args.tune_mm_mlp}")

        from qwenvl.train.train_qwen import set_model
        set_model(model_args, model)
        logger.info("✓ Model component states configured successfully")

        # Apply LoRA if enabled
        if training_args.lora_enable:
            logger.info("🔄 Applying LoRA configuration...")
            try:
                from peft import LoraConfig, get_peft_model, TaskType

                # Create LoRA configuration
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=training_args.lora_r,
                    lora_alpha=training_args.lora_alpha,
                    lora_dropout=training_args.lora_dropout,
                    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # Common attention and MLP layers
                    bias=training_args.lora_bias,
                )

                # Apply LoRA to model
                model = get_peft_model(model, lora_config)

                # Set training mode (PEFT handles parameter freezing correctly)
                model.train()

                logger.info("✅ LoRA applied successfully!")
                logger.info(f"   LoRA rank: {training_args.lora_r}")
                logger.info(f"   LoRA alpha: {training_args.lora_alpha}")
                logger.info(f"   Target modules: {lora_config.target_modules}")

                # Use PEFT's built-in method for accurate parameter statistics
                logger.info("   PEFT Parameter Statistics:")
                model.print_trainable_parameters()

                # Additional debug: Check if LoRA parameters actually require gradients
                logger.info("🔍 LoRA Parameter Gradient Debug:")
                lora_params_with_grad = 0
                lora_params_total = 0
                for name, param in model.named_parameters():
                    if 'lora' in name.lower():
                        lora_params_total += 1
                        if param.requires_grad:
                            lora_params_with_grad += 1
                        logger.info(f"   {name}: requires_grad={param.requires_grad}")

                logger.info(f"   LoRA parameters with gradient: {lora_params_with_grad}/{lora_params_total}")
                if lora_params_with_grad == 0:
                    logger.warning("⚠️  WARNING: No LoRA parameters have gradients enabled!")
                else:
                    logger.info(f"✅ {lora_params_with_grad} LoRA parameters ready for training")

                # CRITICAL: In LoRA mode, freeze lm_head to prevent gradient conflicts
                logger.info("🔒 Freezing lm_head for LoRA compatibility...")
                model.lm_head.requires_grad = False
                logger.info(f"   lm_head.requires_grad: {model.lm_head.requires_grad}")

                # Double-check: ensure only LoRA parameters have gradients
                non_lora_trainable = []
                for name, param in model.named_parameters():
                    if param.requires_grad and 'lora' not in name.lower():
                        non_lora_trainable.append(name)

                if non_lora_trainable:
                    logger.warning(f"⚠️  WARNING: Found {len(non_lora_trainable)} non-LoRA trainable parameters:")
                    for name in non_lora_trainable[:5]:  # Show first 5
                        logger.warning(f"     {name}")
                    if len(non_lora_trainable) > 5:
                        logger.warning(f"     ... and {len(non_lora_trainable) - 5} more")
                else:
                    logger.info("✅ All trainable parameters are LoRA parameters")

            except Exception as e:
                logger.error(f"❌ Failed to apply LoRA: {e}")
                logger.info("💡 Falling back to full parameter training")

            print_gpu_memory_usage(logger, "After LoRA Application")

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )
        logger.info("✓ Tokenizer loaded successfully")

        # Load processor
        processor = AutoProcessor.from_pretrained(MODEL_NAME)
        logger.info("✓ Processor loaded successfully")

    except Exception as e:
        logger.error(f"Failed to load model/tokenizer: {e}")
        raise

    # Set image processor in data_args
    data_args.image_processor = processor.image_processor

    # Set model type for Qwen2.5-VL
    data_args.model_type = "qwen2.5vl"

    # Create data module
    logger.info(f"Loading dataset: {dataset_name}")

    try:
        data_module = make_supervised_data_module(
            tokenizer=tokenizer,
            data_args=data_args
        )
        train_dataset = data_module.get('train_dataset')
        if train_dataset:
            logger.info(f"✓ Dataset loaded successfully. Size: {len(train_dataset)}")
            print_gpu_memory_usage(logger, "After Dataset Loading")
        else:
            logger.warning("No training dataset found in data module")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

    # Create trainer
    logger.info("Creating trainer...")
    try:
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            **data_module
        )
        logger.info("✓ Trainer created successfully")

        # Verify final LoRA configuration with PEFT's method
        logger.info("Final LoRA Parameter Verification:")
        if hasattr(model, 'print_trainable_parameters'):
            model.print_trainable_parameters()
        else:
            # Fallback for non-PEFT models
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"  Trainable parameters: {trainable_params:,}")
            logger.info(f"  Total parameters: {total_params:,}")
            logger.info(f"  Trainable ratio: {trainable_params/total_params*100:.2f}%")

        print_gpu_memory_usage(logger, "After Trainer Creation")
    except Exception as e:
        logger.error(f"Failed to create trainer: {e}")
        raise

    # Check for existing checkpoints
    checkpoint_dir = Path(OUTPUT_DIR)
    existing_checkpoints = list(checkpoint_dir.glob("checkpoint-*"))

    try:
        if existing_checkpoints:
            logger.info(f"Found {len(existing_checkpoints)} existing checkpoints, resuming training...")
            trainer.train(resume_from_checkpoint=True)
        else:
            logger.info("Starting training from scratch...")
            print_gpu_memory_usage(logger, "Before Training Start")
            trainer.train()

        logger.info("✓ Training completed successfully")

        # Save final model
        logger.info("Saving final model...")
        trainer.save_state()
        trainer.save_model(output_dir=OUTPUT_DIR)
        logger.info(f"✓ Model saved to {OUTPUT_DIR}")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.error("Full traceback:", exc_info=True)
        raise

    finally:
        logger.info("="*60)
        logger.info("TRAINING SESSION ENDED")
        logger.info("="*60)

if __name__ == "__main__":
    main()