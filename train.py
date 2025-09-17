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

# ====== CONFIGURATION SECTION ======
# Change this to switch between datasets:
DATASET_TYPE = "grounding"  # Options: "grounding", "text", "text_grounding"

# Model configuration
MODEL_NAME = "./models/Qwen2.5-VL-7B-Instruct"
OUTPUT_DIR = f"./output_{DATASET_TYPE}"
RUN_NAME = f"qwen2vl-medical-{DATASET_TYPE}"

# Training hyperparameters
LEARNING_RATE = 2e-7
BATCH_SIZE = 1  # Simplified for debugging CUDA index issues
GRAD_ACCUM_STEPS = 4
NUM_EPOCHS = 0.5
MAX_PIXELS = 1024*28*28    # 802,816 pixels (safe value that's multiple of 28 for spatial_merge_unit)
MIN_PIXELS = 56*56         # 3,136 pixels (consistent with data generation)

# Hardware configuration
USE_DEEPSPEED = False  # Temporarily disabled to debug tensor dimension issues
DEEPSPEED_CONFIG = "./scripts/zero3.json"

# Compatibility configuration
ENABLE_SPATIAL_MERGE_COMPATIBILITY = True  # Enable fix for spatial merge tensor dimension issues
# ===================================

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
    logger.info(f"Spatial merge compatibility: {ENABLE_SPATIAL_MERGE_COMPATIBILITY}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        logger.info(f"Current GPU: {torch.cuda.current_device()}")
        logger.info(f"GPU name: {torch.cuda.get_device_name()}")
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
        freeze_backbone=False,
        tune_mm_mlp_adapter=True,
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

    # Add compatibility configuration
    data_args.enable_spatial_merge_compatibility = ENABLE_SPATIAL_MERGE_COMPATIBILITY

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
        lora_enable=False,
        lora_r=64,
        lora_alpha=16,
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
        logging_steps=1,
        tf32=False,  # Disabled for V100 compatibility (TF32 requires Ampere+)
        dataloader_num_workers=4,
        gradient_checkpointing=True,
        max_grad_norm=1.0,

        # DeepSpeed
        deepspeed=DEEPSPEED_CONFIG if USE_DEEPSPEED else None,

        # Logging
        run_name=RUN_NAME,
        report_to=["wandb"] if "WANDB_API_KEY" in os.environ else []
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

            # Report compatibility statistics if enabled
            if ENABLE_SPATIAL_MERGE_COMPATIBILITY and hasattr(train_dataset.data_args.image_processor, 'get_stats'):
                try:
                    # Process a few samples to get initial statistics
                    for i in range(min(10, len(train_dataset))):
                        _ = train_dataset[i]  # This will trigger image processing

                    stats = train_dataset.data_args.image_processor.get_stats()
                    if stats['total_processed'] > 0:
                        logger.info(f"Compatibility mode stats: {stats['adjusted_images']}/{stats['total_processed']} images adjusted ({stats['adjustment_rate_percent']:.1f}%)")
                        if stats['adjusted_images'] > 0:
                            logger.info(f"✓ Spatial merge compatibility fix is working - adjusted {stats['adjusted_images']} images")
                except Exception as e:
                    logger.warning(f"Could not get compatibility statistics: {e}")
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