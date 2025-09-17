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
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4
NUM_EPOCHS = 0.5
MAX_PIXELS = 1024*28*28    # 802,816 pixels (safe value that's multiple of 28 for spatial_merge_unit)
MIN_PIXELS = 56*56         # 3,136 pixels (consistent with data generation)

# Hardware configuration
USE_DEEPSPEED = False  # Temporarily disabled to debug tensor dimension issues
DEEPSPEED_CONFIG = "./scripts/zero3.json"
# ===================================

def main():
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

    print(f"Starting training with dataset: {DATASET_TYPE}")
    print(f"Model: {MODEL_NAME}")
    print(f"Output directory: {OUTPUT_DIR}")

    # Dataset mapping
    dataset_mapping = {
        "grounding": "datasets_grounding",
        "text": "datasets_text",
        "text_grounding": "datasets_text_grounding"
    }

    if DATASET_TYPE not in dataset_mapping:
        raise ValueError(f"Invalid DATASET_TYPE: {DATASET_TYPE}. Must be one of {list(dataset_mapping.keys())}")

    dataset_name = dataset_mapping[DATASET_TYPE]

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
    print("Loading model and tokenizer...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        cache_dir=training_args.cache_dir,
        attn_implementation="sdpa",  # 使用PyTorch原生注意力替代Flash Attention
        torch_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    # Load processor
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    # Set image processor in data_args
    data_args.image_processor = processor.image_processor

    # Set model type for Qwen2.5-VL
    data_args.model_type = "qwen2.5vl"

    # Create data module
    print(f"Loading dataset: {dataset_name}")
    data_module = make_supervised_data_module(
        tokenizer=tokenizer,
        data_args=data_args
    )

    # Create trainer
    print("Creating trainer...")
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module
    )

    # Check for existing checkpoints
    if list(Path(OUTPUT_DIR).glob("checkpoint-*")):
        print("Found existing checkpoints, resuming training...")
        trainer.train(resume_from_checkpoint=True)
    else:
        print("Starting training from scratch...")
        trainer.train()

    # Save final model
    print("Saving final model...")
    trainer.save_state()
    trainer.save_model(output_dir=OUTPUT_DIR)

if __name__ == "__main__":
    main()