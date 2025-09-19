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
import subprocess
from datetime import datetime
from pathlib import Path
from transformers import TrainerCallback

# ====== AUTO MULTI-GPU CONFIGURATION ======
# Automatically detects and uses all available GPUs
# No need for torchrun - just run: python train.py
# ============================================

def auto_launch_distributed():
    """Auto-launch distributed training if multiple GPUs detected"""
    if 'RANK' in os.environ:
        # Already in distributed mode
        return False

    gpu_count = torch.cuda.device_count()
    if gpu_count <= 1:
        print(f"Single GPU mode: {gpu_count} GPU detected")
        return False

    print(f"Auto-launching {gpu_count}-GPU distributed training...")
    cmd = [
        sys.executable, '-m', 'torch.distributed.run',
        '--nproc_per_node', str(gpu_count),
        '--master_port', '29500',
        __file__
    ] + sys.argv[1:]

    subprocess.run(cmd)
    return True

# ====== CONFIGURATION SECTION ======
# Change this to switch between datasets:
DATASET_TYPE = "grounding"  # Options: "grounding", "text", "text_grounding"

# Model configuration
MODEL_NAME = "./models/Qwen2.5-VL-7B-Instruct"

# LoRA configuration
LORA_R = 32          # LoRA rank: 16 (faster), 32 (balanced), 64 (better quality)
LORA_ALPHA = 16      # LoRA alpha: typically r/2 or r
LORA_METHOD = "lora" # Training method identifier for output directory

# Multi-GPU Training hyperparameters (optimized for 4x V100)
LEARNING_RATE = 1e-5  # Increased for better LoRA convergence
BATCH_SIZE = 4  # Reduced to avoid CUDA memory allocation failures
GRAD_ACCUM_STEPS = 1  # Simplified for faster convergence
NUM_EPOCHS = 3.0

# Examples for different configurations:
# Current (r=32, lr=2e-7, ep=0.5, bs=4): ./output_grounding_lora_r32_alpha16_lr2e-7_ep0p5_bs4
# - Small/Fast:     LORA_R=16,  LR=1e-6,  EP=0.5, BS=8  -> output_grounding_lora_r16_alpha8_lr1e-6_ep0p5_bs8
# - High Quality:   LORA_R=64,  LR=1e-7,  EP=1.0, BS=2  -> output_grounding_lora_r64_alpha32_lr1e-7_ep1p0_bs2
# - Fast Iteration: LORA_R=32,  LR=5e-7,  EP=0.3, BS=4  -> output_grounding_lora_r32_alpha16_lr5e-7_ep0p3_bs4
# - Linear Probing: METHOD="linear", LR=1e-5, EP=1.0     -> output_grounding_linear_r0_alpha0_lr1e-5_ep1p0_bs4

# Dynamic output directory based on training configuration
# Format: ./output_{dataset}_{method}_r{rank}_alpha{alpha}_lr{lr}_ep{epochs}_bs{batch_size}
lr_str = f"{LEARNING_RATE:.0e}".replace('e-0', 'e-').replace('e+0', 'e+')  # Clean format: 2e-7
ep_str = f"{NUM_EPOCHS}".replace('.', 'p')  # Replace . with p: 0.5 -> 0p5
effective_batch_size = 4 * BATCH_SIZE * GRAD_ACCUM_STEPS  # 4 GPUs * 4 batch * 1 accum = 16

OUTPUT_DIR = f"./output_{DATASET_TYPE}_{LORA_METHOD}_r{LORA_R}_alpha{LORA_ALPHA}_lr{lr_str}_ep{ep_str}_bs{effective_batch_size}"
RUN_NAME = f"qwen2vl-medical-{DATASET_TYPE}-{LORA_METHOD}-r{LORA_R}-lr{lr_str}-bs{effective_batch_size}"
MAX_PIXELS = 256*28*28     # 200,704 pixels (reduced from 451,584 for memory efficiency)
MIN_PIXELS = 16*28*28      # 12,544 pixels (keep same)

# Hardware configuration
USE_DEEPSPEED = True  # Enable DeepSpeed ZeRO for multi-GPU
DEEPSPEED_CONFIG = "./scripts/zero2.json"  # ZeRO-2 for 4x V100

# Simplified: No longer needed with 700x1400 resized images
# ===================================

class TrainingMonitorCallback(TrainerCallback):
    """自定义训练监控回调，支持多GPU训练监控"""

    def __init__(self, logger):
        self.logger = logger
        self.start_time = None
        self.last_log_time = None
        # Get distributed training info
        self.rank = int(os.environ.get('RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.is_distributed = self.world_size > 1

    def on_train_begin(self, args, state, control, **kwargs):
        import time
        self.start_time = time.time()
        self.last_log_time = self.start_time
        if self.rank == 0:  # Only log from master process
            gpu_info = f"{self.world_size}x GPU" if self.is_distributed else "Single GPU"
            self.logger.info(f"🚀 Training started on {gpu_info}...")

    def _log_progress(self, state, logs=None):
        """Helper method to log training progress"""
        import time
        current_time = time.time()

        # Calculate progress
        current_step = state.global_step
        max_steps = state.max_steps
        progress = (current_step / max_steps) * 100 if max_steps > 0 else 0

        # Calculate time estimates
        elapsed_time = current_time - self.start_time
        time_per_step = elapsed_time / current_step if current_step > 0 else 0
        remaining_steps = max_steps - current_step
        eta = remaining_steps * time_per_step

        # Format time
        def format_time(seconds):
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            seconds = int(seconds % 60)
            if hours > 0:
                return f"{hours}h{minutes}m{seconds}s"
            elif minutes > 0:
                return f"{minutes}m{seconds}s"
            else:
                return f"{seconds}s"

        # Extract key metrics
        loss = 'N/A'
        lr = 'N/A'
        if logs:
            loss = logs.get('train_loss', logs.get('loss', 'N/A'))
            lr = logs.get('learning_rate', 'N/A')

        # Log training progress
        progress_bar = "█" * int(progress // 5) + "░" * (20 - int(progress // 5))

        self.logger.info(
            f"Step {current_step:4d}/{max_steps} [{progress_bar}] {progress:5.1f}% | "
            f"Loss: {loss} | LR: {lr} | "
            f"Elapsed: {format_time(elapsed_time)} | ETA: {format_time(eta)}"
        )

        # GPU memory update (less frequent to avoid spam)
        if current_step % 25 == 0 and self.rank == 0:
            if torch.cuda.is_available():
                if self.is_distributed:
                    # Show average memory across all GPUs
                    total_memory = 0
                    for i in range(min(self.world_size, torch.cuda.device_count())):
                        allocated = torch.cuda.memory_allocated(i) / (1024**3)
                        total_memory += allocated
                    avg_memory = total_memory / self.world_size
                    self.logger.info(f"Avg GPU Memory: {avg_memory:.1f}GB across {self.world_size} GPUs")
                else:
                    allocated = torch.cuda.memory_allocated() / (1024**3)
                    self.logger.info(f"GPU Memory: {allocated:.1f}GB")

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when logging occurs - this has the complete data we need"""
        if logs is not None and self.rank == 0:  # Only log from master process
            self._log_progress(state, logs)

    def on_train_end(self, args, state, control, **kwargs):
        if self.rank == 0:  # Only log from master process
            import time
            total_time = time.time() - self.start_time
            gpu_info = f"on {self.world_size} GPUs" if self.is_distributed else ""
            self.logger.info(f"✅ Training completed {gpu_info} in {total_time/60:.1f} minutes")

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
    """Setup comprehensive logging to file and console (master process only writes to file)"""
    # Get distributed training info
    rank = int(os.environ.get('RANK', 0))

    # Create logs directory (only from master process)
    if rank == 0:
        log_dir = Path("./logs")
        log_dir.mkdir(exist_ok=True)

        # Create log filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"training_{DATASET_TYPE}_{timestamp}.log"

        # Master process: log to both file and console
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
    else:
        # Worker processes: only log to console with rank info
        logging.basicConfig(
            level=logging.WARNING,  # Reduce verbosity for worker processes
            format=f'[GPU-{rank}] %(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)]
        )

        logger = logging.getLogger(__name__)

    return logger

def main():
    # Auto-launch distributed training if needed
    if auto_launch_distributed():
        return  # Exit if we launched distributed training

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
    logger.info(f"Dataset: {DATASET_TYPE} | Model: Qwen2.5-VL-7B | Mode: {LORA_METHOD.upper()} Fine-tuning")
    # Calculate GPU count for accurate batch size display
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    current_effective_batch = world_size * BATCH_SIZE * GRAD_ACCUM_STEPS
    logger.info(f"Learning Rate: {LEARNING_RATE} | Per-GPU Batch: {BATCH_SIZE} | Effective Batch: {current_effective_batch} | Epochs: {NUM_EPOCHS}")
    gpu_info = f"{world_size}x {torch.cuda.get_device_name()}" if torch.cuda.is_available() and world_size > 1 else torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'
    logger.info(f"LoRA Config: r={LORA_R}, alpha={LORA_ALPHA} | GPU: {gpu_info}")
    logger.info(f"Output Directory: {OUTPUT_DIR}")
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
        lora_r=LORA_R,      # LoRA rank (configurable)
        lora_alpha=LORA_ALPHA,  # LoRA alpha参数 (configurable)
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
        save_steps=500,  # More frequent saves for multi-GPU training
        save_total_limit=1,
        learning_rate=LEARNING_RATE,
        weight_decay=0.,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",

        # 禁用外部日志避免wandb配置问题
        report_to=[],  # 禁用wandb等所有报告
        logging_dir=None,  # 禁用tensorboard
        run_name=RUN_NAME,  # 设置运行名称
        logging_steps=1,  # Log every step for detailed monitoring
        tf32=False,  # Disabled for V100 compatibility (TF32 requires Ampere+)
        dataloader_num_workers=4,
        gradient_checkpointing=True,
        max_grad_norm=1.0,

        # DeepSpeed
        deepspeed=DEEPSPEED_CONFIG if USE_DEEPSPEED else None,

        # Multi-GPU DDP configuration
        ddp_backend="nccl",                # NCCL backend for multi-GPU
        ddp_find_unused_parameters=False,  # Improves performance for VLM
        dataloader_drop_last=True,        # Ensures consistent batch sizes across GPUs
        ddp_timeout=7200,                 # 2 hours timeout for large model loading

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

        # Configure model components (ONLY for full parameter training, NOT LoRA)
        if not training_args.lora_enable:
            logger.info("Configuring model for full parameter training...")
            from qwenvl.train.train_qwen import set_model
            set_model(model_args, model)
            logger.info("✓ Model configured successfully")
        else:
            logger.info("LoRA Mode: Using PEFT for parameter-efficient training")

        # Apply LoRA if enabled
        if training_args.lora_enable:
            logger.info("🔄 Applying LoRA configuration...")
            try:
                from peft import LoraConfig, get_peft_model, TaskType

                # Enable embedding gradients for LoRA compatibility
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)
                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

                # Create LoRA configuration
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=LORA_R,
                    lora_alpha=LORA_ALPHA,
                    lora_dropout=training_args.lora_dropout,
                    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # Common attention and MLP layers
                    bias=training_args.lora_bias,
                )

                # Apply LoRA to model
                model = get_peft_model(model, lora_config)
                model.train()

                logger.info("✓ LoRA applied successfully")
                # Use PEFT's built-in method for parameter statistics
                model.print_trainable_parameters()

                # Verify LoRA parameters have gradients
                lora_params_with_grad = sum(1 for name, param in model.named_parameters()
                                          if 'lora' in name.lower() and param.requires_grad)
                lora_params_total = sum(1 for name, param in model.named_parameters()
                                      if 'lora' in name.lower())

                if lora_params_with_grad == 0:
                    logger.warning("⚠️  WARNING: No LoRA parameters have gradients enabled!")
                else:
                    logger.info(f"✓ {lora_params_with_grad}/{lora_params_total} LoRA parameters ready for training")

                # Freeze lm_head for LoRA compatibility
                model.lm_head.requires_grad = False

                # Freeze Vision LoRA parameters if tune_mm_vision=False
                if not model_args.tune_mm_vision:
                    vision_lora_frozen = 0
                    for name, param in model.named_parameters():
                        if 'visual' in name and 'lora' in name.lower():
                            param.requires_grad = False
                            vision_lora_frozen += 1
                    if vision_lora_frozen > 0:
                        logger.info(f"Frozen {vision_lora_frozen} Vision LoRA parameters")

                # Freeze MLP LoRA parameters if tune_mm_mlp=False
                if not model_args.tune_mm_mlp:
                    mlp_lora_frozen = 0
                    for name, param in model.named_parameters():
                        if 'merger' in name and 'lora' in name.lower():
                            param.requires_grad = False
                            mlp_lora_frozen += 1
                    if mlp_lora_frozen > 0:
                        logger.info(f"Frozen {mlp_lora_frozen} MLP LoRA parameters")

                # Verify only LoRA parameters are trainable
                non_lora_trainable = [name for name, param in model.named_parameters()
                                    if param.requires_grad and 'lora' not in name.lower()]

                if non_lora_trainable:
                    logger.warning(f"⚠️  WARNING: {len(non_lora_trainable)} non-LoRA parameters are trainable")
                else:
                    logger.info("✓ All trainable parameters are LoRA parameters")

                # Parameter state verification (detailed logging commented out for cleaner output)
                # embedding_params = [name for name, param in model.named_parameters() if 'embed' in name.lower()]
                # lm_head_params = [name for name, param in model.named_parameters() if 'lm_head' in name.lower()]
                # logger.info(f"Embedding parameters: {len(embedding_params)}, LM Head parameters: {len(lm_head_params)}")

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

    # Create trainer with custom monitoring callback
    logger.info("Creating trainer...")
    try:
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            callbacks=[TrainingMonitorCallback(logger)],
            **data_module
        )
        logger.info("✓ Trainer created successfully")

        # Final parameter verification
        if hasattr(model, 'print_trainable_parameters'):
            model.print_trainable_parameters()
        else:
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"Trainable: {trainable_params:,}/{total_params:,} ({trainable_params/total_params*100:.2f}%)")

        # Forward pass debugging (commented out for cleaner logs)
        # first_forward_done = [False]
        # def debug_forward_hook(module, input, output):
        #     if not first_forward_done[0]:
        #         logger.info(f"First forward pass: {module.__class__.__name__}")
        #         first_forward_done[0] = True
        # model.register_forward_hook(debug_forward_hook)

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