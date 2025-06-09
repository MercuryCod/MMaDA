# ---------------------------------------------------------------------
# Train MMada-style transformer on motion-code sequences with LoRA
# 
# VOCABULARY ALLOCATION:
# - Text tokens:   [0, len(text_tokenizer) - 1]
# - Image tokens:  [len(text_tokenizer), len(text_tokenizer) + image_codebook_size - 1]  
# - Motion tokens: [len(text_tokenizer) + image_codebook_size, len(text_tokenizer) + image_codebook_size + motion_vocab_size - 1]
# 
# This ensures no vocabulary collision between any modalities.
# ---------------------------------------------------------------------
import os
# Fix HuggingFace tokenizers parallelism warning in multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys, json, time, math, shutil, logging
import warnings
# Suppress scipy RuntimeWarning from matrix square root in FID calculation
warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide", category=RuntimeWarning)
from pathlib import Path
from typing import Tuple
from tqdm import tqdm
import torch
import numpy as np
from torch.optim import AdamW
from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, set_seed
import wandb
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AutoConfig
from transformers.utils import logging as transformers_logging
import clip
from os.path import join as pjoin
import torch.nn.functional as F
import random
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

# ---------------------------------------------------------------------
# project imports
# ---------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import dataset_tokenize, dataset_TM_eval_fixed, dataset_TM_train
from training.prompting_utils import UniversalPrompting
from training.utils import (
    get_config,
    flatten_omega_conf,
    AverageMeter,
)
from motion_vqvae.models import vqvae  # HumanVQVAE
from models import MMadaModelLM, get_mask_schedule, MMadaConfig
from models.lr_schedulers import get_scheduler
from utils import eval_trans
from models.evaluator_wrapper import EvaluatorModelWrapper
from options.get_eval_option import get_opt
# Import the custom model class instead of MMadaModelLM
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models'))
from models.modelling_ours import MMadaModelLM as MMadaModelLMOurs


# ---------------------------------------------------------------------

logger = get_logger(__name__, log_level="INFO")


# ---------------------------------------------------------------------
# helpers (same as train_t2m.py)
# ---------------------------------------------------------------------
@torch.no_grad()
def build_mlm_batch(
    code_seq: torch.LongTensor, mask_id: int, config, mask_schedule, text_vocab_size: int, image_codebook_size: int, is_train: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Corrupt the discrete motion code sequence with **mask / random-replace**
    noise and produce (input_ids, labels, mask_prob).
    
    FIXED: code_seq should be in OFFSET range [text_vocab_size + image_codebook_size, ...].
    The vocabulary offset has already been applied before calling this function.
    """
    motion_vocab_size = config.model.motion_vq_model.nb_code
    
    inp, lbl, _, mprob = mask_or_random_replace_tokens_motion(
        code_seq, mask_id, config, mask_schedule, 
        motion_vocab_size, text_vocab_size, image_codebook_size, is_train=is_train
    )
    return inp, lbl, mprob


@torch.no_grad()
def mask_or_random_replace_tokens_motion(
    motion_tokens, mask_id, config, mask_schedule, 
    motion_vocab_size: int, text_vocab_size: int, image_codebook_size: int, is_train=True, seed=None
):
    """
    FIXED: Motion-specific masking that correctly handles vocabulary ranges.
    
    Args:
        motion_tokens: (B, L) - Motion tokens in OFFSET range [text_vocab + image_vocab, ...]
        mask_id: The mask token ID 
        motion_vocab_size: Size of motion vocabulary (512)
        text_vocab_size: Size of text vocabulary 
        image_codebook_size: Size of image vocabulary
    """
    batch_size, seq_len = motion_tokens.shape
    
    if not is_train and seed is not None:
        rng_state = torch.get_rng_state()
        if torch.cuda.is_available():
            cuda_rng_state = torch.cuda.get_rng_state()
        python_rng_state = random.getstate()
        
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        random.seed(seed)

    if not is_train and config.training.get("eval_mask_ratios", None):
        mask_prob = random.choices(config.training.eval_mask_ratios, k=batch_size)
        mask_prob = torch.tensor(mask_prob, device=motion_tokens.device)
    else:
        timesteps = torch.rand(batch_size, device=motion_tokens.device)
        mask_prob = mask_schedule(timesteps)
        mask_prob = mask_prob.clip(config.training.min_masking_rate)

    num_token_masked = (seq_len * mask_prob).round().clamp(min=1)
    batch_randperm = torch.rand(batch_size, seq_len, device=motion_tokens.device).argsort(dim=-1)
    mask = batch_randperm < num_token_masked.unsqueeze(-1)

    # Handle noise type
    noise_type = config.training.get("noise_type", "mask")
    if noise_type == "mask":
        input_ids = torch.where(mask, mask_id, motion_tokens)
    elif noise_type == "random_replace":
        # CRITICAL FIX: Generate random tokens in the OFFSET motion token range 
        # to match the input motion_tokens which are already offset
        motion_token_start = text_vocab_size + image_codebook_size
        motion_token_end = motion_token_start + motion_vocab_size
        
        # CRITICAL: Only replace with actual motion codes, not special tokens
        # Generate random tokens only in the actual motion code range [0-511]
        # which maps to [motion_token_start, motion_token_start + 512) in vocabulary space
        random_tokens = torch.randint(
            low=motion_token_start, 
            high=motion_token_start + motion_vocab_size,  # This ensures we only get codes 0-511 in offset space
            size=motion_tokens.shape, 
            device=motion_tokens.device
        )
        input_ids = torch.where(mask, random_tokens, motion_tokens)
    else:
        raise ValueError(f"noise_type {noise_type} not supported")

    # For motion tokens, we predict all tokens or only masked tokens
    if config.training.get("predict_all_tokens", False) or noise_type == "random_replace":
        labels = motion_tokens
    else:
        labels = torch.where(mask, motion_tokens, -100)

    if not is_train and seed is not None:
        torch.set_rng_state(rng_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(cuda_rng_state)
        random.setstate(python_rng_state)

    return input_ids, labels, None, mask_prob


def save_checkpoint(model, accel: Accelerator, config, gstep: int, lora_only=True):
    """
    Save a sharded / fp16-safe checkpoint that can be resumed later.
    
    Args:
        gstep: Can be an integer step number or a string like "best-100"
        lora_only: If True, only save LoRA weights. If False, save the full model.
    """
    out_dir = Path(config.experiment.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle both integer steps and special checkpoint names
    if isinstance(gstep, str):
        ckpt_dir = out_dir / f"checkpoint-{gstep}"
    else:
        ckpt_dir = out_dir / f"checkpoint-{gstep}"
    
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if lora_only:
        # Save only LoRA weights
        if accel.is_main_process:
            accel.unwrap_model(model).save_pretrained(
                ckpt_dir,
                save_function=accel.save,
                safe_serialization=True,
            )
            json.dump({"global_step": gstep}, (ckpt_dir / "meta.json").open("w"))
            logger.info(f"✔ saved LoRA checkpoint to {ckpt_dir}")
    else:
        # Save full model (for final checkpoint)
        state_dict = accel.get_state_dict(model)
        if accel.is_main_process:
            # Merge LoRA weights and save full model
            merged_model = accel.unwrap_model(model).merge_and_unload()
            merged_model.save_pretrained(
                ckpt_dir,
                save_function=accel.save,
                state_dict=merged_model.state_dict(),
                safe_serialization=True,
            )
            json.dump({"global_step": gstep}, (ckpt_dir / "meta.json").open("w"))
            logger.info(f"✔ saved full merged checkpoint to {ckpt_dir}")


def validate_vocabulary_config(text_vocab_size: int, image_codebook_size: int, motion_vocab_size_with_special: int, expected_total: int):
    """
    Validate that vocabulary configuration is consistent.
    
    Args:
        text_vocab_size: Size of text vocabulary (includes text special tokens)
        image_codebook_size: Size of image codebook
        motion_vocab_size_with_special: Motion vocab (512) + EOM (1) + PAD (1) = 514
        expected_total: Expected total vocabulary size
    """
    calculated_total = text_vocab_size + image_codebook_size + motion_vocab_size_with_special
    if calculated_total != expected_total:
        raise ValueError(
            f"Vocabulary size mismatch! "
            f"Calculated: {calculated_total} (text: {text_vocab_size} + image: {image_codebook_size} + motion: {motion_vocab_size_with_special}) "
            f"vs Expected: {expected_total}"
        )
    
    logger.info(f"✅ Vocabulary validation passed: {calculated_total} tokens total")
    logger.info(f"   Text vocabulary (with special tokens): {text_vocab_size}")
    logger.info(f"   Image codebook: {image_codebook_size}")
    logger.info(f"   Motion vocabulary (512 codes + 2 special): {motion_vocab_size_with_special}")
    return True


def validate_loss(loss: torch.Tensor, step: int):
    """
    Check for problematic loss values and log warnings.
    """
    if torch.isnan(loss).any():
        logger.error(f"❌ NaN loss detected at step {step}!")
        raise ValueError(f"NaN loss at step {step}")
    
    if torch.isinf(loss).any():
        logger.error(f"❌ Infinite loss detected at step {step}!")
        raise ValueError(f"Infinite loss at step {step}")
    
    if loss.item() > 100.0:
        logger.warning(f"⚠️  Very high loss detected at step {step}: {loss.item():.3f}")
    
    return True


# ───────────────────────────────────────────────────────────────────
# main
# ───────────────────────────────────────────────────────────────────
def main():
    #########################
    # SETUP Accelerator     #
    #########################
    config = get_config()
    
    vq_dir = os.path.join("./dataset/KIT-ML" if config.dataset.params.dataset_name == 'kit' else "./dataset/HumanML3D", f'{config.model.motion_vq_model.vq_model_name}')
    config.model.motion_vq_model.vq_dir = vq_dir
    os.makedirs(vq_dir, exist_ok=True)
    
    writer = SummaryWriter(config.experiment.output_dir)

    # Enable TF32 on Ampere GPUs
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    config.experiment.logging_dir = str(Path(config.experiment.output_dir) / "logs")
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with="wandb",
        project_dir=config.experiment.logging_dir,
        split_batches=True,
    )

    total_batch_size_per_gpu = config.training.batch_size_t2m
    total_batch_size = (
        (config.training.batch_size_t2m)
        * accelerator.num_processes
        * config.training.gradient_accumulation_steps
    )

    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        accelerator.state.deepspeed_plugin.deepspeed_config[
            "train_micro_batch_size_per_gpu"
        ] = total_batch_size_per_gpu

    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        level=logging.INFO,
        datefmt="%H:%M:%S",
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers_logging.set_verbosity_info()
    else:
        transformers_logging.set_verbosity_error()

    if config.training.seed is not None:
        set_seed(config.training.seed)

    # ============== init wandb ==============
    if accelerator.is_main_process:
        run_id = config.wandb.get("run_id", wandb.util.generate_id())
        config.wandb.run_id = run_id
        accelerator.init_trackers(
            config.experiment.project,
            config={k: v for k, v in flatten_omega_conf(config, resolve=True)},
            init_kwargs={
                "wandb": dict(
                    id=run_id,
                    name=config.experiment.name,
                    resume=config.wandb.resume,
                    entity=config.wandb.get("entity", None),
                )
            },
        )

    # ============== data ==============
    from utils.word_vectorizer import WordVectorizer

    wvec = WordVectorizer("./glove", "our_vab")
    val_loader = dataset_TM_eval_fixed.DATALoader(
        config.dataset.params.dataset_name,
        is_test=False,
        batch_size=32,
        w_vectorizer=wvec,
        unit_length=2**config.dataset.params.down_t,
    )

    # evaluator wrapper
    dataset_opt_path = (
        "checkpoints/kit/Comp_v6_KLD005/opt.txt"
        if config.dataset.params.dataset_name == "kit"
        else "checkpoints/t2m/Comp_v6_KLD005/opt.txt"
    )
    eval_wrapper = EvaluatorModelWrapper(
        get_opt(dataset_opt_path, torch.device("cuda"))
    )

    # ---- VQ-VAE encoder (frozen)
    vq_model = (
        vqvae.HumanVQVAE(
            config,
            config.model.motion_vq_model.nb_code,
            config.model.motion_vq_model.code_dim,
            config.model.motion_vq_model.output_emb_width,
            config.dataset.params.down_t,
            config.dataset.params.stride_t,
            config.model.motion_vq_model.width,
            config.model.motion_vq_model.depth,
            config.model.motion_vq_model.dilation_growth_rate,
        )
        .eval()
        .requires_grad_(False)
        .to(accelerator.device)
    )
    
    logger.info('loading checkpoint from {}'.format(config.model.motion_vq_model.resume_pth))
    ckpt = torch.load(config.model.motion_vq_model.resume_pth, map_location='cpu')
    vq_model.load_state_dict(ckpt['net'], strict=True)

    train_loader = dataset_TM_train.DATALoader(
        config.dataset.params.dataset_name,
        config.training.batch_size_t2m,
        config.model.motion_vq_model.nb_code,
        config.model.motion_vq_model.vq_model_name,
        unit_length=2**config.dataset.params.down_t,
    )
    # ============== models ==============

    # ---- LM backbone

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.mmada.tokenizer_path, padding_side="left"
    )

    uni_prompting = UniversalPrompting(
        tokenizer,
        max_text_len=config.dataset.preprocessing.max_seq_length,
        special_tokens=(
            "<|soi|>",
            "<|eoi|>",
            "<|sov|>",
            "<|eov|>",
            "<|t2i|>",
            "<|mmu|>",
            "<|t2v|>",
            "<|v2v|>",
            "<|lvg|>",
            "<|t2m|>",
            "<|som|>",
            "<|eom|>",
        ),
        ignore_id=-100,
        cond_dropout_prob=config.training.cond_dropout_prob,
        use_reserved_token=True,
    )

    # CRITICAL: Calculate vocabulary sizes correctly
    text_vocab_size = len(uni_prompting.text_tokenizer)
    image_codebook_size = config.model.mmada.image_codebook_size
    motion_vocab_size = config.model.motion_vq_model.nb_code
    
    logger.info(f"Vocabulary: Text={text_vocab_size}, Image={image_codebook_size}, Motion={motion_vocab_size}, EOM+PAD=2, Total={text_vocab_size + image_codebook_size + motion_vocab_size + 2}")
    
    # FIXED: Validate vocabulary configuration including special tokens
    # The total should include text + image + motion + EOM + PAD
    validate_vocabulary_config(
        text_vocab_size, image_codebook_size, motion_vocab_size + 2,  # +2 for EOM and PAD
        config.model.mmada.new_vocab_size
    )

    base_config = AutoConfig.from_pretrained(config.model.mmada.pretrained_model_path).to_dict()
    mmada_config_dict = {k: v for k, v in config.model.mmada.items()}
    merged_config = {**base_config, **mmada_config_dict}
    mmada_config = MMadaConfig(**merged_config)
    
    # Use the custom MMadaModelLM from modelling_ours.py
    model = MMadaModelLMOurs.from_pretrained(
        config.model.mmada.pretrained_model_path, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16, 
        config=mmada_config
    )
    model.resize_token_embeddings(mmada_config.new_vocab_size)
    
    # FIXED: Ensure model vocabulary size is properly set
    model.config.vocab_size = mmada_config.new_vocab_size
    if hasattr(model.config, 'embedding_size'):
        model.config.embedding_size = mmada_config.new_vocab_size
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=config.model.lora.r,
        lora_alpha=config.model.lora.lora_alpha,
        target_modules=list(config.model.lora.target_modules),  # Convert ListConfig to list
        lora_dropout=config.model.lora.lora_dropout,
        bias=config.model.lora.bias,
        task_type=TaskType.FEATURE_EXTRACTION,  # Using feature extraction as task type for masked language modeling
        modules_to_save=["embed_tokens", "lm_head"] if config.model.lora.get("train_embeddings", True) else None,
    )
    
    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    model = model.to(accelerator.device)

    # Enable gradient checkpointing if configured
    if config.model.get('gradient_checkpointing', False):
        try:
            model.gradient_checkpointing_enable()
            logger.info("✅ Gradient checkpointing enabled")
        except (ValueError, NotImplementedError) as e:
            logger.warning(f"⚠️  Gradient checkpointing not supported: {e}")
            logger.info("   Continuing training without gradient checkpointing")

    # ids we'll need
    mask_id = model.config.mask_token_id
    t2m_id = uni_prompting.sptids_dict["<|t2m|>"].item()

    logger.info(f"Model loaded: vocab_size={model.config.vocab_size}, mask_token_id={mask_id}")
    
    ##################################
    #   Optimizer and LR scheduler   #
    #################################
    optimizer_config = config.optimizer.params

    # Only get parameters that require grad (LoRA parameters)
    optimizer_grouped_parameters = filter(lambda p: p.requires_grad, model.parameters())
    
    optimizer_type = config.optimizer.name
    if optimizer_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=optimizer_config.learning_rate,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
            weight_decay=optimizer_config.weight_decay,
            eps=optimizer_config.epsilon,
        )
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")

    # Create mask scheduler
    if config.get("mask_schedule", None) is not None:
        schedule = config.mask_schedule.schedule
        args = config.mask_schedule.get("params", {})
        mask_schedule = get_mask_schedule(schedule, **args)
    else:
        mask_schedule = get_mask_schedule(
            config.training.get("mask_schedule", "cosine")
        )

    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=config.training.max_train_steps,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps,
        min_lr_scale=config.lr_scheduler.params.min_lr_scale,
    )
    # ============== prepare (DDP / fp16) ==============
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    # ============== training loop ==============
    steps_per_epoch = len(train_loader)
    num_epochs = math.ceil(config.training.max_train_steps / (steps_per_epoch // config.training.gradient_accumulation_steps))

    batch_t = AverageMeter()
    end = time.time()
    global_step = 0
    batch_count = 0  # Counter for manual gradient accumulation
    best_fid = float('inf')
    best_iter = 0  # Track the step when best FID was achieved
    best_div = 0.0  # Best diversity value (will be updated by eval function)
    best_top1 = 0.0  # Best top-1 R-precision (maximize)
    best_top2 = 0.0  # Best top-2 R-precision (maximize) 
    best_top3 = 0.0  # Best top-3 R-precision (maximize)
    best_matching = float('inf')  # Best matching score (minimize)
    

    logger.info("***** Train Text-to-Motion with LoRA *****")
    logger.info(f"Total training steps: {config.training.max_train_steps}")
    logger.info(f"Steps per epoch: {steps_per_epoch}")
    logger.info(f"Total epochs: {num_epochs}")
    logger.info(f"LoRA rank: {config.model.lora.r}")
    logger.info(f"LoRA alpha: {config.model.lora.lora_alpha}")
    logger.info(f"LoRA target modules: {config.model.lora.target_modules}")
    
    # Debug: Log vocabulary ranges
    logger.info(f"📊 Vocabulary ranges:")
    logger.info(f"   Text tokens: [0, {text_vocab_size-1}]")
    logger.info(f"   Image tokens: [{text_vocab_size}, {text_vocab_size + image_codebook_size - 1}]")
    logger.info(f"   Motion tokens: [{text_vocab_size + image_codebook_size}, {text_vocab_size + image_codebook_size + motion_vocab_size - 1}]")
    logger.info(f"   EOM token: {text_vocab_size + image_codebook_size + motion_vocab_size}")
    logger.info(f"   PAD token: {text_vocab_size + image_codebook_size + motion_vocab_size + 1}")
    logger.info(f"   Mask token ID: {mask_id}")
    
    try:
        first_batch = True  # DEBUG flag
        for epoch in range(num_epochs):
            model.train()
            
            for batch in train_loader:
                # Initialize sync_gradients flag at the start of each batch
                sync_gradients = False
                
                # Initialize variables that are used in logging to avoid undefined errors
                unscaled_loss = 0.0
                mprob = torch.tensor([0.0], device=accelerator.device)
                
                try:
                    captions, m_tokens, m_tokens_len = batch
                    
                    # Move to accelerator device (consistent device placement)
                    m_tokens = m_tokens.to(accelerator.device).long()  # (B, T)
                    m_tokens_len = m_tokens_len.to(accelerator.device)
                    
                    # Store current batch info for later use in validation
                    current_batch_size = m_tokens.shape[0]
                    current_motion_seq_len = m_tokens.shape[1]
                    current_captions = captions
                    
                    # CRITICAL FIX: Apply vocabulary offset to motion tokens ONCE
                    # Motion tokens need to be offset to their vocabulary range
                    motion_token_offset = text_vocab_size + image_codebook_size
                    
                    # CRITICAL: Handle special tokens from dataset
                    # The dataset uses 512 for end-of-motion and 513 for padding
                    # We need to map these to appropriate vocabulary tokens
                    eom_token_dataset = config.model.motion_vq_model.nb_code  # 512
                    pad_token_dataset = config.model.motion_vq_model.nb_code + 1  # 513
                    
                    # FIXED: Map special tokens to the END of motion vocabulary space
                    # Motion tokens [0-511] map to [134541, 135052]
                    # EOM token 512 maps to 135053
                    # PAD token 513 maps to 135054
                    eom_token_vocab = motion_token_offset + eom_token_dataset  # 134541 + 512 = 135053
                    pad_token_vocab = motion_token_offset + pad_token_dataset  # 134541 + 513 = 135054
                    
                    # Verify these fit in our expanded vocabulary (should always pass now)
                    assert eom_token_vocab < config.model.mmada.new_vocab_size, f"EOM token {eom_token_vocab} exceeds vocab size {config.model.mmada.new_vocab_size}"
                    assert pad_token_vocab < config.model.mmada.new_vocab_size, f"PAD token {pad_token_vocab} exceeds vocab size {config.model.mmada.new_vocab_size}"
                    
                    # Debug logging for special token mapping
                    if first_batch and accelerator.is_main_process:
                        logger.info(f"🔍 Special token mapping debug:")
                        logger.info(f"   EOM dataset token: {eom_token_dataset} -> vocab token: {eom_token_vocab}")
                        logger.info(f"   PAD dataset token: {pad_token_dataset} -> vocab token: {pad_token_vocab}")
                        logger.info(f"   Motion token offset: {motion_token_offset}")
                        logger.info(f"   Expected motion range: [{motion_token_offset}, {motion_token_offset + motion_vocab_size - 1}]")
                        logger.info(f"   Total motion + special tokens: [{motion_token_offset}, {pad_token_vocab}]")
                    
                    # Create offset tokens, but handle special tokens separately
                    m_tokens_offset = m_tokens.clone()
                    
                    # Offset regular motion tokens (0-511)
                    regular_motion_mask = (m_tokens < eom_token_dataset)
                    m_tokens_offset[regular_motion_mask] = m_tokens[regular_motion_mask] + motion_token_offset
                    
                    # Map special tokens to vocabulary special tokens
                    m_tokens_offset[m_tokens == eom_token_dataset] = eom_token_vocab
                    m_tokens_offset[m_tokens == pad_token_dataset] = pad_token_vocab
                    
                    # Validate token ranges - now special tokens should be in valid range
                    if torch.any(m_tokens_offset >= config.model.mmada.new_vocab_size):
                        logger.error(f"Motion tokens out of vocabulary range! Max token: {m_tokens_offset.max()}, Vocab size: {config.model.mmada.new_vocab_size}")
                        # Log which tokens are problematic
                        out_of_bounds = m_tokens_offset >= config.model.mmada.new_vocab_size
                        if out_of_bounds.any():
                            logger.error(f"Out of bounds positions: {out_of_bounds.nonzero().tolist()}")
                            logger.error(f"Out of bounds values: {m_tokens_offset[out_of_bounds].unique().tolist()}")
                        continue
                    
                    # Build masked language modeling batch from OFFSET motion tokens
                    inp, lbl, mprob = build_mlm_batch(
                        m_tokens_offset, mask_id, config, mask_schedule, 
                        text_vocab_size, image_codebook_size, True
                    )
                    
                    # Use UniversalPrompting system to format text-to-motion sequences
                    input_ids, attention_mask, labels = uni_prompting((captions, inp, lbl), 't2m')
                    
                    # Debug logging for first batch
                    if first_batch and accelerator.is_main_process:
                        logger.info(f"🔍 First batch debug info:")
                        logger.info(f"   Original motion tokens shape: {m_tokens.shape}")
                        logger.info(f"   Motion token range: [{m_tokens.min().item()}, {m_tokens.max().item()}]")
                        logger.info(f"   Offset motion token range: [{m_tokens_offset.min().item()}, {m_tokens_offset.max().item()}]")
                        logger.info(f"   Masked input token range: [{inp.min().item()}, {inp.max().item()}]")
                        logger.info(f"   Final input_ids shape: {input_ids.shape}")
                        logger.info(f"   Mask probability: {mprob.mean().item():.3f}")
                        first_batch = False
                    
                    # Move to device
                    input_ids = input_ids.to(accelerator.device)
                    attention_mask = attention_mask.to(accelerator.device) 
                    labels = labels.to(accelerator.device)
                    
                    # Use the dedicated forward_t2m method for text-to-motion training
                    # FIXED: Handle DeepSpeed wrapper by accessing the underlying model
                    if hasattr(model, 'forward_t2m'):
                        # Direct access if not wrapped
                        loss = model.forward_t2m(
                            input_ids=input_ids,
                            labels=labels,
                            attention_mask=attention_mask,
                            mask_token_id=mask_id,
                            p_mask=mprob.mean()
                        )
                    else:
                        # Access through accelerator.unwrap_model for DeepSpeed/DDP wrapped models
                        unwrapped_model = accelerator.unwrap_model(model)
                        loss = unwrapped_model.forward_t2m(
                            input_ids=input_ids,
                            labels=labels,
                            attention_mask=attention_mask,
                            mask_token_id=mask_id,
                            p_mask=mprob.mean()
                        )
                    
                    # FIXED: Validate loss before proceeding
                    validate_loss(loss, global_step)
                    
                    # Store unscaled loss for logging
                    unscaled_loss = loss.item()
                    
                    loss = loss / config.training.gradient_accumulation_steps
                    accelerator.backward(loss)

                except Exception as batch_error:
                    logger.error(f"❌ Error processing batch at step {global_step}: {batch_error}")
                    continue

                # Increment batch counter
                batch_count += 1

                # Only step optimizer every gradient_accumulation_steps
                if batch_count % config.training.gradient_accumulation_steps == 0:
                    if config.training.max_grad_norm:
                        accelerator.clip_grad_norm_(
                            model.parameters(), config.training.max_grad_norm
                        )

                    optimizer.step()
                    lr_scheduler.step()
                    
                    # Log gradient norm if configured (before incrementing global_step)
                    if config.experiment.get('log_grad_norm_every', 0) > 0 and (global_step + 1) % config.experiment.log_grad_norm_every == 0:
                        log_grad_norm(model, accelerator, global_step + 1)
                    
                    optimizer.zero_grad(set_to_none=True)
                    
                    # Increment global_step only when we actually step the optimizer
                    global_step += 1
                    
                    # Reset batch counter after optimizer step for correct gradient accumulation
                    batch_count = 0
                    
                    # Set sync_gradients to True after optimizer step
                    sync_gradients = True
                    
                    # ---- Validation check: Ensure generated tokens are in valid range ----
                    if global_step % config.experiment.val_every == 0:  # Check every val_every steps
                        with torch.no_grad():
                            # Sample a few captions for quick generation test
                            test_idx = min(6, current_batch_size)  # Increased from 3 to 6 for better diversity assessment
                            test_captions = current_captions[:test_idx]
                            
                            # Create test input with all motion tokens masked
                            # FIXED: mask_id is already in full vocabulary space, so we need to create masked motion tokens differently
                            # First create motion tokens filled with zeros, then pass them through build_mlm_batch
                            test_motion_tokens_raw = torch.zeros((test_idx, current_motion_seq_len), dtype=torch.long, device=accelerator.device)
                            # Add offset to place them in motion vocabulary range
                            test_motion_tokens_offset = test_motion_tokens_raw + motion_token_offset
                            
                            # Create fully masked version using build_mlm_batch with 100% masking
                            test_mask_schedule = lambda t: torch.ones_like(t)  # Always return 1.0 for full masking
                            test_inp, test_lbl, _ = build_mlm_batch(
                                test_motion_tokens_offset, mask_id, config, test_mask_schedule,
                                text_vocab_size, image_codebook_size, False
                            )
                            
                            # Format with uni_prompting
                            test_input_ids, test_attention_mask, _ = uni_prompting(
                                (test_captions, test_inp, test_lbl), 't2m'
                            )
                            
                            # Quick generation with fewer timesteps
                            try:
                                # FIXED: Handle DeepSpeed wrapper for generation
                                if hasattr(model, 't2m_generate'):
                                    generated_tokens = model.t2m_generate(
                                        input_ids=test_input_ids,
                                        attention_mask=test_attention_mask,
                                        uni_prompting=uni_prompting,
                                        mask_token_id=mask_id,
                                        motion_vocab_size=motion_vocab_size,
                                        seq_len=current_motion_seq_len,
                                        timesteps=12,  # Increased from 5 to 12 for better diversity
                                        image_codebook_size=image_codebook_size,
                                    )
                                else:
                                    unwrapped_model = accelerator.unwrap_model(model)
                                    generated_tokens = unwrapped_model.t2m_generate(
                                        input_ids=test_input_ids,
                                        attention_mask=test_attention_mask,
                                        uni_prompting=uni_prompting,
                                        mask_token_id=mask_id,
                                        motion_vocab_size=motion_vocab_size,
                                        seq_len=current_motion_seq_len,
                                        timesteps=12,  # Increased from 5 to 12 for better diversity
                                        image_codebook_size=image_codebook_size,
                                    )
                                
                                # Validate generated tokens
                                min_gen = generated_tokens.min().item()
                                max_gen = generated_tokens.max().item()
                                
                                if min_gen < 0 or max_gen >= motion_vocab_size:
                                    logger.warning(f"⚠️  Generated tokens out of range! Min: {min_gen}, Max: {max_gen}, Expected: [0, {motion_vocab_size-1}]")
                                else:
                                    logger.info(f"✅ Generated tokens in valid range: [{min_gen}, {max_gen}] ⊆ [0, {motion_vocab_size-1}]")
                                    
                                    # Check for special tokens in generated output
                                    unique_tokens = generated_tokens.unique()
                                    total_tokens = generated_tokens.numel()
                                    diversity_ratio = len(unique_tokens) / min(total_tokens, motion_vocab_size)
                                    
                                    if motion_vocab_size in unique_tokens:
                                        logger.info(f"   Found end-of-motion token (512) in generation")
                                    logger.info(f"   Unique tokens: {len(unique_tokens)}/{total_tokens} (diversity: {diversity_ratio:.1%})")
                                    
                                    # Warning for potential mode collapse
                                    if len(unique_tokens) < 10:
                                        logger.warning(f"⚠️  Low diversity detected! Only {len(unique_tokens)} unique tokens generated.")
                                        if len(unique_tokens) <= 3:
                                            logger.warning(f"🚨 Possible mode collapse! Consider checking learning rate, loss weights, or generation parameters.")
                                    
                            except Exception as gen_error:
                                logger.warning(f"⚠️  Quick generation test failed: {gen_error}")
                    
                else:
                    sync_gradients = False

                # ---- logging & validation ----------------------------------
                if sync_gradients:
                    batch_t.update(time.time() - end)
                    end = time.time()

                    if global_step % config.experiment.log_every == 0:
                        accelerator.log(
                            {
                                "loss": unscaled_loss,
                                "lr": lr_scheduler.get_last_lr()[0],
                                "mask_rate": mprob.mean().item(),
                                "batch_t": batch_t.val,
                            },
                            step=global_step,
                        )
                        batch_t.reset()
                        logger.info(f"Step {global_step}: loss={unscaled_loss:.4f}, lr={lr_scheduler.get_last_lr()[0]:.2e}")

                    # ---- evaluation every eval_every ----------------------
                    if (global_step % config.experiment.eval_every) == 0:
                        logger.info(f"🔍 Starting evaluation at step {global_step}")
                        model.eval()
                        with torch.no_grad():
                            try:
                                best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching = (
                                    eval_trans.evaluation_mmada_t2m(
                                        config.experiment.output_dir,
                                        val_loader,
                                        vq_model,
                                        model,  # same signature as old code
                                        uni_prompting,  # UniversalPrompting system
                                        logger,
                                        writer,  # tensorboard writer – not used
                                        global_step,
                                        best_fid,
                                        best_iter,
                                        best_div,
                                        best_top1,
                                        best_top2,
                                        best_top3,
                                        best_matching,
                                        eval_wrapper,
                                        mask_token_id=mask_id,
                                        motion_vocab_size=config.model.motion_vq_model.nb_code,
                                        motion_seq_len=256,
                                        image_codebook_size=config.model.mmada.image_codebook_size,
                                        savegif=True,
                                    )
                                )

                                accelerator.log(
                                    {
                                        "val/fid": best_fid,
                                        "val/div": best_div,
                                        "val/top1": best_top1,
                                        "val/top2": best_top2,
                                        "val/top3": best_top3,
                                        "val/matching": best_matching,
                                    },
                                    step=global_step,
                                )

                                # Save checkpoint when FID improves (best_iter indicates when FID improved)
                                if best_iter == global_step:
                                    # FIXED: Use string formatting instead of f-string for checkpoint name
                                    save_checkpoint(model, accelerator, config, f"best-{global_step}", lora_only=True)
                                    
                            except Exception as eval_error:
                                logger.error(f"❌ Evaluation failed at step {global_step}: {eval_error}")

                        model.train()

                    # ---- checkpoint ---------------------------------------
                    if global_step % config.experiment.save_every == 0:
                        save_checkpoint(model, accelerator, config, global_step, lora_only=True)

                    # Check if we've reached max steps at the end of each epoch
                    if global_step >= config.training.max_train_steps:
                        logger.info(f"✅ Reached max training steps ({config.training.max_train_steps})")
                        break

    except Exception as training_error:
        logger.error(f"❌ Training failed: {training_error}")
        raise

    # ============== final save ==============
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # Save final LoRA weights
        accelerator.unwrap_model(model).save_pretrained(
            config.experiment.output_dir, safe_serialization=True
        )
        
        # Also save merged model for easier inference
        merged_model_dir = Path(config.experiment.output_dir) / "merged_model"
        merged_model = accelerator.unwrap_model(model).merge_and_unload()
        merged_model.save_pretrained(merged_model_dir, safe_serialization=True)
        
        logger.info(f"✅ Training completed successfully!")
        logger.info(f"   LoRA weights saved to: {config.experiment.output_dir}")
        logger.info(f"   Merged model saved to: {merged_model_dir}")
    accelerator.end_training()


def log_grad_norm(model, accelerator, global_step):
    """
    Log gradient norms for each parameter.
    """
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads = param.grad.detach().data
            grad_norm = (grads.norm(p=2) / grads.numel()).item()
            accelerator.log({"grad_norm/" + name: grad_norm}, step=global_step)


if __name__ == "__main__":
    main() 