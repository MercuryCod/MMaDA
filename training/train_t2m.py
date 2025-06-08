# ---------------------------------------------------------------------
# Train MMada-style transformer on motion-code sequences only
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
import clip
from os.path import join as pjoin
import torch.nn.functional as F

# ---------------------------------------------------------------------
# project imports
# ---------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import dataset_tokenize, dataset_TM_eval, dataset_TM_train
from training.prompting_utils import UniversalPrompting
from training.utils import (
    get_config,
    flatten_omega_conf,
    mask_or_random_replace_tokens,
    AverageMeter,
)
from motion_vqvae.models import vqvae  # HumanVQVAE
from models import MMadaModelLM, get_mask_schedule, MMadaConfig
from models.lr_schedulers import get_scheduler
from models.logging import set_verbosity_info, set_verbosity_error
from utils import eval_trans
from models.evaluator_wrapper import EvaluatorModelWrapper
from options.get_eval_option import get_opt


# ---------------------------------------------------------------------

logger = get_logger(__name__, log_level="INFO")


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
@torch.no_grad()
def build_mlm_batch(
    code_seq: torch.LongTensor, mask_id: int, config, mask_schedule, is_train: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Corrupt the discrete motion code sequence with **mask / random-replace**
    noise and produce (input_ids, labels, mask_prob).
    """
    inp, lbl, _, mprob = mask_or_random_replace_tokens(
        code_seq, mask_id, config, mask_schedule, is_train=is_train
    )
    return inp, lbl, mprob


def save_checkpoint(model, accel: Accelerator, config, gstep: int):
    """
    Save a sharded / fp16-safe checkpoint that can be resumed later.
    """
    out_dir = Path(config.experiment.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / f"checkpoint-{gstep}"
    ckpt_dir.mkdir(exist_ok=True)

    state_dict = accel.get_state_dict(model)
    if accel.is_main_process:
        accel.unwrap_model(model).save_pretrained(
            ckpt_dir,
            save_function=accel.save,
            state_dict=state_dict,
            safe_serialization=True,
        )
        json.dump({"global_step": gstep}, (ckpt_dir / "meta.json").open("w"))
        logger.info(f"✔ saved checkpoint to {ckpt_dir}")


# ───────────────────────────────────────────────────────────────────
# main
# ───────────────────────────────────────────────────────────────────
def main():
    #########################
    # SETUP Accelerator     #
    #########################
    config = get_config()
    
    vq_dir = os.path.join("./dataset/KIT-ML" if config.dataset.params.dataset_name == 'kit' else "./dataset/HumanML3D", f'{config.model.vq_model.vq_model_name}')
    config.model.vq_model.vq_dir = vq_dir
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
        set_verbosity_info()
    else:
        set_verbosity_error()

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
    train_loader_token = dataset_tokenize.DATALoader(
        config.dataset.params.dataset_name,
        # batch_size   = config.training.batch_size_t2m,
        batch_size=1,
        unit_length=2**config.dataset.params.down_t,
        num_workers=config.dataset.params.num_workers,
    )

    from utils.word_vectorizer import WordVectorizer

    wvec = WordVectorizer("./glove", "our_vab")
    val_loader = dataset_TM_eval.DATALoader(
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
            config.model.vq_model.nb_code,
            config.model.vq_model.code_dim,
            config.model.vq_model.output_emb_width,
            config.dataset.params.down_t,
            config.dataset.params.stride_t,
            config.model.vq_model.width,
            config.model.vq_model.depth,
            config.model.vq_model.dilation_growth_rate,
        )
        .eval()
        .requires_grad_(False)
        .to(accelerator.device)
    )
    
    logger.info('loading checkpoint from {}'.format(config.model.vq_model.resume_pth))
    ckpt = torch.load(config.model.vq_model.resume_pth, map_location='cpu')
    vq_model.load_state_dict(ckpt['net'], strict=True)

    nb_iter, avg_loss_cls, avg_acc = 0, 0.0, 0.0
    right_num = 0
    nb_sample_train = 0
    
    
    ##### ---- get code ---- #####
    # logger.info("Encoding motion codes...")
    # for batch in tqdm(train_loader_token):
    #     pose, name = batch
    #     bs, seq = pose.shape[0], pose.shape[1]

    #     pose = pose.cuda().float()  # bs, nb_joints, joints_dim, seq_len
    #     target = vq_model.encode(pose)
    #     target = target.cpu().numpy()
    #     np.save(pjoin(vq_dir, name[0] + ".npy"), target)
    
    train_loader = dataset_TM_train.DATALoader(
        config.dataset.params.dataset_name,
        config.training.batch_size_t2m,
        config.model.vq_model.nb_code,
        config.model.vq_model.vq_model_name,
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

    print("special tokens : \n", uni_prompting.sptids_dict)

    base_config = AutoConfig.from_pretrained(config.model.mmada.pretrained_model_path).to_dict()
    mmada_config_dict = {k: v for k, v in config.model.mmada.items()}
    merged_config = {**base_config, **mmada_config_dict}
    mmada_config = MMadaConfig(**merged_config)
    model = MMadaModelLM.from_pretrained(config.model.mmada.pretrained_model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, config=mmada_config)
    model.resize_token_embeddings(mmada_config.new_vocab_size)
    model.config.embedding_size = model.config.vocab_size
    model = model.to(accelerator.device)

    # ids we'll need
    mask_id = model.config.mask_token_id
    t2m_id = uni_prompting.sptids_dict["<|t2m|>"].item()

    ##################################
    #   Optimizer and LR scheduler   #
    #################################
    optimizer_config = config.optimizer.params

    # no decay on bias and layernorm and embedding
    no_decay = ["bias", "layer_norm.weight", "mlm_ln.weight", "embeddings.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad and not any(nd in n for nd in no_decay)
            ],
            "weight_decay": optimizer_config.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad and any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
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
    steps_per_epoch = math.ceil(
        len(train_loader) / config.training.gradient_accumulation_steps
    )
    num_epochs = math.ceil(config.training.max_train_steps / steps_per_epoch)

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
    

    # clip_model, _ = clip.load("ViT-B/32", device=accelerator.device, jit=False)

    logger.info("***** Train Text-to-Motion *****")
    for epoch in range(num_epochs):
        model.train()
        
        for batch in train_loader:
            captions, m_tokens, m_tokens_len = batch
            
            # Move to accelerator device (consistent device placement)
            m_tokens = m_tokens.to(accelerator.device).long()  # (B, T)
            m_tokens_len = m_tokens_len.to(accelerator.device)
            
            # Build masked language modeling batch from motion tokens
            inp, lbl, mprob = build_mlm_batch(
                m_tokens, mask_id, config, mask_schedule, True
            )
            
            # Use UniversalPrompting system to format text-to-motion sequences
            input_ids, attention_mask, labels = uni_prompting((captions, inp, lbl), 't2m')
            
            # Move to device
            input_ids = input_ids.to(accelerator.device)
            attention_mask = attention_mask.to(accelerator.device) 
            labels = labels.to(accelerator.device)

            # Manual gradient accumulation instead of accelerator.accumulate()
            # to avoid conflict with DeepSpeed ZeRO stage 3
            
            # Use the dedicated forward_t2m method for text-to-motion training
            loss = model.forward_t2m(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
                mask_token_id=mask_id,
                p_mask=mprob.mean()
            )
            
            loss = loss / config.training.gradient_accumulation_steps
            accelerator.backward(loss)

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
                optimizer.zero_grad(set_to_none=True)
                
                # Update sync_gradients flag for logging
                sync_gradients = True
                # Increment global_step only when we actually step the optimizer
                global_step += 1
            else:
                sync_gradients = False

            # ---- logging & validation ----------------------------------
            if sync_gradients:
                batch_t.update(time.time() - end)
                end = time.time()

                if global_step % config.experiment.log_every == 0:
                    accelerator.log(
                        {
                            "loss": loss.item(),
                            "lr": lr_scheduler.get_last_lr()[0],
                            "mask_rate": mprob.mean().item(),
                            "batch_t": batch_t.val,
                        },
                        step=global_step,
                    )
                    batch_t.reset()

                # ---- evaluation every eval_every ----------------------
                if (global_step % config.experiment.eval_every) == 0:
                    model.eval()
                    with torch.no_grad():
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
                                motion_vocab_size=config.model.vq_model.nb_code,
                                motion_seq_len=256,
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

                    # Update best metrics
                    # Best metrics are now tracked by evaluation function
                        # Save checkpoint when FID improves (best_iter indicates when FID improved)
                    if best_iter == global_step:
                        save_checkpoint(
                            model, accelerator, config, f"best-{global_step}"
                        )
                    

                    model.train()

                # ---- checkpoint ---------------------------------------
                if global_step % config.experiment.save_every == 0:
                    save_checkpoint(model, accelerator, config, global_step)

                if global_step >= config.training.max_train_steps:
                    break
        if global_step >= config.training.max_train_steps:
            break

    # ============== final save ==============
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.unwrap_model(model).save_pretrained(
            config.experiment.output_dir, safe_serialization=True
        )
    accelerator.end_training()


if __name__ == "__main__":
    main()
