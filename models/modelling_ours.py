from __future__ import annotations

import logging
import math
import sys
from abc import abstractmethod
from collections import defaultdict
from functools import partial
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    cast,
)
from dataclasses import fields
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import torch.backends.cuda
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.auto import AutoModel, AutoConfig, AutoModelForCausalLM
from transformers.cache_utils import Cache
from PIL import Image
from .configuration_llada import (
    LLaDAConfig,
    StrEnum,
    InitFnType,
    ActivationType,
    BlockType,
    LayerNormType,
    ModelConfig,
    ActivationCheckpointingStrategy,
)

from .modeling_llada import LLaDAModelLM
from .sampling import cosine_schedule, mask_by_random_topk
from transformers import PretrainedConfig

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens

class MMadaConfig(PretrainedConfig):
    model_type = "mmada"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        allowed_keys = [
            "vocab_size",
            "llm_vocab_size",
            "llm_model_path",
            "codebook_size",
            "num_vq_tokens",
            "num_new_special_tokens",
            "gradient_checkpointing",
            "new_vocab_size",
            "motion_vocab_size",
            "image_codebook_size",
        ]

        for key in allowed_keys:
            if key in kwargs:
                setattr(self, key, kwargs[key])
        
        # Only calculate total vocabulary size if neither vocab_size nor new_vocab_size is explicitly set
        if not hasattr(self, 'vocab_size') and not hasattr(self, 'new_vocab_size'):
            llm_vocab_size = getattr(self, 'llm_vocab_size', 126000)
            image_codebook_size = getattr(self, 'codebook_size', 8192)
            motion_vocab_size = getattr(self, 'motion_vocab_size', 512)
            num_special_tokens = getattr(self, 'num_new_special_tokens', 10)
            buffer_size = 100  # Safety buffer
            
            total_vocab_size = llm_vocab_size + image_codebook_size + motion_vocab_size + num_special_tokens + buffer_size
            self.vocab_size = total_vocab_size
            self.new_vocab_size = total_vocab_size
            
            print(f"MMadaConfig: Auto-calculated vocab_size = {total_vocab_size}")
            print(f"  - LLM vocab: {llm_vocab_size}")
            print(f"  - Image codebook: {image_codebook_size}")
            print(f"  - Motion vocab: {motion_vocab_size}")
            print(f"  - Special tokens: {num_special_tokens}")
            print(f"  - Buffer: {buffer_size}")
        else:
            # If vocab_size or new_vocab_size was provided, ensure both are set to the same value
            if hasattr(self, 'new_vocab_size') and not hasattr(self, 'vocab_size'):
                self.vocab_size = self.new_vocab_size
            elif hasattr(self, 'vocab_size') and not hasattr(self, 'new_vocab_size'):
                self.new_vocab_size = self.vocab_size
            # Also update embedding_size if needed
            if hasattr(self, 'vocab_size'):
                self.embedding_size = self.vocab_size



class MMadaModelLM(LLaDAModelLM):
    config_class = MMadaConfig
    base_model_prefix = "model"
    def __init__(self, config: MMadaConfig, *args, **kwargs):
        print(f"Initializing MMadaModelLM with config: {config}")
        super().__init__(config, *args, **kwargs)

        # # resize token embeddings
        # print(f"Resizing token embeddings to {config.new_vocab_size}")
        # self.resize_token_embeddings(config.new_vocab_size)

    def _get_vocab_size(self):
        """Get the vocabulary size from config or model"""
        # First try to get new_vocab_size from config (this is what we set in training)
        vocab_size = getattr(self.config, 'new_vocab_size', None)
        
        # If not found, try vocab_size
        if vocab_size is None:
            vocab_size = getattr(self.config, 'vocab_size', None)
        
        # If still not found, try to get from embeddings
        if vocab_size is None and hasattr(self, 'model') and hasattr(self.model, 'embed_tokens'):
                vocab_size = self.model.embed_tokens.num_embeddings
            
        # If all else fails, use the expected value
        if vocab_size is None:
            vocab_size = 135055  # Our expected total: text (126349) + image (8192) + motion (512) + special (2)
            print(f"WARNING: Could not determine vocab_size from config, using default: {vocab_size}")
            
        return vocab_size

    def _validate_token_ids(self, input_ids, labels=None, mask_token_id=None):
        """Validate and clamp token IDs to prevent CUDA indexing errors"""
        vocab_size = self._get_vocab_size()
        
        # Debug info about token ranges
        max_token_id = input_ids.max().item()
        min_token_id = input_ids.min().item()
        
        # print(f"DEBUG: vocab_size={vocab_size}, input_ids range=[{min_token_id}, {max_token_id}]")
        # print(f"DEBUG: input_ids shape={input_ids.shape}")
        
        # Get token type ranges for better debugging
        text_vocab_size = getattr(self.config, 'llm_vocab_size', 126000)
        image_codebook_size = getattr(self.config, 'codebook_size', 8192)
        image_start = text_vocab_size
        motion_start = text_vocab_size + image_codebook_size
        
        # print(f"DEBUG: Token ranges - Text: [0, {text_vocab_size-1}], Image: [{image_start}, {image_start + image_codebook_size - 1}], Motion: [{motion_start}, {motion_start + 512 - 1}]")
        
        if max_token_id >= vocab_size:
            print(f"WARNING: Found token ID {max_token_id} >= vocab_size {vocab_size}")
            # Find which tokens are out of bounds
            out_of_bounds = input_ids >= vocab_size
            if out_of_bounds.any():
                oob_ids = input_ids[out_of_bounds].unique()
                print(f"WARNING: Out-of-bounds token IDs: {oob_ids.tolist()}")
            
            # Clamp invalid token IDs to valid range
            input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
            print(f"Clamped input_ids to range [0, {vocab_size-1}]")
        
        if min_token_id < 0:
            print(f"WARNING: Found negative token ID {min_token_id}")
            input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
        
        # Validate mask_token_id
        if mask_token_id is not None and mask_token_id >= vocab_size:
            print(f"WARNING: mask_token_id {mask_token_id} >= vocab_size {vocab_size}")
            mask_token_id = vocab_size - 1  # Use last token as mask token
        elif mask_token_id is None:
            mask_token_id = getattr(self.config, 'mask_token_id', vocab_size - 1)
        
        # Validate labels
        if labels is not None:
            # Labels can have -100 (ignore index), so only check positive values
            valid_labels = labels[labels >= 0]
            if len(valid_labels) > 0:
                max_label_id = valid_labels.max().item()
                if max_label_id >= vocab_size:
                    print(f"WARNING: Found label ID {max_label_id} >= vocab_size {vocab_size}")
                    # Find which label tokens are out of bounds
                    out_of_bounds_labels = labels >= vocab_size
                    if out_of_bounds_labels.any():
                        oob_label_ids = labels[out_of_bounds_labels].unique()
                        print(f"WARNING: Out-of-bounds label IDs: {oob_label_ids.tolist()}")
                    labels = torch.where(labels >= vocab_size, torch.tensor(-100, device=labels.device), labels)
        
        return input_ids, labels, mask_token_id

    @torch.no_grad()
    def t2i_generate(
            self,
            input_ids: torch.LongTensor = None,
            uncond_input_ids: torch.LongTensor = None,
            attention_mask=None,
            uncond_attention_mask=None,
            temperature=1.0,
            timesteps=18,  # ideal number of steps is 18 in maskgit paper
            guidance_scale=0,
            noise_schedule=cosine_schedule,
            generator: torch.Generator = None,
            config=None,
            seq_len=1024,
            mask_token_id = 126336,
            resolution = 512,
            codebook_size = 8192,
            **kwargs,
    ):
        """
        Generate 1:1 similar to the original MaskGit repo
        https://github.com/google-research/maskgit/blob/main/maskgit/libml/parallel_decode.py#L79
        """

        # begin with all image token ids masked
        # 计算有多少个mask token
        mask_count = (input_ids == mask_token_id).sum().item()
        num_vq_tokens = seq_len
        # CRITICAL FIX: num_new_special_tokens should be 0 since special tokens use reserved IDs
        num_new_special_tokens = 0
        uni_prompting = kwargs.get("uni_prompting", None)
        # print(f"config.model.mmada.llm_vocab_size: {config.model.mmada.llm_vocab_size}, {len(uni_prompting.text_tokenizer)}")
        input_ids_minus_lm_vocab_size = input_ids[:, -(num_vq_tokens + 1):-1].clone()
        input_ids_minus_lm_vocab_size = torch.where(input_ids_minus_lm_vocab_size == mask_token_id, mask_token_id, input_ids_minus_lm_vocab_size - len(uni_prompting.text_tokenizer) - num_new_special_tokens)

        # for classifier-free guidance
        if uncond_input_ids is not None:
            uncond_prefix = uncond_input_ids[:, :resolution + 1]

        for step in range(timesteps):
            if uncond_input_ids is not None and guidance_scale > 0:
                uncond_input_ids = torch.cat(
                    [uncond_prefix, input_ids[:, resolution + 1:]], dim=1)
                model_input = torch.cat([input_ids, uncond_input_ids])
                attention_mask = torch.cat([attention_mask, uncond_attention_mask], dim=0)
                attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
                logits = self(model_input, attention_bias=attention_bias).logits 
                # print(f"logits.shape: {logits.shape}")
                cond_logits, uncond_logits = torch.chunk(logits, 2, dim=0)
                # logits = uncond_logits + guidance_scale * (cond_logits - uncond_logits)
                # it seems that muse has a different cfg setting
                logits = (1 + guidance_scale) * cond_logits - guidance_scale * uncond_logits
                logits = logits[:, -(num_vq_tokens + 1):-1, len(uni_prompting.text_tokenizer) + num_new_special_tokens: len(uni_prompting.text_tokenizer) + num_new_special_tokens + codebook_size]
            else:
                attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
                logits = self(input_ids, attention_bias=attention_bias).logits
                logits = logits[:, -(num_vq_tokens + 1):-1, len(uni_prompting.text_tokenizer) + num_new_special_tokens: len(uni_prompting.text_tokenizer) + num_new_special_tokens + codebook_size]

            # logits: 1, 1024, 8192
            # print(f"logits.shape: {logits.shape}")
            probs = logits.softmax(dim=-1)
            sampled = probs.reshape(-1, logits.size(-1))
            # print(f"probs: {probs}, probs.shape: {probs.shape}, sampled: {sampled}, sampled.shape: {sampled.shape}")
            sampled_ids = torch.multinomial(sampled, 1, generator=generator)[:, 0].view(*logits.shape[:-1]) # 1, 1024

            unknown_map = input_ids_minus_lm_vocab_size == mask_token_id
            # print(f"unknown_map.sum(dim=-1, keepdim=True): {unknown_map.sum(dim=-1, keepdim=True)}")
            sampled_ids = torch.where(unknown_map, sampled_ids, input_ids_minus_lm_vocab_size)
            # Defines the mask ratio for the next round. The number to mask out is
            # determined by mask_ratio * unknown_number_in_the_beginning.
            ratio = 1.0 * (step + 1) / timesteps
            mask_ratio = noise_schedule(torch.tensor(ratio))
            # Computes the probabilities of each selected tokens.
            selected_probs = torch.gather(probs, -1, sampled_ids.long()[..., None])
            selected_probs = selected_probs.squeeze(-1)

            # Ignores the tokens given in the input by overwriting their confidence.
            selected_probs = torch.where(unknown_map, selected_probs, torch.finfo(selected_probs.dtype).max)
            # Gets mask lens for each sample in the batch according to the mask ratio.
            mask_len = (num_vq_tokens * mask_ratio).floor().unsqueeze(0).to(logits.device)
            # Keeps at least one of prediction in this round and also masks out at least
            # one and for the next iteration
            mask_len = torch.max(
                torch.tensor([1], device=logits.device), torch.min(unknown_map.sum(dim=-1, keepdim=True) - 1, mask_len)
            )
            # print(f"mask_len: {mask_len}, mask_len.shape: {mask_len.shape}")
            # Adds noise for randomness
            temperature = temperature * (1.0 - ratio)
            masking = mask_by_random_topk(mask_len, selected_probs, temperature, generator=generator)
            # Masks tokens with lower confidence.
            input_ids[:, -(num_vq_tokens + 1):-1] = torch.where(masking, mask_token_id,
                                                          sampled_ids + len(uni_prompting.text_tokenizer)
                                                          + num_new_special_tokens)
            input_ids_minus_lm_vocab_size = torch.where(masking, mask_token_id, sampled_ids)

        return sampled_ids
    
    def forward_process(
            self,
            input_ids, 
            labels,
            batch_size_t2i=0,
            batch_size_lm=0,
            batch_size_mmu=0,
            batch_size_t2m=0,
            max_seq_length=128,
            p_mask_lm=None,
            p_mask_mmu=None,
            p_mask_t2m=None,
            answer_lengths=None,
            t2i_masks=None,
            answer_lengths_lm=None,
            answer_lengths_t2m=None
            ):
        # Validate token IDs to prevent CUDA indexing errors
        input_ids, labels, _ = self._validate_token_ids(input_ids, labels)
        
        # attention bias, True for batch_size, 1, seq_len, seq_len  
        attention_bias = torch.ones(input_ids.shape[0], 1, input_ids.shape[1], input_ids.shape[1])
        if t2i_masks is not None:
            attention_bias_t2i = (t2i_masks[:, :, None] & t2i_masks[:, None, :]).bool().unsqueeze(1)
            attention_bias[:batch_size_t2i] = attention_bias_t2i
        logits = self(input_ids, attention_bias=attention_bias).logits 
        self.output_size = logits.shape[-1]
        if batch_size_t2i == 0:
            loss_t2i = torch.tensor(0.0, device=input_ids.device)
        else:
            loss_t2i = F.cross_entropy(
                logits[:batch_size_t2i, max_seq_length + 1:].contiguous().view(-1, self.output_size),
                labels[:batch_size_t2i, max_seq_length + 1:].contiguous().view(-1), ignore_index=-100,
                )
        masked_indices = input_ids == self.config.mask_token_id 
        masked_indices_lm = masked_indices[batch_size_t2i:batch_size_t2i + batch_size_lm]
        masked_indices_mmu = masked_indices[batch_size_t2i + batch_size_lm:batch_size_t2i + batch_size_lm + batch_size_mmu]
        masked_indices_t2m = masked_indices[-batch_size_t2m:] if batch_size_t2m > 0 else None
        p_mask_lm = p_mask_lm.to(masked_indices_lm.device) if p_mask_lm is not None else None
        p_mask_mmu = p_mask_mmu.to(masked_indices_mmu.device) if p_mask_mmu is not None else None
        p_mask_t2m = p_mask_t2m.to(masked_indices_t2m.device) if (p_mask_t2m is not None and masked_indices_t2m is not None) else None
        answer_lengths = answer_lengths.to(masked_indices_mmu.device) if answer_lengths is not None else None
        loss_lm = torch.tensor(0.0, device=input_ids.device)
        if batch_size_lm > 0:
            loss_lm = F.cross_entropy(
                logits[batch_size_t2i:batch_size_t2i + batch_size_lm][masked_indices_lm].contiguous().view(-1, self.output_size),
                labels[batch_size_t2i:batch_size_t2i + batch_size_lm][masked_indices_lm].contiguous().view(-1), ignore_index=-100, reduction='none'
                )/p_mask_lm[masked_indices_lm]
            loss_lm = loss_lm.sum() / (logits[batch_size_t2i:batch_size_t2i + batch_size_lm].shape[0] * logits[batch_size_t2i:batch_size_t2i + batch_size_lm].shape[1])
            if answer_lengths_lm is not None:
                answer_lengths_lm = answer_lengths_lm.to(masked_indices_lm.device)
                loss_lm = torch.sum(loss_lm / answer_lengths_lm[masked_indices_lm]) / (logits[batch_size_t2i:batch_size_t2i + batch_size_lm].shape[0])  
        loss_mmu = torch.tensor(0.0, device=input_ids.device)
        if batch_size_mmu > 0:
            loss_mmu = F.cross_entropy(
                logits[-batch_size_mmu:][masked_indices_mmu].contiguous().view(-1, self.output_size),
                labels[-batch_size_mmu:][masked_indices_mmu].contiguous().view(-1), ignore_index=-100, reduction='none'
                )/p_mask_mmu[masked_indices_mmu]
            loss_mmu = torch.sum(loss_mmu/answer_lengths[masked_indices_mmu]) / (logits[-batch_size_mmu:].shape[0])
        loss_t2m = torch.tensor(0.0, device=input_ids.device)
        if batch_size_t2m > 0 and masked_indices_t2m is not None:
            loss_t2m = F.cross_entropy(
                logits[-batch_size_t2m:][masked_indices_t2m].contiguous().view(-1, self.output_size),
                labels[-batch_size_t2m:][masked_indices_t2m].contiguous().view(-1), ignore_index=-100, reduction='none'
                )
            if p_mask_t2m is not None:
                loss_t2m = loss_t2m / p_mask_t2m[masked_indices_t2m]
            if answer_lengths_t2m is not None:
                answer_lengths_t2m = answer_lengths_t2m.to(masked_indices_t2m.device)
                loss_t2m = torch.sum(loss_t2m / answer_lengths_t2m[masked_indices_t2m]) / (logits[-batch_size_t2m:].shape[0])
            else:
                loss_t2m = loss_t2m.mean()
        return logits, loss_t2i, loss_lm, loss_mmu, loss_t2m

    def forward_process_with_r2i(
            self,
            input_ids, 
            labels,
            t2i_masks=None,
            max_seq_length=128,
            batch_size_t2i=0,
            batch_size_lm=0,
            batch_size_mmu=0,
            batch_size_r2i=0,
            p_mask_lm=None,
            p_mask_mmu=None,
            p_mask_r2i=None,
            answer_lengths=None,
            answer_lengths_lm=None,
            answer_lengths_r2i=None,
            ):
        # Validate token IDs to prevent CUDA indexing errors
        input_ids, labels, _ = self._validate_token_ids(input_ids, labels)
        
        # attention bias, True for batch_size, 1, seq_len, seq_len  
        attention_bias = torch.ones(input_ids.shape[0], 1, input_ids.shape[1], input_ids.shape[1])
        attention_bias_t2i = (t2i_masks[:, :, None] & t2i_masks[:, None, :]).bool().unsqueeze(1)
        attention_bias[:batch_size_t2i] = attention_bias_t2i
        logits = self(input_ids, attention_bias=attention_bias).logits 
        # logits = self(input_ids).logits
        self.output_size = logits.shape[-1]

        # print(f"logits shape: {logits.shape}") B, 359, vocab_size

        if batch_size_t2i == 0:
            loss_t2i = torch.tensor(0.0, device=input_ids.device)
        else:
            # t2i loss
            loss_t2i = F.cross_entropy(
                logits[:batch_size_t2i, max_seq_length + 1:].contiguous().view(-1, self.output_size),
                labels[:batch_size_t2i, max_seq_length + 1:].contiguous().view(-1), ignore_index=-100,
                )
        
        # llada loss  

        start_lm = batch_size_t2i
        end_lm = start_lm + batch_size_lm
        start_mmu = end_lm
        end_mmu = start_mmu + batch_size_mmu
        start_r2i = end_mmu
        end_r2i = start_r2i + batch_size_r2i

        masked_indices = input_ids == self.config.mask_token_id 
        masked_indices_lm = masked_indices[start_lm:end_lm]
        masked_indices_mmu = masked_indices[start_mmu:end_mmu]
        masked_indices_r2i = masked_indices[start_r2i:end_r2i]

        p_mask_lm = p_mask_lm.to(masked_indices_lm.device)
        p_mask_mmu = p_mask_mmu.to(masked_indices_mmu.device)
        p_mask_r2i = p_mask_r2i.to(masked_indices_r2i.device)

        answer_lengths = answer_lengths.to(masked_indices_mmu.device) 
        answer_lengths_lm = answer_lengths_lm.to(masked_indices_lm.device)
        answer_lengths_r2i = answer_lengths_r2i.to(masked_indices_r2i.device)

        loss_lm = F.cross_entropy(
            logits[start_lm:end_lm][masked_indices_lm].contiguous().view(-1, self.output_size),
            labels[start_lm:end_lm][masked_indices_lm].contiguous().view(-1), ignore_index=-100, reduction='none'
            )/p_mask_lm[masked_indices_lm]
        # print(f"logits lm shape: {logits[batch_size_t2i:batch_size_t2i + batch_size_lm].shape}")
        loss_lm = loss_lm.sum() / (logits[start_lm:end_lm].shape[0] * logits[start_lm:end_lm].shape[1])
        loss_lm = torch.sum(loss_lm / answer_lengths_lm[masked_indices_lm]) / (logits[start_lm:end_lm].shape[0]) 

        loss_mmu = F.cross_entropy(
            logits[start_mmu:end_mmu][masked_indices_mmu].contiguous().view(-1, self.output_size),
            labels[start_mmu:end_mmu][masked_indices_mmu].contiguous().view(-1), ignore_index=-100, reduction='none'
            )/p_mask_mmu[masked_indices_mmu]
        loss_mmu = torch.sum(loss_mmu/answer_lengths[masked_indices_mmu]) / (logits[start_mmu:end_mmu].shape[0])
        
        loss_r2i = F.cross_entropy(
            logits[start_r2i:end_r2i][masked_indices_r2i].contiguous().view(-1, self.output_size),
            labels[start_r2i:end_r2i][masked_indices_r2i].contiguous().view(-1), ignore_index=-100, reduction='none'
            )/p_mask_r2i[masked_indices_r2i]
        loss_r2i = torch.sum(loss_r2i/answer_lengths_r2i[masked_indices_r2i]) / (logits[start_r2i:end_r2i].shape[0])
        
        return logits, loss_t2i, loss_lm, loss_mmu, loss_r2i


    def forward_t2i(
            self,
            input_ids, 
            labels,
            batch_size_t2i=0,
            max_seq_length=128,
            t2i_masks=None
            ):
        # Validate token IDs to prevent CUDA indexing errors
        input_ids, labels, _ = self._validate_token_ids(input_ids, labels)
        
        # attention bias, True for batch_size, 1, seq_len, seq_len  
        attention_bias = torch.ones(input_ids.shape[0], 1, input_ids.shape[1], input_ids.shape[1])
        attention_bias_t2i = (t2i_masks[:, :, None] & t2i_masks[:, None, :]).bool().unsqueeze(1)
        attention_bias[:batch_size_t2i] = attention_bias_t2i
        logits = self(input_ids, attention_bias=attention_bias).logits 
        # logits = self(input_ids).logits
        self.output_size = logits.shape[-1]

        # print(f"logits shape: {logits.shape}") B, 359, vocab_size

        loss_t2i = F.cross_entropy(
            logits[:batch_size_t2i, max_seq_length + 1:].contiguous().view(-1, self.output_size),
            labels[:batch_size_t2i, max_seq_length + 1:].contiguous().view(-1), ignore_index=-100,
            )
        
        return loss_t2i

    def forward_t2m(
            self,
            input_ids, 
            labels,
            attention_mask=None,
            mask_token_id=None,
            p_mask=None
            ):
        """
        Forward pass for text-to-motion training.
        Handles masked language modeling for motion token prediction given text context.
        """
        # Validate all token IDs to prevent CUDA indexing errors
        input_ids, labels, mask_token_id = self._validate_token_ids(input_ids, labels, mask_token_id)
        
        # Create attention bias from attention mask if provided
        if attention_mask is not None:
            attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
            logits = self(input_ids, attention_bias=attention_bias).logits
        else:
            logits = self(input_ids).logits
        
        self.output_size = logits.shape[-1]
        
        # Compute loss only on masked motion tokens
        masked_indices = (input_ids == mask_token_id)
        
        # Ensure we have some masked tokens
        if not masked_indices.any():
            print("WARNING: No masked tokens found for training")
            return torch.tensor(0.0, device=input_ids.device, requires_grad=True)
        
        # Cross-entropy loss on masked positions
        loss_t2m = F.cross_entropy(
            logits[masked_indices].view(-1, self.output_size),
            labels[masked_indices].view(-1),
            ignore_index=-100,
            reduction='mean'
        )
        
        # Account for masking probability if provided
        if p_mask is not None and masked_indices.sum() > 0:
            # Ensure p_mask is not zero to avoid division by zero
            p_mask_safe = torch.clamp(p_mask, min=1e-8)
            loss_t2m = loss_t2m / p_mask_safe
        
        return loss_t2m

    @torch.no_grad()
    def t2m_generate(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask=None,
            temperature=1.0,
            timesteps=18,
            noise_schedule=cosine_schedule,
            generator: torch.Generator = None,
            config=None,
            seq_len=256,
            mask_token_id=126336,
            motion_vocab_size=512,
            num_new_special_tokens=0,
            **kwargs,
    ):
        
        """
        Iterative text-to-motion generation using MaskGIT-style sampling.
        input_ids: (batch, seq_len) with masked motion tokens
        Returns: generated motion token ids (batch, motion_seq_len)
        """
        num_motion_tokens = seq_len
        uni_prompting = kwargs.get("uni_prompting", None)
        
        # Find where motion tokens start and end in the sequence
        som_token = uni_prompting.sptids_dict["<|som|>"].item() if uni_prompting else None
        eom_token = uni_prompting.sptids_dict["<|eom|>"].item() if uni_prompting else None
        
        # Find motion token positions
        motion_start_idx = None
        motion_end_idx = None
        
        if som_token is not None:
            som_positions = (input_ids == som_token).nonzero(as_tuple=True)
            if len(som_positions[1]) > 0:
                motion_start_idx = som_positions[1][0].item() + 1
        
        if eom_token is not None:
            eom_positions = (input_ids == eom_token).nonzero(as_tuple=True)
            if len(eom_positions[1]) > 0:
                motion_end_idx = eom_positions[1][0].item()
        
        # Fallback: assume motion tokens are at the end
        if motion_start_idx is None or motion_end_idx is None:
            motion_start_idx = input_ids.shape[1] - num_motion_tokens
            motion_end_idx = input_ids.shape[1]
        
        # Extract only the motion token region
        motion_tokens = input_ids[:, motion_start_idx:motion_end_idx].clone()
        
        # Initialize motion_tokens_local to track current state (already in offset space)
        motion_tokens_local = motion_tokens.clone()
        
        # Iterative generation using MaskGIT schedule
        for step in range(timesteps):
            # Forward pass through the model
            if attention_mask is not None:
                attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
                logits = self(input_ids, attention_bias=attention_bias).logits
            else:
                logits = self(input_ids).logits
            
            # CRITICAL FIX: Extract logits for motion tokens only, and motion vocab range only
            # Motion tokens are offset by text_vocab_size + image_codebook_size
            text_vocab_size = len(uni_prompting.text_tokenizer) if uni_prompting else 126000
            image_codebook_size = kwargs.get("image_codebook_size", 8192)
            
            # CRITICAL: The motion token start in vocabulary space should NOT include num_new_special_tokens
            # because special tokens use reserved IDs within base vocabulary (not additional space)
            motion_token_start = text_vocab_size + image_codebook_size
            motion_token_end = motion_token_start + motion_vocab_size
            motion_logits = logits[:, motion_start_idx:motion_end_idx, motion_token_start:motion_token_end]
            
            # Sample from the logits
            probs = motion_logits.softmax(dim=-1)
            sampled = probs.reshape(-1, motion_logits.size(-1))
            sampled_ids = torch.multinomial(sampled, 1, generator=generator)[:, 0].view(*motion_logits.shape[:-1])
            
            # Add correct motion token offset to sampled tokens for consistency with training
            motion_token_offset = text_vocab_size + image_codebook_size
            sampled_ids_offset = sampled_ids + motion_token_offset
            
            # Only update masked positions
            unknown_map = motion_tokens_local == mask_token_id
            sampled_ids_offset = torch.where(unknown_map, sampled_ids_offset, motion_tokens_local)
            
            # Update the motion tokens in input_ids with offset tokens
            input_ids[:, motion_start_idx:motion_end_idx] = sampled_ids_offset
            
            # If not the last step, apply MaskGIT masking for next iteration
            if step < timesteps - 1:
                ratio = 1.0 * (step + 1) / timesteps
                mask_ratio = noise_schedule(torch.tensor(ratio))
                
                # Get confidence scores for the sampled tokens
                selected_probs = torch.gather(probs, -1, sampled_ids.long()[..., None])
                selected_probs = selected_probs.squeeze(-1)
                
                # Don't mask tokens that were already given
                selected_probs = torch.where(unknown_map, selected_probs, torch.finfo(selected_probs.dtype).max)
                
                # Calculate how many tokens to mask
                mask_len = (num_motion_tokens * mask_ratio).floor().unsqueeze(0).to(motion_logits.device)
                mask_len = torch.max(
                    torch.tensor([1], device=motion_logits.device), 
                    torch.min(unknown_map.sum(dim=-1, keepdim=True) - 1, mask_len)
                )
                
                # Apply temperature and mask low-confidence tokens
                temperature_adj = temperature * (1.0 - ratio)
                masking = mask_by_random_topk(mask_len, selected_probs, temperature_adj, generator=generator)
                
                # Update local and global token representations
                motion_tokens_local = torch.where(masking, mask_token_id, sampled_ids_offset)
                input_ids[:, motion_start_idx:motion_end_idx] = torch.where(
                    masking, 
                    mask_token_id,
                    sampled_ids_offset  # Use offset tokens for consistency
                )
        
        # CRITICAL FIX: Return motion tokens in VQ-VAE space [0, motion_vocab_size)
        # Convert from offset space back to VQ-VAE space for decoding
        # The final sampled_ids are already in VQ-VAE range [0, motion_vocab_size-1]
        # since they were sampled from motion_logits which is indexed correctly
        return sampled_ids

    @torch.no_grad()
    def mmu_generate(self, idx=None, input_embeddings=None, max_new_tokens=128, steps=128,block_length=128, temperature=0.0, top_k=None, eot_token=None, cfg_scale=0.0, remasking='low_confidence', mask_id=126336, attention_mask=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """

        if attention_mask is not None and 0.0 in attention_mask:
            attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
            # print(f"attention_bias: {attention_bias}")
        else:
            attention_bias = None
        try:
            device = idx.device
        except:
            device = input_embeddings.device

        result = []
        batch_size = idx.shape[0]
        x = torch.full((batch_size, idx.shape[1] + max_new_tokens), mask_id, dtype=torch.long).to(self.device)
        x[:, :idx.shape[1]] = idx.clone()
        prompt_index = (x != mask_id)
        
        
        assert max_new_tokens % block_length == 0
        num_blocks = max_new_tokens // block_length

        assert steps % num_blocks == 0
        steps = steps // num_blocks
        
        # print(f"num_blocks: {num_blocks}, steps: {steps}")
        # num_transfer_tokens = get_num_transfer_tokens(prompt_index, steps)
        for num_block in range(num_blocks):
            block_mask_index = (x[:, idx.shape[1] + num_block * block_length: idx.shape[1] + (num_block + 1) * block_length:] == mask_id)
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
            # num_transfer_tokens = get_num_transfer_tokens(prompt_index, steps)
            # print(f"num_transfer_tokens: {num_transfer_tokens}, num_transfer_tokens.shape: {num_transfer_tokens.shape}")
            for i in range(steps):
                mask_index = (x == mask_id) 
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = self(x_).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = self(x, attention_bias=attention_bias).logits
                
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
                if remasking == 'low_confidence':
                    p = F.softmax(logits.to(torch.float64), dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
                elif remasking == 'random':
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                x0_p[:, idx.shape[1] + (num_block + 1) * block_length:] = -np.inf

                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)

                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for j in range(confidence.shape[0]):
                    _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                    transfer_index[j, select_index] = True
                x[transfer_index] = x0[transfer_index]
                
            
            # logits = logits[:, -1, :] / temperature
            # # optionally crop the logits to only the top k options
            # if top_k is not None:
            #     v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            #     logits[logits < v[:, [-1]]] = -float('Inf')
            # # apply softmax to convert logits to (normalized) probabilities
            # probs = F.softmax(logits, dim=-1)
            # # sample from the distribution
            # idx_next = torch.multinomial(probs, num_samples=1)
            # result.append(idx_next[0][0])
            # # append sampled index to the running sequence and continue
            # if self.config.w_clip_vit:
            #     idx_next_embeddings = self.mmada.model.embed_tokens(idx_next)
            #     input_embeddings = torch.cat([input_embeddings, idx_next_embeddings], dim=1)
            # else:
            #     idx = torch.cat((idx, idx_next), dim=1)

            # if eot_token is not None and idx_next.cpu() == eot_token:
            #     break

        return x

    @torch.no_grad()
    def mmu_generate_fast(self, idx=None, input_embeddings=None, max_new_tokens=128, steps=128,block_length=128, temperature=0.0, top_k=None, eot_token=None, cfg_scale=0.0, remasking='low_confidence', mask_id=126336, attention_mask=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """

        if attention_mask is not None and 0.0 in attention_mask:
            attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
            # print(f"attention_bias: {attention_bias}")
        else:
            attention_bias = None
        try:
            device = idx.device
        except:
            device = input_embeddings.device

        result = []
        batch_size = idx.shape[0]
        x = torch.full((batch_size, idx.shape[1] + max_new_tokens), mask_id, dtype=torch.long).to(self.device)
        x[:, :idx.shape[1]] = idx.clone()
        prompt_index = (x != mask_id)
        
        
        assert max_new_tokens % block_length == 0
        num_blocks = max_new_tokens // block_length

        assert steps % num_blocks == 0
        steps = steps // num_blocks
        
        for num_block in range(num_blocks):
            block_mask_index = (x[:, idx.shape[1] + num_block * block_length: idx.shape[1] + (num_block + 1) * block_length:] == mask_id)
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
            for i in range(steps):
                mask_index = (x == mask_id) 
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = self(x_).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = self(x, attention_bias=attention_bias).logits
                
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
                if remasking == 'low_confidence':
                    p = F.softmax(logits.to(torch.float64), dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
                elif remasking == 'random':
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                x0_p[:, idx.shape[1] + (num_block + 1) * block_length:] = -np.inf

                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)

                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for j in range(confidence.shape[0]):
                    _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                    transfer_index[j, select_index] = True
                x[transfer_index] = x0[transfer_index]
            if eot_token is not None:
                last_token_index_in_current_block = idx.shape[1] + (num_block + 1) * block_length - 1
                if last_token_index_in_current_block < x.shape[1]:
                    tokens_at_block_end = x[:, last_token_index_in_current_block]
                    if torch.all(tokens_at_block_end == eot_token):
                        break
        return x

    @torch.no_grad()
    def t2i_generate_decoding_stepwise(
            self,
            input_ids: torch.LongTensor = None,
            uncond_input_ids: torch.LongTensor = None,
            attention_mask=None,
            uncond_attention_mask=None,
            temperature=1.0,
            timesteps=18,  # ideal number of steps is 18 in maskgit paper
            guidance_scale=0,
            noise_schedule=cosine_schedule,
            generator: torch.Generator = None,
            config=None,
            seq_len=1024,
            mask_token_id = 126336,
            resolution = 512,
            codebook_size = 8192,
            vq_model = None,
            **kwargs,
    ):
        """
        Generate 1:1 similar to the original MaskGit repo
        https://github.com/google-research/maskgit/blob/main/maskgit/libml/parallel_decode.py#L79
        """

        # begin with all image token ids masked
        # 计算有多少个mask token
        mask_count = (input_ids == mask_token_id).sum().item()
        num_vq_tokens = seq_len
        # CRITICAL FIX: num_new_special_tokens should be 0 since special tokens use reserved IDs
        num_new_special_tokens = 0
        uni_prompting = kwargs.get("uni_prompting", None)
        # print(f"config.model.mmada.llm_vocab_size: {config.model.mmada.llm_vocab_size}, {len(uni_prompting.text_tokenizer)}")
        input_ids_minus_lm_vocab_size = input_ids[:, -(num_vq_tokens + 1):-1].clone()
        input_ids_minus_lm_vocab_size = torch.where(input_ids_minus_lm_vocab_size == mask_token_id, mask_token_id, input_ids_minus_lm_vocab_size - len(uni_prompting.text_tokenizer) - num_new_special_tokens)

        # for classifier-free guidance
        if uncond_input_ids is not None:
            uncond_prefix = uncond_input_ids[:, :resolution + 1]

        for step in range(timesteps):
            if uncond_input_ids is not None and guidance_scale > 0:
                uncond_input_ids = torch.cat(
                    [uncond_prefix, input_ids[:, resolution + 1:]], dim=1)
                model_input = torch.cat([input_ids, uncond_input_ids])
                attention_mask = torch.cat([attention_mask, uncond_attention_mask], dim=0)
                attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
                logits = self(model_input, attention_bias=attention_bias).logits 
                # print(f"logits.shape: {logits.shape}")
                cond_logits, uncond_logits = torch.chunk(logits, 2, dim=0)
                # logits = uncond_logits + guidance_scale * (cond_logits - uncond_logits)
                # it seems that muse has a different cfg setting
                logits = (1 + guidance_scale) * cond_logits - guidance_scale * uncond_logits
                logits = logits[:, -(num_vq_tokens + 1):-1, len(uni_prompting.text_tokenizer) + num_new_special_tokens: len(uni_prompting.text_tokenizer) + num_new_special_tokens + codebook_size]
            else:
                attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
                logits = self(input_ids, attention_bias=attention_bias).logits
                logits = logits[:, -(num_vq_tokens + 1):-1, len(uni_prompting.text_tokenizer) + num_new_special_tokens: len(uni_prompting.text_tokenizer) + num_new_special_tokens + codebook_size]

            # logits: 1, 1024, 8192
            # print(f"logits.shape: {logits.shape}")
            probs = logits.softmax(dim=-1)
            sampled = probs.reshape(-1, logits.size(-1))
            # print(f"probs: {probs}, probs.shape: {probs.shape}, sampled: {sampled}, sampled.shape: {sampled.shape}")
            sampled_ids = torch.multinomial(sampled, 1, generator=generator)[:, 0].view(*logits.shape[:-1]) # 1, 1024

            unknown_map = input_ids_minus_lm_vocab_size == mask_token_id
            # print(f"unknown_map.sum(dim=-1, keepdim=True): {unknown_map.sum(dim=-1, keepdim=True)}")
            sampled_ids = torch.where(unknown_map, sampled_ids, input_ids_minus_lm_vocab_size)
            # Defines the mask ratio for the next round. The number to mask out is
            current_image_vq_indices = sampled_ids.clone()
            # print(f"current_image_vq_indices: {current_image_vq_indices}")
            current_image_vq_indices = torch.clamp(current_image_vq_indices, 0, 8192 - 1)
            current_image = vq_model.decode_code(current_image_vq_indices)
            images = torch.clamp((current_image + 1.0) / 2.0, min=0.0, max=1.0)
            images *= 255.0
            images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            pil_images = Image.fromarray(images[0]) 
            yield pil_images, f"Step {step + 1}/{timesteps}"
            # determined by mask_ratio * unknown_number_in_the_beginning.
            ratio = 1.0 * (step + 1) / timesteps
            mask_ratio = noise_schedule(torch.tensor(ratio))
            # Computes the probabilities of each selected tokens.
            selected_probs = torch.gather(probs, -1, sampled_ids.long()[..., None])
            selected_probs = selected_probs.squeeze(-1)

            # Ignores the tokens given in the input by overwriting their confidence.
            selected_probs = torch.where(unknown_map, selected_probs, torch.finfo(selected_probs.dtype).max)
            # Gets mask lens for each sample in the batch according to the mask ratio.
            mask_len = (num_vq_tokens * mask_ratio).floor().unsqueeze(0).to(logits.device)
            # Keeps at least one of prediction in this round and also masks out at least
            # one and for the next iteration
            mask_len = torch.max(
                torch.tensor([1], device=logits.device), torch.min(unknown_map.sum(dim=-1, keepdim=True) - 1, mask_len)
            )
            # print(f"mask_len: {mask_len}, mask_len.shape: {mask_len.shape}")
            # Adds noise for randomness
            temperature = temperature * (1.0 - ratio)
            masking = mask_by_random_topk(mask_len, selected_probs, temperature, generator=generator)
            # Masks tokens with lower confidence.
            input_ids[:, -(num_vq_tokens + 1):-1] = torch.where(masking, mask_token_id,
                                                          sampled_ids + len(uni_prompting.text_tokenizer)
                                                          + num_new_special_tokens)
            input_ids_minus_lm_vocab_size = torch.where(masking, mask_token_id, sampled_ids)
            

        return sampled_ids

AutoConfig.register("mmada", MMadaConfig)
AutoModelForCausalLM.register(MMadaConfig, MMadaModelLM)
AutoModel.register(MMadaConfig, MMadaModelLM)
