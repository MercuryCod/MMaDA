#!/usr/bin/env python3
"""
Debug script for testing motion token vocabulary allocation in MMaDA.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from training.prompting_utils import UniversalPrompting

def test_vocab_allocation():
    """Test the vocabulary allocation for text, image, and motion tokens."""
    
    # Load config
    config = OmegaConf.load("configs/t2m_test.yaml")
    
    # Initialize tokenizer and prompting
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.mmada.tokenizer_path, padding_side="left"
    )
    
    uni_prompting = UniversalPrompting(
        tokenizer,
        max_text_len=config.dataset.preprocessing.max_seq_length,
        special_tokens=(
            "<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>",
            "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>", "<|t2m|>", 
            "<|som|>", "<|eom|>"
        ),
        ignore_id=-100,
        cond_dropout_prob=config.training.cond_dropout_prob,
        use_reserved_token=True,
    )
    
    # Calculate vocabulary allocation
    text_vocab_size = len(uni_prompting.text_tokenizer)
    image_codebook_size = config.model.mmada.image_codebook_size
    motion_vocab_size = config.model.motion_vq_model.nb_code
    
    print("=== VOCABULARY ALLOCATION TEST ===")
    print(f"Text tokens: [0, {text_vocab_size - 1}] (size: {text_vocab_size})")
    print(f"Image tokens: [{text_vocab_size}, {text_vocab_size + image_codebook_size - 1}] (size: {image_codebook_size})")
    print(f"Motion tokens: [{text_vocab_size + image_codebook_size}, {text_vocab_size + image_codebook_size + motion_vocab_size - 1}] (size: {motion_vocab_size})")
    print(f"Total calculated: {text_vocab_size + image_codebook_size + motion_vocab_size}")
    print(f"Config new_vocab_size: {config.model.mmada.new_vocab_size}")
    
    # Check if they match
    if text_vocab_size + image_codebook_size + motion_vocab_size == config.model.mmada.new_vocab_size:
        print("✅ Vocabulary allocation is correct!")
    else:
        print("❌ Vocabulary allocation mismatch!")
        print(f"   Expected: {config.model.mmada.new_vocab_size}")
        print(f"   Calculated: {text_vocab_size + image_codebook_size + motion_vocab_size}")
    
    # Test motion token offset
    dummy_motion_tokens = torch.randint(0, motion_vocab_size, (2, 10))  # Batch=2, seq=10
    print(f"\nOriginal motion tokens: {dummy_motion_tokens}")
    
    # Apply offset (as done in training)
    motion_token_offset = text_vocab_size + image_codebook_size
    offset_motion_tokens = dummy_motion_tokens + motion_token_offset
    print(f"Offset motion tokens: {offset_motion_tokens}")
    print(f"Motion token range: [{motion_token_offset}, {motion_token_offset + motion_vocab_size - 1}]")
    
    # Verify range
    min_val = offset_motion_tokens.min().item()
    max_val = offset_motion_tokens.max().item()
    expected_min = motion_token_offset
    expected_max = motion_token_offset + motion_vocab_size - 1
    
    if min_val >= expected_min and max_val <= expected_max:
        print("✅ Motion token offset range is correct!")
    else:
        print("❌ Motion token offset range is incorrect!")
        print(f"   Actual range: [{min_val}, {max_val}]")
        print(f"   Expected range: [{expected_min}, {expected_max}]")
    
    # Test special tokens and their allocation
    print(f"\n=== SPECIAL TOKENS TEST ===")
    special_token_ids = []
    for key, value in uni_prompting.sptids_dict.items():
        token_id = value.item()
        special_token_ids.append(token_id)
        print(f"  {key}: {token_id}")
    
    # Check if special tokens are within base LLM vocabulary
    max_special_id = max(special_token_ids)
    min_special_id = min(special_token_ids)
    
    print(f"\nSpecial token range: [{min_special_id}, {max_special_id}]")
    print(f"LLM vocab size: {config.model.mmada.llm_vocab_size}")
    
    if max_special_id < config.model.mmada.llm_vocab_size:
        print("✅ Special tokens are within base LLM vocabulary (no additional vocab needed)")
    else:
        print("⚠️  Some special tokens extend beyond base LLM vocabulary")
    
    # Test T2M prompting functionality
    print(f"\n=== T2M PROMPTING TEST ===")
    test_captions = ["a person is walking forward", "a dancer spins gracefully"]
    test_motion_tokens = torch.randint(0, motion_vocab_size, (2, 20)) + motion_token_offset  # Pre-offset
    test_labels = torch.randint(0, motion_vocab_size, (2, 20)) + motion_token_offset
    
    try:
        input_ids, attention_mask, labels = uni_prompting((test_captions, test_motion_tokens, test_labels), 't2m')
        print(f"✅ T2M prompting successful!")
        print(f"   Input shape: {input_ids.shape}")
        print(f"   Attention mask shape: {attention_mask.shape}")
        print(f"   Labels shape: {labels.shape}")
        
        # Check if special tokens are present
        t2m_token = uni_prompting.sptids_dict["<|t2m|>"].item()
        som_token = uni_prompting.sptids_dict["<|som|>"].item()
        eom_token = uni_prompting.sptids_dict["<|eom|>"].item()
        
        has_t2m = (input_ids == t2m_token).any()
        has_som = (input_ids == som_token).any()
        has_eom = (input_ids == eom_token).any()
        
        print(f"   Contains <|t2m|>: {has_t2m}")
        print(f"   Contains <|som|>: {has_som}")
        print(f"   Contains <|eom|>: {has_eom}")
        
        if has_t2m and has_som and has_eom:
            print("✅ All motion special tokens present in sequence!")
        else:
            print("⚠️  Some motion special tokens missing")
            
    except Exception as e:
        print(f"❌ T2M prompting failed: {e}")
    
    print("\n=== TEST COMPLETE ===")

if __name__ == "__main__":
    test_vocab_allocation() 