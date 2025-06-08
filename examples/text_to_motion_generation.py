#!/usr/bin/env python3
"""
Text-to-Motion Generation Example

This script demonstrates how to use the MMaDA model for text-to-motion generation.
It shows both basic generation and advanced generation with classifier-free guidance.
"""

import torch
from transformers import AutoTokenizer
from models.modeling_mmada import MMadaModelLM, MMadaConfig
from utils.motion_process import recover_from_ric  # Assuming this exists
import numpy as np


def load_model_and_tokenizer(model_path: str, device: str = "cuda"):
    """Load the MMaDA model and tokenizer."""
    config = MMadaConfig.from_pretrained(model_path)
    model = MMadaModelLM.from_pretrained(model_path, config=config)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    model.to(device)
    model.eval()
    
    return model, tokenizer, config


def basic_text_to_motion_generation():
    """Basic example of text-to-motion generation."""
    
    # Load model and tokenizer
    model_path = "path/to/your/mmada/model"  # Replace with actual path
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model, tokenizer, config = load_model_and_tokenizer(model_path, device)
    
    # Example text descriptions
    text_descriptions = [
        "A person walks forward slowly",
        "Someone jumps up and down",
        "A person waves their hand",
        "Walking in a circle"
    ]
    
    # Configuration for motion generation
    motion_vocab_start = config.llm_vocab_size + config.num_new_special_tokens
    motion_vocab_size = config.codebook_size
    motion_sequence_length = 196  # Adjust based on your motion representation
    
    print("Generating motion from text descriptions...")
    
    # Generate motion tokens
    motion_tokens = model.generate_motion_from_text(
        text_descriptions=text_descriptions,
        tokenizer=tokenizer,
        motion_vocab_start=motion_vocab_start,
        motion_vocab_size=motion_vocab_size,
        motion_sequence_length=motion_sequence_length,
        timesteps=18,
        temperature=1.0,
        guidance_scale=2.0,
        use_cfg=True,  # Use classifier-free guidance for better quality
        device=device
    )
    
    print(f"Generated motion tokens shape: {motion_tokens.shape}")
    print(f"Sample motion tokens: {motion_tokens[0][:10]}")  # First 10 tokens of first example
    
    return motion_tokens


def advanced_text_to_motion_generation():
    """Advanced example with custom parameters and post-processing."""
    
    # Load model and tokenizer
    model_path = "path/to/your/mmada/model"  # Replace with actual path
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model, tokenizer, config = load_model_and_tokenizer(model_path, device)
    
    # Single text description for detailed analysis
    text_description = "A person performs a graceful dance movement"
    
    # Tokenize the text manually for more control
    text_tokens = tokenizer(
        text_description,
        max_length=77,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(device)
    
    # Configuration
    motion_vocab_start = config.llm_vocab_size + config.num_new_special_tokens
    motion_vocab_size = config.codebook_size
    motion_sequence_length = 196
    
    print(f"Generating motion for: '{text_description}'")
    
    # Generate with different parameters
    with torch.no_grad():
        # High-quality generation with CFG
        motion_tokens_high_quality = model.t2m_generate_with_cfg(
            input_ids=text_tokens.input_ids,
            uncond_input_ids=tokenizer("", max_length=77, padding=True, 
                                     truncation=True, return_tensors="pt").input_ids.to(device),
            attention_mask=text_tokens.attention_mask,
            uncond_attention_mask=torch.ones_like(text_tokens.attention_mask),
            seq_len=motion_sequence_length,
            motion_vocab_start=motion_vocab_start,
            motion_vocab_size=motion_vocab_size,
            timesteps=18,
            temperature=1.0,
            guidance_scale=3.0  # Higher guidance for more text-faithful generation
        )
        
        # Fast generation without CFG
        motion_tokens_fast = model.t2m_generate(
            input_ids=text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            seq_len=motion_sequence_length,
            motion_vocab_start=motion_vocab_start,
            motion_vocab_size=motion_vocab_size,
            timesteps=12,  # Fewer steps for faster generation
            temperature=0.8
        )
    
    print(f"High-quality motion tokens shape: {motion_tokens_high_quality.shape}")
    print(f"Fast generation tokens shape: {motion_tokens_fast.shape}")
    
    return motion_tokens_high_quality, motion_tokens_fast


def convert_tokens_to_motion(motion_tokens, vq_model, motion_dataset):
    """
    Convert motion tokens back to 3D motion coordinates.
    
    Args:
        motion_tokens: Generated motion token indices
        vq_model: VQ-VAE model for decoding tokens
        motion_dataset: Dataset with inv_transform method
    
    Returns:
        3D motion coordinates
    """
    # This is a placeholder - you'll need to implement this based on your setup
    # The exact implementation depends on your VQ-VAE model and motion representation
    
    # Example workflow:
    # 1. Use VQ-VAE decoder to convert tokens to motion features
    # 2. Apply inverse normalization using dataset.inv_transform
    # 3. Convert to 3D coordinates using recover_from_ric
    
    with torch.no_grad():
        # Decode motion tokens using VQ-VAE decoder
        motion_features = vq_model.forward_decoder(motion_tokens)
        
        # Apply inverse transformation to denormalize
        motion_denorm = motion_dataset.inv_transform(motion_features.detach().cpu().numpy())
        
        # Convert to 3D joint coordinates
        num_joints = 21 if motion_features.shape[-1] == 251 else 22
        motion_3d = recover_from_ric(torch.from_numpy(motion_denorm).float(), num_joints)
        
    return motion_3d


def batch_generation_example():
    """Example of generating motion for multiple texts efficiently."""
    
    model_path = "path/to/your/mmada/model"  # Replace with actual path
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model, tokenizer, config = load_model_and_tokenizer(model_path, device)
    
    # Large batch of text descriptions
    text_descriptions = [
        "A person walks forward",
        "Someone runs quickly",
        "A person jumps high",
        "Walking sideways",
        "A person sits down",
        "Someone stands up",
        "A person claps hands",
        "Walking backwards"
    ]
    
    # Configuration
    motion_vocab_start = config.llm_vocab_size + config.num_new_special_tokens
    motion_vocab_size = config.codebook_size
    
    print(f"Generating motion for {len(text_descriptions)} descriptions...")
    
    # Generate in batches to manage memory
    batch_size = 4
    all_motion_tokens = []
    
    for i in range(0, len(text_descriptions), batch_size):
        batch_texts = text_descriptions[i:i + batch_size]
        
        motion_tokens = model.generate_motion_from_text(
            text_descriptions=batch_texts,
            tokenizer=tokenizer,
            motion_vocab_start=motion_vocab_start,
            motion_vocab_size=motion_vocab_size,
            motion_sequence_length=196,
            timesteps=18,
            temperature=1.0,
            guidance_scale=2.0,
            use_cfg=True,
            device=device
        )
        
        all_motion_tokens.append(motion_tokens)
        print(f"Generated batch {i//batch_size + 1}/{(len(text_descriptions) + batch_size - 1)//batch_size}")
    
    # Combine all batches
    final_motion_tokens = torch.cat(all_motion_tokens, dim=0)
    print(f"Final motion tokens shape: {final_motion_tokens.shape}")
    
    return final_motion_tokens


if __name__ == "__main__":
    print("=== Text-to-Motion Generation Examples ===")
    
    try:
        print("\n1. Basic Text-to-Motion Generation:")
        basic_motion_tokens = basic_text_to_motion_generation()
        
        print("\n2. Advanced Text-to-Motion Generation:")
        high_quality_tokens, fast_tokens = advanced_text_to_motion_generation()
        
        print("\n3. Batch Generation Example:")
        batch_tokens = batch_generation_example()
        
        print("\n✅ All examples completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure to:")
        print("1. Replace 'path/to/your/mmada/model' with actual model path")
        print("2. Ensure all dependencies are installed")
        print("3. Check that the model configuration matches your setup") 