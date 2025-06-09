#!/bin/bash

#################################################
# Full LoRA Training Script for MMaDA T2M
#################################################

# Set environment variables
export PYTHONPATH=/homes/55/junlin/MMaDA:$PYTHONPATH
export TRITON_CACHE_DIR=/tmp/triton_cache_$USER
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # Use all 8 GPUs

# Create cache directory
mkdir -p $TRITON_CACHE_DIR

# Configuration
CONFIG_FILE="configs/t2m_instruct_lora.yaml"
ACCELERATE_CONFIG="accelerate_configs/1_node_8_gpus_deepspeed_zero3.yaml"
PORT=29401

echo "=========================================="
echo "MMaDA LoRA Training for Text-to-Motion"
echo "=========================================="
echo "Config: $CONFIG_FILE"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Accelerate Config: $ACCELERATE_CONFIG"
echo "Port: $PORT"
echo "=========================================="

# Run training
/homes/55/junlin/miniconda3/envs/mmada/bin/accelerate launch \
    --config_file $ACCELERATE_CONFIG \
    --main_process_port $PORT \
    training/train_t2m_lora.py \
    config=$CONFIG_FILE

# Check exit status
if [ $? -eq 0 ]; then
    echo "✅ Training completed successfully!"
else
    echo "❌ Training failed. Check the logs for details."
    exit 1
fi

# accelerate launch --config_file accelerate_configs/test_run.yaml --main_process_port=8888 training/train_t2m.py config=configs/t2m_test.yaml
