#!/bin/bash

#################################################
# Test Script for MMaDA T2M LoRA Training
#################################################

# Set environment variables
export PYTHONPATH=/homes/55/junlin/MMaDA:$PYTHONPATH
export TRITON_CACHE_DIR=/tmp/triton_cache_$USER
export CUDA_VISIBLE_DEVICES=0  # Use only first GPU for testing

# Create cache directory
mkdir -p $TRITON_CACHE_DIR

# Configuration
CONFIG_FILE="configs/t2m_lora_test.yaml"
ACCELERATE_CONFIG="accelerate_configs/1_gpu.yaml"
PORT=29403
LOG_FILE="test_run_$(date +%Y%m%d_%H%M%S).log"

echo "=========================================="
echo "MMaDA LoRA Test Run"
echo "=========================================="
echo "Config: $CONFIG_FILE"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Accelerate Config: $ACCELERATE_CONFIG"
echo "Port: $PORT"
echo "Log file: $LOG_FILE"
echo "=========================================="
echo ""
echo "This will run for 100 steps to verify setup..."
echo ""

# Run test
/homes/55/junlin/miniconda3/envs/mmada/bin/accelerate launch \
    --config_file $ACCELERATE_CONFIG \
    --main_process_port $PORT \
    training/train_t2m_lora.py \
    config=$CONFIG_FILE \
    2>&1 | tee $LOG_FILE

# Check exit status
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo "✅ Test run completed successfully!"
    echo ""
    echo "Check the following:"
    echo "- Log file: $LOG_FILE"
    echo "- Output directory: test_lora_output/"
    echo "- Generated samples (if any)"
    echo ""
    echo "If everything looks good, run full training with: bash run.sh"
else
    echo ""
    echo "❌ Test run failed. Check $LOG_FILE for errors."
    echo ""
    # Show last 20 lines of error
    echo "Last 20 lines of log:"
    echo "===================="
    tail -20 $LOG_FILE
    exit 1
fi
