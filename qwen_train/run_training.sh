#!/bin/bash

# Export the environment variable
# export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# Run the accelerate launch command
# accelerate launch train_script.py -c train_configs/debug_qwen.yml
accelerate launch train_script.py -c train_configs/prm800k_qwen_alt_lora.yml

