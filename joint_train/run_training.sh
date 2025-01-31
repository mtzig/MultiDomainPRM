#!/bin/bash

# Export the environment variable
# export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0


# Run the accelerate launch command
accelerate launch train_script.py -c train_configs/prm800k_qwen_lora.yml
