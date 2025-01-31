#!/bin/bash
# set -e

declare -A CONFIGS=(
  ["./eval_data/transformed_mmlupro"]="./full_precision_results/transformed_mmlupro_reward_results_test ./full_precision_figures"
)

MODELS=("mmlu_noaugs_llamabase_lora" "mmlu_noaugs_qwen_lora")
BATCH_NUM=4

for DATASET in "${!CONFIGS[@]}"; do

  IFS=' ' read -r OUTPUT_DIR METRIC_FILE_DIR <<< "${CONFIGS[$DATASET]}"
  
  for MODEL in "${MODELS[@]}"; do

    PIDS=()
    mkdir -p $OUTPUT_DIR/log/$MODEL

    # model loading test. download if not yet.
    /opt/conda/envs/prm/bin/python \
    search/get_rewards_reasoning_step_in_parallel.py \
    --example_file_path_dir="$DATASET" \
    --test_prm="$MODEL" \
    --output_dir="$OUTPUT_DIR" \
    --metric_file_dir="$METRIC_FILE_DIR" \
    --loading_test

    for BATCH_ID in {0..3}; do
      echo "Processing dataset $DATASET batch $(($BATCH_ID+1))/$BATCH_NUM with model: $MODEL"
      CUDA_VISIBLE_DEVICES=$BATCH_ID nohup \
      /opt/conda/envs/prm/bin/python -u search/get_rewards_reasoning_step_in_parallel.py \
      --example_file_path_dir="$DATASET" \
      --batch_id=$BATCH_ID \
      --batch_num=$BATCH_NUM \
      --test_prm="$MODEL" \
      --output_dir="$OUTPUT_DIR" \
      --metric_file_dir="$METRIC_FILE_DIR" \
      --do_not_calculate_metric \
      > $OUTPUT_DIR/log/$MODEL/$MODEL\_batch_$BATCH_ID.log 2>&1 &
      PIDS+=($!)
    done

    for PID_EVAL in "${PIDS[@]}"; do
        wait $PID_EVAL
    done
    echo "Finished processing dataset $DATASET with model: $MODEL"

    # merge the batch outputs
    /opt/conda/envs/prm/bin/python \
    batch_outputs_merge.py \
    --model="$MODEL" \
    --batch_outputs_dir="$OUTPUT_DIR"
    echo "Finished merging batch outputs of model: $MODEL"

    # calculate metric
    METRIC_SAVE_DIR="/mnt/data/results_by_category"
    mkdir -p $METRIC_SAVE_DIR/log/$MODEL
    nohup \
    /opt/conda/envs/prm/bin/python -u calculate_metric_by_category.py \
    --results_dir=$OUTPUT_DIR \
    --save_dir=$METRIC_SAVE_DIR \
    --model=$MODEL \
    --ignore="./mmlu_overlap.json" \
    > $METRIC_SAVE_DIR/log/$MODEL/metric_on_${DATASET##*/}.log 2>&1 &

  done

done

echo "All tasks completed successfully."