#!/bin/bash

DATASET=transformed_mmlupro
OUTPUT_DIR=results_backup/full_precision_results/$DATASET\_reward_results
METRIC_SAVE_DIR=results_backup/results_by_category/$DATASET

echo "calculating metric for $DATASET"
echo "output dir: $OUTPUT_DIR"
echo "save dir: $METRIC_SAVE_DIR"

PIDS=()
for MODEL in "mmlu_noaugs_llama_lora" "prm800k_llama_fulltune" "qwen2.5_math_7b_prm800k"; do
    echo "running $MODEL"
    mkdir -p $METRIC_SAVE_DIR/log/$MODEL
    nohup \
    python -u calculate_metric_by_category.py \
    --results_dir=$OUTPUT_DIR \
    --save_dir=$METRIC_SAVE_DIR \
    --dataset=$DATASET \
    --model=$MODEL \
    --ignore="./mmlu_overlap.json" \
    --N_max=128 \
    > $METRIC_SAVE_DIR/log/$MODEL/metric_on_${DATASET##*/}.log 2>&1 &
    PIDS+=($!)
done
for PID in "${PIDS[@]}"; do
        wait $PID
done
echo "all models done"

# PIDS=()
# for MODEL in "mmlu_noaugs_llama_lora" "mmlu_onlyaugs_llama_lora" "mmlu_small_noaugs_llama_lora" "prm800k_llama_fulltune"; do
#     echo "running $MODEL"
#     mkdir -p $METRIC_SAVE_DIR/log/$MODEL
#     nohup \
#     /opt/conda/envs/prm/bin/python -u calculate_metric_by_category.py \
#     --results_dir=$OUTPUT_DIR \
#     --save_dir=$METRIC_SAVE_DIR \
#     --model=$MODEL \
#     --ignore="./mmlu_overlap.json" \
#     > $METRIC_SAVE_DIR/log/$MODEL/metric_on_${DATASET##*/}.log 2>&1 &
#     PIDS+=($!)
# done
# for PID in "${PIDS[@]}"; do
#         wait $PID
# done

# # PIDS=()
# # for MODEL in "sciqqa_augs_llama_lora_rewards" "sciqqa_noaugs_llama_lora" "sciqqa_noaugs_masked_qwen_lora" "sciqqa_noaugs_qwen_lora"; do
# #     echo "running $MODEL"
# #     mkdir -p $METRIC_SAVE_DIR/log/$MODEL
# #     nohup \
# #     /opt/conda/envs/prm/bin/python -u calculate_metric_by_category.py \
# #     --results_dir=$OUTPUT_DIR \
# #     --save_dir=$METRIC_SAVE_DIR \
# #     --model=$MODEL \
# #     --ignore="./mmlu_overlap.json" \
# #     > $METRIC_SAVE_DIR/log/$MODEL/metric_on_${DATASET##*/}.log 2>&1 &
# #     PIDS+=($!)
# # done
# # for PID in "${PIDS[@]}"; do
# #         wait $PID
# # done

# # PIDS=()
# # for MODEL in "sciqqa_onlyaugs_llama_lora" "v4_llama_lora" "v4_qwen_lora" "v5_llama_lora"; do
# #     echo "running $MODEL"
# #     mkdir -p $METRIC_SAVE_DIR/log/$MODEL
# #     nohup \
# #     /opt/conda/envs/prm/bin/python -u calculate_metric_by_category.py \
# #     --results_dir=$OUTPUT_DIR \
# #     --save_dir=$METRIC_SAVE_DIR \
# #     --model=$MODEL \
# #     --ignore="./mmlu_overlap.json" \
# #     > $METRIC_SAVE_DIR/log/$MODEL/metric_on_${DATASET##*/}.log 2>&1 &
# #     PIDS+=($!)
# # done
# # for PID in "${PIDS[@]}"; do
# #         wait $PID
# # done

# # PIDS=()
# # for MODEL in "v5_qwen_lora" "v6_llama_lora" "v7_noaugs_llama_lora" "v7_noaugs_qwen_lora", "v7_onlyaugs_llama_lora"; do
# #     echo "running $MODEL"
# #     mkdir -p $METRIC_SAVE_DIR/log/$MODEL
# #     nohup \
# #     /opt/conda/envs/prm/bin/python -u calculate_metric_by_category.py \
# #     --results_dir=$OUTPUT_DIR \
# #     --save_dir=$METRIC_SAVE_DIR \
# #     --model=$MODEL \
# #     --ignore="./mmlu_overlap.json" \
# #     > $METRIC_SAVE_DIR/log/$MODEL/metric_on_${DATASET##*/}.log 2>&1 &
# #     PIDS+=($!)
# # done
# # for PID in "${PIDS[@]}"; do
# #         wait $PID
# # done

# # tar -czvf /mnt/data/$DATASET\_by_category.tar.gz .