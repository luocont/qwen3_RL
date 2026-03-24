#!/bin/bash
# 启动 GRPO 训练脚本

cd "$(dirname "$0")"

python data/generate_dataset.py \
    --output_path data/counseling_dataset.json

python train_grpo.py \
    --model_name_or_path "Qwen/Qwen3-8B" \
    --dataset_path "data/counseling_dataset.json" \
    --output_dir "outputs/grpo_counselor" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-6 \
    --num_generations 4 \
    --max_prompt_length 2048 \
    --max_completion_length 1024 \
    --max_dialogue_turns 12 \
    --panas_weight 0.7 \
    --format_weight 0.3 \
    --agent_api_base "http://localhost:8000/v1" \
    --agent_api_key "EMPTY" \
    --five_ps_model "qwen2.5-72b-instruct" \
    --counselor_model "qwen2.5-72b-instruct" \
    --client_model "qwen2.5-72b-instruct" \
    --panas_model "qwen2.5-72b-instruct" \
    --save_best_samples True \
    --best_samples_path "outputs/best_samples.jsonl"
