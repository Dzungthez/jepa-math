#!/bin/bash

# Train with unmask_user and num_prediction_steps=100
# Using gsm8k_step_jepa.jsonl dataset

python train.py \
    --train_file datasets/gsm8k_step_jepa.jsonl \
    --model_name meta-llama/Llama-3.2-1B-Instruct \
    --output_dir ./checkpoints_unmask_user \
    --max_length 2048 \
    --batch_size 4 \
    --grad_accum 4 \
    --learning_rate 1e-5 \
    --num_epochs 3 \
    --eval_steps 50 \
    --predictors 4 \
    --lbd 0.5 \
    --gamma 1.0 \
    --unmask_user \
    --num_prediction_steps 100 \
    --seed 42 \
    --debug 5

