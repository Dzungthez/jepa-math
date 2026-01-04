#!/bin/bash

# Step-JEPA Training Script (Full Options)
# This version supports all training configurations

set -e

cd "$(dirname "$0")"

echo "=========================================="
echo "Step-JEPA Training (Full Options)"
echo "=========================================="
echo ""
echo "This script supports:"
echo "  - Regular mode (no JEPA)"
echo "  - Step-JEPA mode (Step 1 ↔ Step 2)"
echo "  - Variable K predictor tokens (1, 2, 3, ...)"
echo "  - Custom seeds"
echo "  - Last token offset"
echo "  - Loss types (cosine, L2, MSE, InfoNCE)"
echo ""

# =============================================================================
# CONFIGURATION
# =============================================================================

# Data
TRAIN_FILE="datasets/gsm8k_step_jepa.jsonl"
OUTPUT_DIR="./checkpoints_adapted"

# Model
MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"

# Training hyperparameters
NUM_EPOCHS=3
LEARNING_RATE=1e-5
BATCH_SIZE=4
GRAD_ACCUM=4
MAX_LENGTH=2048

# Mode selection (choose ONE)
MODE="step_jepa"  # Options: "regular" or "step_jepa"

# Step-JEPA parameters (only used if MODE == "step_jepa")
LBD=0.5              # Lambda (JEPA loss weight)
GAMMA=1.0            # Gamma (LM loss weight)
PREDICTORS=2         # K predictor tokens after Step 1
LAST_TOKEN=-1        # Last token offset for embedding extraction
JEPA_RATIO=-1.0      # JEPA dropout ratio: -1.0=always use JEPA, 0.5=50% chance to skip JEPA, 1.0=never skip JEPA

# Loss type (choose ONE)
# LOSS_TYPE="cosine"   # Default: 1 - cosine_similarity
# LOSS_TYPE="l2"       # L2 norm
# LOSS_TYPE="mse"      # Mean squared error
# LOSS_TYPE="infonce"  # InfoNCE loss
LOSS_TYPE="cosine"

# LoRA settings
USE_LORA=0
LORA_RANK=16

# Random seed
SEED=42

# Debug level (5 = show losses during training)
DEBUG=5

# =============================================================================
# BUILD COMMAND
# =============================================================================

CMD="python train.py \
  --train_file=$TRAIN_FILE \
  --output_dir=$OUTPUT_DIR \
  --model_name=$MODEL_NAME \
  --num_epochs=$NUM_EPOCHS \
  --learning_rate=$LEARNING_RATE \
  --batch_size=$BATCH_SIZE \
  --grad_accum=$GRAD_ACCUM \
  --max_length=$MAX_LENGTH \
  --seed=$SEED \
  --debug=$DEBUG"

# Add mode-specific flags
if [ "$MODE" == "regular" ]; then
    echo "Mode: Regular (No JEPA)"
    CMD="$CMD --regular"
elif [ "$MODE" == "step_jepa" ]; then
    echo "Mode: Step-JEPA (Step 1 ↔ Step 2)"
    echo "  K predictor tokens: $PREDICTORS"
    echo "  JEPA ratio: $JEPA_RATIO (probability to apply JEPA)"
    CMD="$CMD --step_jepa --predictors=$PREDICTORS --lbd=$LBD --gamma=$GAMMA --last_token=$LAST_TOKEN --jepa_ratio=$JEPA_RATIO"
else
    echo "❌ Error: Invalid MODE=$MODE. Must be 'regular' or 'step_jepa'"
    exit 1
fi

# Add loss type flags
if [ "$LOSS_TYPE" == "l2" ]; then
    CMD="$CMD --jepa_l2"
elif [ "$LOSS_TYPE" == "mse" ]; then
    CMD="$CMD --jepa_mse"
elif [ "$LOSS_TYPE" == "infonce" ]; then
    CMD="$CMD --infonce"
fi

# Add LoRA if enabled
if [ "$USE_LORA" == 1 ]; then
    CMD="$CMD --lora --lora_rank=$LORA_RANK"
fi

echo ""
echo "Command:"
echo "$CMD"
echo ""

# Execute
$CMD

# =============================================================================
# EXAMPLES OF OTHER CONFIGURATIONS
# =============================================================================

# Example 1: Regular training (no JEPA)
# MODE="regular"

# Example 2: Original JEPA with 2 predictor tokens
# MODE="jepa"
# PREDICTORS=2
# LBD=0.5

# Example 3: Step-JEPA with L2 loss
# MODE="step_jepa"
# LOSS_TYPE="l2"
# LBD=0.2

# Example 4: Step-JEPA with different last_token (for different models)
# MODE="step_jepa"
# LAST_TOKEN=-2  # For Gemma models
# LAST_TOKEN=-3  # For Qwen models
# LAST_TOKEN=-4  # For OpenELM models

# Example 5: Multiple seeds experiment (like run.sh)
# for seed in 82 23 37 84 4; do
#     for lbd in 0.5 2.0; do
#         for predictors in 0 1 2; do
#             # Run training with these parameters
#         done
#     done
# done

