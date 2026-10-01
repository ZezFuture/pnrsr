#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Environment
# ============================================================
export CUDA_VISIBLE_DEVICES=0,1,2
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# ============================================================
# Auto distributed config
# gradient_accumulation_steps * GPU_NUM = 12
# ============================================================
TARGET_GLOBAL_ACCUM=12

IFS=',' read -ra GPU_LIST <<< "${CUDA_VISIBLE_DEVICES}"
GPU_NUM=${#GPU_LIST[@]}

if (( TARGET_GLOBAL_ACCUM % GPU_NUM != 0 )); then
  echo "ERROR: TARGET_GLOBAL_ACCUM=${TARGET_GLOBAL_ACCUM} cannot be divided by GPU_NUM=${GPU_NUM}"
  exit 1
fi

GRAD_ACCUM_STEPS=$((TARGET_GLOBAL_ACCUM / GPU_NUM))

echo "GPU_NUM=${GPU_NUM}"
echo "GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS}"
echo "GPU_NUM * GRAD_ACCUM_STEPS = $((GPU_NUM * GRAD_ACCUM_STEPS))"

# ============================================================
# Paths: modify these paths before training
# ============================================================
TRAIN_PY="pnrsr/train_co.py"
PRETRAINED_PATH="path/to/save/Z-Image_warm_up" # Path to the Z-Image weights with the SR LoRA merged

SFT_CKPT_PATH="path/to/your/sr_lora_checkpoint" # Path to the warm-up training checkpoint
ENCODER_LORA_PATH=""
if [[ -n "${SFT_CKPT_PATH}" ]]; then
  ENCODER_LORA_PATH="${SFT_CKPT_PATH}/vae_encoder_lora"
fi

REWARD_MODEL_PATH="siglip2-so400m-patch16-512"
LR_DIR="path/to/your/train/lr_images" # Path to the LR training images
HR_DIR="path/to/your/train/hr_images" # Path to the HR training images
PROMPT_DIR="path/to/your/train/prompt" # Path to the training prompts; can be empty

OUTPUT_DIR="exp/output"

# Z-Image: t=0 is the clean endpoint; a larger t is a harder start.
T_VALUES="0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.64, 0.66, 0.68,
0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82, 0.84, 0.86, 0.88,
0.90, 0.92, 0.94, 0.96, 0.98"

mkdir -p "${OUTPUT_DIR}"

# Build arguments conditionally so an empty optional checkpoint is not passed.
ARGS=(
  --reward_bound_target 0.98
  --rankrandom
  --reward_bound_absolute_weight 0.25
  --pretrained_path "${PRETRAINED_PATH}"
  --reward_model_name "${REWARD_MODEL_PATH}"
  --reward_ckpt_path "${REWARD_CKPT_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --lr_dir "${LR_DIR}"
  --hr_dir "${HR_DIR}"
  --prompt_dir "${PROMPT_DIR}"

  --t_values "${T_VALUES}"
  --default_infer_timestep 500

  --flow_loss_weight 0
  --clean_loss_weight 1.0
  --lpips_loss_weight 2.0
  --fdl_loss_weight 0.005
  --lpips_net vgg

  --lora_rank 64
  --lora_alpha 128
  --lora_dropout 0.0
  --encoder_lora_rank 64
  --encoder_lora_alpha 128
  --encoder_lora_dropout 0.0

  --reward_image_size 512
  --reward_hidden_layers=-1,-3,-6,-9
  --reward_token_pool_size 32
  --reward_attn_dim 512
  --reward_num_heads 8
  --reward_num_fusion_layers 1
  --reward_sr_quality_layers 1
  --reward_head_hidden_dim 1024
  --reward_dropout 0

  --lambda_reward 0.1
  --reward_guidance_start_step 500
  --lambda_reward_ramp_steps 1000
  --reward_updates_per_sr_update 1

  --reward_score_l2_weight 0
  --reward_margin_reg_weight 5e-2
  --reward_max_margin 2

  --train_batch_size 1
  --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}"
  --num_training_epochs 100
  --max_train_steps 30000
  --mixed_precision bf16
  --seed 12345

  --learning_rate_sr_lora 2e-5
  --learning_rate_encoder_lora 2e-5
  --reward_learning_rate 2e-5
  --weight_decay 1e-2
  --reward_weight_decay 1e-2
  --beta1 0.9
  --beta2 0.999
  --adam_epsilon 1e-8
  --max_grad_norm 1.0
  --reward_max_grad_norm 3.0

  --dataloader_num_workers 4
  --report_to tensorboard
  --tracker_project_name zimage
  --checkpointing_steps 500
  --empty_cache_steps 500

  --allow_tf32
  --set_grads_to_none
  --reward_upper_dir "calibration/data/good"
  --reward_lower_dir "calibration/data/bad"
  --reward_bound_lr_dir "calibration/data/lr"

  --reward_bound_gap_weight 0
  --reward_bound_center_weight 0
)

if [[ -n "${ENCODER_LORA_PATH}" ]]; then
  ARGS+=(--resume_encoder_lora_path "${ENCODER_LORA_PATH}")
fi

accelerate launch \
  --num_processes "${GPU_NUM}" \
  --mixed_precision bf16 \
  "${TRAIN_PY}" \
  "${ARGS[@]}"
