#!/usr/bin/env bash
set -e
export CUDA_VISIBLE_DEVICES=0,1,2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "-1" ]]; then
  CUDA_VISIBLE_DEVICES_CLEAN=$(echo "$CUDA_VISIBLE_DEVICES" | tr -d ' ')
  IFS=',' read -ra GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES_CLEAN"
  NUM_GPUS=${#GPU_ARRAY[@]}
else
  NUM_GPUS=$(nvidia-smi -L | wc -l)
fi



if (( 12 % NUM_GPUS != 0 )); then
  echo "Error: 12 is not divisible by the current number of GPUs (${NUM_GPUS})"
  echo "Please manually set gradient_accumulation_steps or adjust the target global batch size"
  exit 1
fi

GRAD_ACC=$((12 / NUM_GPUS))

echo "Detected NUM_GPUS=${NUM_GPUS}"
echo "Set num_processes=${NUM_GPUS}"
echo "Set gradient_accumulation_steps=${GRAD_ACC}"


PRETRAINED_PATH="Tongyi-MAI/Z-Image"

TRAIN_DIR=""

OUTPUT_DIR="exp/warm_up_output"

ONE_STEP_TIMESTEPS="500,480,460,440,420,400,380,360,340,320,300,280,260,240,220,200,180,160,140,120,100,80,60,40,20"


accelerate launch \
  --num_processes "${NUM_GPUS}" \
  --mixed_precision bf16 \
  pnrsr/train_warm.py \
  --pretrained_path "${PRETRAINED_PATH}" \
  --train_dir "${TRAIN_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --resolution 512 \
  --train_batch_size 1 \
  --gradient_accumulation_steps "${GRAD_ACC}" \
  --num_training_epochs 100 \
  --max_train_steps 40000 \
  --mixed_precision bf16 \
  --seed 1234 \
  --allow_tf32 \
  --one_step_timesteps "${ONE_STEP_TIMESTEPS}" \
  --default_infer_timestep 500 \
  --flow_loss_weight 0 \
  --clean_loss_weight 1.0 \
  --lpips_loss_weight 2 \
  --fdl_loss_weight 0.005 \
  --lora_rank 64 \
  --lora_alpha 128 \
  --lora_dropout 0.0 \
  --learning_rate_sr_lora 5e-5 \
  --encoder_lora_rank 64 \
  --encoder_lora_alpha 128 \
  --encoder_lora_dropout 0.0 \
  --learning_rate_encoder_lora 5e-5  \
  --lr_scheduler constant_with_warmup \
  --lr_warmup_steps 500 \
  --adam_beta1 0.9 \
  --adam_beta2 0.999 \
  --adam_weight_decay 1e-2 \
  --adam_epsilon 1e-8 \
  --max_grad_norm 1.0 \
  --dataloader_num_workers 4 \
  --checkpointing_steps 1000 \
  --eval_freq 500 \
  --loss_window_size 100 \
  --report_to tensorboard \
  --tracker_project_name z-image \