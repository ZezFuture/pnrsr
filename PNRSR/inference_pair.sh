ONE_STEP_TIMESTEPS="500, 20"

python pnrsr/inference_pair.py \
  --lr_input path/LR \
  --hr_input path/GT \
  --output_dir path/output \
  --sr_lora_path path/warm_up_checkpoint \
  --pretrained_path  Tongyi-MAI/Z-Image \
  --device cuda \
  --t_values "0.50, 0.98" \
  --one_step_timesteps "${ONE_STEP_TIMESTEPS}" \
  --dtype bf16
