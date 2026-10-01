python pnrsr/inference.py \
  --input_image path/LR \
  --output_dir path/output \
  --pretrained_path path/Z-image_warm_up \
  --sr_lora_path path/checkpoint \
  --timestep 500 \
  --start_from lr_noise \
  --block_size 512 \
  --overlap 32 \
  --upscale 4  \
  --dtype bf16 \
  --device cuda:0 \
  --seed 42 \
  --align_method adain \
  --block_size 512 \

