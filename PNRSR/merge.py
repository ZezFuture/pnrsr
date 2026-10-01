import os
import torch
from diffusers import ZImagePipeline
from peft import PeftModel

base_path = "Tongyi-MAI/Z-Image"
lora_path = "path/to/your/sr_lora_checkpoint"  # Set the path to the warm-up checkpoint


save_path = "path/to/save/Z-Image_warm_up/transformer"  # Copy the Z-Image weights to a new folder and replace the original transformer weights with the merged weights

pipe = ZImagePipeline.from_pretrained(
    base_path,
    torch_dtype=torch.bfloat16,
).to("cuda")

pipe.transformer = PeftModel.from_pretrained(
    pipe.transformer,
    lora_path,
    adapter_name="sr",
    is_trainable=False,
)

pipe.transformer.set_adapter("sr")


merged_transformer = pipe.transformer.merge_and_unload()


merged_transformer.save_pretrained(
    save_path,
    safe_serialization=True,
    max_shard_size="5GB",
)

print(f"Only merged transformer saved to: {save_path}")
