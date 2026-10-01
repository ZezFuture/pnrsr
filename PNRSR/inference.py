import argparse
import os
import glob
import time
import math
import gc
from typing import Optional, List

import cv2
import numpy as np
import torch
from tqdm import tqdm

from sr_utils.wavelet_color_fix import adain_color_fix_cv2, wavelet_color_fix_cv2
import sr_utils.patch_utils as utils

from srmodel import Zimage_SRmodel


def parse_float_list(s: Optional[str]) -> Optional[List[float]]:
    if s is None:
        return None
    s = str(s).strip()
    if s == "":
        return None
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def get_dtype(dtype_name):
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "fp32":
        return torch.float32
    raise ValueError(f"Unknown dtype: {dtype_name}")


def list_images(input_path):
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")

    if os.path.isfile(input_path):
        if input_path.lower().endswith(exts):
            return [input_path]
        raise ValueError(f"Input file is not a supported image: {input_path}")

    if not os.path.isdir(input_path):
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    img_paths = []
    for ext in exts:
        img_paths.extend(glob.glob(os.path.join(input_path, f"*{ext}")))
        img_paths.extend(glob.glob(os.path.join(input_path, f"*{ext.upper()}")))

    return sorted(set(img_paths))


def to_nchw_tensor(x, channels, device):
    x = torch.from_numpy(x).float().to(device)

    if x.ndim != 4:
        raise ValueError(f"Expected 4D tensor, got shape: {x.shape}")

    if x.shape[1] == channels:
        return x.contiguous()

    if x.shape[-1] == channels:
        return x.permute(0, 3, 1, 2).contiguous()

    raise ValueError(f"Cannot infer channel dim from shape: {x.shape}, expected channels={channels}")


def sharpen_image(image):
    ycr_cb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycr_cb)

    blurred_y = cv2.GaussianBlur(y, (0, 0), 5)
    unsharp_y = cv2.addWeighted(y, 1.2, blurred_y, -0.2, 0)

    sharpened_ycr_cb = cv2.merge([unsharp_y, cr, cb])
    return cv2.cvtColor(sharpened_ycr_cb, cv2.COLOR_YCrCb2BGR)




def get_cuda_device_index(device: torch.device) -> Optional[int]:
    if device.type != "cuda":
        return None
    return device.index if device.index is not None else torch.cuda.current_device()


def build_model(args, device):
    sr_lora_path = args.sr_lora_path or args.lora_path
    if sr_lora_path is None:
        raise ValueError("Please provide --sr_lora_path or --lora_path.")

    model = Zimage_SRmodel(
        pretrained_path=args.pretrained_path,
        sr_lora_path=sr_lora_path,
        dtype=get_dtype(args.dtype),
        device=str(device),

        encoder_lora_trainable=False,
        sr_lora_trainable=False,

        compile=args.compile,
        one_step_timesteps=[args.timestep],
        default_infer_timestep=args.timestep,

        
    )

    model.eval()
    model.vae.eval()
    model.text_encoder.eval()
    model.transformer.eval()

    print(f"Single-step scheduler timestep: {args.timestep}")
    print(f"Corresponding Z-Image t: {(1000.0 - float(args.timestep)) / 1000.0:.6f}")
    print(f"start_from: {args.start_from}")
    print(f"SR adapter: {sr_lora_path}")
    print(f"VAE Encoder LoRA: {os.path.join(sr_lora_path, 'vae_encoder_lora')}")

    return model


def run_one_image(model, img_path, args, device, image_idx: int):
    bname = os.path.basename(img_path)

    input_image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if input_image is None:
        print(f"[skip] failed to read image: {img_path}")
        return 0.0

    in_h, in_w = input_image.shape[:2]
    ori_h = in_h * args.upscale
    ori_w = in_w * args.upscale

    start_time = time.time()

    overlap = args.overlap
    block_h = args.block_size
    block_w = args.block_size
    stride = block_h - overlap

    assert block_h % 16 == 0, "block_size should be divisible by 16"
    assert stride % 16 == 0, "stride should be divisible by 16"
    assert overlap < block_h, "overlap must be smaller than block_size"

    min_side = min(ori_h, ori_w)

    if min_side < args.block_size:
        infer_scale = args.block_size / float(min_side)
    else:
        infer_scale = 1.0

    scaled_h = int(round(ori_h * infer_scale))
    scaled_w = int(round(ori_w * infer_scale))

    resize_h = max(
        block_h,
        math.ceil(scaled_h / 16) * 16,
    )
    resize_w = max(
        block_w,
        math.ceil(scaled_w / 16) * 16,
    )

    input_image_hr = cv2.resize(
        input_image,
        (resize_w, resize_h),
        interpolation=cv2.INTER_CUBIC,
    )

    h0, w0 = input_image_hr.shape[:2]

    if args.seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(args.seed + image_idx)

    init_noise = rng.standard_normal(
        (1, h0 // 8, w0 // 8, 16),
    ).astype(np.float32)

    input_rgb = cv2.cvtColor(input_image_hr, cv2.COLOR_BGR2RGB)
    input_rgb = input_rgb.astype(np.float32) / 127.5 - 1.0
    input_rgb = np.expand_dims(input_rgb, axis=0)

    net_input_list = utils.image2patchs(
        input_rgb,
        [block_h, block_w],
        [stride, stride],
    )

    noise_list = utils.image2patchs(
        init_noise,
        [block_h // 8, block_w // 8],
        [stride // 8, stride // 8],
    )

    assert net_input_list.shape[0] == noise_list.shape[0], (
        net_input_list.shape,
        noise_list.shape,
    )

    print(f"\nImage: {bname}")
    print("Input size:", (in_h, in_w))
    print("Target output size:", (ori_h, ori_w))
    print("Inference resized size:", (h0, w0))
    print("Num of patches:", len(net_input_list))

    net_output_list = []

    with torch.no_grad():
        for idx in tqdm(range(0, len(net_input_list), args.test_batch)):
            patch_np = np.concatenate(
                net_input_list[idx:idx + args.test_batch],
                axis=0,
            )

            noise_np = np.concatenate(
                noise_list[idx:idx + args.test_batch],
                axis=0,
            )

            c_t = to_nchw_tensor(
                patch_np,
                channels=3,
                device=device,
            )

            noise = to_nchw_tensor(
                noise_np,
                channels=16,
                device=device,
            )

            output_image = model.inference(
                lr_img=c_t,
                prompt=args.prompt,
                timestep=args.timestep,
                max_sequence_length=args.max_sequence_length,
                noise=noise,
            )

            output_np = output_image.detach().cpu().float().numpy()
            output_np = output_np * 127.5 + 127.5

            for patch_id in range(output_np.shape[0]):
                net_output_list.append(np.expand_dims(output_np[patch_id], axis=0))

    net_output = utils.patchs2image(
        np.stack(net_output_list, axis=0),
        [h0, w0],
        [stride, stride],
        padding=0,
        mode=2,
    )[0]

    net_output = net_output.transpose((1, 2, 0))
    net_output = cv2.cvtColor(net_output, cv2.COLOR_RGB2BGR)

    if args.sharpen:
        net_output = sharpen_image(net_output)

    if args.align_method == "wavelet":
        net_output = wavelet_color_fix_cv2(net_output, input_image_hr)
    elif args.align_method == "adain":
        net_output = adain_color_fix_cv2(net_output, input_image_hr)
    elif args.align_method == "nofix":
        pass
    else:
        raise ValueError(f"Unknown align_method: {args.align_method}")

    net_output = cv2.resize(
        net_output,
        (ori_w, ori_h),
        interpolation=cv2.INTER_CUBIC,
    )

    net_output = np.clip(net_output, 0, 255).astype(np.uint8)

    suffix = args.suffix
    if suffix is None or suffix == "":
        suffix = f"_t{int(round(args.timestep))}_{args.start_from}"

    save_path = os.path.join(
        args.output_dir,
        os.path.splitext(bname)[0] + suffix + ".png",
    )

    cv2.imwrite(save_path, net_output)

    elapsed = time.time() - start_time
    print(f"Processing time: {elapsed:.3f}s")
    print(f"Saved to: {save_path}")

    if args.empty_cache_each_image:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return elapsed


def main(args):
    device = torch.device(args.device)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    os.makedirs(args.output_dir, exist_ok=True)

    model = build_model(args, device)

    cuda_idx = get_cuda_device_index(device)
    if cuda_idx is not None:
        torch.cuda.reset_peak_memory_stats(cuda_idx)

    img_paths = list_images(args.input_image)
    if len(img_paths) == 0:
        raise RuntimeError(f"No images found in: {args.input_image}")

    print(f"Found {len(img_paths)} images.")

    all_time = 0.0

    for image_idx, img_path in enumerate(img_paths):
        all_time += run_one_image(
            model=model,
            img_path=img_path,
            args=args,
            device=device,
            image_idx=image_idx,
        )

    print("all_time:", all_time)

    if cuda_idx is not None:
        peak_alloc = torch.cuda.max_memory_allocated(cuda_idx) / 1024 ** 3
        peak_reserved = torch.cuda.max_memory_reserved(cuda_idx) / 1024 ** 3
        print(f"GPU{cuda_idx} peak allocated: {peak_alloc:.3f} GB")
        print(f"GPU{cuda_idx} peak reserved : {peak_reserved:.3f} GB")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_image", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)


    parser.add_argument("--pretrained_path", type=str, default=None)
    parser.add_argument("--sr_lora_path", type=str, default=None)
    parser.add_argument("--lora_path", type=str, default=None)

    parser.add_argument(
        "--prompt",
        type=str,
        default=(
            "high quality, sharp, detailed image, realistic, natural texture, "
            "photorealistic, clean edges"
        ),
    )

    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--allow_tf32", action="store_true")

    parser.add_argument("--align_method", type=str, default="wavelet", choices=["wavelet", "adain", "nofix"])
    parser.add_argument("--test_batch", type=int, default=1)
    parser.add_argument("--max_sequence_length", type=int, default=512)

    parser.add_argument("--timestep", type=float, default=1000.0)
    parser.add_argument("--start_from", type=str, default="lr_noise", choices=["lr_noise", "noise", "lr"])

    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--sharpen", action="store_true")
    parser.add_argument("--suffix", type=str, default=None)
    parser.add_argument("--empty_cache_each_image", action="store_true")



    args = parser.parse_args()

    if args.negative_prompt not in ("", None):
        print("[warning] negative_prompt is not used.")

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)