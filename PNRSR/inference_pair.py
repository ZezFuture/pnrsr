import argparse
import os
import glob
import csv
import gc
from typing import Optional, List, Tuple

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from srmodel import Zimage_SRmodel


def parse_float_list(s: Optional[str]) -> Optional[List[float]]:
    if s is None:
        return None
    s = str(s).strip()
    if s == "":
        return None
    return [float(x.strip()) for x in s.split(",") if x.strip() != ""]


def get_dtype(dtype_name: str):
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "fp32":
        return torch.float32
    raise ValueError(f"Unknown dtype: {dtype_name}")


def list_images(input_path: str) -> List[str]:
    exts = (
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".webp",
        ".tif",
        ".tiff",
    )

    if os.path.isfile(input_path):
        if input_path.lower().endswith(exts):
            return [input_path]
        raise ValueError(f"Unsupported image file: {input_path}")

    if not os.path.isdir(input_path):
        raise FileNotFoundError(f"Path does not exist: {input_path}")

    paths = []

    for ext in exts:
        paths.extend(
            glob.glob(
                os.path.join(input_path, f"*{ext}")
            )
        )

        paths.extend(
            glob.glob(
                os.path.join(input_path, f"*{ext.upper()}")
            )
        )

    return sorted(set(paths))


def normalize_stem(path: str) -> str:
    stem = os.path.splitext(
        os.path.basename(path)
    )[0]

    for suffix in (
        "_LR",
        "_lr",
        "-LR",
        "-lr",
        "_HR",
        "_hr",
        "-HR",
        "-hr",
        "_GT",
        "_gt",
        "-GT",
        "-gt",
    ):
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]

    return stem


def build_pairs(
    lr_input: str,
    hr_input: str,
) -> List[Tuple[str, str]]:

    lr_paths = list_images(lr_input)
    hr_paths = list_images(hr_input)

    if len(lr_paths) == 1 and len(hr_paths) == 1:
        return [(lr_paths[0], hr_paths[0])]

    hr_by_basename = {
        os.path.basename(p): p
        for p in hr_paths
    }

    hr_by_norm = {
        normalize_stem(p): p
        for p in hr_paths
    }

    pairs = []
    missing = []

    for lr_path in lr_paths:

        basename = os.path.basename(lr_path)
        norm = normalize_stem(lr_path)

        if basename in hr_by_basename:

            pairs.append(
                (
                    lr_path,
                    hr_by_basename[basename],
                )
            )

        elif norm in hr_by_norm:

            pairs.append(
                (
                    lr_path,
                    hr_by_norm[norm],
                )
            )

        else:
            missing.append(lr_path)

    if (
        len(pairs) == 0
        and len(lr_paths) == len(hr_paths)
    ):
        print(
            "[warning] No basename match found. "
            "Fallback to sorted pairing."
        )

        pairs = list(
            zip(
                sorted(lr_paths),
                sorted(hr_paths),
            )
        )

        missing = []

    if len(missing) > 0:

        print(
            f"[warning] Missing HR for "
            f"{len(missing)} LR images. "
            f"First few:"
        )

        for p in missing[:10]:
            print("  ", p)

    if len(pairs) == 0:
        raise RuntimeError(
            "No LR-HR pairs found."
        )

    return pairs


def to_nchw_tensor(
    x: np.ndarray,
    channels: int,
    device: torch.device,
) -> torch.Tensor:

    x = torch.from_numpy(x).float().to(
        device,
        non_blocking=True,
    )

    if x.ndim != 4:
        raise ValueError(
            f"Expected 4D tensor, "
            f"got shape: {x.shape}"
        )

    if x.shape[1] == channels:
        return x.contiguous()

    if x.shape[-1] == channels:
        return x.permute(
            0,
            3,
            1,
            2,
        ).contiguous()

    raise ValueError(
        f"Cannot infer channel dim from "
        f"shape: {x.shape}, "
        f"channels={channels}"
    )


def bgr_to_rgb_float01_to_m11(
    img_bgr: np.ndarray,
) -> np.ndarray:

    img_rgb = cv2.cvtColor(
        img_bgr,
        cv2.COLOR_BGR2RGB,
    )

    img_rgb = (
        img_rgb.astype(np.float32)
        / 127.5
        - 1.0
    )

    return img_rgb


def output_tensor_to_bgr_uint8(
    output_image: torch.Tensor,
) -> np.ndarray:

    output_np = (
        output_image
        .detach()
        .cpu()
        .float()
        .numpy()
    )

    output_np = (
        output_np * 127.5 + 127.5
    )

    output_np = np.clip(
        output_np,
        0,
        255,
    ).astype(np.uint8)

    output_np = output_np.transpose(
        1,
        2,
        0,
    )

    output_bgr = cv2.cvtColor(
        output_np,
        cv2.COLOR_RGB2BGR,
    )

    return output_bgr


def format_t(t: float) -> str:
    return f"{float(t):.3f}"


def sample_t_pair(
    rng: np.random.Generator,
    t_values: List[float],
    fixed_t_pair: Optional[List[float]] = None,
) -> Tuple[float, float]:

    if fixed_t_pair is not None:

        assert len(fixed_t_pair) == 2, (
            "--t_pair must contain exactly "
            "two values."
        )

        t0 = float(fixed_t_pair[0])
        t1 = float(fixed_t_pair[1])

    else:

        assert t_values is not None, (
            "Please provide --t_values "
            "when --t_pair is not set."
        )

        assert len(t_values) >= 2, (
            "Need at least two candidate "
            "t values."
        )

        idx = rng.choice(
            len(t_values),
            size=2,
            replace=False,
        )

        t0 = float(t_values[idx[0]])
        t1 = float(t_values[idx[1]])

    if abs(t0 - t1) < 1e-8:
        raise ValueError(
            f"The two t values must be "
            f"different, got {t0}, {t1}."
        )

    low_t = min(t0, t1)
    high_t = max(t0, t1)

    return low_t, high_t


def build_model(
    args,
    device: torch.device,
) -> Zimage_SRmodel:

    sr_lora_path = (
        args.sr_lora_path
        or args.lora_path
    )

    if sr_lora_path is None:
        raise ValueError(
            "Please provide "
            "--sr_lora_path or --lora_path."
        )

    one_step_timesteps = parse_float_list(
        args.one_step_timesteps
    )

    model = Zimage_SRmodel(
        pretrained_path=args.pretrained_path,
        sr_lora_path=sr_lora_path,
        dtype=get_dtype(args.dtype),
        device=str(device),

        encoder_lora_trainable=False,
        sr_lora_trainable=False,

        compile=args.compile,

        one_step_timesteps=one_step_timesteps,
        default_infer_timestep=500.0,

        flow_loss_weight=1.0,
        clean_loss_weight=1.0,
        lpips_loss_weight=0.0,
    )

    model.eval()
    model.vae.eval()
    model.text_encoder.eval()
    model.transformer.eval()

    return model


def run_one_pair(
    model: Zimage_SRmodel,
    lr_path: str,
    hr_path: str,
    args,
    device: torch.device,
    image_idx: int,
    rng: np.random.Generator,
    rank: int = 0,
):

    lr_bgr = cv2.imread(
        lr_path,
        cv2.IMREAD_COLOR,
    )

    hr_bgr = cv2.imread(
        hr_path,
        cv2.IMREAD_COLOR,
    )

    if lr_bgr is None:
        print(
            f"[GPU {rank}] [skip] "
            f"failed to read LR: {lr_path}"
        )
        return None

    if hr_bgr is None:
        print(
            f"[GPU {rank}] [skip] "
            f"failed to read HR: {hr_path}"
        )
        return None

    hr_h, hr_w = hr_bgr.shape[:2]

    # All inputs use a single 512 x 512 inference canvas.
    resize_h = 512
    resize_w = 512

    lr_bgr_aligned = cv2.resize(
        lr_bgr,
        (resize_w, resize_h),
        interpolation=cv2.INTER_CUBIC,
    )

    hr_bgr_aligned = cv2.resize(
        hr_bgr,
        (resize_w, resize_h),
        interpolation=cv2.INTER_CUBIC,
    )

    lr_rgb = bgr_to_rgb_float01_to_m11(
        lr_bgr_aligned
    )

    hr_rgb = bgr_to_rgb_float01_to_m11(
        hr_bgr_aligned
    )

    lr_rgb = np.expand_dims(
        lr_rgb,
        axis=0,
    )

    hr_rgb = np.expand_dims(
        hr_rgb,
        axis=0,
    )


    if args.seed is None:

        noise_rng = (
            np.random.default_rng()
        )

    else:

        noise_rng = (
            np.random.default_rng(
                args.seed + image_idx
            )
        )

    init_noise = (
        noise_rng
        .standard_normal(
            (
                1,
                resize_h // 8,
                resize_w // 8,
                16,
            )
        )
        .astype(np.float32)
    )

    fixed_t_pair = parse_float_list(
        args.t_pair
    )

    t_values = parse_float_list(
        args.t_values
    )

    low_t, high_t = sample_t_pair(
        rng=rng,
        t_values=t_values,
        fixed_t_pair=fixed_t_pair,
    )

    stem = os.path.splitext(
        os.path.basename(lr_path)
    )[0]

    print(
        f"[GPU {rank}] "
        f"{stem} | "
        f"t={low_t:.3f}/{high_t:.3f} | "
        f"size={resize_w}x{resize_h}"
    )

    with torch.inference_mode():
        lr_tensor = to_nchw_tensor(lr_rgb, channels=3, device=device)
        hr_tensor = to_nchw_tensor(hr_rgb, channels=3, device=device)
        noise_tensor = to_nchw_tensor(init_noise, channels=16, device=device)

        out = model.generate_realism_pair(
            lr_img=lr_tensor,
            hr_img=hr_tensor,
            prompt=args.prompt,
            t_pair=[
                low_t,
                high_t,
            ],
            max_sequence_length=(
                args.max_sequence_length
            ),
            noise=noise_tensor,
            decode=True,
        )

        low_output = out["low_image"][0].detach().cpu().float().numpy()
        high_output = out["high_image"][0].detach().cpu().float().numpy()

        low_output = low_output * 127.5 + 127.5
        high_output = high_output * 127.5 + 127.5

        del lr_tensor, hr_tensor, noise_tensor, out

    low_output = np.clip(
        low_output,
        0,
        255,
    ).astype(np.uint8)

    high_output = np.clip(
        high_output,
        0,
        255,
    ).astype(np.uint8)

    low_output = low_output.transpose(
        1,
        2,
        0,
    )

    high_output = high_output.transpose(
        1,
        2,
        0,
    )

    low_output = cv2.cvtColor(
        low_output,
        cv2.COLOR_RGB2BGR,
    )

    high_output = cv2.cvtColor(
        high_output,
        cv2.COLOR_RGB2BGR,
    )

    low_output = cv2.resize(
        low_output,
        (hr_w, hr_h),
        interpolation=cv2.INTER_CUBIC,
    )

    high_output = cv2.resize(
        high_output,
        (hr_w, hr_h),
        interpolation=cv2.INTER_CUBIC,
    )


    bad_dir = os.path.join(
        args.output_dir,
        "bad",
    )

    good_dir = os.path.join(
        args.output_dir,
        "good",
    )

    os.makedirs(
        bad_dir,
        exist_ok=True,
    )

    os.makedirs(
        good_dir,
        exist_ok=True,
    )

    # low_t = bad
    low_save_path = os.path.join(
        bad_dir,
        f"{stem}_{format_t(low_t)}.png",
    )

    # high_t = good
    high_save_path = os.path.join(
        good_dir,
        f"{stem}_{format_t(high_t)}.png",
    )

    cv2.imwrite(
        low_save_path,
        low_output,
    )

    cv2.imwrite(
        high_save_path,
        high_output,
    )

    if args.empty_cache_each_image:

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "image_idx": image_idx,
        "lr_path": lr_path,
        "hr_path": hr_path,
        "low_t": low_t,
        "high_t": high_t,
        "low_save_path": low_save_path,
        "high_save_path": high_save_path,
        "label": "high_t_is_more_real",
    }



def gpu_worker(
    rank: int,
    world_size: int,
    pairs,
    args,
):



    torch.cuda.set_device(rank)

    device = torch.device(
        f"cuda:{rank}"
    )

    if args.seed is not None:

        worker_seed = (
            args.seed + rank * 100000
        )

        torch.manual_seed(
            worker_seed
        )

        np.random.seed(
            worker_seed
        )

        rng = (
            np.random.default_rng(
                worker_seed
            )
        )

    else:

        rng = (
            np.random.default_rng()
        )

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    print(
        f"[GPU {rank}] "
        f"Loading model..."
    )

    model = build_model(
        args,
        device,
    )

    local_indices = list(
        range(
            rank,
            len(pairs),
            world_size,
        )
    )

    print(
        f"[GPU {rank}] "
        f"Processing "
        f"{len(local_indices)} "
        f"pairs."
    )

    meta_rows = []

    for image_idx in local_indices:

        lr_path, hr_path = (
            pairs[image_idx]
        )

        try:

            row = run_one_pair(
                model=model,
                lr_path=lr_path,
                hr_path=hr_path,
                args=args,
                device=device,
                image_idx=image_idx,
                rng=rng,
                rank=rank,
            )

            if row is not None:
                meta_rows.append(row)

        except Exception as e:

            print(
                f"[GPU {rank}] "
                f"[ERROR] "
                f"image_idx={image_idx}"
            )

            print(
                f"[GPU {rank}] "
                f"{lr_path}"
            )

            print(
                f"[GPU {rank}] "
                f"{repr(e)}"
            )


    temp_meta_path = os.path.join(
        args.output_dir,
        f".realism_pair_meta_rank{rank}.csv",
    )

    with open(
        temp_meta_path,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_idx",
                "lr_path",
                "hr_path",
                "low_t",
                "high_t",
                "low_save_path",
                "high_save_path",
                "label",
            ],
        )

        writer.writeheader()
        writer.writerows(
            meta_rows
        )

    print(
        f"[GPU {rank}] Done."
    )

    del model

    gc.collect()

    torch.cuda.empty_cache()


def merge_meta_csv(
    output_dir: str,
    world_size: int,
):

    rows = []

    for rank in range(world_size):

        path = os.path.join(
            output_dir,
            f".realism_pair_meta_rank{rank}.csv",
        )

        if not os.path.exists(path):
            continue

        with open(
            path,
            "r",
            encoding="utf-8",
            newline="",
        ) as f:

            reader = csv.DictReader(f)

            for row in reader:
                rows.append(row)

    rows.sort(
        key=lambda x: int(
            x["image_idx"]
        )
    )

    meta_path = os.path.join(
        output_dir,
        "realism_pair_meta.csv",
    )

    with open(
        meta_path,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "lr_path",
                "hr_path",
                "low_t",
                "high_t",
                "low_save_path",
                "high_save_path",
                "label",
            ],
        )

        writer.writeheader()

        for row in rows:

            row.pop(
                "image_idx",
                None,
            )

            writer.writerow(row)


    for rank in range(world_size):

        path = os.path.join(
            output_dir,
            f".realism_pair_meta_rank{rank}.csv",
        )

        if os.path.exists(path):
            os.remove(path)

    print(
        f"Saved metadata to: "
        f"{meta_path}"
    )




def run_single(
    args,
    pairs,
):

    device = torch.device(
        args.device
    )

    if args.seed is not None:

        torch.manual_seed(
            args.seed
        )

        np.random.seed(
            args.seed
        )

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    model = build_model(
        args,
        device,
    )

    rng = np.random.default_rng(
        args.seed
        if args.seed is not None
        else None
    )

    meta_rows = []

    for image_idx, (
        lr_path,
        hr_path,
    ) in enumerate(pairs):

        row = run_one_pair(
            model=model,
            lr_path=lr_path,
            hr_path=hr_path,
            args=args,
            device=device,
            image_idx=image_idx,
            rng=rng,
            rank=0,
        )

        if row is not None:

            row.pop(
                "image_idx",
                None,
            )

            meta_rows.append(row)

    meta_path = os.path.join(
        args.output_dir,
        "realism_pair_meta.csv",
    )

    with open(
        meta_path,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "lr_path",
                "hr_path",
                "low_t",
                "high_t",
                "low_save_path",
                "high_save_path",
                "label",
            ],
        )

        writer.writeheader()
        writer.writerows(
            meta_rows
        )

    print(
        f"Saved metadata to: "
        f"{meta_path}"
    )


def main(args):

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )


    os.makedirs(
        os.path.join(
            args.output_dir,
            "good",
        ),
        exist_ok=True,
    )

    os.makedirs(
        os.path.join(
            args.output_dir,
            "bad",
        ),
        exist_ok=True,
    )

    pairs = build_pairs(
        args.lr_input,
        args.hr_input,
    )

    print(
        f"Found {len(pairs)} "
        f"LR-HR pairs."
    )


    use_cuda = (
        args.device.startswith("cuda")
        and torch.cuda.is_available()
    )

    if use_cuda:

        world_size = (
            torch.cuda.device_count()
        )

    else:

        world_size = 1

    if (
        use_cuda
        and world_size > 1
    ):

        print(
            "\n============================"
        )

        print(
            f"Multi-GPU inference enabled"
        )

        print(
            f"Number of GPUs: "
            f"{world_size}"
        )

        for i in range(
            world_size
        ):

            print(
                f"GPU {i}: "
                f"{torch.cuda.get_device_name(i)}"
            )

        print(
            "============================\n"
        )

        mp.spawn(
            gpu_worker,
            args=(
                world_size,
                pairs,
                args,
            ),
            nprocs=world_size,
            join=True,
        )

        merge_meta_csv(
            args.output_dir,
            world_size,
        )

    else:

        print(
            f"Single device mode: "
            f"{args.device}"
        )

        run_single(
            args,
            pairs,
        )

    print()
    print(
        "========== Finished =========="
    )

    print(
        "Good images: "
        f"{os.path.join(args.output_dir, 'good')}"
    )

    print(
        "Bad images : "
        f"{os.path.join(args.output_dir, 'bad')}"
    )


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--lr_input",
        type=str,
        required=True,
        help=(
            "LR image folder or "
            "single LR image."
        ),
    )

    parser.add_argument(
        "--hr_input",
        type=str,
        required=True,
        help=(
            "HR image folder or "
            "single HR image."
        ),
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--pretrained_path",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--sr_lora_path",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default=(
            "high quality, sharp, "
            "detailed image"
        ),
    )

    parser.add_argument(
        "--t_values",
        type=str,
        default=None,
        help=(
            "Candidate Z-Image t values, "
            "comma-separated. "
            "Example: "
            "'0,0.045,0.1,0.1667,"
            "0.25,0.3571,0.5,0.7'."
        ),
    )

    parser.add_argument(
        "--t_pair",
        type=str,
        default=None,
        help=(
            "Fixed Z-Image t pair, "
            "comma-separated. "
            "Example: '0.045,0.7'. "
            "If set, no random sampling."
        ),
    )

    parser.add_argument(
        "--one_step_timesteps",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=[
            "bf16",
            "fp16",
            "fp32",
        ],
    )

    parser.add_argument(
        "--compile",
        action="store_true",
    )

    parser.add_argument(
        "--allow_tf32",
        action="store_true",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--test_batch",
        type=int,
        default=1,
        help="Compatibility option; unused in fixed 512x512 whole-image inference.",
    )

    parser.add_argument(
        "--block_size",
        type=int,
        default=512,
        help="Compatibility option; unused in fixed 512x512 whole-image inference.",
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=128,
        help="Compatibility option; unused in fixed 512x512 whole-image inference.",
    )

    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--empty_cache_each_image",
        action="store_true",
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    main(args)