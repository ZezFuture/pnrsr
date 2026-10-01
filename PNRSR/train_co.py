# -*- coding: utf-8 -*-
import math
import os
import csv
import gc
import cv2
import contextlib
import re
import random
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageOps

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import transformers
import diffusers
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.utils import (
    DistributedDataParallelKwargs,
    ProjectConfiguration,
    set_seed,
)
from diffusers.training_utils import cast_training_params

from sr_utils.training_utils import MyDataset_blind_plus_withprompt
from srmodel import Zimage_SRmodel


from reward import (
    SigLIP2PairSRReward,
    save_reward_model,
    unfreeze_reward_head_only,
)




# ============================================================
# Constants
# ============================================================

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}


# ============================================================
# Basic helpers
# ============================================================

def get_weight_dtype(mixed_precision: str) -> torch.dtype:
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def parse_float_list(value: Optional[str]) -> List[float]:
    if value is None:
        return []
    return [float(x.strip()) for x in str(value).split(",") if x.strip()]


def parse_hidden_layers(value: str) -> Tuple[int, ...]:
    value = value.strip()
    if not value:
        return tuple()
    return tuple(int(x.strip()) for x in value.split(",") if x.strip())


def sanitize_config(args) -> Dict[str, Any]:
    config = {}
    for key, value in vars(args).items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            config[key] = value
        else:
            config[key] = str(value)
    return config


def save_args(args, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "args.txt", "w", encoding="utf-8") as f:
        for key, value in sorted(vars(args).items()):
            f.write(f"{key}: {value}\n")


REWARD_METRICS_CSV_NAME = "reward_metrics.csv"
REWARD_METRICS_CSV_FIELDS = (
    "reward_update",
    "sr_step",
    "sr_updated",
    "lambda_reward_cur",
    "reward_good_score",
    "reward_bad_score",
    "reward_accuracy",
    "reward_raw_good",
    "reward_raw_bad",
)


def init_reward_metrics_csv(output_dir: str) -> Path:
    """Create (overwriting any existing file) the CSV and write its header."""
    csv_path = Path(output_dir, "logs", REWARD_METRICS_CSV_NAME)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(REWARD_METRICS_CSV_FIELDS)
    print(f"[reward_metrics_csv] writing to: {csv_path}")
    return csv_path


def append_reward_metrics_csv(
    csv_path: Optional[Path],
    global_step: int,
    logs: Dict[str, Any],
) -> None:
    """Append one row. Opened/closed per call so the file stays readable
    while training is still running."""
    if csv_path is None:
        return
    row = [global_step]
    row.extend(logs[key] for key in REWARD_METRICS_CSV_FIELDS[1:])
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


def split_sr_and_encoder_params(model: Zimage_SRmodel):
    encoder_params = model.vae_encoder_lora_trainable_parameters()
    encoder_ids = {id(p) for p in encoder_params}
    sr_params = [
        p for p in model.sr_trainable_parameters()
        if id(p) not in encoder_ids
    ]
    return sr_params, encoder_params


def count_params(params: Sequence[nn.Parameter]) -> int:
    return sum(p.numel() for p in params)


def to_reward_range(
    img: torch.Tensor,
    image_size: int,
) -> torch.Tensor:

    img = (img.float() * 0.5 + 0.5).clamp(0.0, 1.0)

    if img.shape[-2:] != (image_size, image_size):
        img = F.interpolate(
            img,
            size=(image_size, image_size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )

    return img.clamp(0.0, 1.0)


def bounded_reward_score(raw_score: torch.Tensor) -> torch.Tensor:
    """
    Map the raw reward logit to the fixed semantic interval [-1, 1].

    All generator guidance, relative ranking, and upper/lower calibration use
    this same transformed score.
    """
    return raw_score.float()
    # return torch.tanh(raw_score.float())


def make_t_batch(
    t_value: float,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.full(
        (batch_size,),
        float(t_value),
        device=device,
        dtype=dtype,
    )



def sample_ordered_t_pair(t_values: Sequence[float]) -> Tuple[float, float]:
    """
    Sample t_good > t_bad.

    Z-Image uses t=0 as the noise endpoint. Therefore, among two sampled
    states, the larger t is treated as the better / cleaner state.
    """
    if len(t_values) < 2:
        raise ValueError("--t_values must contain at least two distinct values.")

    t_a, t_b = random.sample(list(t_values), 2)
    t_bad, t_good = sorted((float(t_a), float(t_b)))
    return t_good, t_bad



def set_sr_train_mode(model: nn.Module) -> None:
    model.train()
    raw = model.module if hasattr(model, "module") else model
    if hasattr(raw, "vae"):
        raw.vae.eval()
    if hasattr(raw, "text_encoder"):
        raw.text_encoder.eval()
    if getattr(raw, "lpips_fn", None) is not None:
        raw.lpips_fn.eval()
    if getattr(raw, "fdl_loss_fn", None) is not None:
        raw.fdl_loss_fn.eval()


def set_reward_train_mode(model: nn.Module) -> None:
    model.train()
    raw = model.module if hasattr(model, "module") else model

    # Frozen DINO backbone stays in eval mode.
    if hasattr(raw, "vision_model"):
        raw.vision_model.eval()
    if hasattr(raw, "cross_blocks"):
        raw.cross_blocks.train()
    if hasattr(raw, "sr_quality_branch") and raw.sr_quality_branch is not None:
        raw.sr_quality_branch.train()
    if hasattr(raw, "reward_head"):
        raw.reward_head.train()


def set_reward_trainable(
    model: nn.Module,
    trainable_names: Sequence[str],
    enabled: bool,
) -> None:
    """Only re-enable parameters that were trainable after head unfreezing."""
    raw = model.module if hasattr(model, "module") else model
    allowed = set(trainable_names)
    for name, param in raw.named_parameters():
        param.requires_grad_(enabled and name in allowed)


def extract_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "model", "reward_model"):
            if isinstance(obj.get(key), dict):
                return obj[key]

        if any(torch.is_tensor(v) for v in obj.values()):
            return obj

    raise RuntimeError("Cannot find reward model state_dict in checkpoint.")


def load_reward_checkpoint(model: nn.Module, checkpoint_path: str) -> None:
    payload = torch.load(checkpoint_path, map_location="cpu")
    state_dict = extract_state_dict(payload)

    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        cleaned[key] = value

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print(f"[Reward] loaded checkpoint: {checkpoint_path}")
    print(f"[Reward] missing={len(missing)}, unexpected={len(unexpected)}")


# ============================================================
# Reward-bound dataset
# ============================================================

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def list_images(root: Path, recursive: bool = False) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Image directory does not exist: {root}")

    iterator = root.rglob("*") if recursive else root.iterdir()
    return sorted(path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def get_bound_key(path: Path) -> str:

    if "_" not in path.stem:
        raise RuntimeError(f"Bound image filename must contain '_': {path}")
    return path.stem.rsplit("_", 1)[0]


class RewardBoundDataset(torch.utils.data.Dataset):


    def __init__(self, upper_dir: str, lower_dir: str, lr_dir: str,
                 image_size: int, recursive: bool = False):
        super().__init__()

        self.upper_dir = Path(upper_dir)
        self.lower_dir = Path(lower_dir)
        self.lr_dir = Path(lr_dir)
        self.image_size = int(image_size)
        self.recursive = bool(recursive)

        lr_images = list_images(self.lr_dir, self.recursive)
        upper_images = list_images(self.upper_dir, self.recursive)
        lower_images = list_images(self.lower_dir, self.recursive)

        self.lr_by_key = self._build_lr_index(lr_images)
        self.upper_by_key = self._build_bound_index(upper_images, "upper")
        self.lower_by_key = self._build_bound_index(lower_images, "lower")

        lr_keys = set(self.lr_by_key)
        upper_keys = set(self.upper_by_key)
        lower_keys = set(self.lower_by_key)

        triplet_keys = sorted(lr_keys & upper_keys & lower_keys)

        self.triplets: List[Tuple[Path, Path, Path]] = [
            (self.lr_by_key[key], self.upper_by_key[key], self.lower_by_key[key])
            for key in triplet_keys
        ]

        self.upper_unmatched = [
            path for key, path in self.upper_by_key.items() if key not in lr_keys
        ]
        self.lower_unmatched = [
            path for key, path in self.lower_by_key.items() if key not in lr_keys
        ]
        self.upper_only_keys = sorted((lr_keys & upper_keys) - lower_keys)
        self.lower_only_keys = sorted((lr_keys & lower_keys) - upper_keys)

        print(
            f"[RewardBoundDataset] LR={len(lr_images)}, "
            f"upper={len(upper_images)}, lower={len(lower_images)}, "
            f"triplets={len(self.triplets)}"
        )

    @staticmethod
    def _build_lr_index(paths: Sequence[Path]) -> Dict[str, Path]:
        index = {}
        for path in paths:
            key = path.stem
            if key in index:
                raise RuntimeError(f"Duplicate LR key '{key}': {index[key]} / {path}")
            index[key] = path
        return index

    @staticmethod
    def _build_bound_index(paths: Sequence[Path], name: str) -> Dict[str, Path]:
        index = {}
        for path in paths:
            key = get_bound_key(path)
            if key in index:
                raise RuntimeError(
                    f"Duplicate {name} key '{key}': {index[key]} / {path}"
                )
            index[key] = path
        return index

    def __len__(self) -> int:
        return len(self.triplets)

    @staticmethod
    def _read_rgb(path: Path) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"cv2.imread failed: {path}")

        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _to_reward_tensor(self, img_rgb: np.ndarray) -> torch.Tensor:
        x = img_rgb.astype(np.float32) / 127.5 - 1.0
        x = torch.from_numpy(
            np.transpose(x, (2, 0, 1))
        ).contiguous()

        x = to_reward_range(
            x.unsqueeze(0),
            self.image_size,
        )

        return x.squeeze(0).contiguous()

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        lr_path, upper_path, lower_path = self.triplets[index]

        lr = self._read_rgb(lr_path)
        upper = self._read_rgb(upper_path)
        lower = self._read_rgb(lower_path)

        if upper.shape[:2] != lower.shape[:2]:
            raise ValueError(
                "Upper/lower size mismatch: "
                f"{upper_path} {upper.shape[:2]} vs "
                f"{lower_path} {lower.shape[:2]}"
            )

        h, w = upper.shape[:2]

        if lr.shape[:2] != (h, w):
            lr = cv2.resize(
                lr,
                (w, h),
                interpolation=cv2.INTER_CUBIC,
            )

        return {
            "bound_lr": self._to_reward_tensor(lr),
            "upper_img": self._to_reward_tensor(upper),
            "lower_img": self._to_reward_tensor(lower),
        }
    
    def write_match_report(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            f.write("Reward bound same-LR triplet report\n")
            f.write("=" * 80 + "\n")
            f.write(f"upper_dir: {self.upper_dir}\n")
            f.write(f"lower_dir: {self.lower_dir}\n")
            f.write(f"lr_dir: {self.lr_dir}\n\n")

            f.write(f"same-LR triplets: {len(self.triplets)}\n")
            f.write(f"upper without LR: {len(self.upper_unmatched)}\n")
            f.write(f"lower without LR: {len(self.lower_unmatched)}\n")
            f.write(f"upper-only keys: {len(self.upper_only_keys)}\n")
            f.write(f"lower-only keys: {len(self.lower_only_keys)}\n\n")

            f.write("[Upper without LR]\n")
            for item in self.upper_unmatched:
                f.write(str(item) + "\n")

            f.write("\n[Lower without LR]\n")
            for item in self.lower_unmatched:
                f.write(str(item) + "\n")

            f.write("\n[Upper-only keys]\n")
            for key in self.upper_only_keys:
                f.write(key + "\n")

            f.write("\n[Lower-only keys]\n")
            for key in self.lower_only_keys:
                f.write(key + "\n")

# ============================================================
# Reward / discriminator objective
# ============================================================

class RelativeRewardLoss(nn.Module):
    """
    Reward objective with two complementary constraints:

    1. Generated-pair ranking:
           R(LR, SR_good) > R(LR, SR_bad)

    2. Same-LR fixed-bound calibration:
           R(LR, upper) -> +target
           R(LR, lower) -> -target
           R(LR, upper) - R(LR, lower) -> 2 * target
           0.5 * (R_upper + R_lower) -> 0

    Scores are bounded by tanh into [-1, 1]. The direct gap term produces
    opposite gradients for upper and lower, so cancellation in the final
    shared bias does not prevent the two classes from separating.
    """

    def __init__(
        self,
        reward_model: nn.Module,
        bound_loss_weight: float = 1.0,
        bound_target: float = 0.9,
        bound_absolute_weight: float = 0.25,
        bound_gap_weight: float = 1.0,
        bound_center_weight: float = 0.1,
        bound_loss_type: str = "mse",
        bound_smooth_l1_beta: float = 0.1,
        score_l2_weight: float = 1.0e-4,
        margin_reg_weight: float = 1.0e-2,
        max_margin: float = 2.0,
    ):
        super().__init__()
        self.reward_model = reward_model
        self.bound_loss_weight = float(bound_loss_weight)
        self.bound_target = float(bound_target)
        self.bound_absolute_weight = float(bound_absolute_weight)
        self.bound_gap_weight = float(bound_gap_weight)
        self.bound_center_weight = float(bound_center_weight)
        self.bound_loss_type = str(bound_loss_type)
        self.bound_smooth_l1_beta = float(bound_smooth_l1_beta)
        self.score_l2_weight = float(score_l2_weight)
        self.margin_reg_weight = float(margin_reg_weight)
        self.max_margin = float(max_margin)

        if not (0.0 < self.bound_target <= 1.0):
            raise ValueError("bound_target must be in (0, 1].")
        if self.bound_loss_type not in {"smooth_l1", "mse"}:
            raise ValueError(
                "bound_loss_type must be either 'smooth_l1' or 'mse'."
            )

    def _bound_regression_loss(
        self,
        score: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        if self.bound_loss_type == "mse":
            return F.mse_loss(score, target)

        return F.smooth_l1_loss(
            score,
            target,
            beta=self.bound_smooth_l1_beta,
        )

    def relative_forward(
        self,
        img_lr: torch.Tensor,
        img_good: torch.Tensor,
        img_bad: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        raw_good = self.reward_model(img_lr, img_good).reshape(-1)
        raw_bad = self.reward_model(img_lr, img_bad).reshape(-1)

        score_good = bounded_reward_score(raw_good)
        score_bad = bounded_reward_score(raw_bad)

        margin = score_good - score_bad
        rank_loss = F.softplus(-margin).mean()

        pair_score_l2 = 0.5 * (
            raw_good.float().square().mean()
            + raw_bad.float().square().mean()
        )

        raw_margin = raw_good.float() - raw_bad.float()
        margin_reg = F.relu(
            raw_margin.abs() - self.max_margin
        ).square().mean()

        loss = (
            rank_loss
            + 0.5 * self.score_l2_weight * pair_score_l2
            + self.margin_reg_weight * margin_reg
        )

        with torch.no_grad():
            accuracy = (score_good > score_bad).float().mean()

        return {
            "loss": loss,
            "rank_loss": rank_loss.detach(),
            "pair_score_l2": pair_score_l2.detach(),
            "margin_reg": margin_reg.detach(),
            "score_good": score_good.detach().mean(),
            "score_bad": score_bad.detach().mean(),
            "margin": margin.detach().mean(),
            "accuracy": accuracy.detach(),
            "raw_good": raw_good.detach().float().mean(),
            "raw_bad": raw_bad.detach().float().mean(),
        }

    def bound_pair_forward(
        self,
        img_lr: torch.Tensor,
        img_upper: torch.Tensor,
        img_lower: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if img_lr.shape[0] != img_upper.shape[0] or img_lr.shape[0] != img_lower.shape[0]:
            raise ValueError(
                "img_lr, img_upper, and img_lower must have the same batch size."
            )

        batch_size = img_lr.shape[0]

        # One joint Reward forward: the same LR is paired with both bounds.
        pair_lr = torch.cat([img_lr, img_lr], dim=0)
        pair_img = torch.cat([img_upper, img_lower], dim=0)
        raw_pair = self.reward_model(pair_lr, pair_img).reshape(-1)

        raw_upper = raw_pair[:batch_size]
        raw_lower = raw_pair[batch_size:]
        score_upper = bounded_reward_score(raw_upper)
        score_lower = bounded_reward_score(raw_lower)

        target_upper = torch.full_like(score_upper, self.bound_target)
        target_lower = torch.full_like(score_lower, -self.bound_target)

        upper_loss = self._bound_regression_loss(score_upper, target_upper)
        lower_loss = self._bound_regression_loss(score_lower, target_lower)
        absolute_loss = 0.5 * (upper_loss + lower_loss)

        gap = score_upper - score_lower
        target_gap = 2.0 * self.bound_target

        # Squared hinge: once the target gap is reached, this term stops
        # pushing scores further toward tanh saturation.
        gap_error = F.relu(target_gap - gap)
        gap_loss = gap_error.square().mean()

        center = 0.5 * (score_upper + score_lower)
        center_loss = center.square().mean()

        calibration_loss = (
            self.bound_absolute_weight * absolute_loss
            + self.bound_gap_weight * gap_loss
            + self.bound_center_weight * center_loss
        )

        bound_score_l2 = 0.5 * (
            raw_upper.float().square().mean()
            + raw_lower.float().square().mean()
        )

        loss = (
            1 * calibration_loss
        )

        with torch.no_grad():
            order_accuracy = (score_upper > score_lower).float().mean()
            upper_accuracy = (score_upper > 0.0).float().mean()
            lower_accuracy = (score_lower < 0.0).float().mean()
            gap_reached = (gap >= target_gap).float().mean()

        return {
            "loss": loss,
            "calibration_loss": calibration_loss.detach(),
            "absolute_loss": absolute_loss.detach(),
            "upper_loss": upper_loss.detach(),
            "lower_loss": lower_loss.detach(),
            "gap_loss": gap_loss.detach(),
            "center_loss": center_loss.detach(),
            "score_l2": bound_score_l2.detach(),
            "score_upper": score_upper.detach().mean(),
            "score_lower": score_lower.detach().mean(),
            "raw_upper": raw_upper.detach().float().mean(),
            "raw_lower": raw_lower.detach().float().mean(),
            "gap": gap.detach().mean(),
            "center": center.detach().mean(),
            "order_accuracy": order_accuracy.detach(),
            "upper_accuracy": upper_accuracy.detach(),
            "lower_accuracy": lower_accuracy.detach(),
            "gap_reached": gap_reached.detach(),
        }


# ============================================================
# Checkpoint
# ============================================================

def save_checkpoint(
    accelerator: Accelerator,
    args,
    sr_model: nn.Module,
    reward_model: nn.Module,
    optimizer_sr,
    optimizer_reward,
    global_step: int,
    final: bool = False,
) -> None:
    if not accelerator.is_main_process:
        return

    name = "final" if final else f"checkpoint-{global_step}"
    save_dir = os.path.join(args.output_dir, "checkpoints", name)
    os.makedirs(save_dir, exist_ok=True)

    raw_sr = accelerator.unwrap_model(sr_model)
    raw_reward = accelerator.unwrap_model(reward_model)

    raw_sr.save_sr_lora(os.path.join(save_dir, "sr"))
    save_reward_model(raw_reward, os.path.join(save_dir, "reward_model.pt"))

    torch.save(
        {
            "global_step": global_step,
            "optimizer_sr": optimizer_sr.state_dict(),
            "optimizer_reward": optimizer_reward.state_dict(),
            "args": vars(args),
        },
        os.path.join(save_dir, "training_state.pt"),
    )
    print(f"[Checkpoint] saved to {save_dir}")


# ============================================================
# Main
# ============================================================

def main(args) -> None:
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=os.path.join(args.output_dir, "logs"),
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=project_config,
        kwargs_handlers=[ddp_kwargs],
    )

    if args.seed is not None:
        set_seed(args.seed)
    if args.rankrandom:
        set_seed(args.seed+accelerator.process_index)

    reward_metrics_csv_path = None
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        save_args(args, Path(args.output_dir, "logs"))
        reward_metrics_csv_path = init_reward_metrics_csv(args.output_dir)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    weight_dtype = get_weight_dtype(args.mixed_precision)

    # Z-Image: t=0 is noise; larger t is closer to the image endpoint.
    t_values = sorted(set(parse_float_list(args.t_values)))
    if len(t_values) < 2:
        raise ValueError("--t_values must contain at least two values.")
    if any(t < 0.0 or t > 1.0 for t in t_values):
        raise ValueError("All normalized Z-Image t values must be in [0, 1].")

    # ------------------------------------------------------------
    # 1. SR model
    # ------------------------------------------------------------
    sr_model = Zimage_SRmodel(
        pretrained_path=args.pretrained_path,
        # Transformer: always load the raw pretrained base model and create a
        # brand new SR LoRA adapter (sr_lora_path=None -> get_peft_model).
        sr_lora_path=None,
        dtype=weight_dtype,
        device=str(accelerator.device),

        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,

        # VAE encoder: resume the previously SFT-trained encoder LoRA.
        encoder_lora_path=args.resume_encoder_lora_path,
        encoder_lora_rank=args.encoder_lora_rank,
        encoder_lora_alpha=args.encoder_lora_alpha,
        encoder_lora_dropout=args.encoder_lora_dropout,
        encoder_lora_trainable=True,

        sr_lora_trainable=True,
        compile=args.compile,

        one_step_timesteps=args.default_infer_timestep,
        default_infer_timestep=args.default_infer_timestep,

        flow_loss_weight=args.flow_loss_weight,
        clean_loss_weight=args.clean_loss_weight,
        lpips_loss_weight=args.lpips_loss_weight,
        lpips_net=args.lpips_net,
        fdl_loss_weight=args.fdl_loss_weight,

    )
    set_sr_train_mode(sr_model)

    cast_training_params(
        [sr_model.transformer, sr_model.vae.encoder],
        dtype=torch.float32,
    )

    sr_lora_params, encoder_lora_params = split_sr_and_encoder_params(sr_model)
    if not sr_lora_params and not encoder_lora_params:
        raise RuntimeError("No trainable SR parameters were found.")

    optimizer_sr_groups = []
    if sr_lora_params:
        optimizer_sr_groups.append(
            {
                "name": "sr_lora",
                "params": sr_lora_params,
                "lr": args.learning_rate_sr_lora,
                "weight_decay": args.weight_decay,
            }
        )
    if encoder_lora_params:
        optimizer_sr_groups.append(
            {
                "name": "vae_encoder_lora",
                "params": encoder_lora_params,
                "lr": args.learning_rate_encoder_lora,
                "weight_decay": args.weight_decay,
            }
        )

    optimizer_sr = torch.optim.AdamW(
        optimizer_sr_groups,
        betas=(args.beta1, args.beta2),
        eps=args.adam_epsilon,
    )


    reward_model = SigLIP2PairSRReward(
        model_name=args.reward_model_name,
        image_size=args.reward_image_size,
        freeze_backbone=True,
        use_pooler=False,
        hidden_layers=parse_hidden_layers(args.reward_hidden_layers),
        token_pool_size=(
            args.reward_token_pool_size
            if args.reward_token_pool_size > 0
            else None
        ),
        attn_dim=args.reward_attn_dim,
        num_heads=args.reward_num_heads,

        num_fusion_layers=args.reward_num_fusion_layers,
        sr_quality_layers=args.reward_sr_quality_layers,
        use_sr_quality_branch=False,#not args.reward_no_sr_quality_branch,
        use_fidelity_branch=True,
        head_hidden_dim=args.reward_head_hidden_dim,
        dropout=args.reward_dropout,
        dtype=weight_dtype,
        local_files_only=args.reward_local_files_only,
        # Backbone parameters are frozen, but gradients must still pass from
        # reward scores to generated SR images during the generator update.
        backbone_no_grad=False,
    ).to(accelerator.device)


    reward_model = unfreeze_reward_head_only(reward_model)
    if args.reward_ckpt_path:
        load_reward_checkpoint(reward_model, args.reward_ckpt_path)

    reward_trainable_names = [
        name for name, p in reward_model.named_parameters() if p.requires_grad
    ]
    cast_training_params([reward_model], dtype=torch.float32)

    reward_params = [p for p in reward_model.parameters() if p.requires_grad]
    if not reward_params:
        raise RuntimeError("No trainable reward parameters were found.")

    optimizer_reward = torch.optim.AdamW(
        reward_params,
        lr=args.reward_learning_rate,
        betas=(args.beta1, args.beta2),
        eps=args.adam_epsilon,
        weight_decay=args.reward_weight_decay,
    )

    reward_loss_fn = RelativeRewardLoss(
        reward_model=reward_model,
        bound_loss_weight=args.reward_bound_loss_weight,
        bound_target=args.reward_bound_target,
        bound_absolute_weight=args.reward_bound_absolute_weight,
        bound_gap_weight=args.reward_bound_gap_weight,
        bound_center_weight=args.reward_bound_center_weight,
        bound_loss_type=args.reward_bound_loss_type,
        bound_smooth_l1_beta=args.reward_bound_smooth_l1_beta,
        score_l2_weight=args.reward_score_l2_weight,
        margin_reg_weight=args.reward_margin_reg_weight,
        max_margin=args.reward_max_margin,
    )

    # ------------------------------------------------------------
    # 3. Data
    # ------------------------------------------------------------
    train_dataset = MyDataset_blind_plus_withprompt(
        lr_dir=args.lr_dir,
        hr_dir=args.hr_dir,
        prompt_dir=args.prompt_dir,
        base_seed=args.seed,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        drop_last=True,
    )

    bound_dataset = RewardBoundDataset(
        upper_dir=args.reward_upper_dir,
        lower_dir=args.reward_lower_dir,
        lr_dir=args.reward_bound_lr_dir,
        image_size=args.reward_image_size,
        recursive=args.reward_bound_recursive,
    )

    bound_loader = torch.utils.data.DataLoader(
        bound_dataset,
        batch_size=args.reward_bound_batch_size,
        shuffle=True,
        num_workers=args.reward_bound_num_workers,
        pin_memory=True,
        drop_last=False,
    )

    if accelerator.is_main_process:
        bound_dataset.write_match_report(
            Path(args.output_dir, "logs", "reward_bound_match_report.txt")
        )

        print(f"Training samples: {len(train_dataset)}")
        print(f"t values: {t_values}")
        print(f"SR LoRA params: {count_params(sr_lora_params) / 1e6:.3f} M")
        print(
            f"Encoder LoRA params: "
            f"{count_params(encoder_lora_params) / 1e6:.3f} M"
        )
        print(
            f"Reward trainable params: "
            f"{count_params(reward_params) / 1e6:.3f} M"
        )
        print(
            "[Reward bounds] "
            f"same-LR triplets={len(bound_dataset.triplets)}, "
            f"upper without LR={len(bound_dataset.upper_unmatched)}, "
            f"lower without LR={len(bound_dataset.lower_unmatched)}, "
            f"upper-only={len(bound_dataset.upper_only_keys)}, "
            f"lower-only={len(bound_dataset.lower_only_keys)}"
        )
        print(
            "[Reward bounds] unmatched details saved to: "
            f"{Path(args.output_dir, 'logs', 'reward_bound_match_report.txt')}"
        )

    # ------------------------------------------------------------
    # 4. Accelerate
    # ------------------------------------------------------------
    (
        sr_model,
        reward_model,
        optimizer_sr,
        optimizer_reward,
        train_loader,
        bound_loader,
    ) = accelerator.prepare(
        sr_model,
        reward_model,
        optimizer_sr,
        optimizer_reward,
        train_loader,
        bound_loader,
    )

    accelerator.unwrap_model(sr_model).device = accelerator.device
    reward_loss_fn.reward_model = reward_model

    if accelerator.is_main_process:
        accelerator.init_trackers(
            args.tracker_project_name,
            config=sanitize_config(args),
        )

    progress_bar = tqdm(
        range(args.max_train_steps),
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    optimizer_sr.zero_grad(set_to_none=args.set_grads_to_none)
    optimizer_reward.zero_grad(set_to_none=args.set_grads_to_none)
    global_step = 0
    reward_update_count = 0
    cycles_since_warmup = 0



    bound_iterator = iter(bound_loader)

    reward_history_decay = 0.99
    reward_history_mean = torch.zeros(
        (),
        dtype=torch.float32,
        device=accelerator.device,
    )

    reward_history_initialized = False
    reward_history_sum = torch.zeros(
        (),
        dtype=torch.float32,
        device=accelerator.device,
    )

    reward_history_count = torch.zeros(
        (),
        dtype=torch.float32,
        device=accelerator.device,
    )

    # ------------------------------------------------------------
    # 5. Two-forward alternating training: SR first, Reward second
    # ------------------------------------------------------------
    for _epoch in range(args.num_training_epochs):
        for batch in train_loader:
            try:
                bound_batch = next(bound_iterator)
            except StopIteration:
                bound_iterator = iter(bound_loader)
                bound_batch = next(bound_iterator)

            x_src = batch["conditioning_pixel_values"]
            x_tgt = batch["output_pixel_values"]

            prompt = batch.get(
                "prompt",
                [args.prompt for _ in range(x_src.shape[0])],
            )

            # Use the same prompt condition in both forwards.
            prompt_for_step = (
                None
                if random.random() < args.prompt_drop_prob
                else prompt
            )

            # Z-Image: t=0 is noise, so t_good > t_bad.
            t_good, t_bad = sample_ordered_t_pair(t_values)

            with accelerator.accumulate(sr_model, reward_model):

                # ====================================================
                # A. Prepare this iteration
                # ====================================================
                set_sr_train_mode(sr_model)

                set_reward_trainable(
                    reward_model,
                    reward_trainable_names,
                    enabled=False,
                )
                reward_model.eval()

                t_good_batch = make_t_batch(
                    t_good,
                    batch_size=x_src.shape[0],
                    device=x_src.device,
                    dtype=x_src.dtype,
                )

                t_bad_batch = make_t_batch(
                    t_bad,
                    batch_size=x_src.shape[0],
                    device=x_src.device,
                    dtype=x_src.dtype,
                )

                lr_reward = to_reward_range(
                    x_src.detach(),
                    args.reward_image_size,
                )

                # ====================================================
                # Reward warmup
                # ====================================================
                warmup_done = (
                    reward_update_count
                    >= args.reward_guidance_start_step
                )

                # ====================================================
                # Reward : SR update ratio
                # ====================================================
                update_sr_this_cycle = warmup_done and (
                    cycles_since_warmup
                    % args.reward_updates_per_sr_update
                    == 0
                )

                # ====================================================
                # Lambda reward ramp
                # ====================================================
                if args.lambda_reward_ramp_steps > 0:
                    lambda_ramp = min(
                        1.0,
                        global_step
                        / args.lambda_reward_ramp_steps,
                    )
                else:
                    lambda_ramp = 1.0

                lambda_reward_cur = (
                    args.lambda_reward
                    * lambda_ramp
                )


                need_generator_reward_score = (
                    update_sr_this_cycle
                    and args.lambda_reward > 0.0
                )

                use_reward_guidance = (
                    need_generator_reward_score
                    and reward_history_initialized
                )

                # Freeze SR while warming up Reward and on
                # Reward-only cycles.
                torch.set_grad_enabled(
                    update_sr_this_cycle
                )


                # ====================================================
                # B1. Good forward + backward
                # ====================================================
                out_good = sr_model(
                    lr_img=x_src,
                    hr_img=x_tgt,
                    prompt=prompt_for_step,
                    max_sequence_length=args.max_sequence_length,
                    prompt_drop_prob=0.0,
                    t=t_good_batch,
                    noise=None,
                    return_outputs=True,
                )

                if "pred_img" not in out_good:
                    raise KeyError(
                        "Zimage_SRmodel.forward(return_outputs=True) "
                        "must return out['pred_img'] without detach."
                    )

                shared_noise = (
                    out_good["noise"]
                    .detach()
                )

                good_for_generator = to_reward_range(
                    out_good["pred_img"],
                    args.reward_image_size,
                )

                good_for_reward = (
                    good_for_generator.detach()
                )

                loss_base_good = (
                    out_good["loss"]
                )

                # ----------------------------------------------------
                # Historical Reward - GOOD
                # ----------------------------------------------------
                if need_generator_reward_score:


                    raw_score_good_generator = reward_model(
                        lr_reward,
                        good_for_generator,
                    ).reshape(-1)

                    with torch.no_grad():
                        reward_history_sum.add_(
                            raw_score_good_generator
                            .detach()
                            .float()
                            .sum()
                        )

                        reward_history_count.add_(
                            float(
                                raw_score_good_generator.numel()
                            )
                        )


                    if reward_history_initialized:

                        score_good_generator = (
                            raw_score_good_generator
                            - reward_history_mean.detach()
                        )



                        loss_reward_good = F.relu(
                            1
                            - score_good_generator
                        ).mean()

                    else:
                        loss_reward_good = (
                            loss_base_good.new_zeros(())
                        )

                else:
                    loss_reward_good = (
                        loss_base_good.new_zeros(())
                    )

                loss_generator_good = 0.5 * (
                    loss_base_good
                    + lambda_reward_cur
                    * loss_reward_good
                )

                if update_sr_this_cycle:
                    accelerator.backward(
                        loss_generator_good
                    )

                loss_base_good_log = (
                    loss_base_good.detach()
                )

                loss_reward_good_log = (
                    loss_reward_good.detach()
                )

                del out_good
                del good_for_generator

                if need_generator_reward_score:
                    del raw_score_good_generator

                    if reward_history_initialized:
                        del score_good_generator

                # ====================================================
                # B2. Bad forward + backward
                # ====================================================
                out_bad = sr_model(
                    lr_img=x_src,
                    hr_img=x_tgt,
                    prompt=prompt_for_step,
                    max_sequence_length=args.max_sequence_length,
                    prompt_drop_prob=0.0,
                    t=t_bad_batch,
                    noise=shared_noise,
                    return_outputs=True,
                )

                if "pred_img" not in out_bad:
                    raise KeyError(
                        "Zimage_SRmodel.forward(return_outputs=True) "
                        "must return out['pred_img'] without detach."
                    )

                bad_for_generator = to_reward_range(
                    out_bad["pred_img"],
                    args.reward_image_size,
                )

                bad_for_reward = (
                    bad_for_generator.detach()
                )

                loss_base_bad = (
                    out_bad["loss"]
                )

                # ----------------------------------------------------
                # Historical Reward - BAD
                # ----------------------------------------------------
                if need_generator_reward_score:

                    raw_score_bad_generator = reward_model(
                        lr_reward,
                        bad_for_generator,
                    ).reshape(-1)

                    with torch.no_grad():
                        reward_history_sum.add_(
                            raw_score_bad_generator
                            .detach()
                            .float()
                            .sum()
                        )

                        reward_history_count.add_(
                            float(
                                raw_score_bad_generator.numel()
                            )
                        )

                    if reward_history_initialized:

                        score_bad_generator = (
                            raw_score_bad_generator
                            - reward_history_mean.detach()
                        )

                        loss_reward_bad = F.relu(
                            1.0
                            - score_bad_generator
                        ).mean()

                    else:
                        loss_reward_bad = (
                            loss_base_bad.new_zeros(())
                        )

                else:
                    loss_reward_bad = (
                        loss_base_bad.new_zeros(())
                    )

                loss_generator_bad = 0.5 * (
                    loss_base_bad
                    + lambda_reward_cur
                    * loss_reward_bad
                )

                if update_sr_this_cycle:
                    accelerator.backward(
                        loss_generator_bad
                    )

                loss_base_bad_log = (
                    loss_base_bad.detach()
                )

                loss_reward_bad_log = (
                    loss_reward_bad.detach()
                )

                # ====================================================
                # B3. Update SR
                # ====================================================

                if update_sr_this_cycle:

                    if accelerator.sync_gradients:

                        sr_params_to_clip = []

                        for group in optimizer_sr.param_groups:
                            sr_params_to_clip.extend(
                                group["params"]
                            )

                        accelerator.clip_grad_norm_(
                            sr_params_to_clip,
                            args.max_grad_norm,
                        )

                    optimizer_sr.step()

                    optimizer_sr.zero_grad(
                        set_to_none=args.set_grads_to_none
                    )

                # Restore grad mode for Reward update.
                torch.set_grad_enabled(True)

                loss_base = 0.5 * (
                    loss_base_good_log
                    + loss_base_bad_log
                )

                loss_adversarial = 0.5 * (
                    loss_reward_good_log
                    + loss_reward_bad_log
                )

                loss_generator = (
                    loss_base
                    + lambda_reward_cur
                    * loss_adversarial
                )

                del out_bad
                del bad_for_generator

                if need_generator_reward_score:
                    del raw_score_bad_generator

                    if reward_history_initialized:
                        del score_bad_generator

                # ====================================================
                # C. Update Reward
                # ====================================================
                set_reward_trainable(
                    reward_model,
                    reward_trainable_names,
                    enabled=True,
                )

                set_reward_train_mode(
                    reward_model
                )


                bound_lr = (
                    bound_batch["bound_lr"]
                    .float()
                )

                upper_img = (
                    bound_batch["upper_img"]
                    .float()
                )

                lower_img = (
                    bound_batch["lower_img"]
                    .float()
                )

                # ====================================================
                # C1. Relative generated pair
                # ====================================================
                reward_pair_out = (
                    reward_loss_fn.relative_forward(
                        img_lr=lr_reward,
                        img_good=good_for_reward,
                        img_bad=bad_for_reward,
                    )
                )

                accelerator.backward(
                    reward_pair_out["loss"]
                )

                reward_pair_loss_log = (
                    reward_pair_out["loss"]
                    .detach()
                )

                reward_good_score_log = (
                    reward_pair_out["score_good"]
                )

                reward_bad_score_log = (
                    reward_pair_out["score_bad"]
                )

                reward_accuracy_log = (
                    reward_pair_out["accuracy"]
                )

                reward_raw_good_log = (
                    reward_pair_out["raw_good"]
                )

                reward_raw_bad_log = (
                    reward_pair_out["raw_bad"]
                )

                del reward_pair_out

                # ====================================================
                # C2. Same-LR upper/lower calibration
                # ====================================================
                reward_bound_out = (
                    reward_loss_fn.bound_pair_forward(
                        img_lr=bound_lr,
                        img_upper=upper_img,
                        img_lower=lower_img,
                    )
                )

                accelerator.backward(
                    reward_bound_out["loss"]
                )

                reward_bound_contribution_log = (
                    reward_bound_out["loss"]
                    .detach()
                )

                reward_upper_score_log = (
                    reward_bound_out["score_upper"]
                )

                reward_lower_score_log = (
                    reward_bound_out["score_lower"]
                )

                reward_bound_gap_log = (
                    reward_bound_out["gap"]
                )

                del reward_bound_out

                # ====================================================
                # Reward total log
                # ====================================================
                reward_model_loss_log = (
                    reward_pair_loss_log
                    + reward_bound_contribution_log
                )

                # ====================================================
                # Reward gradient clipping
                # ====================================================
                if accelerator.sync_gradients:

                    trainable_reward_params = [
                        p
                        for p in accelerator.unwrap_model(
                            reward_model
                        ).parameters()
                        if p.requires_grad
                    ]

                    accelerator.clip_grad_norm_(
                        trainable_reward_params,
                        args.reward_max_grad_norm,
                    )

                optimizer_reward.step()

                optimizer_reward.zero_grad(
                    set_to_none=args.set_grads_to_none
                )

                del good_for_reward
                del bad_for_reward

                # ====================================================
                # C3. Update Historical Reward Mean
                # ====================================================
                if (
                    accelerator.sync_gradients
                    and update_sr_this_cycle
                    and need_generator_reward_score
                ):
                    with torch.no_grad():
                        # --------------------------------------------
                        global_history_sum = accelerator.reduce(
                            reward_history_sum,
                            reduction="sum",
                        )

                        global_history_count = accelerator.reduce(
                            reward_history_count,
                            reduction="sum",
                        )

                        current_policy_reward_mean = (
                            global_history_sum
                            / global_history_count.clamp_min(1.0)
                        )


                        if not reward_history_initialized:

                            reward_history_mean.copy_(
                                current_policy_reward_mean
                            )

                            reward_history_initialized = True

                        else:
                            reward_history_mean.mul_(
                                reward_history_decay
                            ).add_(
                                current_policy_reward_mean,
                                alpha=(
                                    1.0
                                    - reward_history_decay
                                ),
                            )
                        reward_history_mean_log = (
                            reward_history_mean
                            .detach()
                            .clone()
                        )

                        current_policy_reward_mean_log = (
                            current_policy_reward_mean
                            .detach()
                            .clone()
                        )

                        reward_history_sum.zero_()
                        reward_history_count.zero_()


            # ========================================================
            # D. Logs / save
            # ========================================================
            if accelerator.sync_gradients:
                reward_update_count += 1


                sr_updated = update_sr_this_cycle
                if sr_updated:
                    global_step += 1
                    progress_bar.update(1)

                if warmup_done:
                    cycles_since_warmup += 1
                logs = {
                    # Read by append_reward_metrics_csv. Keys and order must
                    # stay in sync with REWARD_METRICS_CSV_FIELDS.
                    "sr_step": global_step,
                    "sr_updated": int(sr_updated),
                    "lambda_reward_cur": lambda_reward_cur,
                    "reward_good_score": reward_good_score_log.item(),
                    "reward_bad_score": reward_bad_score_log.item(),
                    "reward_accuracy": reward_accuracy_log.item(),
                    "reward_raw_good": reward_raw_good_log.item(),
                    "reward_raw_bad": reward_raw_bad_log.item(),

                    # Read by progress_bar.set_postfix only.
                    "loss_generator": loss_generator.item(),
                    "loss_reward_model": reward_model_loss_log.item(),
                    "reward_upper_score": reward_upper_score_log.item(),
                    "reward_lower_score": reward_lower_score_log.item(),
                    "reward_bound_gap": reward_bound_gap_log.item(),
                }

                progress_bar.set_postfix(
                    g=f"{logs['loss_generator']:.4f}",
                    d=f"{logs['loss_reward_model']:.4f}",
                    rank=f"{logs['reward_accuracy']:.3f}",
                    upper=f"{logs['reward_upper_score']:.3f}",
                    lower=f"{logs['reward_lower_score']:.3f}",
                    gap=f"{logs['reward_bound_gap']:.3f}",
                    pair=f"{t_good:.2f}/{t_bad:.2f}",
                    rup=reward_update_count,
                    lam=f"{lambda_reward_cur:.4f}",
                )
                # Logged against the Reward update counter so the warm-up phase
                # is not collapsed onto step 0 (global_step is frozen there).
                accelerator.log(logs, step=reward_update_count)

                if accelerator.is_main_process:
                    append_reward_metrics_csv(
                        reward_metrics_csv_path, reward_update_count, logs
                    )

                if sr_updated and global_step % args.checkpointing_steps == 0:
                    save_checkpoint(
                        accelerator=accelerator,
                        args=args,
                        sr_model=sr_model,
                        reward_model=reward_model,
                        optimizer_sr=optimizer_sr,
                        optimizer_reward=optimizer_reward,
                        global_step=global_step,
                    )

                # Keyed on the Reward counter so the cadence is preserved during
                # warm up too (global_step is frozen at 0 there).
                if reward_update_count % args.empty_cache_steps == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                if global_step >= args.max_train_steps:
                    break

        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    save_checkpoint(
        accelerator=accelerator,
        args=args,
        sr_model=sr_model,
        reward_model=reward_model,
        optimizer_sr=optimizer_sr,
        optimizer_reward=optimizer_reward,
        global_step=global_step,
        final=True,
    )
    accelerator.end_training()


# ============================================================
# Args
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    # Paths / regular SR training data
    parser.add_argument("--pretrained_path", type=str, default=None)
    parser.add_argument("--resume_encoder_lora_path", type=str, default=None)
    parser.add_argument("--reward_ckpt_path", type=str, default=None)
    parser.add_argument(
        "--output_dir",
        type=str,
    )
    parser.add_argument(
        "--lr_dir",
        type=str,
    )
    parser.add_argument(
        "--hr_dir",
        type=str,
    )
    parser.add_argument(
        "--prompt_dir",
        type=str,
    )

    # Reward upper/lower calibration data
    parser.add_argument(
        "--reward_upper_dir",
        type=str,
        help="Images calibrated to reward +1.",
    )
    parser.add_argument(
        "--reward_lower_dir",
        type=str,
        help="Images calibrated to reward -1.",
    )
    parser.add_argument(
        "--reward_bound_lr_dir",
        type=str,
        help=(
            "LR directory used to match both upper- and lower-bound images. "
            "Unmatched bound images are skipped."
        ),
    )
    parser.add_argument(
        "--reward_bound_recursive",
        action="store_true",
        help="Recursively scan upper/lower/LR bound directories.",
    )

    # Multi-start times. Z-Image: t=0 is noise; larger t is cleaner.
    parser.add_argument(
        "--t_values",
        type=str,
        default=(
            "0.50"
        ),
    )

    # SR model
    parser.add_argument(
        "--prompt",
        type=str,
        default="high quality, sharp, detailed image",
    )
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--prompt_drop_prob", type=float, default=0.0)
    parser.add_argument("--default_infer_timestep", type=float, default=500.0)

    parser.add_argument("--flow_loss_weight", type=float, default=0.0)
    parser.add_argument("--clean_loss_weight", type=float, default=1.0)
    parser.add_argument("--lpips_loss_weight", type=float, default=2.0)
    parser.add_argument("--fdl_loss_weight", type=float, default=0.001)
    parser.add_argument(
        "--lpips_net",
        type=str,
        default="vgg",
        choices=["alex", "vgg", "squeeze"],
    )

    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--encoder_lora_rank", type=int, default=16)
    parser.add_argument("--encoder_lora_alpha", type=int, default=32)
    parser.add_argument("--encoder_lora_dropout", type=float, default=0.0)

    # Reward model
    parser.add_argument(
        "--reward_model_name",
        type=str,
    )
    parser.add_argument("--reward_image_size", type=int, default=512)
    parser.add_argument("--reward_hidden_layers", type=str, default="-1,-7,-13")
    parser.add_argument("--reward_token_pool_size", type=int, default=32)
    parser.add_argument("--reward_attn_dim", type=int, default=512)
    parser.add_argument("--reward_num_heads", type=int, default=8)
    parser.add_argument("--reward_num_fusion_layers", type=int, default=1)
    parser.add_argument("--reward_sr_quality_layers", type=int, default=1)
    parser.add_argument("--reward_head_hidden_dim", type=int, default=1024)
    parser.add_argument("--reward_dropout", type=float, default=0)
    parser.add_argument("--reward_no_pooler", action="store_true")
    parser.add_argument("--reward_no_sr_quality_branch", action="store_true")
    parser.add_argument("--reward_local_files_only", action="store_true")

    # Joint objective
    parser.add_argument("--lambda_reward", type=float, default=0.1)
    parser.add_argument(
        "--reward_guidance_start_step",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--lambda_reward_ramp_steps",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--reward_updates_per_sr_update",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--reward_bound_loss_weight",
        type=float,
        default=1.0,
        help=(
            "Weight of the absolute +1/-1 calibration loss relative to "
            "the generated-pair ranking loss."
        ),
    )
    parser.add_argument(
        "--reward_bound_target",
        type=float,
        default=1,
        help="Calibrated upper/lower target magnitude after tanh.",
    )
    parser.add_argument(
        "--reward_bound_absolute_weight",
        type=float,
        default=0.25,
        help="Weight of direct upper/lower absolute regression.",
    )
    parser.add_argument(
        "--reward_bound_gap_weight",
        type=float,
        default=0,
        help="Weight of the same-LR upper-minus-lower gap loss.",
    )
    parser.add_argument(
        "--reward_bound_center_weight",
        type=float,
        default=0,
        help="Weight keeping the midpoint of upper/lower scores near zero.",
    )
    parser.add_argument(
        "--reward_bound_loss_type",
        type=str,
        default="mse",
        choices=["smooth_l1", "mse"],
    )
    parser.add_argument(
        "--reward_bound_smooth_l1_beta",
        type=float,
        default=0.1,
    )

    parser.add_argument("--reward_score_l2_weight", type=float, default=0.0)
    parser.add_argument(
        "--reward_margin_reg_weight",
        type=float,
        default=0,
    )
    parser.add_argument("--reward_max_margin", type=float, default=5.0)

    # Training
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--reward_bound_batch_size", type=int, default=1)
    parser.add_argument("--num_training_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=50000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--compile", action="store_true")

    # Optimizers
    parser.add_argument(
        "--learning_rate_sr_lora",
        type=float,
        default=1.0e-4,
    )
    parser.add_argument(
        "--learning_rate_encoder_lora",
        type=float,
        default=2.0e-5,
    )
    parser.add_argument(
        "--reward_learning_rate",
        type=float,
        default=1.0e-4,
    )
    parser.add_argument("--weight_decay", type=float, default=1.0e-2)
    parser.add_argument("--reward_weight_decay", type=float, default=1.0e-2)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--adam_epsilon", type=float, default=1.0e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--reward_max_grad_norm", type=float, default=1.0)
    parser.add_argument("--set_grads_to_none", action="store_true")

    # Runtime / logging
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--reward_bound_num_workers", type=int, default=2)
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="pnrsr",
    )
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--empty_cache_steps", type=int, default=500)
    parser.add_argument("--rankrandom", action="store_true")

    parsed = parser.parse_args()

    if parsed.reward_updates_per_sr_update < 1:
        raise ValueError(
            "--reward_updates_per_sr_update must be >= 1 "
            f"(got {parsed.reward_updates_per_sr_update})."
        )
    if parsed.lambda_reward_ramp_steps < 0:
        raise ValueError(
            "--lambda_reward_ramp_steps must be >= 0 "
            f"(got {parsed.lambda_reward_ramp_steps})."
        )

    return parsed


if __name__ == "__main__":
    main(parse_args())
