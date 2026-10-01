import os
import gc
import argparse
from pathlib import Path
from typing import Optional, List

import torch
import torch.utils.data
import transformers
import diffusers
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.utils import DistributedDataParallelKwargs
from accelerate.utils import ProjectConfiguration

from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params

from sr_utils.training_utils import MyDataset_blind_plus,MyDataset_blind_plus_withprompt

from srmodel import Zimage_SRmodel


def get_weight_dtype(mixed_precision):
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def sanitize_config(args):
    out = {}
    for k, v in vars(args).items():
        if isinstance(v, (int, float, str, bool)):
            out[k] = v
        elif v is None:
            out[k] = "None"
        else:
            out[k] = str(v)
    return out


def parse_float_list(s: Optional[str]) -> Optional[List[float]]:
    if s is None:
        return None
    s = str(s).strip()
    if s == "":
        return None
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def count_params(params):
    return sum(p.numel() for p in params)


def split_sr_and_encoder_params(model: Zimage_SRmodel):

    encoder_params = model.vae_encoder_lora_trainable_parameters()
    encoder_ids = {id(p) for p in encoder_params}

    sr_params = [
        p for p in model.sr_trainable_parameters()
        if id(p) not in encoder_ids
    ]

    return sr_params, encoder_params


def save_args(args, logging_out_dir: Path):
    logging_out_dir.mkdir(parents=True, exist_ok=True)
    args_txt_path = logging_out_dir / "args.txt"

    with open(args_txt_path, "w", encoding="utf-8") as f:
        for k, v in sorted(vars(args).items()):
            f.write(f"{k}: {v}\n")

    print(f"[args] saved to: {args_txt_path}")


def main(args):
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    logging_out_dir = Path(args.output_dir, "logs")
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=str(logging_out_dir),
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    if accelerator.is_main_process:
        save_args(args, logging_out_dir)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    weight_dtype = get_weight_dtype(args.mixed_precision)

    one_step_timesteps = parse_float_list(args.one_step_timesteps)
    sr_lora_path = args.resume_lora_path

    # ------------------------------------------------------------
    # 1. Model
    # ------------------------------------------------------------
    net_pix2pix = Zimage_SRmodel(
        pretrained_path=args.pretrained_path,
        sr_lora_path=sr_lora_path,
        dtype=weight_dtype,
        device=str(accelerator.device),

        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,

        encoder_lora_rank=args.encoder_lora_rank,
        encoder_lora_alpha=args.encoder_lora_alpha,
        encoder_lora_dropout=args.encoder_lora_dropout,
        encoder_lora_trainable=True,

        sr_lora_trainable=True,

        compile=args.compile,
        one_step_timesteps=one_step_timesteps,
        default_infer_timestep=args.default_infer_timestep,

        flow_loss_weight=args.flow_loss_weight,
        clean_loss_weight=args.clean_loss_weight,
        lpips_loss_weight=args.lpips_loss_weight,
        lpips_net=args.lpips_net,
        fdl_loss_weight=args.fdl_loss_weight,
    )

    # net_pix2pix.train()
    net_pix2pix.vae.eval()
    net_pix2pix.text_encoder.eval()

    if getattr(net_pix2pix, "lpips_fn", None) is not None:
        net_pix2pix.lpips_fn.eval()
    if getattr(net_pix2pix, "fdl_loss_fn", None) is not None:
        net_pix2pix.fdl_loss_fn.eval()

    # 只 cast trainable params 到 fp32。
    modules_to_cast = [net_pix2pix.transformer, net_pix2pix.vae.encoder]
    cast_training_params(modules_to_cast, dtype=torch.float32)

    sr_lora_params, encoder_lora_params = split_sr_and_encoder_params(net_pix2pix)

    if accelerator.is_main_process:
        print(f"SR LoRA parameters:          {count_params(sr_lora_params) / 1e6:.3f} M")
        print(f"VAE encoder LoRA parameters: {count_params(encoder_lora_params) / 1e6:.3f} M")
        print(
            "Total optimized parameters: "
            f"{(count_params(sr_lora_params) + count_params(encoder_lora_params)) / 1e6:.3f} M"
        )

    if len(sr_lora_params) == 0 and len(encoder_lora_params) == 0:
        raise RuntimeError("No SR trainable parameters found.")


    # ------------------------------------------------------------
    # 2. Optimizers
    # ------------------------------------------------------------
    optimizer_sr_groups = []

    if len(sr_lora_params) > 0:
        optimizer_sr_groups.append(
            {
                "name": "sr_lora",
                "params": sr_lora_params,
                "lr": args.learning_rate_sr_lora,
                "weight_decay": args.adam_weight_decay,
            }
        )

    if len(encoder_lora_params) > 0:
        optimizer_sr_groups.append(
            {
                "name": "vae_encoder_lora",
                "params": encoder_lora_params,
                "lr": args.learning_rate_encoder_lora,
                "weight_decay": args.adam_weight_decay,
            }
        )

    optimizer_sr = torch.optim.AdamW(
        optimizer_sr_groups,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
    )

    lr_scheduler_sr = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer_sr,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )



    # ------------------------------------------------------------
    # 3. Dataset
    # ------------------------------------------------------------
    dataset_train =  MyDataset_blind_plus_withprompt(
        lr_dir="path/to/your/train/lr_images",
        hr_dir="path/to/your/train/hr_images",
        prompt_dir="path/to/your/train/prompt",
        base_seed=args.seed,
    )

    if accelerator.is_main_process:
        print(f"Num of training data: {len(dataset_train)}")

    dl_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # ------------------------------------------------------------
    # 4. Prepare
    # ------------------------------------------------------------
    net_pix2pix, optimizer_sr, dl_train, lr_scheduler_sr = accelerator.prepare(
            net_pix2pix,
            optimizer_sr,
            dl_train,
            lr_scheduler_sr,
        )

    accelerator.unwrap_model(net_pix2pix).device = accelerator.device

    if accelerator.is_main_process:
        accelerator.init_trackers(
            args.tracker_project_name,
            config=sanitize_config(args),
        )

    progress_bar = tqdm(
        range(args.max_train_steps),
        initial=0,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    global_step = 0
    loss_window = []

    # ------------------------------------------------------------
    # 5. Train
    # ------------------------------------------------------------
    for epoch in range(args.num_training_epochs):
        for step, batch in enumerate(dl_train):
            with accelerator.accumulate(net_pix2pix):
                x_src = batch["conditioning_pixel_values"]
                x_tgt = batch["output_pixel_values"]
                prompt = batch.get("prompt", [args.prompt for _ in range(len(x_src))])


                out = net_pix2pix(
                    lr_img=x_src,
                    hr_img=x_tgt,
                    prompt=prompt,
                    max_sequence_length=args.max_sequence_length,
                    prompt_drop_prob=args.prompt_drop_prob,
                    return_outputs=True,
                    t=None,
                )

                loss = out["loss"]
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    params_to_clip = []
                    for group in optimizer_sr.param_groups:
                        params_to_clip.extend(group["params"])

                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer_sr.step()
                lr_scheduler_sr.step()
                optimizer_sr.zero_grad(set_to_none=args.set_grads_to_none)


            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                loss_value = out["loss"].detach().item()
                loss_window.append(loss_value)
                if len(loss_window) > args.loss_window_size:
                    loss_window.pop(0)

                logs = {
                    "loss": loss_value,
                    "loss_sr": out["loss_sr_total"].detach().item(),
                    "loss_base": out["loss_base"].detach().item(),
                    "loss_clean": out["loss_clean"].detach().item(),
                    "loss_lpips": out.get("loss_lpips", torch.tensor(0.0)).detach().item(),
                    "window_loss": sum(loss_window) / len(loss_window),
                }

                sr_lrs = lr_scheduler_sr.get_last_lr()
                for i, group in enumerate(optimizer_sr.param_groups):
                    if i < len(sr_lrs):
                        group_name = group.get("name", f"sr_group_{i}")
                        logs[f"lr_{group_name}"] = sr_lrs[i]

                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                if accelerator.is_main_process and global_step % args.checkpointing_steps == 0:
                    save_dir = os.path.join(args.output_dir, "checkpoints", f"checkpoint-{global_step}")
                    unwrapped_model = accelerator.unwrap_model(net_pix2pix)

                    unwrapped_model.save_sr_lora(os.path.join(save_dir, "sr"))

                    print(f"Saved checkpoint to: {save_dir}")

                if global_step % args.eval_freq == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                if global_step >= args.max_train_steps:
                    break

        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        final_save_dir = os.path.join(args.output_dir, "checkpoints", "final")
        unwrapped_model = accelerator.unwrap_model(net_pix2pix)

        unwrapped_model.save_sr_lora(os.path.join(final_save_dir, "sr"))


        print(f"Saved final checkpoint to: {final_save_dir}")

    accelerator.end_training()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pretrained_path", type=str, default=None)
    parser.add_argument(
        "--output_dir",
        type=str,
    )

    parser.add_argument("--resume_lora_path", type=str, default=None)

    parser.add_argument("--train_dir", required=True, nargs="+")
    parser.add_argument("--degradation_config", type=str, default="params_realesrgan_seesr.yml")
    parser.add_argument("--resolution", type=int, default=512)

    parser.add_argument("--prompt", type=str, default="high quality, sharp, detailed image")
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--prompt_drop_prob", type=float, default=0)

    parser.add_argument("--one_step_timesteps", type=str, default=None)
    parser.add_argument("--default_infer_timestep", type=float, default=500.0)

    parser.add_argument("--flow_loss_weight", type=float, default=1.0)
    parser.add_argument("--clean_loss_weight", type=float, default=1.0)
    parser.add_argument("--lpips_loss_weight", type=float, default=1.0)
    parser.add_argument("--fdl_loss_weight", type=float, default=0.001)
    parser.add_argument("--lpips_net", type=str, default="vgg", choices=["alex", "vgg", "squeeze"])


    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_training_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=10000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--compile", action="store_true")

    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)

    parser.add_argument("--encoder_lora_rank", type=int, default=16)
    parser.add_argument("--encoder_lora_alpha", type=int, default=16)
    parser.add_argument("--encoder_lora_dropout", type=float, default=0.0)

    parser.add_argument("--learning_rate_sr_lora", type=float, default=1e-4)
    parser.add_argument("--learning_rate_encoder_lora", type=float, default=2e-5)


    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--set_grads_to_none", action="store_true")

    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--lr_power", type=float, default=1.0)

    parser.add_argument("--dataloader_num_workers", type=int, default=4)

    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--tracker_project_name", type=str, default="zimage_one_step_sr_vsd")
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--eval_freq", type=int, default=500)
    parser.add_argument("--loss_window_size", type=int, default=100)

    args = parser.parse_args()


    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)