import os
import sys
from contextlib import contextmanager
from typing import Optional, List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model

current_file = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(current_file))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.utils import AttentionBackend, load_from_local_dir


EXTRA_WEIGHTS_NAME = "non_lora_trainable.pt"

VAE_ENCODER_LORA_DIR = "vae_encoder_lora"
VAE_ENCODER_ADAPTER_NAME = "lr_encoder"


class Zimage_SRmodel(nn.Module):


    def __init__(
        self,
        pretrained_path: Optional[str] = None,
        sr_lora_path: Optional[str] = None,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        lora_rank: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        encoder_lora_path: Optional[str] = None,
        encoder_lora_rank: int = 16,
        encoder_lora_alpha: int = 16,
        encoder_lora_dropout: float = 0.0,
        encoder_lora_trainable: bool = True,
        sr_lora_trainable: bool = True,
        compile: bool = False,
        one_step_timesteps: Optional[List[float]] = None,
        default_infer_timestep: float = 1000.0,
        flow_loss_weight: float = 1.0,
        clean_loss_weight: float = 1.0,
        lpips_loss_weight: float = 1.0,
        fdl_loss_weight: float = 0.001,
        lpips_net: str = "vgg",
        lora_path: Optional[str] = None,
        lora_trainable: Optional[bool] = None,
    ):
        super().__init__()


        if sr_lora_path is None and lora_path is not None:
            sr_lora_path = lora_path
        if lora_trainable is not None:
            sr_lora_trainable = bool(lora_trainable)

        default_zimage_path = "Tongyi-MAI/Z-Image"
        zimage_path = pretrained_path if pretrained_path is not None else default_zimage_path

        self.dtype = dtype
        self.device = torch.device(device)

        self.default_infer_timestep = float(default_infer_timestep)
        self.fdl_loss_weight=float(fdl_loss_weight)
        self.flow_loss_weight = float(flow_loss_weight)
        self.clean_loss_weight = float(clean_loss_weight)
        self.lpips_loss_weight = float(lpips_loss_weight)
        self.lpips_net = lpips_net
        self.lpips_fn = None

        scheduler_ts = torch.tensor(one_step_timesteps, dtype=torch.float32)
        zimage_t = (1000.0 - scheduler_ts) / 1000.0

        self.register_buffer("one_step_timesteps", scheduler_ts, persistent=False)
        self.register_buffer("one_step_t_values", zimage_t, persistent=False)

        components = load_from_local_dir(
            zimage_path,
            device=device,
            dtype=dtype,
            compile=compile,
        )
        AttentionBackend.print_available_backends()

        self.transformer = components["transformer"]
        self.vae = components["vae"].to(dtype)
        self.text_encoder = components["text_encoder"].to(dtype)
        self.tokenizer = components["tokenizer"]
        self.scheduler = components["scheduler"]

        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.transformer.requires_grad_(False)
        self.midt = (1000-default_infer_timestep)/1000
        prompt_payload = torch.load(
            "pnrsr/fixed_prompt_feat.pt",
            map_location="cpu",
        )

        self.register_buffer(
            "stu_cap_feat",
            prompt_payload["cap_feat"].to(dtype=self.dtype),
            persistent=False,
        )

        if encoder_lora_path is None and sr_lora_path is not None:
            candidate = os.path.join(sr_lora_path, VAE_ENCODER_LORA_DIR)
            if os.path.exists(os.path.join(candidate, "adapter_config.json")):
                encoder_lora_path = candidate

        self._init_vae_encoder_lora(
            encoder_lora_path=encoder_lora_path,
            rank=encoder_lora_rank,
            alpha=encoder_lora_alpha,
            dropout=encoder_lora_dropout,
            trainable=encoder_lora_trainable,
        )

        target_modules = ["attention.to_q","attention.to_k","attention.to_v","attention.to_out.0","feed_forward.w1","feed_forward.w2","feed_forward.w3",]


        self._init_transformer_adapters(
            target_modules=target_modules,
            sr_lora_path=sr_lora_path,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            sr_trainable=sr_lora_trainable,
        )

        if self.lpips_loss_weight > 0:
            self._init_lpips()
            self.lpips_fn.requires_grad_(False)
        

        from FDL_pytorch import FDL_loss

        self.fdl_loss_fn = FDL_loss(
            patch_size=5,
            stride=1,
            num_proj=256,
            model="VGG",
            phase_weight=1,
        ).to(device).eval()
        self.fdl_loss_fn.requires_grad_(False)

        
        self.print_trainable_parameters()

    # ============================================================
    # LoRA helpers
    # ============================================================
    def _make_noisy_latent_2(cls, x0: torch.Tensor, mid_noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_view = cls._expand_t(t, x0)
        v= (mid_noise-x0)/(1-cls.midt)
        return x0+(1-t_view)*v
    

    @staticmethod
    def _make_lora_config(
        target_modules: List[str],
        r: int,
        alpha: int,
        dropout: float,
    ) -> LoraConfig:
        return LoraConfig(
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules,
            bias="none",
        )

    @staticmethod
    def _is_lora_key(name: str) -> bool:
        return (
            "lora_A" in name
            or "lora_B" in name
            or "lora_embedding_A" in name
            or "lora_embedding_B" in name
            or "lora_magnitude_vector" in name
        )

    @staticmethod
    def _has_adapter_name(name: str, adapter_name: str) -> bool:
        patterns = (
            f"lora_A.{adapter_name}",
            f"lora_B.{adapter_name}",
            f"lora_embedding_A.{adapter_name}",
            f"lora_embedding_B.{adapter_name}",
            f"lora_magnitude_vector.{adapter_name}",
            f".{adapter_name}.",
            f".{adapter_name}.weight",
        )
        return any(p in name for p in patterns)

    def _is_adapter_lora_param(self, name: str, adapter_name: str) -> bool:
        return self._is_lora_key(name) and self._has_adapter_name(name, adapter_name)

    @staticmethod
    def _snapshot_lora_requires_grad(module: nn.Module) -> Dict[int, bool]:
        return {
            id(p): p.requires_grad
            for name, p in module.named_parameters()
            if Zimage_SRmodel._is_lora_key(name)
        }

    @staticmethod
    def _restore_lora_requires_grad(module: nn.Module, states: Dict[int, bool]):
        if not states:
            return

        for name, p in module.named_parameters():
            if Zimage_SRmodel._is_lora_key(name) and id(p) in states:
                p.requires_grad_(states[id(p)])

    def _set_lora_trainable(
        self,
        module: nn.Module,
        adapter_name: str,
        trainable: bool,
        verbose: bool = True,
        prefix: str = "lora",
    ):
        n_tensors = 0
        n_params = 0

        for name, p in module.named_parameters():
            if self._is_adapter_lora_param(name, adapter_name):
                p.requires_grad_(trainable)
                n_tensors += 1
                n_params += p.numel()

        if verbose:
            print(
                f"[{prefix}] adapter={adapter_name}, trainable={trainable}, "
                f"tensors={n_tensors}, params={n_params / 1e6:.3f} M"
            )

    def _set_adapter_trainable(self, adapter_name: str, trainable: bool, verbose: bool = True):
        self._set_lora_trainable(
            module=self.transformer,
            adapter_name=adapter_name,
            trainable=trainable,
            verbose=verbose,
            prefix="transformer-lora",
        )

    @contextmanager
    def use_adapter(self, adapter_name: str):
        if not isinstance(self.transformer, PeftModel):
            yield
            return

        states = self._snapshot_lora_requires_grad(self.transformer)
        old_adapter = getattr(self.transformer, "active_adapter", None)

        self.transformer.set_adapter(adapter_name)
        self._restore_lora_requires_grad(self.transformer, states)

        try:
            yield
        finally:
            if old_adapter is not None:
                try:
                    self.transformer.set_adapter(old_adapter)
                except Exception:
                    pass
            self._restore_lora_requires_grad(self.transformer, states)


    # ============================================================
    # VAE encoder LoRA
    # ============================================================

    @staticmethod
    def _find_vae_encoder_targets(encoder: nn.Module) -> List[str]:
        targets = []

        for name, module in encoder.named_modules():
            if name and isinstance(module, (nn.Conv2d, nn.Linear)):
                targets.append(name)

        if len(targets) == 0:
            print("[vae-encoder-lora][warning] no Conv2d/Linear target modules found.")

        return targets

    def _init_vae_encoder_lora(
        self,
        encoder_lora_path: Optional[str],
        rank: int,
        alpha: int,
        dropout: float,
        trainable: bool,
    ):
        if encoder_lora_path is None and not trainable:
            print("[vae-encoder-lora] disabled.")
            return

        targets = self._find_vae_encoder_targets(self.vae.encoder)
        self.vae.requires_grad_(False)
        if encoder_lora_path is not None:
            self.vae.encoder = PeftModel.from_pretrained(
                self.vae.encoder,
                encoder_lora_path,
                is_trainable=trainable,
                adapter_name=VAE_ENCODER_ADAPTER_NAME,
            )
            print(f"[vae-encoder-lora] loaded: {encoder_lora_path}")
        else:
            config = LoraConfig(
                r=rank,
                lora_alpha=alpha,
                lora_dropout=dropout,
                target_modules=targets,
                bias="none",
            )
            self.vae.encoder = get_peft_model(
                self.vae.encoder,
                config,
                adapter_name=VAE_ENCODER_ADAPTER_NAME,
            )
            print(f"[vae-encoder-lora] created: {VAE_ENCODER_ADAPTER_NAME}")

        self.vae.encoder.to(device=self.device, dtype=self.dtype)


        self._set_lora_trainable(
            module=self.vae.encoder,
            adapter_name=VAE_ENCODER_ADAPTER_NAME,
            trainable=trainable,
            verbose=True,
            prefix="vae-encoder-lora",
        )

    @contextmanager
    def use_vae_encoder_lora(self, enabled: bool):
        if not isinstance(self.vae.encoder, PeftModel):
            yield
            return

        states = self._snapshot_lora_requires_grad(self.vae.encoder)

        if enabled:
            old_adapter = getattr(self.vae.encoder, "active_adapter", None)

            self.vae.encoder.set_adapter(VAE_ENCODER_ADAPTER_NAME)
            self._restore_lora_requires_grad(self.vae.encoder, states)

            try:
                yield
            finally:
                if old_adapter is not None:
                    try:
                        self.vae.encoder.set_adapter(old_adapter)
                    except Exception:
                        pass
                self._restore_lora_requires_grad(self.vae.encoder, states)
        else:
            with self.vae.encoder.disable_adapter():
                yield

            self._restore_lora_requires_grad(self.vae.encoder, states)

    def vae_encoder_lora_trainable_parameters(self) -> List[nn.Parameter]:
        if not isinstance(self.vae.encoder, PeftModel):
            return []

        return [
            p
            for name, p in self.vae.encoder.named_parameters()
            if p.requires_grad and self._is_adapter_lora_param(name, VAE_ENCODER_ADAPTER_NAME)
        ]

    def _save_vae_encoder_lora(self, save_dir: str):
        if not isinstance(self.vae.encoder, PeftModel):
            return

        encoder_save_dir = os.path.join(save_dir, VAE_ENCODER_LORA_DIR)
        os.makedirs(encoder_save_dir, exist_ok=True)

        try:
            self.vae.encoder.save_pretrained(
                encoder_save_dir,
                selected_adapters=[VAE_ENCODER_ADAPTER_NAME],
            )
        except TypeError:
            self.vae.encoder.set_adapter(VAE_ENCODER_ADAPTER_NAME)
            self.vae.encoder.save_pretrained(encoder_save_dir)

        print(f"[save_vae_encoder_lora] saved to: {encoder_save_dir}")

    # ============================================================
    # Transformer LoRA adapters
    # ============================================================

    def _init_transformer_adapters(
        self,
        target_modules: List[str],
        sr_lora_path: Optional[str],
        rank: int,
        alpha: int,
        dropout: float,
        sr_trainable: bool,
    ):
        sr_config = self._make_lora_config(
            target_modules,
            rank,
            alpha,
            dropout,
        )

        if sr_lora_path is not None:
            self.transformer = PeftModel.from_pretrained(
                self.transformer,
                sr_lora_path,
                is_trainable=sr_trainable,
                adapter_name="sr",
            )
            print(f"[transformer-lora] loaded SR adapter: {sr_lora_path}")
        else:
            self.transformer = get_peft_model(
                self.transformer,
                sr_config,
                adapter_name="sr",
            )
            print("[transformer-lora] created SR adapter: sr")

        self.transformer.requires_grad_(False)
        self.transformer.set_adapter("sr")

        self._set_adapter_trainable(
            "sr",
            sr_trainable,
        )

        print(
            f"[transformer-lora] target Linear modules: "
            f"{len(target_modules)}"
        )

    # ============================================================
    # Trainable parameter groups
    # ============================================================

    def sr_trainable_parameters(self) -> List[nn.Parameter]:
        self._set_adapter_trainable("sr", True, verbose=False)

        params = [
            p
            for name, p in self.transformer.named_parameters()
            if p.requires_grad and self._is_adapter_lora_param(name, "sr")
        ]

        params.extend(self.vae_encoder_lora_trainable_parameters())

        return params

    def print_trainable_parameters(self):
        total = 0
        trainable = 0

        for _, p in self.named_parameters():
            n = p.numel()
            total += n
            if p.requires_grad:
                trainable += n

        print(f"[trainable] {trainable:,} / {total:,} params ({100.0 * trainable / max(total, 1):.4f}%)")

        for name, p in self.named_parameters():
            if p.requires_grad:
                print(f"  - {name}")

    # ============================================================
    # Save
    # ============================================================

    def save_sr_lora(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)

        old_adapter = getattr(self.transformer, "active_adapter", None)
        try:
            self.transformer.save_pretrained(save_dir, selected_adapters=["sr"])
        except TypeError:
            self.transformer.set_adapter("sr")
            self.transformer.save_pretrained(save_dir)
        finally:
            if old_adapter is not None:
                try:
                    self.transformer.set_adapter(old_adapter)
                except Exception:
                    pass

        torch.save(
            {
                "format_version": 1,
                "role": "zimage_one_step_sr_adapter",
                "conditioning_mode": "one_step",
                "one_step_timesteps": self.one_step_timesteps.detach().cpu().tolist(),
                "one_step_t_values": self.one_step_t_values.detach().cpu().tolist(),
                "default_infer_timestep": self.default_infer_timestep,
                "flow_loss_weight": self.flow_loss_weight,
                "clean_loss_weight": self.clean_loss_weight,
                "lpips_loss_weight": self.lpips_loss_weight,
                "state_dict": {},
            },
            os.path.join(save_dir, EXTRA_WEIGHTS_NAME),
        )

        self._save_vae_encoder_lora(save_dir)

        print(f"[save_sr_lora] saved to: {save_dir}")


    # ============================================================
    # Time / flow helpers
    # ============================================================

    def scheduler_timestep_to_zimage_t(self, timestep: torch.Tensor) -> torch.Tensor:
        timestep = timestep.to(device=self.device, dtype=torch.float32)
        return (1000.0 - timestep) / 1000.0

    def zimage_t_to_scheduler_timestep(self, t: torch.Tensor) -> torch.Tensor:
        t = t.to(device=self.device, dtype=torch.float32)
        return 1000.0 - 1000.0 * t

    def sample_one_step_t(self, batch_size: int) -> torch.Tensor:
        idx = torch.randint(
            0,
            self.one_step_t_values.shape[0],
            (batch_size,),
            device=self.device,
        )
        return self.one_step_t_values.to(self.device)[idx].float()


    @staticmethod
    def _expand_t(t: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return t.view(target.shape[0], 1, 1, 1).to(device=target.device, dtype=target.dtype)

    @classmethod
    def _make_noisy_latent(cls, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_view = cls._expand_t(t, x0)
        return t_view * x0 + (1.0 - t_view) * noise

    # ============================================================
    # Encode / decode / LPIPS
    # ============================================================

    def _init_lpips(self):
        try:
            import lpips
        except Exception as e:
            raise ImportError(
                "lpips_loss_weight > 0 but package `lpips` is not installed. "
                "Install it with `pip install lpips`, or set lpips_loss_weight=0."
            ) from e

        self.lpips_fn = lpips.LPIPS(net=self.lpips_net).to(self.device).eval()
        self.lpips_fn.requires_grad_(False)

        print(f"[LPIPS] enabled, net={self.lpips_net}, weight={self.lpips_loss_weight}")

    def encode_prompt_zimage(self, prompt, batch_size: int, max_sequence_length: int = 512):
        if prompt is None:
            prompt = [""] * batch_size
        elif isinstance(prompt, str):
            prompt = [prompt] * batch_size

        formatted = []

        for p in prompt:
            messages = [{"role": "user", "content": p}]
            formatted.append(
                self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
            )

        text_inputs = self.tokenizer(
            formatted,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = text_inputs.input_ids.to(self.device)
        attention_mask = text_inputs.attention_mask.to(self.device).bool()

        with torch.no_grad():
            hidden = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            ).hidden_states[-2]

        hidden = hidden.to(device=self.device, dtype=self.dtype)

        return [hidden[i][attention_mask[i]] for i in range(batch_size)]

    def encode_image_to_latent(
        self,
        img: torch.Tensor,
        sample: bool = True,
        use_encoder_lora: bool = False,
        encoder_lora_grad: bool = False,
    ) -> torch.Tensor:
        img = img.to(device=self.device, dtype=self.dtype)

        grad_enabled = bool(use_encoder_lora and encoder_lora_grad)
        grad_ctx = torch.enable_grad() if grad_enabled else torch.no_grad()

        with self.use_vae_encoder_lora(enabled=use_encoder_lora):
            with grad_ctx:
                if hasattr(self.vae, "encode"):
                    enc_out = self.vae.encode(img)

                    if hasattr(enc_out, "latent_dist"):
                        latent = enc_out.latent_dist.sample() if sample else enc_out.latent_dist.mode()
                    else:
                        latent = enc_out[0] if isinstance(enc_out, tuple) else enc_out
                else:
                    h = self.vae.encoder(img)
                    moments = self.vae.quant_conv(h) if self.vae.quant_conv is not None else h
                    mean, logvar = torch.chunk(moments, 2, dim=1)
                    logvar = torch.clamp(logvar, -30.0, 20.0)

                    if sample:
                        latent = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)
                    else:
                        latent = mean

        shift = getattr(self.vae.config, "shift_factor", 0.0) or 0.0
        scale = getattr(self.vae.config, "scaling_factor", 1.0)

        return ((latent - shift) * scale).to(device=self.device, dtype=self.dtype)

    def decode_latent_to_image_grad(self, latents: torch.Tensor) -> torch.Tensor:
        vae_dtype = getattr(self.vae, "dtype", self.dtype)
        shift = getattr(self.vae.config, "shift_factor", 0.0) or 0.0
        scale = getattr(self.vae.config, "scaling_factor", 1.0)

        latents = latents.to(device=self.device, dtype=vae_dtype)
        latents = latents / scale + shift

        image = self.vae.decode(latents, return_dict=False)[0]
        return image.clamp(-1, 1)

    @torch.no_grad()
    def decode_latent_to_image(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decode_latent_to_image_grad(latents)

    def compute_lpips_loss_from_image(
        self,
        pred_img: torch.Tensor,
        hr_img: torch.Tensor,
    ) -> torch.Tensor:
        if self.lpips_loss_weight <= 0:
            return pred_img.new_tensor(0.0, dtype=torch.float32)

        if self.lpips_fn is None:
            self._init_lpips()

        hr_img = hr_img.to(device=pred_img.device, dtype=pred_img.dtype).clamp(-1, 1)

        return self.lpips_fn(
            pred_img.float(),
            hr_img.float(),
        ).mean()

    # ============================================================
    # Transformer
    # ============================================================

    def _latent_to_list(self, latent: torch.Tensor) -> List[torch.Tensor]:
        latent = latent.to(device=self.device, dtype=self.dtype)
        return list(latent.unsqueeze(2).unbind(dim=0))

    def _cond_to_list(self, cond_latent: Optional[torch.Tensor]) -> Optional[List[torch.Tensor]]:
        if cond_latent is None:
            return None

        cond_latent = cond_latent.to(device=self.device, dtype=self.dtype)
        return list(cond_latent.unsqueeze(2).unbind(dim=0))

    def _run_transformer(
        self,
        latent: torch.Tensor,
        t: torch.Tensor,
        cap_feats: List[torch.Tensor],
        cond_latent: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out_list, _ = self.transformer(
            self._latent_to_list(latent),
            t.to(device=self.device, dtype=torch.float32),
            cap_feats,
            cond=self._cond_to_list(cond_latent),
        )

        return torch.stack(out_list, dim=0).squeeze(2)

    def _predict_one_step(
        self,
        start_latent: torch.Tensor,
        lr_latent: torch.Tensor,
        t: torch.Tensor,
        cap_feats: List[torch.Tensor],
    ):
        with self.use_adapter("sr"):
            model_flow = self._run_transformer(
                latent=start_latent,
                t=t,
                cap_feats=cap_feats,
                cond_latent=lr_latent,
            )

        t_view = self._expand_t(t, start_latent)
        pred_clean_latent = start_latent + (1.0 - t_view) * model_flow

        return model_flow, pred_clean_latent

    # ============================================================
    # Forward
    # ============================================================
    
    
    def forward(
        self,
        lr_img: torch.Tensor,
        hr_img: torch.Tensor,
        prompt=None,
        negative_prompt=None,
        max_sequence_length: int = 512,
        prompt_drop_prob: float = 0.3,
        t: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        return_outputs: bool = False,
    ):

        batch_size = lr_img.shape[0]

        lr_img = lr_img.to(device=self.device, dtype=self.dtype)
        hr_img = hr_img.to(device=self.device, dtype=self.dtype)

        lr_latent = self.encode_image_to_latent(
            lr_img,
            sample=False,
            use_encoder_lora=True,
            encoder_lora_grad=True,
        )

        hr_latent = self.encode_image_to_latent(
            hr_img,
            sample=False,
            use_encoder_lora=False,
            encoder_lora_grad=False,
        )


        stu_cap_feats = [self.stu_cap_feat]* batch_size
        if noise is None:
            noise = torch.randn_like(hr_latent)
        else:
            noise = noise.to(device=self.device, dtype=self.dtype)
        
        if t is None:
            t = self.sample_one_step_t(batch_size)
        else:
            t = t.to(device=self.device, dtype=self.dtype)


        mid_noise = self.midt*lr_latent +(1-self.midt)*noise

        noisy_input = self._make_noisy_latent_2(hr_latent, mid_noise, t)

 
        model_flow, pred_latent = self._predict_one_step(
            start_latent=noisy_input,
            lr_latent=lr_latent,
            t=t,
            cap_feats=stu_cap_feats,
        )

        pred_img = self.decode_latent_to_image_grad(pred_latent)

        hr_img_for_loss = hr_img.to(
            device=pred_img.device,
            dtype=pred_img.dtype,
        ).clamp(-1, 1)

        loss_clean = F.mse_loss(
            pred_latent.float(),
            hr_latent.float(),
            reduction="mean",
        )

        pred_fdl = (pred_img.float() * 0.5 + 0.5).clamp(0, 1)
        gt_fdl = (hr_img_for_loss.float() * 0.5 + 0.5).clamp(0, 1)



        with torch.cuda.amp.autocast(enabled=False):
            loss_fdl = self.fdl_loss_fn(
                pred_fdl.contiguous(),
                gt_fdl.contiguous(),
            ).mean()

        loss_lpips = self.compute_lpips_loss_from_image(
            pred_img=pred_img,
            hr_img=hr_img_for_loss,
        )

        loss_base = (
            + self.clean_loss_weight * loss_clean
            + self.lpips_loss_weight * loss_lpips
            + self.fdl_loss_weight*loss_fdl

        )

        loss = loss_base
        loss_sr_total = loss_base


        if not return_outputs:
            return loss

        out: Dict[str, Any] = {
            "loss": loss,
            "loss_base": loss_base,
            "loss_sr_total": loss_sr_total,
            "loss_clean": loss_clean.detach(),
            "loss_lpips": loss_lpips.detach(),
            "loss_fdl": loss_fdl.detach(),
            "pred_img": pred_img,

            "model_flow": model_flow,
            "pred_latent": pred_latent,
            "hr_latent": hr_latent,
            "lr_latent": lr_latent,
            "noise": noise,
        }


        return out


    
    # ============================================================
    # Inference
    # ============================================================

    @torch.no_grad()
    def inference(
        self,
        lr_img: torch.Tensor,
        prompt=None,
        timestep: Optional[float] = None,
        max_sequence_length: int = 512,
        noise: Optional[torch.Tensor] = None,
    ):
        self.eval()

        lr_img = lr_img.to(device=self.device, dtype=self.dtype)

        # 1. LR -> latent
        lr_latent = self.encode_image_to_latent(
            lr_img,
            sample=False,
            use_encoder_lora=True,
            encoder_lora_grad=False,
        )

        batch_size = lr_latent.shape[0]

        # 2. text condition
        cap_feats = self.encode_prompt_zimage(
            prompt=prompt,
            batch_size=batch_size,
            max_sequence_length=max_sequence_length,
        )

        # 3. noise
        if noise is None:
            noise = torch.randn_like(lr_latent)
        else:
            noise = noise.to(device=self.device, dtype=self.dtype)

        # 4. scheduler timestep -> Z-Image t
        if timestep is None:
            timestep = self.default_infer_timestep

        scheduler_t = torch.full(
            (batch_size,),
            float(timestep),
            device=self.device,
            dtype=torch.float32,
        )
        t = self.scheduler_timestep_to_zimage_t(scheduler_t)

        # 5. LR + noise starting state
        x = self._make_noisy_latent(
            lr_latent,
            noise,
            t,
        )

        # 6. exactly one SR forward
        _, pred_latent = self._predict_one_step(
            start_latent=x,
            lr_latent=lr_latent,
            t=t,
            cap_feats=cap_feats,
        )

        # 7. latent -> image
        return self.decode_latent_to_image(pred_latent)
    

    @torch.no_grad()
    def generate_realism_pair(
        self,
        lr_img: torch.Tensor,
        hr_img: torch.Tensor,
        prompt: str = "high quality, sharp, detailed image",
        t_pair: Optional[List[float]] = None,
        t_values: Optional[List[float]] = None,
        max_sequence_length: int = 512,
        noise: Optional[torch.Tensor] = None,
        decode: bool = True,
    ):
        self.eval()

        lr_img = lr_img.to(device=self.device, dtype=self.dtype)
        hr_img = hr_img.to(device=self.device, dtype=self.dtype)

        batch_size = lr_img.shape[0]

        lr_latent = self.encode_image_to_latent(
            lr_img,
            sample=False,
            use_encoder_lora=True,
            encoder_lora_grad=False,
        )

        hr_latent = self.encode_image_to_latent(
            hr_img,
            sample=False,
            use_encoder_lora=False,
            encoder_lora_grad=False,
        )

        cap_feats = [self.stu_cap_feat.to(lr_latent.device)] * batch_size

        if noise is None:
            noise = torch.randn_like(hr_latent)
        else:
            noise = noise.to(device=self.device, dtype=self.dtype)

        if noise.ndim == 4 and noise.shape[-1] == hr_latent.shape[1]:
            noise = noise.permute(0, 3, 1, 2).contiguous()

        assert noise.shape == hr_latent.shape, (
            f"noise shape must match hr_latent shape. "
            f"noise={tuple(noise.shape)}, hr_latent={tuple(hr_latent.shape)}"
        )

        base_noise = noise
        mid_noise = self.midt*lr_latent +(1-self.midt)*base_noise

        if t_pair is None:
            if t_values is None:
                t_values_tensor = self.one_step_t_values.detach().float().to(self.device)
            else:
                t_values_tensor = torch.tensor(
                    t_values,
                    device=self.device,
                    dtype=torch.float32,
                )

            assert t_values_tensor.numel() >= 2, "Need at least two t values."

            perm = torch.randperm(t_values_tensor.numel(), device=self.device)
            t0 = float(t_values_tensor[perm[0]].item())
            t1 = float(t_values_tensor[perm[1]].item())
        else:
            assert len(t_pair) == 2, "t_pair must contain exactly two Z-Image t values."
            t0 = float(t_pair[0])
            t1 = float(t_pair[1])

        assert 0.0 <= t0 <= 1.0 and 0.0 <= t1 <= 1.0, (
            f"Z-Image t must be in [0, 1], got {t0}, {t1}."
        )
        assert abs(t0 - t1) > 1e-8, "The two t values must be different."

        low_t_value = min(t0, t1)
        high_t_value = max(t0, t1)

        low_t = torch.full(
            (batch_size,),
            low_t_value,
            device=self.device,
            dtype=torch.float32,
        )
        high_t = torch.full(
            (batch_size,),
            high_t_value,
            device=self.device,
            dtype=torch.float32,
        )

        def solve_one(t: torch.Tensor):
            noisy_input = self._make_noisy_latent_2(
                hr_latent,
                mid_noise,
                t,
            )

            model_flow, pred_latent = self._predict_one_step(
                start_latent=noisy_input,
                lr_latent=lr_latent,
                t=t,
                cap_feats=cap_feats,
            )

            pred_img = self.decode_latent_to_image(pred_latent) if decode else None

            return {
                "model_flow": model_flow,
                "pred_latent": pred_latent,
                "pred_img": pred_img,
                "noisy_input": noisy_input,
            }

        low_out = solve_one(low_t)
        high_out = solve_one(high_t)

        result = {
            "low_t": low_t,
            "high_t": high_t,
            "low_scheduler_timestep": self.zimage_t_to_scheduler_timestep(low_t),
            "high_scheduler_timestep": self.zimage_t_to_scheduler_timestep(high_t),
            "low_latent": low_out["pred_latent"],
            "high_latent": high_out["pred_latent"],
            "low_noisy_input": low_out["noisy_input"],
            "high_noisy_input": high_out["noisy_input"],
            "hr_latent": hr_latent,
            "lr_latent": lr_latent,
            "base_noise": base_noise,
            "gaussian_noise": noise,
            "label": "high_t_is_more_real",
        }

        if decode:
            result["low_image"] = low_out["pred_img"]
            result["high_image"] = high_out["pred_img"]

        return result