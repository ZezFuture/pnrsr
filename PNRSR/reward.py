import math
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SiglipVisionModel, AutoImageProcessor

class TokenCrossAttentionFusionBlock(nn.Module):
    """
    LR-conditioned SR plausibility block.

    SR tokens form the primary representation, while LR tokens
    provide conditional information through gated cross-attention.
    """

    def __init__(
        self,
        in_dim: int,
        attn_dim: int = 512,
        num_heads: int = 8,
        num_fusion_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        if attn_dim % num_heads != 0:
            raise ValueError(
                f"attn_dim={attn_dim} must be divisible by num_heads={num_heads}"
            )

        self.in_dim = in_dim
        self.attn_dim = attn_dim
        self.num_heads = num_heads
        self.out_dim = attn_dim

        # -------------------------
        # LR / SR feature projection
        # -------------------------
        self.lr_norm = nn.LayerNorm(in_dim)
        self.sr_norm = nn.LayerNorm(in_dim)

        self.lr_proj = nn.Linear(in_dim, attn_dim)
        self.sr_proj = nn.Linear(in_dim, attn_dim)

        # -------------------------
        # LR-conditioned interaction
        # SR <- LR
        # -------------------------
        self.lr_condition_attn = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Learn how much LR information should modify each SR token
        self.gate = nn.Sequential(
            nn.Linear(attn_dim * 2, attn_dim),
            nn.Sigmoid(),
        )

        self.cond_proj = nn.Linear(attn_dim, attn_dim)

        # -------------------------
        # SR-centered refinement
        # -------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=attn_dim,
            nhead=num_heads,
            dim_feedforward=attn_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.fusion_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_fusion_layers,
        )

        # -------------------------
        # Pooling
        # -------------------------
        self.pool_query = nn.Parameter(
            torch.randn(1, 1, attn_dim) * 0.02
        )

        self.pool_attn = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.out_norm = nn.LayerNorm(attn_dim)

    def forward(
        self,
        lr_tokens: torch.Tensor,
        sr_tokens: torch.Tensor,
    ) -> torch.Tensor:

        lr_tokens = self.lr_norm(lr_tokens.float())
        sr_tokens = self.sr_norm(sr_tokens.float())

        lr = self.lr_proj(lr_tokens)
        sr = self.sr_proj(sr_tokens)

        # --------------------------------------------------
        # LR is used as condition for the SR representation
        # --------------------------------------------------
        lr_context, _ = self.lr_condition_attn(
            query=sr,
            key=lr,
            value=lr,
            need_weights=False,
        )

        lr_context = self.cond_proj(lr_context)

        # --------------------------------------------------
        # Gated conditional correction
        # --------------------------------------------------
        gate = self.gate(
            torch.cat([sr, lr_context], dim=-1)
        )

        conditioned_sr = sr + gate * lr_context

        # --------------------------------------------------
        # Further reasoning is performed on SR-centered tokens
        # --------------------------------------------------
        fused_tokens = self.fusion_encoder(conditioned_sr)

        # --------------------------------------------------
        # Attention pooling
        # --------------------------------------------------
        batch_size = fused_tokens.shape[0]
        query = self.pool_query.expand(batch_size, -1, -1)

        pooled, _ = self.pool_attn(
            query=query,
            key=fused_tokens,
            value=fused_tokens,
            need_weights=False,
        )

        pooled = pooled.squeeze(1)
        pooled = self.out_norm(pooled)

        return pooled
        

class SRQualityBranch(nn.Module):
    """SR-only perceptual quality branch."""

    def __init__(
        self,
        in_dim: int,
        attn_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 1,
        dropout: float = 0.1,
        use_pooler: bool = True,
    ):
        super().__init__()

        if attn_dim % num_heads != 0:
            raise ValueError(
                f"attn_dim={attn_dim} must be divisible by num_heads={num_heads}"
            )

        self.in_dim = in_dim
        self.attn_dim = attn_dim
        self.use_pooler = use_pooler

        self.token_norm = nn.LayerNorm(in_dim)
        self.token_proj = nn.Linear(in_dim, attn_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=attn_dim,
            nhead=num_heads,
            dim_feedforward=attn_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.pool_query = nn.Parameter(torch.randn(1, 1, attn_dim) * 0.02)
        self.pool_attn = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        if use_pooler:
            self.pooler_proj = nn.Linear(in_dim, attn_dim)
            self.out_dim = attn_dim * 2
        else:
            self.pooler_proj = None
            self.out_dim = attn_dim

        self.out_norm = nn.LayerNorm(self.out_dim)

    def forward(
        self,
        sr_tokens: torch.Tensor,
        sr_pooler: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        sr_tokens = self.token_norm(sr_tokens.float())
        sr_tokens = self.token_proj(sr_tokens)
        sr_tokens = self.encoder(sr_tokens)

        batch_size = sr_tokens.shape[0]
        query = self.pool_query.expand(batch_size, -1, -1)

        pooled, _ = self.pool_attn(
            query=query,
            key=sr_tokens,
            value=sr_tokens,
            need_weights=False,
        )
        pooled = pooled.squeeze(1)

        if self.use_pooler:
            sr_pooler = self.pooler_proj(sr_pooler.float())
            quality_feat = torch.cat([pooled, sr_pooler], dim=-1)
        else:
            quality_feat = pooled

        return self.out_norm(quality_feat)


class SigLIP2PairSRReward(nn.Module):
    """SigLIP2-based LR-conditioned SR reward model."""

    def __init__(
        self,
        model_name: str = "google/siglip2-so400m-patch16-512",
        image_size: int = 512,
        freeze_backbone: bool = True,
        use_pooler: bool = True,
        hidden_layers: Tuple[int, ...] = (-1, -3, -6, -9),
        token_pool_size: Optional[int] = 32,
        attn_dim: int = 512,
        num_heads: int = 8,
        num_fusion_layers: int = 3,
        sr_quality_layers: int = 3,
        use_fidelity_branch: bool = True,
        use_sr_quality_branch: bool = True,
        head_hidden_dim: int = 1024,
        dropout: float = 0.1,
        dtype: Optional[torch.dtype] = torch.bfloat16,
        local_files_only: bool = False,
        normalize_global_feature: bool = True,
        backbone_no_grad: bool = False,
        normalize_feature: Optional[bool] = None,
    ):
        super().__init__()

        self.model_name = model_name
        self.image_size = image_size
        self.use_pooler = use_pooler
        self.hidden_layers = tuple(hidden_layers)
        self.token_pool_size = token_pool_size
        self.attn_dim = attn_dim
        self.num_heads = num_heads
        self.num_fusion_layers = num_fusion_layers
        self.sr_quality_layers = sr_quality_layers
        self.use_fidelity_branch = use_fidelity_branch
        self.use_sr_quality_branch = use_sr_quality_branch
        self.head_hidden_dim = head_hidden_dim
        self.dropout = dropout
        self.normalize_global_feature = normalize_global_feature
        self.backbone_no_grad = backbone_no_grad

        if normalize_feature is not None:
            self.normalize_global_feature = normalize_feature

        self.vision_model = SiglipVisionModel.from_pretrained(
            model_name,
            torch_dtype=dtype,
            local_files_only=local_files_only,
        )
        self.image_processor = AutoImageProcessor.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )

        if freeze_backbone:
            self.vision_model.requires_grad_(False)

        vision_dim = self.vision_model.config.hidden_size
        self.vision_dim = vision_dim

        if self.use_fidelity_branch:
            self.cross_blocks = nn.ModuleList(
                [
                    TokenCrossAttentionFusionBlock(
                        in_dim=vision_dim,
                        attn_dim=attn_dim,
                        num_heads=num_heads,
                        num_fusion_layers=num_fusion_layers,
                        dropout=dropout,
                    )
                    for _ in self.hidden_layers
                ]
            )
            cross_feat_dim = len(self.hidden_layers) * attn_dim
        else:
            self.cross_blocks = nn.ModuleList()
            cross_feat_dim = 0

        global_pair_dim = vision_dim * 5 if self.use_pooler else 0

        if self.use_sr_quality_branch:
            self.sr_quality_branch = SRQualityBranch(
                in_dim=vision_dim,
                attn_dim=attn_dim,
                num_heads=num_heads,
                num_layers=sr_quality_layers,
                dropout=dropout,
                use_pooler=use_pooler,
            )
            quality_feat_dim = self.sr_quality_branch.out_dim
        else:
            self.sr_quality_branch = None
            quality_feat_dim = 0

        pair_feat_dim = global_pair_dim + cross_feat_dim + quality_feat_dim

        self.pair_feat_dim = pair_feat_dim

        self.reward_head = nn.Sequential(
            nn.LayerNorm(pair_feat_dim),
            nn.Linear(pair_feat_dim, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, head_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim // 2, 1,bias=False),
        )

        image_mean = getattr(
            self.image_processor,
            "image_mean",
            [0.5, 0.5, 0.5],
        )
        image_std = getattr(
            self.image_processor,
            "image_std",
            [0.5, 0.5, 0.5],
        )

        self.register_buffer(
            "image_mean",
            torch.tensor(image_mean, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor(image_std, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    @staticmethod
    def _ensure_nchw(x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expect 4D image tensor, got shape={tuple(x.shape)}")

        if x.shape[1] != 3 and x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2).contiguous()

        if x.shape[1] != 3:
            raise ValueError(
                f"Expect RGB image with 3 channels, got shape={tuple(x.shape)}"
            )

        return x

    def preprocess_tensor(self, x: torch.Tensor) -> torch.Tensor:
        x = self._ensure_nchw(x).float()

        if x.detach().max() > 10:
            x = x / 255.0

        if x.detach().min() < -0.05:
            x = (x + 1.0) * 0.5

        x = x.clamp(0.0, 1.0)
        x = F.interpolate(
            x,
            size=(self.image_size, self.image_size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )

        mean = self.image_mean.to(device=x.device, dtype=x.dtype)
        std = self.image_std.to(device=x.device, dtype=x.dtype)

        return (x - mean) / std

    @staticmethod
    def _remove_cls_if_needed(tokens: torch.Tensor) -> torch.Tensor:
        _, num_tokens, _ = tokens.shape

        root = int(math.sqrt(num_tokens))
        if root * root == num_tokens:
            return tokens

        root_minus = int(math.sqrt(num_tokens - 1))
        if root_minus * root_minus == num_tokens - 1:
            return tokens[:, 1:]

        return tokens

    def _pool_tokens_2d(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = self._remove_cls_if_needed(tokens)

        if self.token_pool_size is None:
            return tokens

        batch_size, num_tokens, channels = tokens.shape
        height = int(math.sqrt(num_tokens))
        width = height

        if height * width != num_tokens:
            return tokens

        x = tokens.transpose(1, 2).reshape(
            batch_size,
            channels,
            height,
            width,
        )

        if height != self.token_pool_size or width != self.token_pool_size:
            x = F.adaptive_avg_pool2d(
                x,
                output_size=(self.token_pool_size, self.token_pool_size),
            )

        return x.flatten(2).transpose(1, 2).contiguous()

    def _vision_forward(self, pixel_values: torch.Tensor):
        backbone_dtype = next(self.vision_model.parameters()).dtype
        pixel_values = pixel_values.to(dtype=backbone_dtype)

        return self.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )

    def encode_image(
        self,
        x: torch.Tensor,
    ) -> Dict[str, Union[torch.Tensor, list]]:
        pixel_values = self.preprocess_tensor(x)

        if self.backbone_no_grad:
            with torch.no_grad():
                outputs = self._vision_forward(pixel_values)
        else:
            outputs = self._vision_forward(pixel_values)

        pooler = None
        if self.use_pooler:
            if (
                hasattr(outputs, "pooler_output")
                and outputs.pooler_output is not None
            ):
                pooler = outputs.pooler_output.float()
            else:
                pooler = outputs.last_hidden_state.mean(dim=1).float()

            if self.normalize_global_feature:
                pooler = F.normalize(pooler, dim=-1)

        token_list = []
        if self.use_fidelity_branch or self.use_sr_quality_branch:
            for layer_idx in self.hidden_layers:
                hidden = outputs.hidden_states[layer_idx].float()
                hidden = self._pool_tokens_2d(hidden)
                token_list.append(hidden)

        return {
            "pooler": pooler,
            "tokens": token_list,
        }

    @staticmethod
    def build_global_pair_feature(
        feat_lr: torch.Tensor,
        feat_sr: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat(
            [
                feat_lr,
                feat_sr,
                feat_sr - feat_lr,
                torch.abs(feat_sr - feat_lr),
                feat_lr * feat_sr,
            ],
            dim=-1,
        )

    def forward(
        self,
        img_lr: torch.Tensor,
        img_sr: torch.Tensor,
        return_dict: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if img_lr.shape[0] != img_sr.shape[0]:
            raise ValueError(
                f"Batch size mismatch: img_lr batch={img_lr.shape[0]}, "
                f"img_sr batch={img_sr.shape[0]}"
            )

        lr_out = self.encode_image(img_lr)
        sr_out = self.encode_image(img_sr)

        pair_feats = []

        global_pair_feat = None
        if self.use_pooler:
            global_pair_feat = self.build_global_pair_feature(
                lr_out["pooler"],
                sr_out["pooler"],
            )
            pair_feats.append(global_pair_feat)

        cross_pair_feat = None
        if self.use_fidelity_branch:
            cross_feats = []

            for block, lr_tokens, sr_tokens in zip(
                self.cross_blocks,
                lr_out["tokens"],
                sr_out["tokens"],
            ):
                cross_feats.append(block(lr_tokens, sr_tokens))

            cross_pair_feat = torch.cat(cross_feats, dim=-1)
            pair_feats.append(cross_pair_feat)

        sr_quality_feat = None
        if self.use_sr_quality_branch:
            quality_idx = self.hidden_layers.index(-1)
            sr_quality_tokens = sr_out["tokens"][quality_idx]

            sr_quality_feat = self.sr_quality_branch(
                sr_tokens=sr_quality_tokens,
                sr_pooler=sr_out["pooler"],
            )
            pair_feats.append(sr_quality_feat)

        if len(pair_feats) == 0:
            raise RuntimeError("No reward feature was generated.")

        pair_feat = torch.cat(pair_feats, dim=-1)
        score = self.reward_head(pair_feat).squeeze(-1)

        if not return_dict:
            return score

        out = {"score": score}

        if global_pair_feat is not None:
            out["global_pair_feat"] = global_pair_feat.detach()
        if cross_pair_feat is not None:
            out["cross_pair_feat"] = cross_pair_feat.detach()
        if sr_quality_feat is not None:
            out["sr_quality_feat"] = sr_quality_feat.detach()

        return out


def unfreeze_reward_head_only(
    reward_model: SigLIP2PairSRReward,
):
    reward_model.vision_model.requires_grad_(False)

    if reward_model.use_fidelity_branch:
        reward_model.cross_blocks.requires_grad_(True)

    if (
        reward_model.use_sr_quality_branch
        and reward_model.sr_quality_branch is not None
    ):
        reward_model.sr_quality_branch.requires_grad_(True)

    reward_model.reward_head.requires_grad_(True)

    return reward_model


def freeze_reward_model_for_refl(
    reward_model: nn.Module,
):
    reward_model.eval()

    if hasattr(reward_model, "backbone_no_grad"):
        reward_model.backbone_no_grad = False

    for parameter in reward_model.parameters():
        parameter.requires_grad_(False)

    return reward_model


def save_reward_model(
    reward_model: SigLIP2PairSRReward,
    save_path: str,
):
    checkpoint = {
        "model_name": reward_model.model_name,
        "image_size": reward_model.image_size,
        "use_pooler": reward_model.use_pooler,
        "hidden_layers": reward_model.hidden_layers,
        "token_pool_size": reward_model.token_pool_size,
        "attn_dim": reward_model.attn_dim,
        "num_heads": reward_model.num_heads,
        "num_fusion_layers": reward_model.num_fusion_layers,
        "sr_quality_layers": reward_model.sr_quality_layers,
        "use_fidelity_branch": reward_model.use_fidelity_branch,
        "use_sr_quality_branch": reward_model.use_sr_quality_branch,
        "head_hidden_dim": reward_model.head_hidden_dim,
        "dropout": reward_model.dropout,
        "normalize_global_feature": reward_model.normalize_global_feature,
        "cross_blocks": reward_model.cross_blocks.state_dict(),
        "sr_quality_branch": (
            reward_model.sr_quality_branch.state_dict()
            if reward_model.sr_quality_branch is not None
            else None
        ),
        "reward_head": reward_model.reward_head.state_dict(),
    }

    torch.save(checkpoint, save_path)


def load_reward_model(
    ckpt_path: str,
    device: str = "cuda",
    dtype: Optional[torch.dtype] = torch.bfloat16,
    freeze_backbone: bool = True,
    local_files_only: bool = False,
):
    checkpoint = torch.load(
        ckpt_path,
        map_location="cpu",
    )

    reward_model = SigLIP2PairSRReward(
        model_name=checkpoint["model_name"],
        image_size=checkpoint["image_size"],
        freeze_backbone=freeze_backbone,
        use_pooler=checkpoint["use_pooler"],
        hidden_layers=tuple(checkpoint["hidden_layers"]),
        token_pool_size=checkpoint["token_pool_size"],
        attn_dim=checkpoint["attn_dim"],
        num_heads=checkpoint["num_heads"],
        num_fusion_layers=checkpoint["num_fusion_layers"],
        sr_quality_layers=checkpoint["sr_quality_layers"],
        use_fidelity_branch=checkpoint.get(
            "use_fidelity_branch",
            True,
        ),
        use_sr_quality_branch=checkpoint["use_sr_quality_branch"],
        head_hidden_dim=checkpoint.get("head_hidden_dim", 1024),
        dropout=checkpoint.get("dropout", 0.1),
        dtype=dtype,
        local_files_only=local_files_only,
        normalize_global_feature=checkpoint.get(
            "normalize_global_feature",
            True,
        ),
    )

    if reward_model.use_fidelity_branch:
        cross_state = checkpoint.get("cross_blocks")
        reward_model.cross_blocks.load_state_dict(
            cross_state,
            strict=True,
        )

    if reward_model.sr_quality_branch is not None:
        quality_state = checkpoint.get("sr_quality_branch")
        reward_model.sr_quality_branch.load_state_dict(
            quality_state,
            strict=True,
        )

    reward_model.reward_head.load_state_dict(
        checkpoint["reward_head"],
        strict=True,
    )
    reward_model.to(device)

    return reward_model


def quick_gradient_check():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SigLIP2PairSRReward(
        model_name="google/siglip2-so400m-patch16-512",
        image_size=512,
        freeze_backbone=True,
        use_pooler=False,
        hidden_layers=(-1, -4),
        token_pool_size=8,
        attn_dim=256,
        num_heads=4,
        num_fusion_layers=1,
        sr_quality_layers=1,
        use_fidelity_branch=True,
        use_sr_quality_branch=True,
        dtype=torch.bfloat16,
        backbone_no_grad=False,
    ).to(device)

    model.eval()

    img_lr = torch.rand(1, 3, 128, 128, device=device)
    img_sr = torch.rand(
        1,
        3,
        512,
        512,
        device=device,
        requires_grad=True,
    )

    out = model(img_lr, img_sr, return_dict=True)
    score = out["score"]

    gradient = torch.autograd.grad(
        score.mean(),
        img_sr,
        retain_graph=True,
    )[0]

    print("score:", score.detach().cpu())
    print("grad abs mean:", gradient.abs().mean().item())
    print("grad abs max:", gradient.abs().max().item())

    if gradient.abs().mean().item() == 0:
        print("[Warning] grad is zero.")
    else:
        print("[OK] score is differentiable with respect to img_sr.")


if __name__ == "__main__":
    quick_gradient_check()





