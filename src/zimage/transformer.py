import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from config import (
    ADALN_EMBED_DIM,
    FREQUENCY_EMBEDDING_SIZE,
    MAX_PERIOD,
    ROPE_AXES_DIMS,
    ROPE_AXES_LENS,
    ROPE_THETA,
    SEQ_MULTI_OF,
)


class TimestepEmbedder(nn.Module):
    def __init__(self, out_size, mid_size=None, frequency_embedding_size=FREQUENCY_EMBEDDING_SIZE):
        super().__init__()
        if mid_size is None:
            mid_size = out_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, mid_size, bias=True),
            nn.SiLU(),
            nn.Linear(mid_size, out_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=MAX_PERIOD):
        with torch.amp.autocast("cuda", enabled=False):
            half = dim // 2
            freqs = torch.exp(
                -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
            )
            args = t[:, None].float() * freqs[None]
            embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
            if dim % 2:
                embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        weight_dtype = self.mlp[0].weight.dtype
        if weight_dtype.is_floating_point:
            t_freq = t_freq.to(weight_dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return output * self.weight


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def apply_rotary_emb(x_in: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    with torch.amp.autocast("cuda", enabled=False):
        x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(2)
        x_out = torch.view_as_real(x * freqs_cis).flatten(3)
        return x_out.type_as(x_in)



class ZImageAttention(nn.Module):
    _attention_backend = None

    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, qk_norm: bool = True, eps: float = 1e-5):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads

        self.to_q = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.to_k = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.to_v = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(n_heads * self.head_dim, dim, bias=False)])

        self.norm_q = RMSNorm(self.head_dim, eps=eps) if qk_norm else None
        self.norm_k = RMSNorm(self.head_dim, eps=eps) if qk_norm else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        

        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        query = query.unflatten(-1, (self.n_heads, -1))
        key = key.unflatten(-1, (self.n_kv_heads, -1))
        value = value.unflatten(-1, (self.n_kv_heads, -1))

        if self.norm_q is not None:
            query = self.norm_q(query)
        if self.norm_k is not None:
            key = self.norm_k(key)

        if freqs_cis is not None:
            query = apply_rotary_emb(query, freqs_cis)
            key = apply_rotary_emb(key, freqs_cis)

        dtype = query.dtype
        query, key = query.to(dtype), key.to(dtype)

        from utils.attention import dispatch_attention

        hidden_states = dispatch_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False, backend=self._attention_backend
        )

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(dtype)

        output = self.to_out[0](hidden_states)

        return output

class ZImageTransformerBlock(nn.Module):
    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        norm_eps: float,
        qk_norm: bool,
        modulation=True,
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = dim // n_heads
        self.layer_id = layer_id
        self.modulation = modulation

        self.attention = ZImageAttention(dim, n_heads, n_kv_heads, qk_norm, norm_eps)
        self.feed_forward = FeedForward(dim=dim, hidden_dim=int(dim / 3 * 8))

        self.attention_norm1 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps)
        self.attention_norm2 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = RMSNorm(dim, eps=norm_eps)

        if modulation:
            self.adaLN_modulation = nn.ModuleList([nn.Linear(min(dim, ADALN_EMBED_DIM), 4 * dim, bias=True)])

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        adaln_input: Optional[torch.Tensor] = None,
    ):
        if self.modulation:
            assert adaln_input is not None
            scale_msa, gate_msa, scale_mlp, gate_mlp = (
                self.adaLN_modulation[0](adaln_input).unsqueeze(1).chunk(4, dim=2)
            )
            gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
            scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp

            attn_out = self.attention(
                self.attention_norm1(x) * scale_msa,
                attention_mask=attn_mask,
                freqs_cis=freqs_cis,
            )
            x = x + gate_msa * self.attention_norm2(attn_out)
            x = x + gate_mlp * self.ffn_norm2(self.feed_forward(self.ffn_norm1(x) * scale_mlp))
        else:
            attn_out = self.attention(
                self.attention_norm1(x),
                attention_mask=attn_mask,
                freqs_cis=freqs_cis,
            )
            x = x + self.attention_norm2(attn_out)
            x = x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))

        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(hidden_size, ADALN_EMBED_DIM), hidden_size, bias=True),
        )

    def forward(self, x, c):
        scale = 1.0 + self.adaLN_modulation(c)
        x = self.norm_final(x) * scale.unsqueeze(1)
        x = self.linear(x)
        return x


class RopeEmbedder:
    def __init__(
        self,
        theta: float = ROPE_THETA,
        axes_dims: List[int] = ROPE_AXES_DIMS,
        axes_lens: List[int] = ROPE_AXES_LENS,
    ):
        self.theta = theta
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        assert len(axes_dims) == len(axes_lens)
        self.freqs_cis = None

    @staticmethod
    def precompute_freqs_cis(dim: List[int], end: List[int], theta: float = ROPE_THETA):
        with torch.device("cpu"):
            freqs_cis = []
            for i, (d, e) in enumerate(zip(dim, end)):
                freqs = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float64, device="cpu") / d))
                timestep = torch.arange(e, device=freqs.device, dtype=torch.float64)
                freqs = torch.outer(timestep, freqs).float()
                freqs_cis_i = torch.polar(torch.ones_like(freqs), freqs).to(torch.complex64)
                freqs_cis.append(freqs_cis_i)
            return freqs_cis

    def __call__(self, ids: torch.Tensor):
        assert ids.ndim == 2
        assert ids.shape[-1] == len(self.axes_dims)
        device = ids.device

        if self.freqs_cis is None:
            self.freqs_cis = self.precompute_freqs_cis(self.axes_dims, self.axes_lens, theta=self.theta)
            self.freqs_cis = [freqs_cis.to(device) for freqs_cis in self.freqs_cis]
        else:
            if self.freqs_cis[0].device != device:
                self.freqs_cis = [freqs_cis.to(device) for freqs_cis in self.freqs_cis]

        result = []
        for i in range(len(self.axes_dims)):
            index = ids[:, i]
            result.append(self.freqs_cis[i][index])
        return torch.cat(result, dim=-1)



class ZImageTransformer2DModel(nn.Module):
    def __init__(
        self,
        all_patch_size=(2,),
        all_f_patch_size=(1,),
        in_channels=16,
        dim=3840,
        n_layers=30,
        n_refiner_layers=2,
        n_heads=30,
        n_kv_heads=30,
        norm_eps=1e-5,
        qk_norm=True,
        cap_feat_dim=2560,
        rope_theta=ROPE_THETA,
        t_scale=1000.0,
        axes_dims=ROPE_AXES_DIMS,
        axes_lens=ROPE_AXES_LENS,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.all_patch_size = all_patch_size
        self.all_f_patch_size = all_f_patch_size
        self.dim = dim
        self.n_heads = n_heads
        self.rope_theta = rope_theta
        self.t_scale = t_scale

        assert len(all_patch_size) == len(all_f_patch_size)

        all_x_embedder = {}
        all_final_layer = {}
        for patch_size, f_patch_size in zip(all_patch_size, all_f_patch_size):
            x_embedder = nn.Linear(f_patch_size * patch_size * patch_size * in_channels, dim, bias=True)
            all_x_embedder[f"{patch_size}-{f_patch_size}"] = x_embedder

            final_layer = FinalLayer(dim, patch_size * patch_size * f_patch_size * self.out_channels)
            all_final_layer[f"{patch_size}-{f_patch_size}"] = final_layer

        self.all_x_embedder = nn.ModuleDict(all_x_embedder)
        self.all_final_layer = nn.ModuleDict(all_final_layer)

        self.noise_refiner = nn.ModuleList(
            [
                ZImageTransformerBlock(
                    1000 + layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    norm_eps,
                    qk_norm,
                    modulation=True,
                )
                for layer_id in range(n_refiner_layers)
            ]
        )

        self.context_refiner = nn.ModuleList(
            [
                ZImageTransformerBlock(
                    layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    norm_eps,
                    qk_norm,
                    modulation=False,
                )
                for layer_id in range(n_refiner_layers)
            ]
        )


        self.t_embedder = TimestepEmbedder(min(dim, ADALN_EMBED_DIM), mid_size=1024)

        self.cap_embedder = nn.Sequential(
            RMSNorm(cap_feat_dim, eps=norm_eps),
            nn.Linear(cap_feat_dim, dim, bias=True),
        )

        self.x_pad_token = nn.Parameter(torch.empty((1, dim)))
        self.cap_pad_token = nn.Parameter(torch.empty((1, dim)))


        self.layers = nn.ModuleList(
            [
                ZImageTransformerBlock(
                    layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    norm_eps,
                    qk_norm,
                )
                for layer_id in range(n_layers)
            ]
        )

        head_dim = dim // n_heads
        assert head_dim == sum(axes_dims)

        self.axes_dims = axes_dims
        self.axes_lens = axes_lens

        self.rope_embedder = RopeEmbedder(
            theta=rope_theta,
            axes_dims=axes_dims,
            axes_lens=axes_lens,
        )

    def unpatchify(
        self,
        x: List[torch.Tensor],
        size: List[Tuple],
        patch_size,
        f_patch_size,
    ) -> List[torch.Tensor]:
        pH = pW = patch_size
        pF = f_patch_size
        bsz = len(x)
        assert len(size) == bsz

        for i in range(bsz):
            F, H, W = size[i]
            ori_len = (F // pF) * (H // pH) * (W // pW)

            x[i] = (
                x[i][:ori_len]
                .view(F // pF, H // pH, W // pW, pF, pH, pW, self.out_channels)
                .permute(6, 0, 3, 1, 4, 2, 5)
                .reshape(self.out_channels, F, H, W)
            )

        return x

    @staticmethod
    def create_coordinate_grid(size, start=None, device=None):
        if start is None:
            start = (0 for _ in size)

        axes = [
            torch.arange(
                x0,
                x0 + span,
                dtype=torch.int32,
                device=device,
            )
            for x0, span in zip(start, size)
        ]

        grids = torch.meshgrid(axes, indexing="ij")
        return torch.stack(grids, dim=-1)

    def prepare_cap_inputs(self, all_cap_feats: List[torch.Tensor]):
        device = all_cap_feats[0].device

        all_cap_feats_out = []
        all_cap_pos_ids = []
        all_cap_pad_mask = []

        for cap_feat in all_cap_feats:
            cap_ori_len = len(cap_feat)
            cap_padding_len = (-cap_ori_len) % SEQ_MULTI_OF
            cap_padded_len = cap_ori_len + cap_padding_len

            cap_padded_pos_ids = self.create_coordinate_grid(
                size=(cap_padded_len, 1, 1),
                start=(1, 0, 0),
                device=device,
            ).flatten(0, 2)

            all_cap_pos_ids.append(cap_padded_pos_ids)

            if cap_padding_len > 0:
                cap_pad_mask = torch.cat(
                    [
                        torch.zeros((cap_ori_len,), dtype=torch.bool, device=device),
                        torch.ones((cap_padding_len,), dtype=torch.bool, device=device),
                    ],
                    dim=0,
                )

                cap_feat = torch.cat(
                    [
                        cap_feat,
                        cap_feat[-1:].repeat(cap_padding_len, 1),
                    ],
                    dim=0,
                )
            else:
                cap_pad_mask = torch.zeros((cap_ori_len,), dtype=torch.bool, device=device)

            all_cap_pad_mask.append(cap_pad_mask)
            all_cap_feats_out.append(cap_feat)

        return all_cap_feats_out, all_cap_pos_ids, all_cap_pad_mask

    def patchify_image_inputs(
        self,
        all_image: List[torch.Tensor],
        cap_item_seqlens: List[int],
        patch_size: int,
        f_patch_size: int,
    ):
        pH = pW = patch_size
        pF = f_patch_size
        device = all_image[0].device

        all_image_out = []
        all_image_size = []
        all_image_pos_ids = []
        all_image_pad_mask = []
        all_grid_shapes = []

        for image, cap_padded_len in zip(all_image, cap_item_seqlens):
            C, F, H, W = image.size()

            assert C == self.in_channels, (
                f"Expected latent channels={self.in_channels}, but got C={C}."
            )

            assert F % pF == 0 and H % pH == 0 and W % pW == 0, (
                f"Image latent size {(F, H, W)} must be divisible by "
                f"f_patch_size={pF}, patch_size={patch_size}."
            )

            all_image_size.append((F, H, W))

            F_tokens = F // pF
            H_tokens = H // pH
            W_tokens = W // pW

            all_grid_shapes.append((F_tokens, H_tokens, W_tokens))

            # [C, F, H, W]
            # -> [C, F_tokens, pF, H_tokens, pH, W_tokens, pW]
            image = image.view(
                C,
                F_tokens,
                pF,
                H_tokens,
                pH,
                W_tokens,
                pW,
            )

            # -> [F_tokens, H_tokens, W_tokens, pF, pH, pW, C]
            # -> [N_img, patch_dim]
            image = image.permute(1, 3, 5, 2, 4, 6, 0).reshape(
                F_tokens * H_tokens * W_tokens,
                pF * pH * pW * C,
            )

            image_ori_len = len(image)
            image_padding_len = (-image_ori_len) % SEQ_MULTI_OF

            # text positions: 1, 2, ..., cap_padded_len
            # image positions: cap_padded_len + 1, ...
            image_ori_pos_ids = self.create_coordinate_grid(
                size=(F_tokens, H_tokens, W_tokens),
                start=(cap_padded_len + 1, 0, 0),
                device=device,
            ).flatten(0, 2)

            if image_padding_len > 0:
                pad_pos_ids = (
                    self.create_coordinate_grid(
                        size=(1, 1, 1),
                        start=(0, 0, 0),
                        device=device,
                    )
                    .flatten(0, 2)
                    .repeat(image_padding_len, 1)
                )

                image_padded_pos_ids = torch.cat(
                    [
                        image_ori_pos_ids,
                        pad_pos_ids,
                    ],
                    dim=0,
                )

                image_pad_mask = torch.cat(
                    [
                        torch.zeros((image_ori_len,), dtype=torch.bool, device=device),
                        torch.ones((image_padding_len,), dtype=torch.bool, device=device),
                    ],
                    dim=0,
                )

                image_padded_feat = torch.cat(
                    [
                        image,
                        image[-1:].repeat(image_padding_len, 1),
                    ],
                    dim=0,
                )
            else:
                image_padded_pos_ids = image_ori_pos_ids
                image_pad_mask = torch.zeros((image_ori_len,), dtype=torch.bool, device=device)
                image_padded_feat = image

            all_image_out.append(image_padded_feat)
            all_image_pos_ids.append(image_padded_pos_ids)
            all_image_pad_mask.append(image_pad_mask)

        return (
            all_image_out,
            all_image_size,
            all_image_pos_ids,
            all_image_pad_mask,
            all_grid_shapes,
        )

    def embed_image_token_list(
        self,
        image_patches: List[torch.Tensor],
        image_pos_ids: List[torch.Tensor],
        image_pad_mask: List[torch.Tensor],
        patch_size: int,
        f_patch_size: int,
        pad_token: torch.Tensor,
    ):
        bsz = len(image_patches)
        device = image_patches[0].device

        item_seqlens = [len(_) for _ in image_patches]
        assert all(_ % SEQ_MULTI_OF == 0 for _ in item_seqlens)

        max_item_seqlen = max(item_seqlens)

        flat = torch.cat(image_patches, dim=0)
        flat = self.all_x_embedder[f"{patch_size}-{f_patch_size}"](flat)

        flat_pad_mask = torch.cat(image_pad_mask, dim=0)
        flat[flat_pad_mask] = pad_token.to(device=flat.device, dtype=flat.dtype)

        tokens = list(flat.split(item_seqlens, dim=0))

        freqs_cis = list(
            self.rope_embedder(torch.cat(image_pos_ids, dim=0)).split(
                [len(_) for _ in image_pos_ids],
                dim=0,
            )
        )

        tokens = pad_sequence(tokens, batch_first=True, padding_value=0.0)
        freqs_cis = pad_sequence(freqs_cis, batch_first=True, padding_value=0.0)
        freqs_cis = freqs_cis[:, : tokens.shape[1]]

        attn_mask = torch.zeros(
            (bsz, max_item_seqlen),
            dtype=torch.bool,
            device=device,
        )

        for i, seq_len in enumerate(item_seqlens):
            attn_mask[i, :seq_len] = 1

        return tokens, freqs_cis, attn_mask, item_seqlens

    def embed_cap_token_list(
        self,
        cap_feats: List[torch.Tensor],
        cap_pos_ids: List[torch.Tensor],
        cap_pad_mask: List[torch.Tensor],
    ):
        bsz = len(cap_feats)
        device = cap_feats[0].device

        item_seqlens = [len(_) for _ in cap_feats]
        assert all(_ % SEQ_MULTI_OF == 0 for _ in item_seqlens)

        max_item_seqlen = max(item_seqlens)

        flat = torch.cat(cap_feats, dim=0)
        flat = self.cap_embedder(flat)

        flat_pad_mask = torch.cat(cap_pad_mask, dim=0)
        flat[flat_pad_mask] = self.cap_pad_token.to(device=flat.device, dtype=flat.dtype)

        tokens = list(flat.split(item_seqlens, dim=0))

        freqs_cis = list(
            self.rope_embedder(torch.cat(cap_pos_ids, dim=0)).split(
                [len(_) for _ in cap_pos_ids],
                dim=0,
            )
        )

        tokens = pad_sequence(tokens, batch_first=True, padding_value=0.0)
        freqs_cis = pad_sequence(freqs_cis, batch_first=True, padding_value=0.0)
        freqs_cis = freqs_cis[:, : tokens.shape[1]]

        attn_mask = torch.zeros(
            (bsz, max_item_seqlen),
            dtype=torch.bool,
            device=device,
        )

        for i, seq_len in enumerate(item_seqlens):
            attn_mask[i, :seq_len] = 1

        return tokens, freqs_cis, attn_mask, item_seqlens

    
    def forward(
        self,
        x: List[torch.Tensor],
        t,
        cap_feats: List[torch.Tensor],
        cond: Optional[List[torch.Tensor]] = None,
        patch_size=2,
        f_patch_size=1,
        return_image_tokens: bool = False,
        image_token_layer_ids: Optional[List[int]] = None,
    ):
        

        assert patch_size in self.all_patch_size
        assert f_patch_size in self.all_f_patch_size

        bsz = len(x)
        device = x[0].device

        assert len(cap_feats) == bsz

        if cond is not None:
            assert len(cond) == bsz

        t = t * self.t_scale
        t = self.t_embedder(t)


        selected_image_token_layer_ids = []
        selected_image_token_layer_set = set()

        if return_image_tokens:
            num_layers = len(self.layers)

            if num_layers < 4:
                raise ValueError(
                    "return_image_tokens=True requires at least 4 main "
                    f"Transformer layers, but got {num_layers}."
                )

            if image_token_layer_ids is None:
               
                candidate_layers = list(range(num_layers // 2, num_layers))

                if len(candidate_layers) < 4:
                    candidate_layers = list(range(num_layers - 4, num_layers))

                positions = [
                    round(i * (len(candidate_layers) - 1) / 3)
                    for i in range(4)
                ]

                selected_image_token_layer_ids = [
                    candidate_layers[pos] for pos in positions
                ]
            else:
                selected_image_token_layer_ids = [
                    int(layer_id) for layer_id in image_token_layer_ids
                ]

                if len(selected_image_token_layer_ids) != 4:
                    raise ValueError(
                        "image_token_layer_ids must contain exactly 4 layer ids, "
                        f"but got {selected_image_token_layer_ids}."
                    )

                if len(set(selected_image_token_layer_ids)) != 4:
                    raise ValueError(
                        "image_token_layer_ids must contain 4 unique layer ids, "
                        f"but got {selected_image_token_layer_ids}."
                    )

                invalid_layer_ids = [
                    layer_id
                    for layer_id in selected_image_token_layer_ids
                    if layer_id < 0 or layer_id >= num_layers
                ]

                if invalid_layer_ids:
                    raise ValueError(
                        "Found invalid layer ids "
                        f"{invalid_layer_ids}. Valid range is [0, {num_layers - 1}]."
                    )

                selected_image_token_layer_ids = sorted(
                    selected_image_token_layer_ids
                )

            if len(set(selected_image_token_layer_ids)) != 4:
                raise RuntimeError(
                    "Automatically selected layer ids are not unique: "
                    f"{selected_image_token_layer_ids}."
                )

            selected_image_token_layer_set = set(
                selected_image_token_layer_ids
            )

        # ------------------------------------------------------------
        # 1. Text tokens
        # ------------------------------------------------------------
        cap_feats, cap_pos_ids, cap_inner_pad_mask = self.prepare_cap_inputs(
            cap_feats
        )
        cap_item_seqlens = [len(_) for _ in cap_feats]

        # ------------------------------------------------------------
        # 2. Noisy latent tokens
        # ------------------------------------------------------------
        (
            x_patches,
            x_size,
            x_pos_ids,
            x_inner_pad_mask,
            x_grid_shapes,
        ) = self.patchify_image_inputs(
            x,
            cap_item_seqlens=cap_item_seqlens,
            patch_size=patch_size,
            f_patch_size=f_patch_size,
        )

        x, x_freqs_cis, x_attn_mask, x_item_seqlens = (
            self.embed_image_token_list(
                x_patches,
                x_pos_ids,
                x_inner_pad_mask,
                patch_size,
                f_patch_size,
                pad_token=self.x_pad_token,
            )
        )

        adaln_input = t.type_as(x)

        for layer in self.noise_refiner:
            x = layer(
                x,
                x_attn_mask,
                x_freqs_cis,
                adaln_input,
            )

        # ------------------------------------------------------------
        # 3. Condition latent tokens
        # ------------------------------------------------------------
        has_cond = cond is not None

        if has_cond:
            (
                cond_patches,
                cond_size,
                cond_pos_ids,
                cond_inner_pad_mask,
                cond_grid_shapes,
            ) = self.patchify_image_inputs(
                cond,
                cap_item_seqlens=cap_item_seqlens,
                patch_size=patch_size,
                f_patch_size=f_patch_size,
            )

            assert cond_size == x_size, (
                "For SR-style spatially aligned conditioning, cond[i] must have "
                "the same latent size as x[i]. "
                f"Got cond_size={cond_size}, x_size={x_size}."
            )

            assert cond_grid_shapes == x_grid_shapes, (
                "For shared RoPE positions, cond token grid must match x token grid. "
                f"Got cond_grid_shapes={cond_grid_shapes}, "
                f"x_grid_shapes={x_grid_shapes}."
            )

            cond_pos_ids = []

            for ids, grid_shape in zip(x_pos_ids, x_grid_shapes):
                F_tokens, H_tokens, W_tokens = grid_shape

                cond_ids = ids.clone()

                cond_ids[:, 0] += F_tokens

                cond_pos_ids.append(cond_ids)

            (
                cond_tokens,
                cond_freqs_cis,
                cond_attn_mask,
                cond_item_seqlens,
            ) = self.embed_image_token_list(
                cond_patches,
                cond_pos_ids,
                cond_inner_pad_mask,
                patch_size,
                f_patch_size,
                pad_token=self.x_pad_token,
            )

            assert cond_item_seqlens == x_item_seqlens, (
                "cond and x should have the same padded token lengths when using "
                "shared spatial positions."
            )

            for layer in self.noise_refiner:
                cond_tokens = layer(
                    cond_tokens,
                    cond_attn_mask,
                    cond_freqs_cis,
                    adaln_input,
                )
        else:
            cond_tokens = None
            cond_freqs_cis = None
            cond_item_seqlens = [0 for _ in range(bsz)]

        # ------------------------------------------------------------
        # 4. Text tokens refine
        # ------------------------------------------------------------
        (
            cap_feats,
            cap_freqs_cis,
            cap_attn_mask,
            cap_item_seqlens,
        ) = self.embed_cap_token_list(
            cap_feats,
            cap_pos_ids,
            cap_inner_pad_mask,
        )

        for layer in self.context_refiner:
            cap_feats = layer(
                cap_feats,
                cap_attn_mask,
                cap_freqs_cis,
            )


        unified = []
        unified_freqs_cis = []

        for i in range(bsz):
            x_len = x_item_seqlens[i]
            cap_len = cap_item_seqlens[i]

            if has_cond:
                cond_len = cond_item_seqlens[i]

                unified.append(
                    torch.cat(
                        [
                            x[i][:x_len],
                            cond_tokens[i][:cond_len],
                            cap_feats[i][:cap_len],
                        ],
                        dim=0,
                    )
                )

                unified_freqs_cis.append(
                    torch.cat(
                        [
                            x_freqs_cis[i][:x_len],
                            cond_freqs_cis[i][:cond_len],
                            cap_freqs_cis[i][:cap_len],
                        ],
                        dim=0,
                    )
                )
            else:
                unified.append(
                    torch.cat(
                        [
                            x[i][:x_len],
                            cap_feats[i][:cap_len],
                        ],
                        dim=0,
                    )
                )

                unified_freqs_cis.append(
                    torch.cat(
                        [
                            x_freqs_cis[i][:x_len],
                            cap_freqs_cis[i][:cap_len],
                        ],
                        dim=0,
                    )
                )

        if has_cond:
            unified_item_seqlens = [
                x_len + cond_len + cap_len
                for x_len, cond_len, cap_len in zip(
                    x_item_seqlens,
                    cond_item_seqlens,
                    cap_item_seqlens,
                )
            ]
        else:
            unified_item_seqlens = [
                x_len + cap_len
                for x_len, cap_len in zip(
                    x_item_seqlens,
                    cap_item_seqlens,
                )
            ]

        assert unified_item_seqlens == [len(_) for _ in unified]

        unified_max_item_seqlen = max(unified_item_seqlens)

        unified = pad_sequence(
            unified,
            batch_first=True,
            padding_value=0.0,
        )

        unified_freqs_cis = pad_sequence(
            unified_freqs_cis,
            batch_first=True,
            padding_value=0.0,
        )

        unified_freqs_cis = unified_freqs_cis[:, :unified.shape[1]]

        unified_attn_mask = torch.zeros(
            (bsz, unified_max_item_seqlen),
            dtype=torch.bool,
            device=device,
        )

        for i, seq_len in enumerate(unified_item_seqlens):
            unified_attn_mask[i, :seq_len] = True


        image_token_mask = None
        max_x_token_length = None

        if return_image_tokens:
            max_x_token_length = max(x_item_seqlens)

            image_token_lengths_tensor = torch.tensor(
                x_item_seqlens,
                dtype=torch.long,
                device=device,
            )

            image_token_positions = torch.arange(
                max_x_token_length,
                device=device,
            ).unsqueeze(0)

            image_token_mask = (
                image_token_positions
                < image_token_lengths_tensor.unsqueeze(1)
            )

        # ------------------------------------------------------------
        # 6. Main Transformer layers
        # ------------------------------------------------------------
        collected_image_tokens = []

        for layer_id, layer in enumerate(self.layers):
            unified = layer(
                unified,
                unified_attn_mask,
                unified_freqs_cis,
                adaln_input,
            )

            if (
                return_image_tokens
                and layer_id in selected_image_token_layer_set
            ):

                current_image_tokens = unified[
                    :,
                    :max_x_token_length,
                    :,
                ]

                current_image_tokens = (
                    current_image_tokens
                    * image_token_mask.unsqueeze(-1).to(
                        dtype=current_image_tokens.dtype
                    )
                )

                collected_image_tokens.append(current_image_tokens)

        if return_image_tokens:
            if len(collected_image_tokens) != 4:
                raise RuntimeError(
                    "Expected to collect exactly 4 image-token tensors, "
                    f"but collected {len(collected_image_tokens)}. "
                    f"Selected layer ids: {selected_image_token_layer_ids}."
                )

        # ------------------------------------------------------------
        # 7. Final layer + unpatchify
        # ------------------------------------------------------------
        unified = self.all_final_layer[
            f"{patch_size}-{f_patch_size}"
        ](
            unified,
            adaln_input,
        )

        unified = list(unified.unbind(dim=0))

        x = self.unpatchify(
            unified,
            x_size,
            patch_size,
            f_patch_size,
        )

        # ------------------------------------------------------------
        # 8. Extra outputs
        # ------------------------------------------------------------
        if return_image_tokens:
            extra_outputs = {

                "image_tokens": collected_image_tokens,
                "image_token_mask": image_token_mask,
                "image_token_layer_ids": selected_image_token_layer_ids,
                "image_token_lengths": list(x_item_seqlens),
                "image_token_grid_shapes": x_grid_shapes,
            }
        else:
            extra_outputs = {}

        return x, extra_outputs