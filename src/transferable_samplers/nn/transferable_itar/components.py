# Shared building blocks for transferable two-time iTarFlow variants.
#
# All components are self-contained here (copied locally rather than imported
# from nn/tarflow/) to keep transferable_itar decoupled from the single-system
# and MLE-transferable code paths.

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


class PermutationFromDict(nn.Module):
    """Semantic permutation looked up in a dict by string key.

    Copied from nn/tarflow/tarflow.py::PermutationFromDict.
    """

    def __init__(self, permutation_key: str) -> None:
        super().__init__()
        self.permutation_key = permutation_key

    def forward(
        self, data: torch.Tensor, data_permutations_dict: dict[str, torch.Tensor], inverse: bool = False
    ) -> torch.Tensor:
        assert self.permutation_key in data_permutations_dict, (
            f"Permutation key {self.permutation_key} not found in data_permutations"
        )
        permutation = data_permutations_dict[self.permutation_key]
        if inverse:
            permutation = torch.argsort(permutation)
        permutation = permutation.unsqueeze(-1).expand(-1, -1, data.shape[-1])
        return torch.gather(data, dim=1, index=permutation)


# ---------------------------------------------------------------------------
# TimestepEmbedding (sinusoidal + MLP)
# ---------------------------------------------------------------------------


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding followed by an MLP projection.

    Copied from nn/tarflow/itarflow_components.py::TimestepEmbedding.
    """

    def __init__(self, channels: int, max_period: int = 10000) -> None:
        super().__init__()
        assert channels % 2 == 0
        half_dim = channels // 2
        self.register_buffer(
            "freqs",
            torch.exp(-math.log(max_period) * torch.arange(half_dim) / half_dim),
        )
        linear1 = nn.Linear(channels, channels)
        nn.init.xavier_uniform_(linear1.weight)
        nn.init.zeros_(linear1.bias)
        linear2 = nn.Linear(channels, channels)
        nn.init.xavier_uniform_(linear2.weight)
        nn.init.zeros_(linear2.bias)
        self.mlp = nn.Sequential(linear1, nn.SiLU(), linear2)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        args = t[:, None].float() * self.freqs[None, :]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(emb)


# ---------------------------------------------------------------------------
# modulate + MLP + SwiGLU + Transition
# ---------------------------------------------------------------------------


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Affine transformation for adaLN zero."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class MLP(nn.Module):
    """LayerNorm + Linear + GELU + Linear feedforward block."""

    def __init__(self, channels: int, dtype: torch.dtype = torch.float32, expansion: int = 4) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels, dtype=dtype)
        linear1 = nn.Linear(channels, channels * expansion, dtype=dtype)
        nn.init.xavier_uniform_(linear1.weight)
        nn.init.zeros_(linear1.bias)
        linear2 = nn.Linear(channels * expansion, channels, dtype=dtype)
        nn.init.xavier_uniform_(linear2.weight)
        nn.init.zeros_(linear2.bias)
        self.main = nn.Sequential(linear1, nn.GELU(), linear2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(self.norm(x.float()).type(x.dtype))


class SwiGLU(nn.Module):
    """SwiGLU activation (AF3 style)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gates = x.chunk(2, dim=-1)
        return F.silu(gates) * x


class Transition(nn.Module):
    """SwiGLU MLP transition block used inside PROSE-style adaLN transition."""

    def __init__(self, channels: int, expansion_factor: int = 4) -> None:
        super().__init__()
        channels_inner = int(channels * expansion_factor)
        self.swish_linear = nn.Sequential(nn.Linear(channels, channels_inner * 2, bias=False), SwiGLU())
        self.linear_out = nn.Linear(channels_inner, channels, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.linear_out(self.swish_linear(x))
        return x * mask[..., None]


# ---------------------------------------------------------------------------
# ConditionalAttentionMasked
# ---------------------------------------------------------------------------


class ConditionalAttentionMasked(nn.Module):
    """Attention with in-context conditioning and padding-mask support.

    Extension of nn/tarflow/itarflow_components.ConditionalAttention adding:
      * optional QK-LayerNorm (``use_qkln``).
      * a padding mask ``pad_mask`` of shape ``[B, T]`` that keeps padded x
        tokens from attending (via the SDPA ``attn_mask``) and zeros the
        output at padded positions.

    Queries come only from the ``T`` x tokens; keys/values come from x tokens
    plus ``T_c`` conditioning tokens.  A causal mask is applied over the x
    tokens; the conditioning tokens are always attendable.
    """

    def __init__(
        self,
        channels: int,
        head_dim: int = 64,
        use_qkln: bool = False,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        assert channels % head_dim == 0, "channels must be divisible by head_dim"
        self.num_heads = channels // head_dim
        self.head_dim = head_dim
        self.dtype = dtype

        self.norm = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, 3 * channels)
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.zeros_(self.qkv.bias)
        self.proj = nn.Linear(channels, channels)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

        self.q_layer_norm = nn.LayerNorm(channels) if use_qkln else nn.Identity()
        self.k_layer_norm = nn.LayerNorm(channels) if use_qkln else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        causal_mask: torch.Tensor | None = None,
        pad_mask: torch.Tensor | None = None,
        k_cache: dict | None = None,
        v_cache: dict | None = None,
        which_cache: str = "cond",
        train: bool = True,
    ) -> tuple[torch.Tensor, dict | None, dict | None]:
        """Forward pass.

        Args:
            x: ``[B, T, C]`` input tokens.
            c: ``[B, T_c, C]`` conditioning tokens (always attendable).
            causal_mask: ``[T, T]`` boolean causal mask (training only).
            pad_mask: ``[B, T]`` boolean/float mask (1 = valid, 0 = padded).
            k_cache: KV-cache dict for inference.
            v_cache: KV-cache dict for inference.
            which_cache: Key into the cache dicts.
            train: Whether in training mode (parallel attention).
        """
        num_heads = self.num_heads
        assert c is not None, "c must be provided"
        B, T, C = x.shape
        B_c, T_c, C_c = c.shape
        assert C == C_c and B == B_c

        # Zero padded x tokens before normalisation for numerical safety.
        if pad_mask is not None:
            x = x * pad_mask[..., None]

        xc = torch.cat([x, c], dim=1)
        xc = self.norm(xc.to(self.dtype))
        qkv = self.qkv(xc)  # [B, T+T_c, 3*C]
        q, k, v = qkv.chunk(3, dim=-1)
        q = self.q_layer_norm(q)
        k = self.k_layer_norm(k)

        q = q.reshape(B, T + T_c, num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, T + T_c, num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, T + T_c, num_heads, self.head_dim).transpose(1, 2)

        # Queries only from x tokens.
        q = q[:, :, :T, :]

        if train:
            assert k_cache is None and v_cache is None
            assert causal_mask is not None, "causal_mask needed during training"
            # Broadcast causal to [B, 1, T, T], then AND with pad_mask keys.
            attn_mask = causal_mask.to(torch.bool)
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0).expand(B, 1, T, T)
            if pad_mask is not None:
                # Keys corresponding to padded positions must not be attended to.
                key_pad = pad_mask.to(torch.bool).unsqueeze(1).unsqueeze(1)  # [B,1,1,T]
                attn_mask = attn_mask & key_pad
            # Append fully-attendable cond keys.
            c_mask = torch.ones((B, 1, T, T_c), dtype=torch.bool, device=attn_mask.device)
            attn_mask = torch.cat([attn_mask, c_mask], dim=-1)
        else:
            # Inference: KV-cache path.
            k_c, v_c = k[:, :, -T_c:], v[:, :, -T_c:]
            k, v = k[:, :, :-T_c], v[:, :, :-T_c]
            k_idx = k_cache["idx"]
            v_idx = v_cache["idx"]
            k_cache[which_cache][:, :, k_idx : k_idx + T, :] = k
            v_cache[which_cache][:, :, v_idx : v_idx + T, :] = v
            k = k_cache[which_cache][:, :, : k_idx + T, :]
            v = v_cache[which_cache][:, :, : v_idx + T, :]
            k = torch.cat([k, k_c], dim=2)
            v = torch.cat([v, v_c], dim=2)
            attn_mask = None

        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        attn = attn.transpose(1, 2).contiguous().view(B, T, C)
        attn = self.proj(attn)
        if pad_mask is not None:
            attn = attn * pad_mask[..., None]
        return attn, k_cache, v_cache


# ---------------------------------------------------------------------------
# InContextLayerMasked (DiT-style block with in-context (s, t) cond + pad mask)
# ---------------------------------------------------------------------------


class InContextLayerMasked(nn.Module):
    """DiT block with in-context conditioning and padding-mask support.

    Copy of nn/tarflow/itarflow_components.InContextLayer with:
      * ``ConditionalAttentionMasked`` in place of ``ConditionalAttention``.
      * optional additive peptide conditioning added before the attention.
      * padding mask threading through attn + MLP.
    """

    def __init__(
        self,
        channels: int,
        head_dim: int = 64,
        use_qkln: bool = False,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.dtype = dtype
        self.norm1 = nn.LayerNorm(channels, bias=False, elementwise_affine=False, dtype=dtype)
        self.norm2 = nn.LayerNorm(channels, bias=False, elementwise_affine=False, dtype=dtype)
        self.attn = ConditionalAttentionMasked(channels=channels, head_dim=head_dim, use_qkln=use_qkln, dtype=dtype)
        self.mlp = MLP(channels, self.dtype, expansion=4)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        peptide_cond: torch.Tensor | None = None,
        causal_mask: torch.Tensor | None = None,
        pad_mask: torch.Tensor | None = None,
        k_cache: dict | None = None,
        v_cache: dict | None = None,
        which_cache: str = "cond",
        train: bool = True,
    ) -> tuple[torch.Tensor, dict | None, dict | None]:
        c = c.reshape(-1, 1, self.channels) if c.ndim == 2 else c

        # Additive peptide conditioning (Variant-1 mechanism).
        if peptide_cond is not None:
            x = x + peptide_cond
            if pad_mask is not None:
                x = x * pad_mask[..., None]

        attn, k_cache, v_cache = self.attn(
            x,
            c,
            causal_mask=causal_mask,
            pad_mask=pad_mask,
            k_cache=k_cache,
            v_cache=v_cache,
            which_cache=which_cache,
            train=train,
        )
        x = x + attn
        mlp_out = self.mlp(x)
        if pad_mask is not None:
            mlp_out = mlp_out * pad_mask[..., None]
        x = x + mlp_out
        if pad_mask is not None:
            x = x * pad_mask[..., None]
        return x, k_cache, v_cache


# ---------------------------------------------------------------------------
# AdaptiveLayerNorm + AdaptiveLayerNormOutputScale (AF3 style, per-token)
# ---------------------------------------------------------------------------


class AdaptiveLayerNorm(nn.Module):
    """AF3-style adaLN: per-token gamma/beta produced by cond.

    Copied from nn/tarflow/adaptive_blocks.py::AdaptiveLayerNorm.
    """

    def __init__(self, *, channels: int, channels_cond: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels, elementwise_affine=False)
        self.norm_cond = nn.LayerNorm(channels_cond)
        self.to_gamma = nn.Sequential(nn.Linear(channels_cond, channels), nn.Sigmoid())
        self.to_beta = nn.Linear(channels_cond, channels, bias=False)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        normed = self.norm(x)
        normed_cond = self.norm_cond(cond)
        gamma = self.to_gamma(normed_cond)
        beta = self.to_beta(normed_cond)
        return (normed * gamma + beta) * mask[..., None]


class AdaptiveLayerNormOutputScale(nn.Module):
    """AF3-style sigmoid output gate.

    Copied from nn/tarflow/adaptive_blocks.py::AdaptiveLayerNormOutputScale.
    """

    def __init__(self, *, channels: int, channels_cond: int, adaln_zero_bias_init_value: float = -2.0) -> None:
        super().__init__()
        lin = nn.Linear(channels_cond, channels)
        nn.init.zeros_(lin.weight)
        nn.init.constant_(lin.bias, adaln_zero_bias_init_value)
        self.to_adaln_zero_gamma = nn.Sequential(lin, nn.Sigmoid())

    def forward(self, x: torch.Tensor, cond: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        gamma = self.to_adaln_zero_gamma(cond)
        return x * gamma * mask[..., None]


# ---------------------------------------------------------------------------
# FinalLayer (DiT-style adaLN + linear projection)
# ---------------------------------------------------------------------------


class FinalLayer(nn.Module):
    """Final layer: DiT-style adaLN modulation + linear projection.

    Copied from nn/tarflow/itarflow_components.py::FinalLayer.
    Zero-init on both proj_out and adaLN_modulation gives the identity
    coupling at initialisation (alpha = mu = 0).
    """

    def __init__(self, hidden_dim: int, out_channels: int, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.norm_final = nn.LayerNorm(hidden_dim, bias=False, elementwise_affine=False, dtype=dtype)
        self.proj_out = nn.Linear(self.hidden_dim, 2 * self.out_channels, dtype=dtype)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim, dtype=dtype))
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.proj_out(x)
