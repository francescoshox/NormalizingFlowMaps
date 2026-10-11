# Shared building blocks for iTarFlow variants.
#
# Ported from nf-flow-map/src/models/components/networks/itarflow_components.py
# for use in two-time NFMD training.

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Affine transformation for adaLN zero."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ---------------------------------------------------------------------------
# Permutation
# ---------------------------------------------------------------------------


class PermutationFlip(nn.Module):
    """Simple sequence flip permutation for alternating coupling blocks."""

    def __init__(self, flip: bool = True) -> None:
        super().__init__()
        self.flip = flip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.flip:
            x = x.flip(dims=[1])
        return x


# ---------------------------------------------------------------------------
# TimestepEmbedding (sinusoidal + MLP)
# ---------------------------------------------------------------------------


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding followed by an MLP projection."""

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
        """Embed scalar timesteps.

        Args:
            t: ``[B]`` scalar timesteps.

        Returns:
            ``[B, channels]`` embeddings.
        """
        args = t[:, None].float() * self.freqs[None, :]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(emb)


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# ConditionalAttention
# ---------------------------------------------------------------------------


class ConditionalAttention(nn.Module):
    """Attention with in-context conditioning.

    Concatenates x and conditioning c, computes QKV jointly, but queries only
    come from x tokens while keys/values come from all tokens (x + c).
    Uses a causal mask over x tokens with full attention to conditioning tokens.
    """

    def __init__(self, channels: int, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.num_heads = max(1, channels // 64)
        self.head_dim = channels // self.num_heads
        assert self.head_dim * self.num_heads == channels
        self.dtype = dtype
        self.norm = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, 3 * channels)
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.zeros_(self.qkv.bias)
        self.proj = nn.Linear(channels, channels)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        mask: torch.Tensor | None = None,
        k_cache: dict | None = None,
        v_cache: dict | None = None,
        which_cache: str = "cond",
        train: bool = True,
    ) -> tuple[torch.Tensor, dict | None, dict | None]:
        """Forward pass.

        Args:
            x: ``[B, T, C]`` input tokens.
            c: ``[B, T_c, C]`` conditioning tokens.
            mask: Causal mask ``[T, T]`` (training only).
            k_cache: KV-cache dict for inference.
            v_cache: KV-cache dict for inference.
            which_cache: Key into the cache dicts.
            train: Whether in training mode.

        Returns:
            ``(attn, k_cache, v_cache)``.
        """
        num_heads = self.num_heads
        assert c is not None, "c must be provided for ConditionalAttention"
        B, T, C = x.shape
        B_c, T_c, C_c = c.shape
        assert C == C_c and B == B_c, "x and c must have the same number of channels and batch size"

        x = torch.cat([x, c], dim=1)
        x = self.norm(x.to(self.dtype))
        qkv = self.qkv(x)  # [B, T+T_c, 3*C]
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to (B, num_heads, seq, head_dim) for torch attn
        q = q.reshape(B, T + T_c, num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, T + T_c, num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, T + T_c, num_heads, self.head_dim).transpose(1, 2)

        # Queries only from x tokens; keys/values from all tokens
        q = q[:, :, :T, :]  # [B, H, T, D]

        if train:
            assert k_cache is None and v_cache is None, "k_cache and v_cache must be None during training"
            assert mask is not None, "mask needed for SPDA training"
            # The incoming mask is (T, T). We append the condition mask.
            # Shape becomes (T, T + T_c)
            c_mask = torch.ones((mask.shape[0], T_c), dtype=torch.bool, device=mask.device)
            attn_mask = torch.cat([mask, c_mask], dim=1)
        else:
            # Inference: split off conditioning, update cache, reassemble
            k_c, v_c = k[:, :, -T_c:], v[:, :, -T_c:]
            k, v = k[:, :, :-T_c], v[:, :, :-T_c]
            k_idx = k_cache["idx"]
            v_idx = v_cache["idx"]
            # In-place update of cache buffer
            k_cache[which_cache][:, :, k_idx : k_idx + T, :] = k
            v_cache[which_cache][:, :, v_idx : v_idx + T, :] = v

            # Extract the sequence history up to the current index
            k = k_cache[which_cache][:, :, : k_idx + T, :]
            v = v_cache[which_cache][:, :, : v_idx + T, :]

            # Reattach conditioning
            k = torch.cat([k, k_c], dim=2)
            v = torch.cat([v, v_c], dim=2)
            attn_mask = None

        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        attn = attn.transpose(1, 2).contiguous().view(B, T, C)
        attn = self.proj(attn)

        return attn, k_cache, v_cache


# ---------------------------------------------------------------------------
# InContextLayer (DiT block)
# ---------------------------------------------------------------------------


class InContextLayer(nn.Module):
    """A DiT block with in-context conditioning."""

    def __init__(self, channels: int, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.channels = channels
        self.dtype = dtype
        # NOTE: norm1 is unused in the itarflow implementation but kept for compatibility
        self.norm1 = nn.LayerNorm(channels, bias=False, elementwise_affine=False, dtype=dtype)
        self.norm2 = nn.LayerNorm(channels, bias=False, elementwise_affine=False, dtype=dtype)
        self.attn = ConditionalAttention(channels)
        self.mlp = MLP(channels, self.dtype, expansion=4)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        k_cache: dict | None = None,
        v_cache: dict | None = None,
        mask: torch.Tensor | None = None,
        which_cache: str = "cond",
        train: bool = True,
    ) -> tuple[torch.Tensor, dict | None, dict | None]:
        c = c.reshape(-1, 1, self.channels) if c.ndim == 2 else c
        attn, k_cache, v_cache = self.attn(
            x,
            c,
            mask,
            k_cache=k_cache,
            v_cache=v_cache,
            which_cache=which_cache,
            train=train,
        )
        x = x + attn
        x = x + self.mlp(x)
        return x, k_cache, v_cache


# ---------------------------------------------------------------------------
# FinalLayer (adaLN + projection)
# ---------------------------------------------------------------------------


class FinalLayer(nn.Module):
    """Final layer of a MetaBlock: adaLN modulation + linear projection."""

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
        x = self.proj_out(x)
        return x
