# Two-time iTarFlow: (s, t)-conditioned affine coupling for molecules.
#
# Ported from nf-flow-map for use in NFMD training.
# Supports both standard and gated variants:
#   Standard (gated=False): y = (x - mu) * exp(-alpha)
#   Gated    (gated=True):  y = (x - dt*mu) * exp(-dt*alpha),  dt = t - s
#
# Identity at s=t is structurally guaranteed only when gated=True.
#
# Convention: forward_and_log_det maps data -> noise (normalizing direction).
#             inverse_and_log_det maps noise -> data (generative direction).

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from transferable_samplers.nn.tarflow.itarflow_components import (
    FinalLayer,
    InContextLayer,
    PermutationFlip,
    TimestepEmbedding,
)

# ---------------------------------------------------------------------------
# TwoTimeInContextMetaBlock
# ---------------------------------------------------------------------------


class TwoTimeInContextMetaBlock(nn.Module):
    """InContextMetaBlock with two-time (s, t) conditioning.

    When ``gated=False`` (default), uses the standard affine coupling:
        y = (x - mu) * exp(-alpha)
    conditioned on (s, t) via 3 attention tokens [c_class, c_s, c_t]
    and adaLN on c_class + c_s + c_t.

    When ``gated=True``, gates the coupling by dt = t - s:
        y = (x - dt*mu) * exp(-dt*alpha)
    which is the identity at s = t.
    """

    def __init__(
        self,
        in_channels: int,
        channels: int,
        num_patches: int,
        num_layers: int,
        num_classes: int,
        permutation: PermutationFlip,
        dtype: Any = torch.float32,
        clip_range: float | None = None,
        gated: bool = False,
        logdet_sum: bool = False,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.num_patches = num_patches
        self.num_classes = num_classes
        self.permutation = permutation
        self.clip_range = clip_range
        self.gated = gated
        self.logdet_sum = logdet_sum

        self.proj_in = nn.Linear(in_channels, channels, dtype=dtype)
        nn.init.xavier_uniform_(self.proj_in.weight)
        nn.init.zeros_(self.proj_in.bias)

        self.pos_embedding = nn.Parameter(torch.randn(num_patches, channels) * 0.02)
        self.class_embedding = nn.Parameter(torch.randn(num_classes, channels) * 0.02)

        self.blocks = nn.ModuleList([InContextLayer(channels=channels, dtype=dtype) for _ in range(num_layers)])

        self.proj_out = FinalLayer(hidden_dim=channels, out_channels=in_channels, dtype=dtype)
        self.attn_mask = lambda device=None: torch.tril(
            torch.ones((self.num_patches, self.num_patches), dtype=torch.bool, device=device)
        )

    def _compute_alpha_mu(
        self,
        x_in: torch.Tensor,
        y: torch.Tensor,
        c_s: torch.Tensor,
        c_t: torch.Tensor,
        train: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run transformer to produce (alpha, mu) parameters.

        Args:
            x_in: ``[B, T, C_in]`` permuted input tokens.
            y: ``[B]`` class labels (negative values = unconditional).
            c_s: ``[B, channels]`` time embedding for s.
            c_t: ``[B, channels]`` time embedding for t.
            train: Whether in training mode (parallel attention).

        Returns:
            ``(alpha, mu)`` each ``[B, T, C_in]``.
        """
        x = self.proj_in(x_in)
        x = x + self.permutation(self.pos_embedding.unsqueeze(0))

        mask = (y < 0).to(torch.float32).reshape(-1, 1)
        c_class = (1 - mask) * self.class_embedding[y] + mask * self.class_embedding.mean(dim=0)
        c_attn = torch.stack([c_class, c_s, c_t], dim=1)  # [B, 3, channels]
        c_adaln = c_class + c_s + c_t  # [B, channels]

        for block in self.blocks:
            x, _, _ = block(x, c_attn, mask=self.attn_mask(x.device), train=train)

        x = self.proj_out(x, c_adaln)  # [B, T, 2*C_in]
        # Causal shift: first token always gets zero parameters
        x = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        if self.clip_range is not None:
            x = torch.clip(x, min=-self.clip_range, max=self.clip_range)
        alpha, mu = x.chunk(2, dim=-1)
        return alpha, mu

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        c_s: torch.Tensor,
        c_t: torch.Tensor,
        s: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass (data -> noise direction).

        Args:
            x: ``[B, T, C]`` input tokens.
            y: ``[B]`` class labels (negative values = unconditional).
            c_s: ``[B, channels]`` time embedding for s.
            c_t: ``[B, channels]`` time embedding for t.
            s: ``[B]`` source time (required when ``gated=True``).
            t: ``[B]`` target time (required when ``gated=True``).

        Returns:
            ``(x_new, log_jacob, alpha, mu)``.
        """
        x_in = self.permutation(x)
        alpha, mu = self._compute_alpha_mu(x_in, y, c_s, c_t, train=True)

        if self.gated:
            assert s is not None and t is not None, "s, t required when gated=True"
            dt = (t - s).view(-1, 1, 1)
            alpha = dt * alpha
            mu = dt * mu

        x_new = (x_in - mu) * torch.exp(-alpha)  # [B, T, C_in]
        log_jacob = -alpha.sum(dim=(1, 2)) if self.logdet_sum else -alpha.mean(dim=(1, 2))
        x_new = self.permutation(x_new)
        return x_new, log_jacob, alpha, mu

    def reverse_step(
        self,
        x: torch.Tensor,
        i: int,
        y: torch.Tensor,
        c_s: torch.Tensor,
        c_t: torch.Tensor,
        k_cache: list[dict],
        v_cache: list[dict],
        which_cache: str = "cond",
    ) -> tuple[torch.Tensor, torch.Tensor, list[dict], list[dict]]:
        """Single autoregressive reverse step (token i)."""
        B, T, C = x.shape
        x_in = x[:, i : i + 1, :]

        x_step = self.proj_in(x_in)
        pos_embed = self.permutation(self.pos_embedding.unsqueeze(0))
        x_step = x_step + pos_embed[:, i : i + 1, :]

        # Handle class conditioning (CFG drops class only, not time)
        if y is not None:
            y = y.long()
            mask_cond = (y < 0).float().view(-1, 1)
            c_class = (1 - mask_cond) * self.class_embedding[y] + mask_cond * self.class_embedding.mean(dim=0)
        else:
            c_class = self.class_embedding.mean(dim=0).unsqueeze(0).expand(B, -1)

        c_attn = torch.stack([c_class, c_s, c_t], dim=1)  # [B, 3, channels]
        c_adaln = c_class + c_s + c_t  # [B, channels]

        for bi, block in enumerate(self.blocks):
            x_step, k_cache[bi], v_cache[bi] = block(
                x_step,
                c_attn,
                mask=None,
                k_cache=k_cache[bi],
                v_cache=v_cache[bi],
                which_cache=which_cache,
                train=False,
            )

        x_step = self.proj_out(x_step, c_adaln)  # [B, 1, 2*C]

        if self.clip_range is not None:
            x_step = torch.clamp(x_step, min=-self.clip_range, max=self.clip_range)

        alpha, mu = x_step.chunk(2, dim=-1)
        return alpha, mu, k_cache, v_cache

    def reverse(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
        c_s: torch.Tensor | None = None,
        c_t: torch.Tensor | None = None,
        s: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
        guidance: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Autoregressive inverse: x = y * exp(alpha) + mu (noise -> data).

        When ``gated=True``, alpha and mu are scaled by dt = t - s.
        """
        x_recon = self.permutation(x)
        B, T, C = x_recon.shape
        num_heads = max(1, self.channels // 64)
        head_dim = self.channels // num_heads

        # Compute dt for gated variant; use 1.0 for standard (no-op multiplier)
        if self.gated:
            assert s is not None and t is not None, "s, t required when gated=True"
            dt = (t - s).view(-1, 1, 1)  # [B, 1, 1]
        else:
            dt = 1.0

        k_cache = [
            {
                "cond": torch.zeros(B, num_heads, T, head_dim, device=x.device, dtype=x.dtype),
                "uncond": torch.zeros(B, num_heads, T, head_dim, device=x.device, dtype=x.dtype),
                "idx": 0,
            }
            for _ in range(len(self.blocks))
        ]
        v_cache = [
            {
                "cond": torch.zeros(B, num_heads, T, head_dim, device=x.device, dtype=x.dtype),
                "uncond": torch.zeros(B, num_heads, T, head_dim, device=x.device, dtype=x.dtype),
                "idx": 0,
            }
            for _ in range(len(self.blocks))
        ]

        alphas = []
        for i in range(T - 1):
            for bi in range(len(self.blocks)):
                k_cache[bi]["idx"] = i
                v_cache[bi]["idx"] = i

            alpha_step, mu_step, k_cache, v_cache = self.reverse_step(
                x_recon,
                i=i,
                y=y,
                c_s=c_s,
                c_t=c_t,
                k_cache=k_cache,
                v_cache=v_cache,
                which_cache="cond",
            )

            if guidance > 0:
                alpha_u, mu_u, k_cache, v_cache = self.reverse_step(
                    x_recon,
                    i=i,
                    y=None,
                    c_s=c_s,
                    c_t=c_t,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    which_cache="uncond",
                )
                scale_cond = 1 + guidance * (i + 1) / (T - 1)
                scale_uncond = guidance * (i + 1) / (T - 1)
                alpha_step = scale_cond * alpha_step - scale_uncond * alpha_u
                mu_step = scale_cond * mu_step - scale_uncond * mu_u

            alphas.append(alpha_step)
            # Reverse coupling: x = y * exp(dt * alpha) + dt * mu
            next_token = x_recon[:, i + 1 : i + 2, :]
            x_recon[:, i + 1 : i + 2, :] = next_token * torch.exp(dt * alpha_step) + dt * mu_step

        # Reverse log-det: dt * reduction(alpha) (negative of forward log-det)
        alphas_tensor = torch.cat([torch.zeros_like(alphas[0]), *alphas], dim=1)
        reverse_logdet = (
            (dt * alphas_tensor).sum(dim=(1, 2)) if self.logdet_sum else (dt * alphas_tensor).mean(dim=(1, 2))
        )

        return self.permutation(x_recon), reverse_logdet


# ---------------------------------------------------------------------------
# TwoTimeITarFlowMolecules — molecule wrapper with NormalizingFlow-like interface
# ---------------------------------------------------------------------------


class TwoTimeITarFlowMolecules(nn.Module):
    """Two-time iTarFlow for molecules.

    Wraps ``TwoTimeInContextMetaBlock`` with 1D patchify/unpatchify.
    Accepts a ``context = [B, 2]`` tensor of ``[s, t]`` time pairs.
    Unconditional: a single null class embedding is used internally.

    Convention:
        - ``forward_and_log_det(x, context)`` maps data -> noise.
        - ``inverse_and_log_det(y, context)`` maps noise -> data.

    Args:
        in_channels: Number of coordinate channels per particle (e.g. 3 for xyz).
        img_size: Total flat dimension of molecule coordinates.
        patch_size: Patch/token size for the 1D patchification.
        channels: Hidden channel dimension.
        num_blocks: Number of autoregressive coupling blocks.
        num_layers: Number of transformer layers per block.
        dtype: Parameter dtype.
        clip_range: Clipping range for coupling parameters.
        gated: If ``True``, use ``(t-s)``-gated coupling (identity at ``s=t``).
        logdet_sum: If ``True``, sum logdet over all dims; else mean.
    """

    def __init__(
        self,
        in_channels: int,
        img_size: int,
        patch_size: int = 1,
        channels: int = 256,
        num_blocks: int = 4,
        num_layers: int = 4,
        dtype: torch.dtype = torch.float32,
        clip_range: float = 1.0,
        gated: bool = False,
        logdet_sum: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = img_size // patch_size // in_channels
        self.in_channels_per_token = in_channels * patch_size
        self.gated = gated
        self.logdet_sum = logdet_sum

        if in_channels != 1:
            assert img_size % in_channels == 0, (
                f"img_size ({img_size}) must be divisible by in_channels ({in_channels})"
            )

        self.time_embed_s = TimestepEmbedding(channels)
        self.time_embed_t = TimestepEmbedding(channels)

        permutations = [PermutationFlip(i % 2 == 1) for i in range(num_blocks)]
        self.blocks = nn.ModuleList(
            [
                TwoTimeInContextMetaBlock(
                    in_channels=self.in_channels_per_token,
                    channels=channels,
                    num_patches=self.num_patches,
                    num_layers=num_layers,
                    num_classes=1,
                    permutation=permutations[i],
                    dtype=dtype,
                    clip_range=clip_range,
                    gated=gated,
                    logdet_sum=logdet_sum,
                )
                for i in range(num_blocks)
            ]
        )

    # -- patchify / unpatchify --

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert flat coordinates ``[B, dim]`` to patches ``[B, T, C]``."""
        if self.in_channels != 1:
            return x.reshape(-1, self.img_size // self.in_channels, self.in_channels)
        else:
            x = x.reshape(-1, 1, self.img_size, 1)
            u = torch.nn.functional.unfold(x, self.patch_size, stride=self.patch_size)
            return u.transpose(1, 2)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert patches ``[B, T, C]`` to flat coordinates ``[B, dim]``."""
        return x.reshape(-1, self.img_size)

    # -- context parsing --

    def _parse_context(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split ``[B, 2]`` context into s, t scalars and time embeddings."""
        s = context[:, 0]  # [B]
        t = context[:, 1]  # [B]
        c_s = self.time_embed_s(s)  # [B, channels]
        c_t = self.time_embed_t(t)  # [B, channels]
        return s, t, c_s, c_t

    # -- forward / inverse with log-det --

    def forward_and_log_det(self, x: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward map with log-det. Data -> noise (normalizing direction).

        Args:
            x: ``[B, dim]`` flat molecule coordinates.
            context: ``[B, 2]`` tensor of ``[s, t]`` time pairs.

        Returns:
            ``(z, total_logdet)`` where ``z`` is ``[B, dim]``.
        """
        s, t, c_s, c_t = self._parse_context(context)
        x = self.patchify(x)
        dummy_y = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        total_logdet = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

        for block in self.blocks:
            x, logdet, _, _ = block(x, dummy_y, c_s, c_t, s, t)
            total_logdet = total_logdet + logdet

        return self.unpatchify(x), total_logdet

    def inverse_and_log_det(self, y: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Inverse map with log-det. Noise -> data (generative direction).

        Args:
            y: ``[B, dim]`` flat noise samples.
            context: ``[B, 2]`` tensor of ``[s, t]`` time pairs.

        Returns:
            ``(x, total_logdet)`` where ``x`` is ``[B, dim]``.
        """
        s, t, c_s, c_t = self._parse_context(context)
        x = self.patchify(y)
        dummy_y = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)

        total_logdet = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        for block in reversed(list(self.blocks)):
            x, block_logdet = block.reverse(x, dummy_y, c_s=c_s, c_t=c_t, s=s, t=t, guidance=0.0)
            total_logdet += block_logdet

        return self.unpatchify(x), total_logdet
