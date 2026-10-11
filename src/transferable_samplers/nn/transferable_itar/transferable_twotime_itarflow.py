# Transferable two-time iTarFlow (Variant 1).
#
# Two-time (s, t)-conditioned affine coupling on peptide point clouds.
#
# Conditioning mechanisms:
#   * (s, t) enter as in-context attention tokens [c_s, c_t] and via DiT-style
#     adaLN (c_s + c_t) on the final layer (iTarFlow / DiT convention).
#   * Peptide identity enters as an additive per-token bias inside every
#     InContextLayer, mirroring how transferable-tarflow injects its condition
#     (nn/tarflow/attention.py::AttentionBlock, line 280-281).
#
# Interfaces:
#   forward_and_log_det(x, context, encodings, mask, permutations) -> (z, logdet)
#   inverse_and_log_det(y, context, encodings, mask=None, permutations)  -> (x, logdet)
# Convention: forward = data -> noise; inverse = noise -> data.
#
# ``context`` is a ``[B, 2]`` tensor of ``[s, t]`` time pairs.

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from transferable_samplers.nn.embedder import SinusoidalEmbedding
from transferable_samplers.nn.transferable_itar.components import (
    FinalLayer,
    InContextLayerMasked,
    PermutationFromDict,
    TimestepEmbedding,
)

MAX_SEQ_LEN = 512


class TransferableTwoTimeInContextMetaBlock(nn.Module):
    """A single autoregressive coupling block with in-context (s, t) tokens
    and additive peptide conditioning.

    Args:
        in_channels: Number of coordinate channels per token (e.g. 3).
        channels: Hidden channel dimension.
        num_patches: Maximum number of tokens (for pos_embed).
        num_layers: Number of ``InContextLayerMasked`` layers.
        permutation: A ``PermutationFromDict`` that permutes tokens by dict key.
        head_dim: Attention head dimension.
        use_qkln: Whether to apply LayerNorm on Q and K.
        lookahead_conditioning: If True, condition on ``[cond_i, cond_{i+1}]``.
        dtype: Parameter dtype.
        clip_range: Clip (alpha, mu) to +/-clip_range.
        gated: If True, coupling is scaled by ``dt = t - s`` (identity at s=t).

    Logdet contract: this block always returns the SUMMED logdet
    ``alpha.sum(dim=(1,2))`` (padded rows have ``alpha=0`` so the sum is
    exact under padding).  The trainer divides once by the per-sample
    active dimension ``num_active_tokens * C_in``.
    """

    def __init__(
        self,
        in_channels: int,
        channels: int,
        num_patches: int,
        num_layers: int,
        permutation: PermutationFromDict,
        head_dim: int = 64,
        use_qkln: bool = False,
        lookahead_conditioning: bool = False,
        dtype: Any = torch.float32,
        clip_range: float | None = None,
        gated: bool = False,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.num_patches = num_patches
        self.permutation = permutation
        self.clip_range = clip_range
        self.gated = gated
        self.lookahead_conditioning = lookahead_conditioning

        self.proj_in = nn.Linear(in_channels, channels, dtype=dtype)
        nn.init.xavier_uniform_(self.proj_in.weight)
        nn.init.zeros_(self.proj_in.bias)

        # Peptide-condition projection (matches transferable-tarflow):
        # - plain Linear when no lookahead;
        # - 2-layer MLP over ``[cond_i, cond_{i+1}]`` when lookahead is on.
        if not lookahead_conditioning:
            self.proj_cond = nn.Linear(channels, channels, dtype=dtype)
        else:
            self.proj_cond = nn.Sequential(
                nn.Linear(channels * 2, channels, dtype=dtype),
                nn.GELU(),
                nn.Linear(channels, channels, dtype=dtype),
            )

        self.register_buffer(
            "pos_embedding",
            SinusoidalEmbedding(embed_size=channels, div_value=10000.0)(torch.arange(MAX_SEQ_LEN)),
            persistent=False,
        )
        self.pos_embed_scale = nn.Parameter(torch.ones(1) * 1e-2)

        self.blocks = nn.ModuleList(
            [
                InContextLayerMasked(channels=channels, head_dim=head_dim, use_qkln=use_qkln, dtype=dtype)
                for _ in range(num_layers)
            ]
        )

        self.proj_out = FinalLayer(hidden_dim=channels, out_channels=in_channels, dtype=dtype)

        self.register_buffer(
            "attn_mask_buf",
            torch.tril(torch.ones(num_patches, num_patches, dtype=torch.bool)),
        )

    # -----------------------------------------------------------------
    # Peptide-condition helper: apply permutation, lookahead, projection
    # -----------------------------------------------------------------

    def _prepare_peptide_cond(
        self,
        cond: torch.Tensor,
        permutations: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Apply block permutation and (optional) lookahead, then project.

        Args:
            cond: ``[B, T, channels]`` per-token peptide encoding.
            permutations: dict of permutation buffers.

        Returns:
            ``[B, T, channels]`` per-token conditioning added inside every layer.
        """
        cond = self.permutation(cond, permutations)
        if self.lookahead_conditioning:
            lookahead_cond = torch.cat([cond[:, 1:], torch.zeros_like(cond[:, :1])], dim=1)
            cond = torch.cat([cond, lookahead_cond], dim=-1)
        return self.proj_cond(cond)

    # -----------------------------------------------------------------
    # Forward pass: compute (alpha, mu) then apply coupling
    # -----------------------------------------------------------------

    def _compute_alpha_mu(
        self,
        x_in: torch.Tensor,
        c_s: torch.Tensor,
        c_t: torch.Tensor,
        peptide_cond: torch.Tensor | None,
        pad_mask: torch.Tensor | None,
        train: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run transformer to produce (alpha, mu) coupling parameters."""
        x = self.proj_in(x_in)
        # Add pos-embedding for the (permuted) sequence positions.
        x = x + self.pos_embedding[: x.shape[1]].unsqueeze(0) * self.pos_embed_scale
        if pad_mask is not None:
            x = x * pad_mask[..., None]

        c_attn = torch.stack([c_s, c_t], dim=1)  # [B, 2, C]
        c_adaln = c_s + c_t  # [B, C]

        attn_mask = self.attn_mask_buf[: x.shape[1], : x.shape[1]]
        for block in self.blocks:
            x, _, _ = block(
                x,
                c_attn,
                peptide_cond=peptide_cond,
                causal_mask=attn_mask,
                pad_mask=pad_mask,
                train=train,
            )

        x = self.proj_out(x, c_adaln)  # [B, T, 2*C_in]
        # Causal shift: first token gets zero parameters (identity).
        x = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        if pad_mask is not None:
            x = x * pad_mask[..., None]
        if self.clip_range is not None:
            x = torch.clip(x, min=-self.clip_range, max=self.clip_range)
        alpha, mu = x.chunk(2, dim=-1)
        return alpha, mu

    def forward(
        self,
        x: torch.Tensor,
        permutations: dict[str, torch.Tensor],
        c_s: torch.Tensor,
        c_t: torch.Tensor,
        peptide_cond_pre: torch.Tensor | None,
        mask: torch.Tensor | None = None,
        s: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward map (data -> noise) for a single block.

        Args:
            x: ``[B, T, C_in]`` tokens (unpermuted).
            permutations: dict with the block's semantic permutation.
            c_s, c_t: ``[B, channels]`` time embeddings.
            peptide_cond_pre: ``[B, T, channels]`` peptide encoding BEFORE
                permutation (this method permutes it internally).  May be None
                for unconditional models.
            mask: ``[B, T]`` padding mask (1=valid, 0=padded).
            s, t: ``[B]`` source/target times (only needed when gated).
        """
        x_in = self.permutation(x, permutations)  # permuted input tokens
        if mask is not None:
            pad_mask = self.permutation(mask.to(x.dtype)[..., None], permutations)
            pad_mask = pad_mask.squeeze(-1)  # [B, T]
        else:
            pad_mask = None

        peptide_cond = (
            self._prepare_peptide_cond(peptide_cond_pre, permutations) if peptide_cond_pre is not None else None
        )

        alpha, mu = self._compute_alpha_mu(x_in, c_s, c_t, peptide_cond=peptide_cond, pad_mask=pad_mask, train=True)

        # Apply tokenization_map mask if present (identity coupling at untokenized positions).
        tokenization_map = permutations.get("tokenization_map", None)
        if tokenization_map is not None:
            tok_mask = (tokenization_map != -1).float().repeat_interleave(x.shape[-1], dim=-1)
            tok_mask = self.permutation(tok_mask, permutations)
            alpha = alpha * tok_mask
            mu = mu * tok_mask

        if self.gated:
            assert s is not None and t is not None, "s, t required when gated=True"
            dt = (t - s).view(-1, 1, 1)
            alpha = dt * alpha
            mu = dt * mu

        scale = (-alpha.float()).exp().type(alpha.dtype)
        x_out = (x_in - mu) * scale

        # Summed logdet contract: padded rows have alpha=0 (masked earlier),
        # so summing over (T, C) is exact under padding.  The trainer divides
        # once by per-sample active dim.
        logdet = -alpha.sum(dim=(1, 2))

        x_out = self.permutation(x_out, permutations, inverse=True)
        if mask is not None:
            x_out = x_out * mask[..., None]
        return x_out, logdet

    # -----------------------------------------------------------------
    # Reverse pass (autoregressive, no padding — single peptide at a time)
    # -----------------------------------------------------------------

    def reverse_step(
        self,
        x_recon: torch.Tensor,
        i: int,
        c_s: torch.Tensor,
        c_t: torch.Tensor,
        peptide_cond: torch.Tensor | None,
        k_cache: list[dict],
        v_cache: list[dict],
        which_cache: str = "cond",
    ) -> tuple[torch.Tensor, torch.Tensor, list[dict], list[dict]]:
        """Autoregressive step producing (alpha, mu) for token ``i+1``."""
        B, T, C = x_recon.shape

        x_step = self.proj_in(x_recon[:, i : i + 1, :])
        x_step = x_step + self.pos_embedding[i : i + 1].unsqueeze(0) * self.pos_embed_scale

        if peptide_cond is not None:
            peptide_cond_i = peptide_cond[:, i : i + 1, :]
        else:
            peptide_cond_i = None

        c_attn = torch.stack([c_s, c_t], dim=1)  # [B, 2, C]
        c_adaln = c_s + c_t  # [B, C]

        for bi, block in enumerate(self.blocks):
            x_step, k_cache[bi], v_cache[bi] = block(
                x_step,
                c_attn,
                peptide_cond=peptide_cond_i,
                causal_mask=None,
                pad_mask=None,
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
        permutations: dict[str, torch.Tensor],
        c_s: torch.Tensor,
        c_t: torch.Tensor,
        peptide_cond_pre: torch.Tensor | None = None,
        s: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Autoregressive inverse: noise -> data."""
        x_recon = self.permutation(x, permutations)
        B, T, C = x_recon.shape

        head_dim = self.blocks[0].attn.head_dim
        num_heads = self.blocks[0].attn.num_heads

        if self.gated:
            assert s is not None and t is not None
            dt = (t - s).view(-1, 1, 1)
        else:
            dt = 1.0

        peptide_cond = (
            self._prepare_peptide_cond(peptide_cond_pre, permutations) if peptide_cond_pre is not None else None
        )

        # Tokenization mask (permuted) if present.
        tokenization_map = permutations.get("tokenization_map", None)
        if tokenization_map is not None:
            tok_mask_full = (tokenization_map != -1).float().repeat_interleave(C, dim=-1)
            tok_mask_full = self.permutation(tok_mask_full, permutations)
        else:
            tok_mask_full = None

        def _new_cache() -> list[dict]:
            return [
                {
                    "cond": torch.zeros(B, num_heads, T, head_dim, device=x.device, dtype=x.dtype),
                    "uncond": torch.zeros(B, num_heads, T, head_dim, device=x.device, dtype=x.dtype),
                    "idx": 0,
                }
                for _ in range(len(self.blocks))
            ]

        k_cache = _new_cache()
        v_cache = _new_cache()

        alphas: list[torch.Tensor] = []
        for i in range(T - 1):
            for bi in range(len(self.blocks)):
                k_cache[bi]["idx"] = i
                v_cache[bi]["idx"] = i

            alpha_step, mu_step, k_cache, v_cache = self.reverse_step(
                x_recon,
                i=i,
                c_s=c_s,
                c_t=c_t,
                peptide_cond=peptide_cond,
                k_cache=k_cache,
                v_cache=v_cache,
                which_cache="cond",
            )

            # Apply tokenization mask at position i+1.
            if tok_mask_full is not None:
                tok_slice = tok_mask_full[:, i + 1 : i + 2, :]
                alpha_step = alpha_step * tok_slice
                mu_step = mu_step * tok_slice

            alphas.append(alpha_step)
            next_token = x_recon[:, i + 1 : i + 2, :]
            x_recon = x_recon.clone()
            x_recon[:, i + 1 : i + 2, :] = next_token * torch.exp(dt * alpha_step) + dt * mu_step

        # Assemble logdet.
        alphas_tensor = torch.cat([torch.zeros_like(alphas[0]), *alphas], dim=1)
        alphas_scaled = dt * alphas_tensor if self.gated else alphas_tensor
        # Summed logdet contract (reverse assumes single peptide, no padding).
        reverse_logdet = alphas_scaled.sum(dim=(1, 2))

        return self.permutation(x_recon, permutations, inverse=True), reverse_logdet


# ---------------------------------------------------------------------------
# TransferableTwoTimeITarFlow (top-level model)
# ---------------------------------------------------------------------------


class TransferableTwoTimeITarFlow(nn.Module):
    """Transferable two-time iTarFlow for variable-size peptides.

    Wraps a stack of ``TransferableTwoTimeInContextMetaBlock``s with 3D
    coordinate tokens (one atom = one token, C=3).

    The NFMD trainer calls ``forward_and_log_det(x, context, encodings, mask,
    permutations)`` where:
      * ``x`` is ``[B, T, 3]`` or ``[B, T*3]`` flat coords.
      * ``context`` is ``[B, 2]`` of ``[s, t]``.
      * ``encodings``, ``permutations``, ``mask`` come from the
        ManyPeptidesMD datamodule batch dict.

    Convention: forward = data -> noise; inverse = noise -> data.

    Logdet contract: the net returns the SUMMED per-sample logdet ``[B]``
    (padded rows have ``alpha=0`` so the sum is exact under padding); the
    trainer divides once by the per-sample active dimension.  This matches
    the "pure-sum end-to-end" contract used by ``NormalizingFlowModule``.
    """

    def __init__(
        self,
        input_dimension: int,
        max_num_tokens: int,
        channels: int,
        num_blocks: int,
        layers_per_block: int,
        head_dim: int = 64,
        use_qkln: bool = False,
        lookahead_conditioning: bool = False,
        permutation_keys: list[str] | None = None,
        cond_embed: nn.Module | None = None,
        clip_range: float = 5.0,
        gated: bool = False,
    ) -> None:
        super().__init__()
        self.input_dimension = input_dimension
        self.max_num_tokens = max_num_tokens
        self.gated = gated
        self.conditional = cond_embed is not None
        self.cond_embed = cond_embed

        if permutation_keys is None:
            permutation_keys = [
                "n2c_residue-by-residue_standard",
                "n2c_residue-by-residue_standard_flip",
            ]
        permutation_keys = list(permutation_keys) * (num_blocks // len(permutation_keys) + 1)

        self.time_embed_s = TimestepEmbedding(channels)
        self.time_embed_t = TimestepEmbedding(channels)

        self.blocks = nn.ModuleList(
            [
                TransferableTwoTimeInContextMetaBlock(
                    in_channels=input_dimension,
                    channels=channels,
                    num_patches=max_num_tokens,
                    num_layers=layers_per_block,
                    permutation=PermutationFromDict(permutation_keys[i]),
                    head_dim=head_dim,
                    use_qkln=use_qkln,
                    lookahead_conditioning=lookahead_conditioning,
                    clip_range=clip_range,
                    gated=gated,
                )
                for i in range(num_blocks)
            ]
        )

    # -----------------------------------------------------------------
    # Coord reshape helpers
    # -----------------------------------------------------------------

    def _to_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``[B, T, C]`` tokens.  Accepts ``[B, T, C]`` or ``[B, T*C]``."""
        if x.ndim == 3:
            return x
        assert x.ndim == 2
        return x.reshape(x.shape[0], -1, self.input_dimension)

    def _to_original_shape(self, x_tokens: torch.Tensor, was_flat: bool) -> torch.Tensor:
        if was_flat:
            return x_tokens.reshape(x_tokens.shape[0], -1)
        return x_tokens

    def _parse_context(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        s = context[:, 0]
        t = context[:, 1]
        return s, t, self.time_embed_s(s), self.time_embed_t(t)

    def _build_peptide_cond(
        self,
        encodings: dict[str, torch.Tensor] | None,
        mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if not self.conditional:
            return None
        assert encodings is not None, "encodings must be provided for conditional model"
        return self.cond_embed(**encodings, mask=mask)  # [B, T, channels]

    # -----------------------------------------------------------------
    # Forward / inverse with log-det
    # -----------------------------------------------------------------

    def forward_and_log_det(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        encodings: dict[str, torch.Tensor] | None = None,
        mask: torch.Tensor | None = None,
        permutations: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        was_flat = x.ndim == 2
        x_tok = self._to_tokens(x)
        s, t, c_s, c_t = self._parse_context(context)
        assert permutations is not None, "permutations dict required"
        peptide_cond = self._build_peptide_cond(encodings, mask)

        total_logdet = torch.zeros(x_tok.shape[0], device=x_tok.device, dtype=x_tok.dtype)
        for block in self.blocks:
            x_tok, logdet = block(
                x_tok,
                permutations=permutations,
                c_s=c_s,
                c_t=c_t,
                peptide_cond_pre=peptide_cond,
                mask=mask,
                s=s,
                t=t,
            )
            total_logdet = total_logdet + logdet

        return self._to_original_shape(x_tok, was_flat), total_logdet

    def inverse_and_log_det(
        self,
        y: torch.Tensor,
        context: torch.Tensor,
        encodings: dict[str, torch.Tensor] | None = None,
        mask: torch.Tensor | None = None,
        permutations: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reverse map (noise -> data).  No padding: single peptide at a time."""
        was_flat = y.ndim == 2
        x_tok = self._to_tokens(y)
        s, t, c_s, c_t = self._parse_context(context)
        assert permutations is not None
        peptide_cond = self._build_peptide_cond(encodings, mask=None)

        total_logdet = torch.zeros(x_tok.shape[0], device=x_tok.device, dtype=x_tok.dtype)
        for block in reversed(list(self.blocks)):
            x_tok, block_logdet = block.reverse(
                x_tok,
                permutations=permutations,
                c_s=c_s,
                c_t=c_t,
                peptide_cond_pre=peptide_cond,
                s=s,
                t=t,
            )
            total_logdet = total_logdet + block_logdet

        return self._to_original_shape(x_tok, was_flat), total_logdet
