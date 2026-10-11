# NFMD training module for transferable two-time iTarFlow variants.
#
# Fork of ``NFMDModule`` for transferable models that thread the batch
# ``permutations``, ``encodings``, and ``mask`` tensors through
# ``forward_and_log_det`` / ``inverse_and_log_det``.
#
# Objective: identical one-time NFMD variant as ``NFMDModule``:
#   sample s in [0, nfmd_t_max], fix t = 1, build x_s = (1-s)*x_0 + s*eps,
#   encode via f_{s,1}, and minimise  0.5/d ||z||^2 - 1/d logdet.
#
# ``x_data`` remains ``[B, T, 3]`` throughout so we can multiply by the padding
# mask without collapsing dimensions.  We rely on ``mask``-aware coord standardisation
# and prior sampling for correctness on padded batches (the datamodule zero-pads
# ``x`` and encodings, and the NormalDistribution prior is mask-aware).
#
# Logdet contract (transferable-only):
#   The transferable pipeline uses padded variable-length batches, so we use
#   the "pure sum end-to-end" contract:
#     * The net returns the SUMMED per-sample logdet ``[B]`` (padded rows
#       contribute zero because ``alpha`` is masked to zero).
#     * The prior returns the SUMMED per-sample log-prob ``[B]`` (mask-aware).
#     * The trainer divides once by the per-sample active dimension
#       ``(mask.int().sum(-1) * spatial)`` to produce the per-dim NFMD/MLE loss.
#   This matches ``NormalizingFlowModule``.  The transferable net classes hard-code
#   the summed-logdet reduction (no ``logdet_sum`` parameter) so misconfiguration
#   is structurally impossible.

from __future__ import annotations

from typing import Any

import torch

from transferable_samplers.models.base_lightning_module import BaseLightningModule
from transferable_samplers.models.priors.prior import Prior
from transferable_samplers.utils.dataclasses import SourceEnergyConfig, SystemCond
from transferable_samplers.utils.pylogger import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=False)


class TransferableNFMDModule(BaseLightningModule):
    """NFMD training for a transferable two-time normalizing flow.

    The network must implement:

        forward_and_log_det(x, context, encodings, mask, permutations) -> (z, logdet)
        inverse_and_log_det(y, context, encodings, mask, permutations) -> (x, logdet)

    where ``x`` is ``[B, T, 3]`` (or ``[B, T*3]`` flat), ``context`` is
    ``[B, 2]`` of ``[s, t]``, and ``encodings``/``permutations``/``mask``
    come from the ManyPeptidesMD-style batch dict.

    Same objective and diagnostics as ``NFMDModule``.
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        prior: Prior,
        scheduler: Any = None,
        compile_net: bool = False,
        source_energy_config: SourceEnergyConfig | None = None,
        train_from_buffer: bool = False,
        mean_free_prior: bool = False,
        objective: str = "nfmd",
        nfmd_t_max: float = 1.0,
        mle_loss: bool = False,
        mle_loss_weight: float = 1.0,
        nfmd_eval_num_s: int = 5,
    ) -> None:
        super().__init__(
            net=net,
            optimizer=optimizer,
            scheduler=scheduler,
            prior=prior,
            compile_net=compile_net,
            source_energy_config=source_energy_config,
            train_from_buffer=train_from_buffer,
            mean_free_prior=mean_free_prior,
        )

        valid_objectives = {"nfmd", "mle", "nfmd_plus_mle"}
        if objective not in valid_objectives:
            raise ValueError(f"Invalid objective {objective!r}; expected one of {sorted(valid_objectives)}")

        self.objective = objective
        self.nfmd_t_max = nfmd_t_max
        self.mle_loss = mle_loss
        self.mle_loss_weight = mle_loss_weight
        self.nfmd_eval_num_s = nfmd_eval_num_s

    # ------------------------------------------------------------------
    # Batch unpacking + system-dimension helpers.
    # ------------------------------------------------------------------

    @staticmethod
    def _unpack_batch(
        batch: dict[str, Any],
    ) -> tuple[
        torch.Tensor,
        dict[str, torch.Tensor] | None,
        dict[str, torch.Tensor] | None,
        torch.Tensor | None,
    ]:
        """Return (x[B,T,3], encodings, permutations, mask)."""
        x = batch["x"]
        assert x.ndim == 3, "molecules must be a pointcloud (batch, num_atoms, 3)"
        return x, batch.get("encodings"), batch.get("permutations"), batch.get("mask")

    @staticmethod
    def _system_dim(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """Per-sample active dimensionality (num_active_tokens * spatial)."""
        if mask is not None:
            return (mask.int().sum(-1).float() * x.shape[-1]).clamp(min=1.0)
        return x.new_full((x.shape[0],), float(x.shape[1] * x.shape[2]))

    @staticmethod
    def _logdet_per_dim(log_det: torch.Tensor, dim: torch.Tensor) -> torch.Tensor:
        """Return logdet on the per-dimension scale used by the NFMD objective.

        The net returns a summed logdet by construction; the per-dim scale is
        one division by the per-sample active dim.
        """
        return log_det.view(-1) / dim

    # ------------------------------------------------------------------
    # Core NFMD / MLE loss helpers.
    # ------------------------------------------------------------------

    def _sample_interpolant(
        self,
        x_data: torch.Tensor,
        mask: torch.Tensor | None,
        s: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (x_s, context) for the one-time interpolant at times s (t=1).

        Both endpoints already respect the padding mask: the data ``x_data``
        is zero-padded by the datamodule, and the sampled noise ``eps`` is
        multiplied by the mask so padded positions stay zero.
        """
        B = x_data.shape[0]
        eps = torch.randn_like(x_data)
        if mask is not None:
            eps = eps * mask[..., None]

        # Broadcast s over token / spatial dims.
        s_broad = s.view(B, 1, 1)
        x_s = (1.0 - s_broad) * x_data + s_broad * eps

        # Context: [s, 1] per sample.
        t_ones = torch.ones(B, 1, device=x_data.device, dtype=x_data.dtype)
        context = torch.cat([s.view(B, 1), t_ones], dim=-1)
        return x_s, context

    def _nfmd_loss_at(
        self,
        x_data: torch.Tensor,
        encodings: dict[str, torch.Tensor] | None,
        permutations: dict[str, torch.Tensor] | None,
        mask: torch.Tensor | None,
        s: torch.Tensor,
    ) -> torch.Tensor:
        """Per-sample NFMD loss with s as an already-drawn ``[B]`` tensor."""
        x_s, context = self._sample_interpolant(x_data, mask, s)
        z, log_det = self.net.forward_and_log_det(
            x_s, context, encodings=encodings, mask=mask, permutations=permutations
        )
        z = z.reshape(x_data.shape)
        if mask is not None:
            z = z * mask[..., None]

        # Per-sample squared norm reduced over active dims.
        sq = z.pow(2).sum(dim=(1, 2))  # sum over T,C so masked terms drop out
        dim = self._system_dim(x_data, mask)
        return 0.5 * sq / dim - self._logdet_per_dim(log_det, dim)

    def _compute_nfmd_loss(
        self,
        x_data: torch.Tensor,
        encodings: dict[str, torch.Tensor] | None,
        permutations: dict[str, torch.Tensor] | None,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Scalar training NFMD loss (random s uniform in [0, nfmd_t_max])."""
        B = x_data.shape[0]
        s = torch.rand(B, device=x_data.device, dtype=x_data.dtype) * self.nfmd_t_max
        return self._nfmd_loss_at(x_data, encodings, permutations, mask, s).mean()

    def _compute_mle_loss_per_sample(
        self,
        x_data: torch.Tensor,
        encodings: dict[str, torch.Tensor] | None,
        permutations: dict[str, torch.Tensor] | None,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Per-sample endpoint MLE loss (context [0, 1])."""
        B = x_data.shape[0]
        s = torch.zeros(B, device=x_data.device, dtype=x_data.dtype)
        # No noise at s=0.
        t_ones = torch.ones(B, 1, device=x_data.device, dtype=x_data.dtype)
        context = torch.cat([s.view(B, 1), t_ones], dim=-1)
        z, log_det = self.net.forward_and_log_det(
            x_data, context, encodings=encodings, mask=mask, permutations=permutations
        )
        z = z.reshape(x_data.shape)
        if mask is not None:
            z = z * mask[..., None]
        sq = z.pow(2).sum(dim=(1, 2))
        dim = self._system_dim(x_data, mask)
        return 0.5 * sq / dim - self._logdet_per_dim(log_det, dim)

    def _compute_mle_loss(
        self,
        x_data: torch.Tensor,
        encodings: dict[str, torch.Tensor] | None,
        permutations: dict[str, torch.Tensor] | None,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        return self._compute_mle_loss_per_sample(x_data, encodings, permutations, mask).mean()

    def _compute_nll_loss_per_sample(
        self,
        x_data: torch.Tensor,
        encodings: dict[str, torch.Tensor] | None,
        permutations: dict[str, torch.Tensor] | None,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Proper per-sample NLL at context [0, 1] using the prior's logp."""
        B = x_data.shape[0]
        context = torch.zeros(B, 2, device=x_data.device, dtype=x_data.dtype)
        context[:, 1] = 1.0
        z, log_det = self.net.forward_and_log_det(
            x_data, context, encodings=encodings, mask=mask, permutations=permutations
        )
        z = z.reshape(x_data.shape)
        if mask is not None:
            z = z * mask[..., None]
        logp_z = self.prior.logp(z, mask=mask)
        # Net returns summed logdet (enforced by guard) and prior.logp is
        # mask-aware summed log-prob, so both are on the summed scale.
        logq = logp_z + log_det.view(-1)
        return -logq

    # ------------------------------------------------------------------
    # Eval-grid helpers + primary-loss / eval-loss / training-step hooks.
    # ------------------------------------------------------------------

    def _nfmd_eval_s_grid(self, device: torch.device) -> torch.Tensor:
        return torch.linspace(0.0, self.nfmd_t_max, self.nfmd_eval_num_s, device=device)

    def _compute_nfmd_loss_at_s(
        self,
        x_data: torch.Tensor,
        encodings: dict[str, torch.Tensor] | None,
        permutations: dict[str, torch.Tensor] | None,
        mask: torch.Tensor | None,
        s_value: torch.Tensor,
    ) -> torch.Tensor:
        B = x_data.shape[0]
        s = torch.full((B,), float(s_value.item()), device=x_data.device, dtype=x_data.dtype)
        return self._nfmd_loss_at(x_data, encodings, permutations, mask, s)

    def _compute_nfmd_eval_loss_stochastic(
        self,
        x_data: torch.Tensor,
        encodings: dict[str, torch.Tensor] | None,
        permutations: dict[str, torch.Tensor] | None,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        B = x_data.shape[0]
        s = torch.rand(B, device=x_data.device, dtype=x_data.dtype) * self.nfmd_t_max
        return self._nfmd_loss_at(x_data, encodings, permutations, mask, s)

    def compute_primary_loss(self, batch: dict[str, Any]) -> torch.Tensor:
        """Per-sample primary loss (used by LossEvaluationCallback)."""
        x, enc, perm, mask = self._unpack_batch(batch)
        if self.objective == "mle":
            return self._compute_mle_loss_per_sample(x, enc, perm, mask)
        return self._compute_nfmd_eval_loss_stochastic(x, enc, perm, mask)

    def compute_eval_losses(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Per-sample validation losses matching NFMDModule."""
        x, enc, perm, mask = self._unpack_batch(batch)
        metrics: dict[str, torch.Tensor] = {}
        metrics["eval-nll"] = self._compute_nll_loss_per_sample(x, enc, perm, mask)

        if self.objective == "mle":
            metrics["eval-loss"] = self._compute_mle_loss_per_sample(x, enc, perm, mask)
            return metrics

        metrics["eval-loss"] = self._compute_nfmd_eval_loss_stochastic(x, enc, perm, mask)

        s_losses = []
        for idx, s_value in enumerate(self._nfmd_eval_s_grid(x.device), start=1):
            s_loss = self._compute_nfmd_loss_at_s(x, enc, perm, mask, s_value)
            s_losses.append(s_loss)
            metrics[f"eval-s{idx}"] = s_loss
        metrics["eval-loss-grid"] = torch.stack(s_losses, dim=0).mean(dim=0)
        return metrics

    # pyrefly: ignore [bad-override]
    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training loss for the configured objective."""
        if self.train_from_buffer:
            assert self._buffer is not None, "Buffer must be set for training from buffer"
            batch = self._buffer.sample(batch["x"].shape[0], device=self.device)

        x, enc, perm, mask = self._unpack_batch(batch)

        loss = x.new_tensor(0.0)
        use_nfmd = self.objective in {"nfmd", "nfmd_plus_mle"}
        use_mle = self.objective in {"mle", "nfmd_plus_mle"} or self.mle_loss

        if use_nfmd:
            nfmd_loss = self._compute_nfmd_loss(x, enc, perm, mask)
            self.log("train/nfmd_loss", nfmd_loss.item(), prog_bar=True, sync_dist=True)
            loss = loss + nfmd_loss

        if use_mle:
            mle_loss = self._compute_mle_loss(x, enc, perm, mask)
            self.log("train/mle_loss", mle_loss.item(), prog_bar=True, sync_dist=True)
            if self.objective == "mle":
                loss = loss + mle_loss
            else:
                loss = loss + self.mle_loss_weight * mle_loss

        batch_value = self.train_metrics(loss)
        self.log_dict(batch_value, prog_bar=True)
        return loss

    # ------------------------------------------------------------------
    # Generative interface (single peptide at a time via SystemCond).
    # ------------------------------------------------------------------

    def generate_proposal(
        self,
        net: torch.nn.Module,
        num_samples: int,
        system_cond: SystemCond | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample via inverse_and_log_det at context [0, 1].

        Uses SystemCond.for_batch to expand permutations/encodings to
        ``batch_size``.  Single peptide -> no padding mask.
        """
        with torch.no_grad():
            assert system_cond is not None, "TransferableNFMDModule requires SystemCond"
            assert system_cond.encodings is not None, "encodings required"
            num_atoms = system_cond.encodings["atom_type"].size(0)

            z = self.prior.sample(num_samples, num_atoms, device=self.device)
            logp_z = self.prior.logp(z)

            batched = system_cond.for_batch(num_samples, self.device)
            context = torch.zeros(num_samples, 2, device=self.device)
            context[:, 1] = 1.0

            x_pred, logdet_inv = net.inverse_and_log_det(
                z,
                context,
                encodings=batched.encodings,
                mask=None,
                permutations=batched.permutations,
            )
            x_pred = x_pred.reshape(num_samples, num_atoms, -1)

            # Net returns summed logdet; prior.logp returns summed log-prob.
            # Correct density uses -logdet_inv because logdet_inv = log|dx/dz|.
            logq = logp_z - logdet_inv.view(-1)
            return x_pred, -logq

    def proposal_energy(
        self,
        net: torch.nn.Module,
        x: torch.Tensor,
        system_cond: SystemCond | None = None,
    ) -> torch.Tensor:
        """Compute -log q(x) at context [0, 1]."""
        assert system_cond is not None
        batched = system_cond.for_batch(x.shape[0], x.device)
        B = x.shape[0]
        context = torch.zeros(B, 2, device=x.device)
        context[:, 1] = 1.0

        z, logdet = net.forward_and_log_det(
            x,
            context,
            encodings=batched.encodings,
            mask=None,
            permutations=batched.permutations,
        )
        z = z.reshape(x.shape)
        logp_z = self.prior.logp(z)
        # Net returns summed logdet; prior.logp returns summed log-prob.
        logq = logp_z + logdet.view(-1)
        return -logq
