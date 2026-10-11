# NFMD training module for two-time iTarFlow.
#
# Trains a bijective flow to minimize the forward KL divergence using
# conditional interpolants.  Only the "one-time" NFMD variant is implemented:
#   sample s in [0, nfmd_t_max], fix t = 1, build x_s = (1-s)*x_0 + s*eps,
#   encode via f_{s,1}, and minimise  0.5/d ||z||^2 - 1/d logdet.
#

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import torch

from transferable_samplers.models.base_lightning_module import BaseLightningModule
from transferable_samplers.models.priors.prior import Prior
from transferable_samplers.utils.dataclasses import SourceEnergyConfig, SystemCond
from transferable_samplers.utils.pylogger import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=False)


class NFMDModule(BaseLightningModule):
    """NFMD training for a two-time normalizing flow.

    The network must implement ``forward_and_log_det(x, context)`` where
    ``x`` is ``[B, dim]`` flat coordinates and ``context`` is ``[B, 2]``
    tensor of ``[s, t]`` time pairs.

    Training uses the one-time NFMD objective:
        1. Sample source time ``s ~ U[0, nfmd_t_max]``.
        2. Build interpolant ``x_s = (1-s) * x_0 + s * eps``.
        3. Encode ``z, logdet = net.forward_and_log_det(x_s, [s, 1])``.
        4. Loss = ``0.5 * mean(z^2) - mean(logdet)`` (per-dim, mean over batch).

    Generation uses ``inverse_and_log_det`` with context ``[0, 1]``.

    Args:
        nfmd_t_max: Upper bound for sampling s.  Default 1.0 (full range).
        objective: Training objective to use: ``nfmd``, ``mle``, or
            ``nfmd_plus_mle``.  The default ``nfmd`` preserves existing behavior.
        mle_loss: Backward-compatible flag.  If True with ``objective=nfmd``,
            also compute MLE loss on data (context ``[0, 1]``).
        mle_loss_weight: Weight for the MLE loss term.
        eval_logdet_sum: If ``True``, switch to sum-reduced log-determinants
            during evaluation (for correct log-probabilities).  Training uses
            whatever ``logdet_sum`` the network was configured with (typically
            ``False`` = per-dim mean for gradient stability).  ``None`` disables
            the override.
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
        eval_logdet_sum: bool | None = None,
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
            raise ValueError(f"Invalid NFMDModule objective {objective!r}; expected one of {sorted(valid_objectives)}")

        self.objective = objective
        self.nfmd_t_max = nfmd_t_max
        self.mle_loss = mle_loss
        self.mle_loss_weight = mle_loss_weight
        self.eval_logdet_sum = eval_logdet_sum
        self.nfmd_eval_num_s = nfmd_eval_num_s

    # -----------------------------------------------------------------
    # Logdet-sum override for evaluation
    # -----------------------------------------------------------------

    @contextmanager
    def _logdet_sum_override(self, net: torch.nn.Module | None = None):
        """Temporarily switch ``logdet_sum`` on the network during evaluation.

        During training, ``logdet_sum=False`` (per-dim mean) is used for
        gradient stability.  During evaluation, ``logdet_sum=True`` (sum)
        gives mathematically correct log-probabilities.

        This patches ``self.net.logdet_sum`` and each ``block.logdet_sum``,
        restoring the originals on exit.  If ``self.eval_logdet_sum`` is
        ``None`` or already matches the network, this is a no-op.
        """
        override = self.eval_logdet_sum
        net = self.net if net is None else net
        if override is not None and hasattr(net, "logdet_sum") and net.logdet_sum != override:
            saved_model = net.logdet_sum
            blocks = getattr(net, "blocks", [])
            saved_blocks = [(b, b.logdet_sum) for b in blocks if hasattr(b, "logdet_sum")]
            net.logdet_sum = override
            for block, _ in saved_blocks:
                block.logdet_sum = override
            try:
                yield
            finally:
                net.logdet_sum = saved_model
                for block, orig in saved_blocks:
                    block.logdet_sum = orig
        else:
            yield

    # -----------------------------------------------------------------
    # NFMD loss (one-time variant)
    # -----------------------------------------------------------------

    def _compute_nfmd_loss(self, x_data: torch.Tensor) -> torch.Tensor:
        """Compute the one-time forward KL loss via conditional interpolant.

        Args:
            x_data: ``[B, dim]`` flat molecule coordinates (training data).

        Returns:
            Scalar NFMD loss (mean over batch).
        """
        B = x_data.shape[0]
        d = x_data.shape[1]  # total coordinate dimension

        # Sample s in [0, nfmd_t_max] uniformly.
        s = torch.rand(B, 1, device=x_data.device) * self.nfmd_t_max

        # Noisy interpolant: x_s = (1 - s) * x_0 + s * eps.
        eps = torch.randn_like(x_data)
        x_s = (1.0 - s) * x_data + s * eps

        # Encode x_s -> z via f_{s,1} with context [s, 1].
        t_ones = torch.ones(B, 1, device=x_data.device)
        context = torch.cat([s, t_ones], dim=-1)  # [B, 2]
        z, log_det = self.net.forward_and_log_det(x_s, context)

        # Per-dimension NLL: 0.5/d ||z||^2 - 1/d logdet.
        loss = 0.5 * z.pow(2).mean(dim=-1) - self._logdet_per_dim(log_det, d)
        return loss.mean()

    # -----------------------------------------------------------------
    # MLE loss (optional, via context [0, 1])
    # -----------------------------------------------------------------

    def _compute_mle_loss(self, x_data: torch.Tensor) -> torch.Tensor:
        """Compute MLE (NLL) loss using context [0, 1] (full data -> noise).

        Args:
            x_data: ``[B, dim]`` flat molecule coordinates.

        Returns:
            Scalar MLE loss (mean over batch, per-dim).
        """
        return self._compute_mle_loss_per_sample(x_data).mean()

    def _compute_mle_loss_per_sample(self, x_data: torch.Tensor) -> torch.Tensor:
        """Compute per-sample endpoint MLE loss on the training scale."""
        B = x_data.shape[0]
        context = torch.zeros(B, 2, device=x_data.device, dtype=x_data.dtype)
        context[:, 1] = 1.0  # [s=0, t=1]

        z, log_det = self.net.forward_and_log_det(x_data, context)

        # Per-dim NLL: 0.5/d ||z||^2 - 1/d logdet
        return 0.5 * z.pow(2).mean(dim=-1) - self._logdet_per_dim(log_det, x_data.shape[1])

    def _logdet_per_dim(self, log_det: torch.Tensor, dim: int) -> torch.Tensor:
        """Return logdet on the per-dimension scale used by the NFMD objective."""
        if getattr(self.net, "logdet_sum", False):
            return log_det.view(-1) / dim
        return log_det.view(-1)

    def _logdet_sum(self, log_det: torch.Tensor, dim: int) -> torch.Tensor:
        """Return logdet on the summed scale used by proper log-probabilities."""
        if getattr(self.net, "logdet_sum", False):
            return log_det.view(-1)
        return log_det.view(-1) * dim

    def _nfmd_eval_s_grid(self, device: torch.device) -> torch.Tensor:
        """Grid of source times used for validation NFMD diagnostics."""
        return torch.linspace(0.0, self.nfmd_t_max, self.nfmd_eval_num_s, device=device)

    def _compute_nfmd_loss_at_s(self, x_data: torch.Tensor, s_value: torch.Tensor) -> torch.Tensor:
        """Compute per-sample one-time NFMD loss at a fixed source time."""
        B, dim = x_data.shape
        s = torch.full((B, 1), float(s_value.item()), device=x_data.device, dtype=x_data.dtype)
        eps = torch.randn_like(x_data)
        x_s = (1.0 - s) * x_data + s * eps

        t_ones = torch.ones(B, 1, device=x_data.device, dtype=x_data.dtype)
        context = torch.cat([s, t_ones], dim=-1)
        z, log_det = self.net.forward_and_log_det(x_s, context)

        return 0.5 * z.pow(2).mean(dim=-1) - self._logdet_per_dim(log_det, dim)

    def _compute_nfmd_eval_loss_grid(self, x_data: torch.Tensor) -> torch.Tensor:
        """Compute per-sample grid-averaged validation NFMD loss."""
        losses = [self._compute_nfmd_loss_at_s(x_data, s_value) for s_value in self._nfmd_eval_s_grid(x_data.device)]
        return torch.stack(losses, dim=0).mean(dim=0)

    def _compute_nfmd_eval_loss_stochastic(self, x_data: torch.Tensor) -> torch.Tensor:
        """Compute per-sample stochastic NFMD loss (same as training objective).

        Samples ``s ~ U[0, nfmd_t_max]`` independently per sample, matching
        the training loss exactly.  Returns per-sample losses (no reduction).
        """
        B, dim = x_data.shape
        s = torch.rand(B, 1, device=x_data.device, dtype=x_data.dtype) * self.nfmd_t_max
        eps = torch.randn_like(x_data)
        x_s = (1.0 - s) * x_data + s * eps

        t_ones = torch.ones(B, 1, device=x_data.device, dtype=x_data.dtype)
        context = torch.cat([s, t_ones], dim=-1)
        z, log_det = self.net.forward_and_log_det(x_s, context)

        return 0.5 * z.pow(2).mean(dim=-1) - self._logdet_per_dim(log_det, dim)

    def _compute_nll_loss_per_sample(self, x_data: torch.Tensor) -> torch.Tensor:
        """Compute proper per-sample NLL at context [0, 1]."""
        B, dim = x_data.shape
        context = torch.zeros(B, 2, device=x_data.device, dtype=x_data.dtype)
        context[:, 1] = 1.0  # [s=0, t=1]

        z, log_det = self.net.forward_and_log_det(x_data, context)
        logp_z = -0.5 * z.pow(2).sum(dim=-1) - 0.5 * dim * torch.log(
            torch.tensor(2 * torch.pi, device=z.device, dtype=z.dtype)
        )
        logq = logp_z + self._logdet_sum(log_det, dim)
        return -logq

    @staticmethod
    def _flatten_batch_x(batch: dict[str, Any]) -> torch.Tensor:
        """Return flat coordinates from a molecular point-cloud batch."""
        x = batch["x"]
        assert len(x.shape) == 3, "molecules must be a pointcloud (batch_size, num_atoms, 3)"
        B, num_atoms, spatial = x.shape
        return x.reshape(B, -1)

    # -----------------------------------------------------------------
    # BaseLightningModule interface
    # -----------------------------------------------------------------

    def compute_primary_loss(self, batch: dict[str, Any]) -> torch.Tensor:
        """Compute per-sample primary loss (used by LossEvaluationCallback).

        NFMD objectives use random ``s ~ U[0, nfmd_t_max]`` sampling, identical
        to the training objective.  MLE-only uses endpoint NLL at context
        ``[0, 1]`` on the same per-dimension scale as training.

        Validation may run with ``eval_logdet_sum=True`` for proper proposal
        energies.  The NFMD loss is explicitly rescaled back to per-dimension
        logdet units so it matches the training objective.
        """
        with self._logdet_sum_override():
            x_flat = self._flatten_batch_x(batch)
            if self.objective == "mle":
                return self._compute_mle_loss_per_sample(x_flat)
            return self._compute_nfmd_eval_loss_stochastic(x_flat)

    def compute_eval_losses(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Compute additional validation losses for NFMD runs.

        Returns:
            ``eval-loss``: Stochastic NFMD loss (random ``s``, same as training).
            ``eval-nll``: Proper endpoint NLL at context [0, 1].
            ``eval-loss-grid``: Grid-averaged per-dimension NFMD loss.
            ``eval-s{i}``: Per-dimension NFMD loss at fixed grid point i.
        """
        with self._logdet_sum_override():
            x_flat = self._flatten_batch_x(batch)
            metrics: dict[str, torch.Tensor] = {}

            # Proper NLL at the endpoint (s=0, t=1).
            metrics["eval-nll"] = self._compute_nll_loss_per_sample(x_flat)

            if self.objective == "mle":
                metrics["eval-loss"] = self._compute_mle_loss_per_sample(x_flat)
                return metrics

            # Stochastic NFMD loss (matches training objective exactly).
            metrics["eval-loss"] = self._compute_nfmd_eval_loss_stochastic(x_flat)

            # Per-grid-point and grid-averaged NFMD losses.
            s_losses = []
            for idx, s_value in enumerate(self._nfmd_eval_s_grid(x_flat.device), start=1):
                s_loss = self._compute_nfmd_loss_at_s(x_flat, s_value)
                s_losses.append(s_loss)
                metrics[f"eval-s{idx}"] = s_loss
            metrics["eval-loss-grid"] = torch.stack(s_losses, dim=0).mean(dim=0)

            return metrics

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Compute training loss for the configured objective."""
        x = batch["x"]
        assert len(x.shape) == 3, "molecules must be a pointcloud (batch_size, num_atoms, 3)"
        B, num_atoms, spatial = x.shape
        x_flat = x.reshape(B, -1)  # [B, dim]

        loss = x_flat.new_tensor(0.0)
        use_nfmd = self.objective in {"nfmd", "nfmd_plus_mle"}
        use_mle = self.objective in {"mle", "nfmd_plus_mle"} or self.mle_loss

        if use_nfmd:
            nfmd_loss = self._compute_nfmd_loss(x_flat)
            self.log("train/nfmd_loss", nfmd_loss.item(), prog_bar=True, sync_dist=True)
            loss = loss + nfmd_loss

        if use_mle:
            mle_loss = self._compute_mle_loss(x_flat)
            self.log("train/mle_loss", mle_loss.item(), prog_bar=True, sync_dist=True)
            if self.objective == "mle":
                loss = loss + mle_loss
            else:
                loss = loss + self.mle_loss_weight * mle_loss

        batch_value = self.train_metrics(loss)
        self.log_dict(batch_value, prog_bar=True)
        return loss

    def generate_proposal(
        self,
        net: torch.nn.Module,
        num_samples: int,
        system_cond: SystemCond | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate samples via inverse_and_log_det with context [0, 1].

        Noise -> data using the generative direction.
        Uses ``eval_logdet_sum`` override for correct log-probabilities.
        """
        with torch.no_grad(), self._logdet_sum_override(net):
            if system_cond and system_cond.encodings:
                num_atoms = system_cond.encodings["atom_type"].size(0)
            else:
                # pyrefly: ignore [missing-attribute]
                num_atoms = self.trainer.datamodule.num_atoms

            z = self.prior.sample(num_samples, num_atoms, device=self.device)
            logp_z = self.prior.logp(z)

            # Flatten for the network
            z_flat = z.reshape(num_samples, -1)  # [B, dim]

            # Context [0, 1] = full noise -> data map
            context = torch.zeros(num_samples, 2, device=self.device)
            context[:, 1] = 1.0

            x_flat, logdet_inv = net.inverse_and_log_det(z_flat, context)

            # Reshape back to [B, num_atoms, 3]
            x_pred = x_flat.reshape(num_samples, num_atoms, -1)

            # logq(x) = log p_Z(z) - log|det df^-1/dz|
            logq = logp_z - logdet_inv

            return x_pred, -logq  # (samples, proposal_energy = -logq)

    def proposal_energy(
        self,
        net: torch.nn.Module,
        x: torch.Tensor,
        system_cond: SystemCond | None = None,
    ) -> torch.Tensor:
        """Compute proposal energy (-log q) via forward_and_log_det at [0, 1].

        Uses ``eval_logdet_sum`` override for correct log-probabilities.
        """
        with self._logdet_sum_override(net):
            B = x.shape[0]
            x_flat = x.reshape(B, -1)

            context = torch.zeros(B, 2, device=x.device)
            context[:, 1] = 1.0

            z, logdet = net.forward_and_log_det(x_flat, context)

            logp_z = self.prior.logp(z.reshape(x.shape))
            logq = logp_z + logdet
            return -logq
