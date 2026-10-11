from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from lightning import Callback, LightningModule, Trainer

from transferable_samplers.callbacks.ema_weight_averaging import EMAWeightAveraging
from transferable_samplers.callbacks.eval_metadata import log_test_checkpoint_metadata
from transferable_samplers.utils.dist_utils import broadcast_metrics, get_rank, get_world_size
from transferable_samplers.utils.pylogger import RankedLogger
from transferable_samplers.utils.standardization import standardize_coords
from transferable_samplers.utils.wandb_utils import compute_mean_metrics

if TYPE_CHECKING:
    from lightning.pytorch.callbacks import WeightAveraging

logger = RankedLogger(__name__, rank_zero_only=False)


class LossEvaluationCallback(Callback):
    """Evaluate model loss on held-out true samples during validation/test.

    Uses the model's ``compute_primary_loss`` without system-size normalization,
    unless the model exposes ``compute_eval_losses`` for a richer metric bundle.

    DDP-safe: all ranks compute loss on a shard of the data, then all_reduce to
    get the global mean. All ranks log metrics so that Lightning's
    ``ModelCheckpoint`` monitors can find them in ``trainer.callback_metrics``.

    Args:
        batch_size: Number of samples per forward pass.
        max_samples: Cap on the number of true samples to evaluate. None = use all.
        run_on_train_start: Also evaluate validation loss before the first
            training epoch, without triggering sampling-based validation.
    """

    def __init__(
        self,
        batch_size: int = 256,
        max_samples: int | None = None,
        run_on_train_start: bool = False,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.max_samples = max_samples
        self.run_on_train_start = run_on_train_start

    # -----------------------------------------------------------------
    # EMA detection
    # -----------------------------------------------------------------

    @staticmethod
    def _find_ema_callback(trainer: Trainer) -> WeightAveraging | None:
        """Return the ``EMAWeightAveraging`` callback if present, else ``None``."""
        for cb in trainer.callbacks:
            if isinstance(cb, EMAWeightAveraging):
                return cb
        return None

    # -----------------------------------------------------------------
    # Hooks
    # -----------------------------------------------------------------

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Optionally log an initial validation loss before any updates."""
        if not self.run_on_train_start:
            return

        logger.info("Running initial validation loss evaluation at train start")
        self._evaluate(trainer, pl_module, "val")

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Evaluate loss on validation data.

        When EMA is active, ``pl_module`` already holds EMA weights (swapped in
        by ``WeightAveraging.on_validation_epoch_start``).  We first evaluate
        with those (the default ``val/`` keys), then temporarily swap training
        weights back in and evaluate again under a ``noema/`` sub-prefix.
        """
        # 1) Normal eval — uses whatever weights are in pl_module (EMA during
        #    validation, training weights during test).
        self._evaluate(trainer, pl_module, "val")

        # 2) If EMA is active, also eval with the non-averaged training weights.
        ema_cb = self._find_ema_callback(trainer)
        if ema_cb is not None and ema_cb._average_model is not None:
            ema_cb._swap_models(pl_module)  # training weights IN
            try:
                self._evaluate(trainer, pl_module, "val", metric_sub_prefix="noema")
            finally:
                ema_cb._swap_models(pl_module)  # EMA weights back IN

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Evaluate loss on test data."""
        prefix = getattr(pl_module, "_test_prefix", "test")
        log_test_checkpoint_metadata(trainer, pl_module, prefix)
        self._evaluate(trainer, pl_module, prefix)

    def _evaluate(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        prefix: str,
        metric_sub_prefix: str | None = None,
    ) -> None:
        datamodule = trainer.datamodule
        # The prefix may be a custom test prefix (e.g. "test_energy_w2");
        # map it to the canonical stage ("val" or "test") for data selection.
        stage = "val" if prefix == "val" else "test"
        eval_sequences = datamodule.val_sequences if stage == "val" else datamodule.test_sequences

        world_size = get_world_size()
        rank = get_rank()

        all_metrics: dict[str, Any] = {}
        for sequence in eval_sequences:
            eval_ctx = datamodule.prepare_eval(sequence=sequence, stage=stage)

            samples = eval_ctx.true_data.samples
            if self.max_samples is not None:
                samples = samples[: self.max_samples]
            x = standardize_coords(samples, eval_ctx.normalization_std).to(
                device=pl_module.device, dtype=pl_module.dtype
            )

            # Shard samples across ranks (drop remainder for even split)
            n = (x.shape[0] // world_size) * world_size
            x = x[:n]
            chunk_size = n // world_size
            x_local = x[rank * chunk_size : (rank + 1) * chunk_size]

            system_cond = eval_ctx.system_cond

            # Compute losses on local shard. Models may optionally expose a
            # full validation metric bundle; otherwise we log the primary loss.
            metric_losses: dict[str, list[torch.Tensor]] = {}
            compute_eval_losses = getattr(pl_module, "compute_eval_losses", None)
            with torch.no_grad():
                for i in range(0, x_local.shape[0], self.batch_size):
                    x_batch = x_local[i : i + self.batch_size]
                    batch = self._build_batch(x_batch, system_cond, pl_module.device)
                    if callable(compute_eval_losses):
                        batch_metrics = compute_eval_losses(batch)
                        if "eval-loss" not in batch_metrics:
                            batch_metrics = {
                                **batch_metrics,
                                "eval-loss": pl_module.compute_primary_loss(batch),
                            }
                    else:
                        batch_metrics = {"eval-loss": pl_module.compute_primary_loss(batch)}

                    for name, value in batch_metrics.items():
                        if not isinstance(value, torch.Tensor):
                            value = torch.tensor(value, device=pl_module.device)
                        metric_losses.setdefault(name, []).append(value.mean().detach())

            local_metrics = {name: torch.stack(values).mean() for name, values in metric_losses.items()}

            # All-reduce to get global mean (each rank has equal-sized shard)
            for metric_name, local_mean in local_metrics.items():
                if world_size > 1:
                    torch.distributed.all_reduce(local_mean, op=torch.distributed.ReduceOp.AVG)

                if metric_sub_prefix:
                    key = f"{prefix}/{sequence}/{metric_sub_prefix}/{metric_name}"
                else:
                    key = f"{prefix}/{sequence}/{metric_name}"
                value = local_mean.item()
                all_metrics[key] = value
                pl_module.log_dict({key: value})
                if trainer.is_global_zero:
                    logger.info(f"{key}: {value:.6f}")

            # Barrier so all ranks are aligned before next sequence
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

            del eval_ctx

        mean_metrics = compute_mean_metrics(all_metrics, prefix=prefix) if all_metrics else {}
        mean_metrics = broadcast_metrics(mean_metrics)
        if mean_metrics:
            pl_module.log_dict(mean_metrics)

        # Barrier so all ranks are aligned before returning
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    @staticmethod
    def _build_batch(
        x: torch.Tensor,
        system_cond: Any | None,
        device: torch.device,
    ) -> dict[str, Any]:
        """Build a batch dict from samples and optional system conditioning."""
        batch: dict[str, Any] = {"x": x}
        if system_cond is not None:
            batched = system_cond.for_batch(x.shape[0], device)
            if batched.encodings is not None:
                batch["encodings"] = batched.encodings
            if batched.permutations is not None:
                batch["permutations"] = batched.permutations
        return batch
