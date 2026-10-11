from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from lightning import Callback, LightningModule, Trainer

from transferable_samplers.callbacks.eval_metadata import log_test_checkpoint_metadata
from transferable_samplers.evaluation.evaluator import PeptideEnsembleEvaluator
from transferable_samplers.samplers.base_sampler import BaseSampler
from transferable_samplers.utils.dist_utils import broadcast_metrics
from transferable_samplers.utils.results import export_metrics
from transferable_samplers.utils.wandb_utils import compute_mean_metrics


class SamplingEvaluationCallback(Callback):
    """Sample on all ranks, evaluate on rank zero, and export numerical metrics.

    Metrics are broadcast before logging so every rank can use the same
    checkpoint monitors. EMA weight swapping is handled by EMAWeightAveraging.
    """

    def __init__(
        self, evaluator: PeptideEnsembleEvaluator, sampler: BaseSampler | None = None, output_dir: str = ""
    ) -> None:
        super().__init__()
        self.evaluator = evaluator
        self.sampler = sampler
        self.output_dir = output_dir

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Evaluate validation sequences."""
        self.evaluate(trainer, pl_module, "val")

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Evaluate test sequences."""
        log_test_checkpoint_metadata(trainer, pl_module, "test")
        self.evaluate(trainer, pl_module, "test")

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Export after every epoch-end callback has logged its metrics."""
        self._export(trainer, pl_module, "val")

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Export sampling and held-out loss metrics together."""
        self._export(trainer, pl_module, "test")

    def evaluate(self, trainer: Trainer, pl_module: LightningModule, prefix: str) -> None:
        """Compute per-system metrics and dataset summaries."""
        if self.sampler is None:
            return
        datamodule = trainer.datamodule
        sequences = datamodule.val_sequences if prefix == "val" else datamodule.test_sequences
        if trainer.sanity_checking:
            sequences = sequences[:1]
        all_metrics: dict[str, Any] = {}
        for sequence in sequences:
            eval_ctx = datamodule.prepare_eval(sequence=sequence, stage=prefix)
            source_energy = pl_module.build_source_energy(eval_ctx.system_cond)
            samples_dict, _ = self.sampler.sample(source_energy, eval_ctx.target_energy)
            seq_metrics = {}
            if trainer.is_global_zero:
                seq_metrics = self.evaluator.evaluate(samples_dict, eval_ctx, prefix=f"{prefix}/{sequence}")
                seq_metrics = {
                    key: value.detach().cpu().item() if isinstance(value, torch.Tensor) else value
                    for key, value in seq_metrics.items()
                }
            seq_metrics = broadcast_metrics(seq_metrics)
            pl_module.log_dict(seq_metrics)
            all_metrics.update(seq_metrics)
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            del eval_ctx, samples_dict

        summary = compute_mean_metrics(all_metrics, prefix=prefix) if all_metrics else {}
        if summary:
            pl_module.log_dict(summary)

    def _export(self, trainer: Trainer, pl_module: LightningModule, prefix: str) -> None:
        datamodule = trainer.datamodule
        sequences = datamodule.val_sequences if prefix == "val" else datamodule.test_sequences
        if trainer.is_global_zero and self.output_dir and not trainer.sanity_checking:
            metadata = {
                "seed": getattr(pl_module, "_run_seed", None),
                "checkpoint": getattr(pl_module, "_checkpoint_path", None),
                "checkpoint_epoch": getattr(pl_module, "_checkpoint_epoch", trainer.current_epoch),
                "checkpoint_global_step": getattr(pl_module, "_checkpoint_global_step", trainer.global_step),
                "sampler": type(self.sampler).__name__ if self.sampler is not None else None,
                "num_proposals": self.sampler.num_samples if self.sampler is not None else None,
                "sampler_settings": vars(self.sampler) if self.sampler is not None else {},
                "proposal_temperature": float(pl_module.prior.std) ** 2,
                "temperature_kelvin": getattr(datamodule, "temperature", None),
                "world_size": trainer.world_size,
                "num_nodes": trainer.num_nodes,
                "sequences": list(sequences),
            }
            destination = Path(self.output_dir) / prefix / f"epoch_{trainer.current_epoch:04d}"
            metrics = {key: value for key, value in trainer.callback_metrics.items() if key.startswith(prefix + "/")}
            export_metrics(destination, metrics, metadata)
