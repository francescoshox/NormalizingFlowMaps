from __future__ import annotations

from lightning import LightningModule, Trainer


def log_test_checkpoint_metadata(trainer: Trainer, pl_module: LightningModule, prefix: str) -> None:
    """Log test-stage checkpoint metadata, falling back to trainer epoch."""
    checkpoint_epoch = getattr(pl_module, "_checkpoint_epoch", None)
    epoch = checkpoint_epoch if checkpoint_epoch is not None else trainer.current_epoch

    metrics = {f"{prefix}/epoch": float(epoch)}
    if checkpoint_epoch is not None:
        metrics[f"{prefix}/checkpoint_epoch"] = float(checkpoint_epoch)

    checkpoint_global_step = getattr(pl_module, "_checkpoint_global_step", None)
    if checkpoint_global_step is not None:
        metrics[f"{prefix}/checkpoint_global_step"] = float(checkpoint_global_step)

    pl_module.log_dict(metrics)
