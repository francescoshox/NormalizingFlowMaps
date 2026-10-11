from typing import Any

from dotenv import load_dotenv

load_dotenv(override=False)

import hydra
import lightning
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from transferable_samplers.utils.hydra_utils import extras, get_metric_value, task_wrapper
from transferable_samplers.utils.checkpoint_utils import augment_state_dict_for_teacher, resolve_init
from transferable_samplers.utils.instantiators import instantiate_callbacks, instantiate_loggers
from transferable_samplers.utils.pylogger import RankedLogger
from transferable_samplers.utils.wandb_utils import log_hyperparameters

# We had issues with invertbility of TarFlow without the following settings.
# Didn't notice any walltime difference for TarFlow or ECNF, but is worth
# benchmarking for any further implemented models.
torch.set_float32_matmul_precision("highest")  # must be at least high
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
logger = RankedLogger(__name__, rank_zero_only=False)
logger.info(
    "Some numerical settings applied for TarFlow invertibility. No slowdown was "
    "observed for ECNF but other neural networks may be slower than expected."
)


@task_wrapper
def train(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Train one continuous run with an explicit Hydra seed."""
    if cfg.get("seed") is None:
        raise ValueError("Set an integer seed in the Hydra configuration.")
    lightning.seed_everything(cfg.seed, workers=True)

    if cfg.get("torch_num_threads"):
        torch.set_num_threads(cfg.torch_num_threads)

    logger.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    logger.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    model._run_seed = int(cfg.seed)

    logger.info("Instantiating callbacks...")
    callbacks: list[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    logger.info("Instantiating loggers...")
    loggers: list[Logger] = instantiate_loggers(cfg.get("logger"))

    logger.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=loggers)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": loggers,
        "trainer": trainer,
    }

    if loggers:
        logger.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    init_state_dict = resolve_init(
        init_ckpt_path=cfg.get("init_ckpt_path"),
        init_hf_state_dict_path=cfg.get("init_hf_state_dict_path"),
        scratch_dir=cfg.paths.scratch_dir,
    )

    if init_state_dict is not None:
        # If the model has a teacher (i.e. teacher regularization), we need to augment the state dict before loading.
        if hasattr(model, "teacher"):
            init_state_dict = augment_state_dict_for_teacher(init_state_dict)
        logger.info("Loading model state dict from init_state_dict")
        model.load_state_dict(init_state_dict)

    logger.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule)

    return dict(trainer.callback_metrics), object_dict


@hydra.main(version_base="1.3", config_path="../../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> float | None:
    """Main entry point for training.

    Args:
        cfg: DictConfig configuration composed by Hydra.

    Returns:
        Optimized metric value, if configured.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(metric_dict=metric_dict, metric_name=cfg.get("optimized_metric"))

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
