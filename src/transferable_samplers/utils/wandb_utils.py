from __future__ import annotations

import statistics as stats
from collections import defaultdict
from collections.abc import Iterator
from typing import Any

import torch
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import OmegaConf

from transferable_samplers.utils.pylogger import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=False)

SUMMARY_METRIC_SUFFIXES = {
    "energy-w2",
    "torus-w2",
    "tica-w2",
    "effective-sample-size",
    "tica-k-jsd",
    "torus-k-jsd",
}


def _sequence_length_group(sequence: str) -> str | None:
    """Return a transferable peptide length group for plain AA sequences."""
    if not sequence.isalpha() or not sequence.isupper():
        return None

    length = len(sequence)
    if length not in {2, 4, 8}:
        return None

    return f"{length}AA"


def _metric_is_summary_candidate(metric_name: str) -> bool:
    """Return whether a metric should get median summaries."""
    return metric_name.rsplit("/", maxsplit=1)[-1] in SUMMARY_METRIC_SUFFIXES


def _iter_sequence_metrics(metrics: dict[str, Any], prefix: str) -> Iterator[tuple[str, str, float]]:
    """Yield numeric per-sequence metrics matching ``prefix``."""
    for key, value in metrics.items():
        parts = key.split("/")
        if len(parts) < 3 or parts[0] != prefix:
            continue

        sequence = parts[1]
        metric_name = "/".join(parts[2:])

        if isinstance(value, torch.Tensor):
            value = value.item()
        elif isinstance(value, int | float):
            value = float(value)
        else:
            continue

        yield sequence, metric_name, value


def compute_mean_metrics(metrics: dict[str, Any], prefix: str = "val") -> dict[str, float]:
    """Aggregate per-sequence metrics by computing global and length summaries.

    Args:
        metrics: Dict of metric values keyed by ``{prefix}/{sequence}/{metric_name}``.
        prefix: Metric key prefix to filter on (e.g. ``"val"``).

    Returns:
        Dict keyed by ``{prefix}/mean/{metric_name}`` with mean values.
        Transferable peptide metrics also get grouped means keyed by
        ``{prefix}/{length}AA/mean/{metric_name}``.
        Selected scalar evaluation metrics also get median summaries keyed by
        ``{prefix}/median/{metric_name}`` and
        ``{prefix}/{length}AA/median/{metric_name}``.
    """
    mean_dict_list: defaultdict[str, list[float]] = defaultdict(list)
    median_dict_list: defaultdict[str, list[float]] = defaultdict(list)

    for sequence, metric_name, value in _iter_sequence_metrics(metrics, prefix):
        mean_dict_list[f"{prefix}/mean/{metric_name}"].append(value)
        if _metric_is_summary_candidate(metric_name):
            median_dict_list[f"{prefix}/median/{metric_name}"].append(value)

        if group := _sequence_length_group(sequence):
            mean_dict_list[f"{prefix}/{group}/mean/{metric_name}"].append(value)
            if _metric_is_summary_candidate(metric_name):
                median_dict_list[f"{prefix}/{group}/median/{metric_name}"].append(value)

    return {
        **{k: stats.mean(v) for k, v in mean_dict_list.items()},
        **{k: stats.median(v) for k, v in median_dict_list.items()},
    }


@rank_zero_only
def log_hyperparameters(object_dict: dict[str, Any], resolve: bool = True) -> None:
    """Control which config parts are saved by Lightning loggers.

    Additionally saves the number of model parameters (total, trainable,
    non-trainable).

    Args:
        object_dict: A dictionary containing the following objects:
            - ``"cfg"``: A DictConfig object containing the main config.
            - ``"model"``: The Lightning model.
            - ``"trainer"``: The Lightning trainer.
        resolve: Whether to resolve OmegaConf references when converting the config.
    """
    hparams: dict[str, Any] = {}
    cfg: dict[str, Any] = OmegaConf.to_container(object_dict["cfg"], resolve=resolve)  # type: ignore[assignment]
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        logger.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["model"] = cfg["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    hparams["model/params/non_trainable"] = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    hparams["data"] = cfg["data"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")
    hparams["paths"] = cfg.get("paths")
    hparams["resolved_config"] = cfg

    # send hparams to all loggers
    for lg in trainer.loggers:
        lg.log_hyperparams(hparams)
