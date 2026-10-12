"""Small explicit configuration; no cluster or account defaults."""

import hashlib
import json
from pathlib import Path

import yaml


def load(path):
    config = yaml.safe_load(Path(path).read_text())
    if config["method"] not in ("nfmd", "nfmr"):
        raise ValueError("method must be nfmd or nfmr")
    if config["stage"] not in ("nfmd", "flow_matching", "mean_flow"):
        raise ValueError("Unknown training stage")
    if (config["method"] == "nfmd") != (config["stage"] == "nfmd"):
        raise ValueError("Stage does not belong to the specified method")
    train = config["training"]
    if train["batch_size"] < 1 or train["epochs"] < 1:
        raise ValueError("Positive batch size and epoch count required")
    if not all(0 < value < 1 for value in train["emas"]):
        raise ValueError("EMA decays must be between zero and one")
    evaluation = config["evaluation"]
    if evaluation.get("every_epochs", 0) < 0:
        raise ValueError("Evaluation interval must be nonnegative; zero disables periodic FID")
    if train["checkpoint_epochs"] < 1 or train["log_steps"] < 1:
        raise ValueError("Checkpoint and logging intervals must be positive")
    if config["method"] == "nfmd":
        objective = config["objective"]
        variant = objective.get("variant", "single_time")
        if variant not in ("single_time", "mix5050"):
            raise ValueError("Unknown NFMD objective variant")
        if variant == "mix5050" and (
            not config["model"].get("gated", False) or objective["s_max"] != 1.0
        ):
            raise ValueError("Mix5050 requires gated maps and uniform times over [0,1]")
        if evaluation.get("split") is not None:
            if not 0 < evaluation["split"] < 1:
                raise ValueError("Two-step split must lie strictly between zero and one")
            if evaluation.get("first_step_unconditional", False) and evaluation.get("first_cfg", 0) != 0:
                raise ValueError("Unconditional first step requires first_cfg=0")
    return config


def digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
