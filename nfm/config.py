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
    return config


def digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
