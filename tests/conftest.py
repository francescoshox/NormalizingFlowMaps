import copy
from pathlib import Path

import pytest

from nfm.config import load


@pytest.fixture
def tiny_config():
    def make(method="nfmd", stage=None, precision="fp32"):
        name = "cifar10_nfmd" if method == "nfmd" else "imagenet64_nfmr_flow_matching"
        config = copy.deepcopy(
            load(Path(__file__).parents[1] / "configs" / f"{name}.yaml")
        )
        config["model"].update(
            img_size=4,
            channels=64,
            num_blocks=2,
            num_layers=1,
            num_classes=3,
            precision=precision,
        )
        config["dataset"].update(epoch_samples=0, flip=True)
        config["training"].update(
            batch_size=2, epochs=1, warmup_steps=0, checkpoint_epochs=1, log_steps=1
        )
        config["evaluation"].update(samples=4, batch_size=2, every_epochs=0)
        if stage == "mean_flow":
            mean = load(
                Path(__file__).parents[1] / "configs/imagenet64_nfmr_mean_flow.yaml"
            )
            config["objective"] = mean["objective"]
            config["objective"].update(use_lpips=False, use_convnext=False)
            config["stage"] = "mean_flow"
            config["handoff_ema"] = 0.999
            config["training"]["lr"] = 1e-4
        return config

    return make
