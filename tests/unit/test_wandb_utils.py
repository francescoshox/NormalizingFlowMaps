import pytest
import torch

from transferable_samplers.utils.wandb_utils import compute_mean_metrics


def test_compute_mean_metrics_adds_transferable_length_groups_and_medians() -> None:
    metrics = {
        "val/AL/resampled/effective-sample-size": 0.2,
        "val/EC/resampled/effective-sample-size": torch.tensor(0.4),
        "val/CAVQ/resampled/energy-w2": 5.0,
        "val/EDWY/resampled/energy-w2": 7.0,
        "val/AHKILWHG/resampled/tica-w2": 9.0,
        "val/AMYNLMIR/resampled/tica-w2": 11.0,
    }

    mean_metrics = compute_mean_metrics(metrics, prefix="val")

    assert mean_metrics["val/mean/resampled/effective-sample-size"] == pytest.approx(0.3)
    assert mean_metrics["val/2AA/mean/resampled/effective-sample-size"] == pytest.approx(0.3)
    assert mean_metrics["val/mean/resampled/energy-w2"] == 6.0
    assert mean_metrics["val/4AA/mean/resampled/energy-w2"] == 6.0
    assert mean_metrics["val/mean/resampled/tica-w2"] == 10.0
    assert mean_metrics["val/8AA/mean/resampled/tica-w2"] == 10.0
    assert mean_metrics["val/median/resampled/effective-sample-size"] == pytest.approx(0.3)
    assert mean_metrics["val/2AA/median/resampled/effective-sample-size"] == pytest.approx(0.3)
    assert mean_metrics["val/median/resampled/energy-w2"] == 6.0
    assert mean_metrics["val/4AA/median/resampled/energy-w2"] == 6.0
    assert mean_metrics["val/median/resampled/tica-w2"] == 10.0
    assert mean_metrics["val/8AA/median/resampled/tica-w2"] == 10.0


def test_compute_mean_metrics_skips_non_transferable_sequence_groups() -> None:
    metrics = {
        "val/Ace-A-Nme/proposal/tica-w2": 1.0,
        "val/GYDPETGTWG/proposal/tica-w2": 3.0,
        "test/AA/proposal/tica-w2": 100.0,
    }

    mean_metrics = compute_mean_metrics(metrics, prefix="val")

    assert mean_metrics == {
        "val/mean/proposal/tica-w2": 2.0,
        "val/median/proposal/tica-w2": 2.0,
    }


def test_compute_mean_metrics_does_not_add_median_for_metadata() -> None:
    metrics = {
        "val/AL/resampled/num-eval-samples": 10_000,
        "val/EC/resampled/num-eval-samples": 20_000,
        "val/AL/resampled/median-energy": 1.0,
        "val/EC/resampled/median-energy": 3.0,
    }

    mean_metrics = compute_mean_metrics(metrics, prefix="val")

    assert mean_metrics["val/mean/resampled/num-eval-samples"] == 15_000
    assert mean_metrics["val/mean/resampled/median-energy"] == 2.0
    assert "val/median/resampled/num-eval-samples" not in mean_metrics
    assert "val/median/resampled/median-energy" not in mean_metrics
