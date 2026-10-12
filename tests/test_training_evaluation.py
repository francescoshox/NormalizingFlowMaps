import json

import jax
import numpy as np
import pytest

from nfm import checkpoint, evaluate
from nfm.state import create_state, learning_rate, make_model
from nfm.train import run
from test_pipeline import data_file, assert_tree_equal


class SmallFeatures:
    weights_digest = "test-feature-weights"
    calls = 0
    fail_on = None

    def __init__(self, weights=None):
        pass

    def __call__(self, pixels):
        type(self).calls += 1
        if type(self).calls == type(self).fail_on:
            raise RuntimeError("simulated interrupted evaluation")
        return np.asarray(pixels, np.float64).mean(axis=(1, 2)) / 255


def test_periodic_fid_resume_preserves_training_and_completed_results(
    tmp_path, tiny_config, monkeypatch
):
    SmallFeatures.calls, SmallFeatures.fail_on = 0, None
    monkeypatch.setattr(evaluate, "FeatureExtractor", SmallFeatures)
    config = tiny_config(precision="bf16")
    config["model"]["gated"] = True
    config["objective"].update(variant="mix5050", s_max=1.0, freeze_ft1=True)
    config["training"].update(epochs=3, checkpoint_epochs=3)
    config["evaluation"].update(
        every_epochs=2, cfg=[3.2], split=0.75, first_cfg=0,
        first_step_unconditional=True,
    )
    dataset = data_file(tmp_path)
    reference = tmp_path / "reference.npz"
    np.savez(reference, mu=np.zeros(3), sigma=np.eye(3))
    full = run(config, dataset, tmp_path / "full", reference=reference)
    directories = sorted((tmp_path / "full/evaluation").glob("epoch_*"))
    assert [p.name for p in directories] == ["epoch_0002", "epoch_0003"]
    best = json.loads((tmp_path / "full/best_fid.json").read_text())
    assert best["sampler"]["split"] == 0.75
    assert best["sampler"]["first_step_unconditional"]
    assert (tmp_path / "full/evaluation/epoch_0002/ema_0.9995_cfg_3.2.json").is_file()

    # Interrupt the second EMA's first batch: the first tuple is durable, and
    # the recovery checkpoint precedes any subsequent training update.
    SmallFeatures.calls, SmallFeatures.fail_on = 0, 3
    with pytest.raises(RuntimeError, match="rerun to resume"):
        run(config, dataset, tmp_path / "resume", reference=reference)
    saved = checkpoint.latest(tmp_path / "resume")
    assert json.loads((saved / "metadata.json").read_text())["step"] == 4
    first_result = tmp_path / "resume/evaluation/epoch_0002/ema_0.9995_cfg_3.2.json"
    first_bytes, first_time = first_result.read_bytes(), first_result.stat().st_mtime_ns
    SmallFeatures.fail_on = None
    resumed = run(config, dataset, tmp_path / "resume", reference=reference)
    # Cached tuples are skipped rather than rewritten on recovery.
    assert first_result.read_bytes() == first_bytes
    assert first_result.stat().st_mtime_ns == first_time
    model = make_model(config)
    template = create_state(model, config, learning_rate(config["training"], 2))
    assert_tree_equal(checkpoint.restore(full, template), checkpoint.restore(resumed, template))
    assert checkpoint.latest(tmp_path / "resume") == resumed
    best = json.loads((tmp_path / "resume/best_fid.json").read_text())
    assert best["status"] == "complete" and np.isfinite(best["fid"])

    # Rerunning a completed training recipe repairs/skips its final evaluation
    # without sampling again or changing the checkpoint.
    before = SmallFeatures.calls
    assert run(config, dataset, tmp_path / "resume", reference=reference) == resumed
    assert SmallFeatures.calls == before
    metrics = [json.loads(line) for line in (tmp_path / "full/metrics.jsonl").read_text().splitlines()]
    assert all(np.isfinite(row["composed_loss"]) for row in metrics)
    assert any(row["grad_norm"] > 0 for row in metrics)
    assert all(np.asarray(p).dtype == np.float32 for p in jax.tree.leaves(template.params))
    np.savez(reference, mu=np.ones(3), sigma=np.eye(3))
    with pytest.raises(RuntimeError, match="rerun to resume"):
        run(config, dataset, tmp_path / "resume", reference=reference)
    assert SmallFeatures.calls == before
