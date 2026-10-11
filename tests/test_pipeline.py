import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from safetensors.numpy import save_file

from nfm import checkpoint
from nfm.data import Dataset, epoch_indices, process_batch
from nfm.evaluate import sample_batches, likelihood
from nfm.state import (
    learning_rate,
    make_model,
    create_state,
    parameter_labels,
)
from nfm.train import run


def assert_tree_equal(left, right):
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right)):
        np.testing.assert_array_equal(a, b)


def data_file(tmp_path, name="pixels", count=4):
    path = tmp_path / f"{name}.safetensors"
    random = np.random.default_rng(2)
    save_file(
        {
            "x1": random.integers(0, 256, (count, 4, 4, 3), dtype=np.uint8),
            "label": np.arange(count, dtype=np.int32) % 3,
            "x0": random.normal(size=(count, 4, 4, 3)).astype(np.float32),
        },
        str(path),
    )
    return path


def test_schedule_and_optimizer_groups():
    schedule = learning_rate(
        {
            "lr": 4e-4,
            "final_lr": 1e-4,
            "warmup_steps": 25000,
            "cooldown_start": 150,
            "cooldown_end": 200,
        },
        5004,
    )
    np.testing.assert_allclose(
        [
            schedule(i)
            for i in [0, 25000, 150 * 5004, 175 * 5004, 200 * 5004, 300 * 5004]
        ],
        [1e-6, 4e-4, 4e-4, 2.5e-4, 1e-4, 1e-4],
        rtol=1e-6,
    )
    labels = parameter_labels(
        {
            "dense": jnp.ones((3, 4)),
            "pos_embedding": jnp.ones((1, 4, 3)),
            "label_embedding": jnp.ones((3, 4)),
            "bias": jnp.ones(4),
        }
    )
    assert labels == {
        "dense": "muon",
        "pos_embedding": "adam_decay",
        "label_embedding": "adam_decay",
        "bias": "adam_no_decay",
    }
    with pytest.raises(ValueError):
        parameter_labels({"unexpected": jnp.zeros((3, 4, 5))})


def test_data_sharding_and_pair_flips(tmp_path):
    data = Dataset(data_file(tmp_path), 4, require_couplings=True, num_classes=3)
    assert data.batch([3, 1])["image"].shape == (2, 4, 4, 3)
    batches = [epoch_indices(41, 32, 42, 3, 8, 2, rank) for rank in range(2)]
    assert len(set(batches[0].ravel()) & set(batches[1].ravel())) == 0
    assert sum(b.size for b in batches) == 32
    np.testing.assert_array_equal(batches[0], epoch_indices(41, 32, 42, 3, 8, 2, 0))
    x = np.tile(np.arange(4)[None, None, :, None], (32, 4, 1, 3)).astype(np.uint8)
    target = jnp.asarray(x, jnp.float32) / 255 * 2 - 1
    processed = process_batch(
        {"image": jnp.asarray(x), "x0": target, "label": jnp.zeros(32)},
        jax.random.PRNGKey(9),
    )
    np.testing.assert_array_equal(processed["image"], processed["x0"])
    assert np.any(np.asarray(processed["image"]) != np.asarray(target))


def test_training_resume_equivalence_and_sampling(tmp_path, tiny_config):
    config = tiny_config()
    dataset = data_file(tmp_path)
    full = run(config, dataset, tmp_path / "full")
    partial = run(config, dataset, tmp_path / "resume", max_steps=1)
    assert not json.loads((partial / "metadata.json").read_text())["complete"]
    resumed = run(config, dataset, tmp_path / "resume")
    model = make_model(config)
    template = create_state(model, config, learning_rate(config["training"], 2))
    full_state = checkpoint.restore(full, template)
    resumed_state = checkpoint.restore(resumed, template)
    assert_tree_equal(full_state, resumed_state)
    for decay, ema in full_state.ema_params.items():
        assert any(
            np.any(np.asarray(a) != np.asarray(b))
            for a, b in zip(jax.tree.leaves(ema), jax.tree.leaves(full_state.params))
        )
    params = checkpoint.load_weights(full, 0.9995)
    samples = list(sample_batches(config, params, 3, 2, 0, 42))
    assert sum(len(x) for x in samples) == 3
    assert samples[0].dtype == np.uint8
    batch = Dataset(dataset, 4).batch([0, 1])
    assert np.isfinite(likelihood(config, params, batch["image"], batch["label"])).all()
    config["training"]["lr"] *= 2
    with pytest.raises(ValueError, match="configuration changed"):
        run(config, dataset, tmp_path / "resume")


def test_nfmr_two_stages_and_fresh_optimizer(tmp_path, tiny_config):
    dataset = data_file(tmp_path)
    fm_config = tiny_config("nfmr")
    source = run(fm_config, dataset, tmp_path / "nfmr/flow_matching")
    params = checkpoint.load_weights(source, 0.999)
    mean_config = tiny_config("nfmr", "mean_flow")
    model = make_model(mean_config)
    initial = create_state(
        model, mean_config, learning_rate(mean_config["training"], 2), params
    )
    assert int(initial.step) == 0
    assert_tree_equal(initial.params, params)
    assert_tree_equal(initial.ema_params[0.999], params)
    first = run(
        mean_config, dataset, tmp_path / "nfmr/mean_flow", init_from=source, max_steps=1
    )
    assert not json.loads((first / "metadata.json").read_text())["complete"]
    # Recovery takes precedence over a missing source checkpoint.
    finished = run(
        mean_config,
        dataset,
        tmp_path / "nfmr/mean_flow",
        init_from=tmp_path / "missing",
    )
    assert json.loads((finished / "metadata.json").read_text())["complete"]
    result = checkpoint.load_weights(finished, "online")
    assert any(
        np.any(np.asarray(a) != np.asarray(b))
        for a, b in zip(jax.tree.leaves(result), jax.tree.leaves(params))
    )
    assert sum(len(x) for x in sample_batches(mean_config, result, 3, 2, 4.5, 42)) == 3


def test_time_limited_run_saves_incomplete_checkpoint(tmp_path, tiny_config):
    config = tiny_config()
    config["training"]["epochs"] = 2
    dataset = data_file(tmp_path)
    with pytest.raises(SystemExit) as exit_info:
        run(config, dataset, tmp_path / "limited", max_seconds=1e-12)
    assert exit_info.value.code == 75
    saved = checkpoint.latest(tmp_path / "limited")
    metadata = json.loads((saved / "metadata.json").read_text())
    assert metadata["step"] == 1 and not metadata["complete"]
