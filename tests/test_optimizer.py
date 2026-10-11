"""Check JAX updates against the actual PyTorch paper optimizer primitives."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from nfm.state import adjusted_muon


def test_adjusted_muon_matches_pytorch_routing_and_scaling():
    torch = pytest.importorskip("torch")
    random = np.random.default_rng(3)
    # PyTorch Linear weights are [out,in]; JAX Dense weights are [in,out].
    arrays = {
        "dense": random.normal(size=(8, 32)).astype(np.float32),
        "embedding": random.normal(size=(5, 8)).astype(np.float32),
        "bias": random.normal(size=(8,)).astype(np.float32),
    }
    params = {key: jnp.asarray(value) for key, value in arrays.items()}
    reference = {
        key: torch.nn.Parameter(torch.tensor(value.T if key == "dense" else value))
        for key, value in arrays.items()
    }
    muon = torch.optim.Muon(
        [reference["dense"]],
        lr=4e-4,
        momentum=0.95,
        nesterov=True,
        weight_decay=0.1,
        eps=1e-7,
        ns_steps=5,
    )
    adam = torch.optim.AdamW(
        [
            {"params": [reference["embedding"]], "weight_decay": 1e-4},
            {"params": [reference["bias"]], "weight_decay": 0},
        ],
        lr=4e-4,
        betas=(0.9, 0.95),
    )
    transform = adjusted_muon(params, lambda step: jnp.asarray(4e-4))
    state = transform.init(params)
    for _ in range(3):
        gradients = {
            key: random.normal(size=value.shape).astype(np.float32)
            for key, value in arrays.items()
        }
        before = params
        updates, state = transform.update(
            jax.tree.map(jnp.asarray, gradients), state, params
        )
        params = optax.apply_updates(params, updates)
        for key, value in gradients.items():
            reference[key].grad = torch.tensor(
                value.T.copy() if key == "dense" else value
            )
        muon.step()
        adam.step()
        expected_dense = reference["dense"].detach().numpy().T
        # Torch's NS iterations use BF16; Optax uses FP32. Compare the update
        # magnitude/direction tightly enough to catch wrong width/RMS scaling.
        expected_update = expected_dense - np.asarray(before["dense"])
        difference = np.asarray(params["dense"]) - expected_dense
        assert np.linalg.norm(difference) / np.linalg.norm(expected_update) < 0.12
        for key in ("embedding", "bias"):
            np.testing.assert_allclose(
                params[key], reference[key].detach().numpy(), atol=5e-7, rtol=1e-6
            )
