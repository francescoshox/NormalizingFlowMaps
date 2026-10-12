import jax
import jax.numpy as jnp
import numpy as np

from nfm.models import nfmd
from nfm.objectives.nfmd import distributional_loss
from nfm.state import make_model, initialize


def test_frozen_terminal_parameters_keep_input_and_logdet_gradients():
    class ToyMap:
        def apply(self, variables, images, labels, s, t):
            w = variables["params"]["w"]
            return w * images, w * jnp.mean(images, axis=(1, 2, 3))

        def unpatchify(self, value):
            return value

    model = ToyMap()
    images = jnp.array([0.4, 0.7]).reshape(2, 1, 1, 1)
    s, t, labels = jnp.full(2, 0.2), jnp.full(2, 0.7), jnp.array([0, -1])
    noise = jnp.zeros_like(images)
    source = 0.8 * images
    w = 1.2

    def loss(weight, freeze):
        return distributional_loss(
            model.apply, {"w": weight}, images, labels, s, noise, t, freeze
        )[0]

    frozen = jax.grad(lambda weight: loss(weight, True))(w)
    full = jax.grad(lambda weight: loss(weight, False))(w)
    expected_frozen = w**3 * jnp.mean(source**2) - (1 + w) * jnp.mean(source)
    expected_full = 2 * w**3 * jnp.mean(source**2) - (1 + 2 * w) * jnp.mean(source)
    np.testing.assert_allclose(frozen, expected_frozen, atol=1e-6)
    np.testing.assert_allclose(full, expected_full, atol=1e-6)
    assert abs(float(frozen)) > 1e-3
    np.testing.assert_array_equal(loss(w, True), loss(w, False))


def test_two_step_sampling_uses_separate_conditioning(monkeypatch, tiny_config):
    config = tiny_config()
    config["model"]["gated"] = True
    model = make_model(config)
    variables = {"params": initialize(model, "nfmd", 42)}
    calls = []

    def reverse(params, model, patches, labels, s, t, guidance=0):
        calls.append((np.asarray(labels), np.asarray(s), np.asarray(t), guidance))
        return model.unpatchify(patches)

    monkeypatch.setattr(nfmd, "reverse", reverse)
    image = nfmd.generate(
        variables, model, jax.random.PRNGKey(0), 2, 0, 3.2, image_idx=1,
        split=0.75, first_guidance=0, first_step_unconditional=True,
    )
    assert image.shape == (2, 4, 4, 3)
    np.testing.assert_array_equal(calls[0][0], [-1, -1])
    np.testing.assert_array_equal(calls[1][0], [2, 0])
    np.testing.assert_array_equal(calls[0][1], [0.75, 0.75])
    np.testing.assert_array_equal(calls[0][2], [1, 1])
    np.testing.assert_array_equal(calls[1][1], [0, 0])
    np.testing.assert_array_equal(calls[1][2], [0.75, 0.75])
    assert calls[0][3] == 0 and calls[1][3] == 3.2
