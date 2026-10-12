import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nfm.models import nfmd, nfmr
from nfm.state import make_model, initialize


def perturbed(params):
    leaves, tree = jax.tree.flatten(params)
    keys = jax.random.split(jax.random.PRNGKey(123), len(leaves))
    return jax.tree.unflatten(
        tree, [p + 0.003 * jax.random.normal(k, p.shape) for p, k in zip(leaves, keys)]
    )


@pytest.mark.parametrize("method", ["nfmd", "nfmd_gated", "nfmr"])
def test_nontrivial_inverse_and_analytic_logdet(method, tiny_config):
    gated = method == "nfmd_gated"
    method = "nfmd" if gated else method
    config = tiny_config(method)
    if gated:
        config["model"]["gated"] = True
    model = make_model(config)
    variables = {"params": perturbed(initialize(model, method, 42))}
    x = jax.random.normal(jax.random.PRNGKey(2), (1, 4, 4, 3)) * 0.1
    label = jnp.array([1], jnp.int32)
    s, t, one, zero = jnp.array([0.2]), jnp.array([0.8]), jnp.ones(1), jnp.zeros(1)
    args = (s, t) if method == "nfmd" else (s, t, one, zero, one)
    z, logdet = model.apply(variables, x, label, *args)
    jacobian = jax.jacrev(
        lambda value: model.apply(variables, value.reshape(x.shape), label, *args)[
            0
        ].reshape(-1)
    )(x.reshape(-1))
    expected = jnp.linalg.slogdet(jacobian)[1]
    np.testing.assert_allclose(logdet[0] * x.size, expected, atol=2e-4, rtol=1e-4)
    assert float(jnp.linalg.norm(z.reshape(-1) - x.reshape(-1))) > 1e-3
    if method == "nfmd":
        reconstructed = nfmd.reverse(variables, model, z, label, s, t)
    else:
        reconstructed, reverse_logdet = nfmr.reverse(variables, model, z, label, *args)
        np.testing.assert_allclose(logdet + reverse_logdet, 0, atol=2e-5)
    np.testing.assert_allclose(reconstructed, x, atol=3e-5, rtol=2e-4)


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_gated_identity_at_equal_times(precision, tiny_config):
    config = tiny_config(precision=precision)
    config["model"]["gated"] = True
    model = make_model(config)
    variables = {"params": perturbed(initialize(model, "nfmd", 42))}
    x = jax.random.normal(jax.random.PRNGKey(8), (2, 4, 4, 3))
    labels, time = jnp.array([0, -1]), jnp.array([0.15, 0.8])
    z, logdet = model.apply(variables, x, labels, time, time)
    np.testing.assert_array_equal(z, model.patchify(x))
    np.testing.assert_array_equal(logdet, jnp.zeros(2))
    np.testing.assert_array_equal(
        nfmd.reverse(variables, model, z, labels, time, time, guidance=3.2), x
    )


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_nfmr_dual_matches_autodiff(precision, tiny_config):
    config = tiny_config("nfmr", precision=precision)
    model = make_model(config)
    variables = {"params": perturbed(initialize(model, "nfmr", 42))}
    x = jax.random.normal(jax.random.PRNGKey(3), (1, 4, 4, 3)) * 0.1
    dx = jnp.ones_like(x) * 0.05
    label = jnp.array([2], jnp.int32)
    s, t, w, low, high = [
        jnp.array([value], jnp.float32) for value in (0.2, 0.8, 2.0, 0.1, 0.9)
    ]
    one, zero = jnp.ones(1), jnp.zeros(1)
    function = lambda image, time: model.apply(
        variables, image, label, time, t, w, low, high, method=model.delta_st
    )
    expected, tangent = jax.jvp(function, (x, s), (dx, one))
    actual, dual = model.apply(
        variables,
        x,
        dx,
        label,
        s,
        one,
        t,
        zero,
        w,
        zero,
        low,
        zero,
        high,
        zero,
        method=model.delta_st_dual,
    )
    tolerance = 2e-3 if precision == "bf16" else 2e-5
    np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance)
    np.testing.assert_allclose(dual, tangent, atol=tolerance, rtol=tolerance)
    if precision == "fp32":
        epsilon = 0.002
        finite_difference = (
            function(x + epsilon * dx, s + epsilon)
            - function(x - epsilon * dx, s - epsilon)
        ) / (2 * epsilon)
        np.testing.assert_allclose(dual, finite_difference, atol=4e-4, rtol=3e-3)
