from collections.abc import Callable
import jax
import jax.numpy as jnp
import optax
from jax import lax, random
from jax.flatten_util import ravel_pytree
from nfm.state import TrainState


def compute_metrics(values):
    return lax.pmean(values, axis_name="batch")


def nfmd_train_step(
    state: TrainState,
    batch,
    rng_init,
    learning_rate_fn,
    s_max,
    dropout_rate=0.0,
    process_batch_on_device_fn: Callable = None,
):
    """One-time NFMD training step.

    Instead of the MLE input-noise augmentation, we encode a stochastic
    interpolant x_s = (1-s)*x0 + s*eps (s ~ U[0, s_max]) at target time t=1.
    Loss = 0.5*mean(z^2) - logdet.mean() -- same scale/form as the MLE NLL,
    reducing to it exactly at s=0.
    """
    rng_step = random.fold_in(rng_init, state.step)
    rng_device = random.fold_in(rng_step, lax.axis_index(axis_name="batch"))
    rng_s, rng_eps, rng_label, rng_process_img = random.split(rng_device, 4)
    batch = process_batch_on_device_fn(batch, rng_key=rng_process_img)
    x0 = batch["image"]
    x0 = x0.astype(jnp.float32)
    B = x0.shape[0]
    s = jax.random.uniform(rng_s, (B,), minval=0.0, maxval=s_max, dtype=jnp.float32)
    eps = jax.random.normal(rng_eps, x0.shape, dtype=jnp.float32)
    s_b = s.reshape(B, 1, 1, 1)
    x_s = (1 - s_b) * x0 + s_b * eps
    t = jnp.ones((B,), dtype=jnp.float32)
    mask = jax.random.bernoulli(rng_label, dropout_rate, batch["label"].shape)
    label = jnp.where(mask, -jnp.ones_like(batch["label"]), batch["label"])

    def loss_fn(wrt):
        """NFMD loss = negative log-likelihood of the interpolant under f_{s,1}."""
        z, logdet = state.apply_fn({"params": wrt}, x=x_s, y=label, s=s, t=t)
        log_prior = 0.5 * jnp.mean(jnp.square(z), dtype=jnp.float32)
        mean_logdet = jnp.mean(logdet, dtype=jnp.float32)
        loss = -mean_logdet + log_prior
        dict_losses = {"loss": loss, "log_det": mean_logdet, "log_prior": log_prior}
        return (loss, dict_losses)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    aux, grads = grad_fn(state.params)
    dict_losses = aux[1]
    flat_grads, unravel = ravel_pytree(grads)
    grads = unravel(lax.pmean(flat_grads, axis_name="batch"))
    new_state = state.apply_gradients(grads=grads)
    new_ema_params = {}
    for k, ema_param in new_state.ema_params.items():
        ema_value = float(k)
        new_ema = jax.tree.map(
            lambda e, p: ema_value * e + (1 - ema_value) * p,
            ema_param,
            new_state.params,
        )
        new_ema_params[k] = new_ema
    new_state = new_state.replace(ema_params=new_ema_params)
    lr = learning_rate_fn(new_state.step)
    metrics = compute_metrics(dict_losses)
    metrics["lr"] = lr
    metrics["grad_norm"] = optax.tree.norm(grads)
    return (new_state, metrics)
