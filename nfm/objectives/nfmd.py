from collections.abc import Callable
import jax
import jax.numpy as jnp
import optax
from jax import lax, random
from jax.flatten_util import ravel_pytree
from nfm.state import TrainState


def compute_metrics(values):
    return lax.pmean(values, axis_name="batch")


def distributional_loss(apply_fn, params, images, labels, s, noise, t=None,
                        freeze_ft1=False):
    """Per-dimension direct or composed NLL; frozen parameters retain input JVPs."""
    source = (1 - s[:, None, None, None]) * images + s[:, None, None, None] * noise
    one = jnp.ones_like(s)
    if t is None:
        z, logdet = apply_fn({"params": params}, source, labels, s, one)
    else:
        intermediate, first_logdet = apply_fn({"params": params}, source, labels, s, t)
        # The model returns patches, but its encoder accepts images.
        terminal_params = jax.tree.map(lax.stop_gradient, params) if freeze_ft1 else params
        model = apply_fn.__self__
        intermediate = model.unpatchify(intermediate)
        z, terminal_logdet = apply_fn(
            {"params": terminal_params}, intermediate, labels, t, one
        )
        logdet = first_logdet + terminal_logdet
    prior = 0.5 * jnp.mean(jnp.square(z), dtype=jnp.float32)
    determinant = jnp.mean(logdet, dtype=jnp.float32)
    return prior - determinant, {"log_prior": prior, "log_det": determinant}


def nfmd_train_step(
    state: TrainState,
    batch,
    rng_init,
    learning_rate_fn,
    s_max,
    dropout_rate=0.0,
    process_batch_on_device_fn: Callable = None,
    variant="single_time",
    freeze_ft1=True,
):
    """Single-time or full-batch Mix5050 NFMD training step.

    Instead of the MLE input-noise augmentation, we encode a stochastic
    interpolant x_s = (1-s)*x0 + s*eps (s ~ U[0, s_max]) at target time t=1.
    Loss = 0.5*mean(z^2) - logdet.mean() -- same scale/form as the MLE NLL,
    reducing to it exactly at s=0.
    Mix5050 adds an equally weighted composed-map NLL from epoch zero.
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
    mask = jax.random.bernoulli(rng_label, dropout_rate, batch["label"].shape)
    label = jnp.where(mask, -jnp.ones_like(batch["label"]), batch["label"])

    def loss_fn(wrt):
        """Distributional NLL, optionally mixed with the composed-map NLL."""
        direct, dict_losses = distributional_loss(state.apply_fn, wrt, x0, label, s, eps)
        loss = direct
        if variant == "mix5050":
            time_key, noise_key = random.split(random.fold_in(rng_device, 5050))
            times = random.uniform(time_key, (2, B), dtype=jnp.float32)
            low, high = jnp.min(times, axis=0), jnp.max(times, axis=0)
            high = jnp.maximum(high, 1e-5)
            low = jnp.minimum(low, high - 1e-5)
            composed_noise = random.normal(noise_key, x0.shape, dtype=jnp.float32)
            composed, composed_metrics = distributional_loss(
                state.apply_fn, wrt, x0, label, low, composed_noise, high, freeze_ft1
            )
            loss = 0.5 * direct + 0.5 * composed
            dict_losses = {
                key: 0.5 * value + 0.5 * composed_metrics[key]
                for key, value in dict_losses.items()
            }
            dict_losses.update(direct_loss=direct, composed_loss=composed)
        dict_losses["loss"] = loss
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
