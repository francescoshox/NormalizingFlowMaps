import jax
import jax.numpy as jnp
import optax
from jax import lax, random
from jax.flatten_util import ravel_pytree


def compute_metrics(values):
    return lax.pmean(values, axis_name="batch")


from nfm.models.nfmr import (
    NFMR,
    _expand_time,
    logit_normal_sample,
    sample_pixel_mean_flow_time,
    sample_pmf_cfg_interval,
    sample_pmf_cfg_scale,
    mf_adaptive_reweight,
)


def x1_noise_magnitude(
    step, steps_per_epoch, magnitude, anneal, max_anneal_epoch, minimum
):
    if magnitude <= 0:
        return jnp.asarray(0.0, dtype=jnp.float32)
    if anneal is None:
        return jnp.asarray(magnitude, dtype=jnp.float32)
    if anneal != "linear":
        raise ValueError(f"{anneal!r} is not a valid x1 noise annealing strategy")
    if max_anneal_epoch <= 0:
        raise ValueError("x_1_noise_max_anneal_epoch must be positive")
    if not 0 <= minimum <= magnitude:
        raise ValueError("x_1_noise_min must be between zero and x_1_noise_mag")
    current_epoch = (step // steps_per_epoch).astype(jnp.float32)
    fraction = jnp.minimum(current_epoch / max_anneal_epoch, 1.0)
    return (magnitude - minimum) * (1.0 - fraction) + minimum


def flow_matching_train_step(
    state,
    batch,
    rng_init,
    learning_rate_fn,
    time_mu,
    time_sigma,
    steps_per_epoch,
    x_1_noise_mag,
    x_1_noise_anneal,
    x_1_noise_max_anneal_epoch,
    x_1_noise_min,
    num_classes,
    class_dropout_prob,
    process_batch_on_device_fn=None,
    clip_target=False,
):
    """Flow-match the diagonal velocity of the same model used for iMF."""
    rng_step = random.fold_in(rng_init, state.step)
    rng_device = random.fold_in(rng_step, lax.axis_index(axis_name="batch"))
    rng_time, rng_noise, rng_process, rng_label, rng_x1_noise = random.split(
        rng_device, 5
    )
    batch = process_batch_on_device_fn(batch, rng_key=rng_process)
    x1 = batch["image"]
    x1 = x1.astype(jnp.float32)
    label = batch["label"].astype(jnp.int32)
    batch_size = x1.shape[0]
    current_noise = x1_noise_magnitude(
        state.step,
        steps_per_epoch,
        x_1_noise_mag,
        x_1_noise_anneal,
        x_1_noise_max_anneal_epoch,
        x_1_noise_min,
    )
    if x_1_noise_mag > 0:
        x1 = x1 + current_noise * random.normal(
            rng_x1_noise, x1.shape, dtype=jnp.float32
        )
    if "x0" in batch:
        x0 = batch["x0"].astype(jnp.float32)
        if x0.shape != x1.shape:
            raise ValueError(
                f"Coupling x0 shape {x0.shape} does not match x1 shape {x1.shape}"
            )
    else:
        x0 = random.normal(rng_noise, x1.shape, dtype=jnp.float32)
    s = logit_normal_sample(rng_time, batch_size, mean=time_mu, std=time_sigma)
    x_s = _expand_time(s, x1) * x1 + _expand_time(1.0 - s, x0) * x0
    target = x1 - x0
    if clip_target:
        target = state.apply_fn(
            {"params": state.params}, x1, x_s, s, method=NFMR.denoiser_velocity
        )
    drop_mask = random.bernoulli(rng_label, class_dropout_prob, (batch_size,))
    label = jnp.where(drop_mask, num_classes, label)

    def loss_fn(parameters):
        velocity = state.apply_fn(
            {"params": parameters},
            x_s,
            label,
            s,
            jnp.ones_like(s),
            jnp.zeros_like(s),
            jnp.ones_like(s),
            method=NFMR.v_t,
        )
        loss = jnp.mean(jnp.square(target - velocity), dtype=jnp.float32)
        return (
            loss,
            {
                "loss": loss,
                "mean_s": jnp.mean(s),
                "class_drop_fraction": jnp.mean(drop_mask, dtype=jnp.float32),
                "v_ss_rms": jnp.sqrt(jnp.mean(jnp.square(velocity))),
                "x_1_noise_mag": current_noise,
            },
        )

    (_loss, losses), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    flat_grads, unravel = ravel_pytree(grads)
    grads = unravel(lax.pmean(flat_grads, axis_name="batch"))
    new_state = state.apply_gradients(grads=grads)
    new_state = new_state.replace(
        ema_params={
            k: jax.tree.map(
                lambda e, p: float(k) * e + (1.0 - float(k)) * p, ema, new_state.params
            )
            for k, ema in new_state.ema_params.items()
        }
    )
    metrics = compute_metrics(losses)
    metrics["lr"] = learning_rate_fn(new_state.step)
    metrics["grad_norm"] = optax.tree.norm(grads)
    return (new_state, metrics)


def mean_flow_train_step(
    state,
    batch,
    rng_init,
    learning_rate_fn,
    norm_eps,
    norm_p,
    diagonal_prop,
    time_mu,
    time_sigma,
    st_uniform,
    steps_per_epoch,
    x_1_noise_mag,
    x_1_noise_anneal,
    x_1_noise_max_anneal_epoch,
    x_1_noise_min,
    use_lpips,
    lpips_lambda,
    perceptual_min_s,
    use_convnext,
    convnext_lambda,
    auxloss_fn,
    use_cfg,
    cfg_beta,
    cfg_scale_max,
    expand_batch_for_cfg,
    num_classes,
    class_dropout_prob,
    process_batch_on_device_fn=None,
    clip_target=False,
):
    rng_step = random.fold_in(rng_init, state.step)
    rng_device = random.fold_in(rng_step, lax.axis_index(axis_name="batch"))
    (
        rng_time,
        rng_noise,
        rng_process,
        rng_cfg_interval,
        rng_cfg_scale,
        rng_label,
        rng_perceptual,
        rng_x1_noise,
    ) = random.split(rng_device, 8)
    batch = process_batch_on_device_fn(batch, rng_key=rng_process)
    x1 = batch["image"]
    x1 = x1.astype(jnp.float32)
    label = batch["label"].astype(jnp.int32)
    batch_size = x1.shape[0]
    current_x1_noise_mag = x1_noise_magnitude(
        state.step,
        steps_per_epoch,
        x_1_noise_mag,
        x_1_noise_anneal,
        x_1_noise_max_anneal_epoch,
        x_1_noise_min,
    )
    if x_1_noise_mag > 0:
        x1 = x1 + current_x1_noise_mag * random.normal(
            rng_x1_noise, x1.shape, dtype=jnp.float32
        )
    if "x0" in batch:
        x0 = batch["x0"].astype(jnp.float32)
        if x0.shape != x1.shape:
            raise ValueError(
                f"Coupling x0 shape {x0.shape} does not match x1 shape {x1.shape}"
            )
    else:
        x0 = random.normal(rng_noise, x1.shape, dtype=jnp.float32)
    s, t, diagonal_mask = sample_pixel_mean_flow_time(
        rng_time,
        batch_size,
        diagonal_prop=diagonal_prop,
        mu=time_mu,
        sigma=time_sigma,
        st_uniform=st_uniform,
    )
    I_s = _expand_time(s, x1) * x1 + _expand_time(1 - s, x0) * x0
    dot_I_s = x1 - x0
    if clip_target:
        dot_I_s = state.apply_fn(
            {"params": state.params}, x1, I_s, s, method=NFMR.denoiser_velocity
        )

    def loss_fn(parameters):
        variables = {"params": parameters}
        model_label = label
        if use_cfg:
            t_min, t_max = sample_pmf_cfg_interval(
                rng_cfg_interval, batch_size, diagonal_mask
            )
            w = sample_pmf_cfg_scale(rng_cfg_scale, batch_size, cfg_beta, cfg_scale_max)
            v_cond, v_uncond = state.apply_fn(
                variables,
                I_s,
                s,
                model_label,
                w,
                num_classes,
                expand_batch_for_cfg,
                method=NFMR.cond_and_uncond_v_t,
            )
            v_cond = lax.stop_gradient(v_cond)
            v_uncond = lax.stop_gradient(v_uncond)
            w_img = _expand_time(w, I_s)
            dot_Is_guided_fm = dot_I_s + (1 - 1 / w_img) * (v_cond - v_uncond)
            w_filtered = jnp.where((s >= t_min) & (s <= t_max), w, 1.0)
            v_cond = state.apply_fn(
                variables,
                I_s,
                model_label,
                s,
                w_filtered,
                jnp.zeros_like(s),
                jnp.ones_like(s),
                method=NFMR.v_t,
            )
            v_cond = lax.stop_gradient(v_cond)
            dot_Is_guided = dot_I_s + (1 - 1 / w_img) * (v_cond - v_uncond)
            dot_Is_guided = jnp.where(
                _expand_time(diagonal_mask, dot_I_s), dot_Is_guided_fm, dot_Is_guided
            )
            v_ss = v_cond
            rand_mask = random.bernoulli(rng_label, class_dropout_prob, (batch_size,))
            num_drop = jnp.sum(rand_mask, dtype=jnp.int32)
            drop_mask = jnp.arange(batch_size) < num_drop
            model_label = jnp.where(drop_mask, num_classes, model_label)
            dot_Is_guided = jnp.where(
                _expand_time(drop_mask, dot_I_s), dot_I_s, dot_Is_guided
            )
        else:
            w = jnp.ones((batch_size,), dtype=jnp.float32)
            t_min = jnp.zeros((batch_size,), dtype=jnp.float32)
            t_max = jnp.ones((batch_size,), dtype=jnp.float32)
            delta_ss = state.apply_fn(
                variables, I_s, model_label, s, s, w, t_min, t_max, method=NFMR.delta_st
            )
            v_ss = jax.lax.stop_gradient(
                state.apply_fn(
                    variables, delta_ss, I_s, s, method=NFMR.denoiser_velocity
                )
            )
            dot_Is_guided = dot_I_s
        s_dot = jnp.ones_like(s)
        t_dot = jnp.zeros_like(t)
        w_dot = jnp.zeros_like(w)
        t_min_dot = jnp.zeros_like(t_min)
        t_max_dot = jnp.zeros_like(t_max)
        delta_st, delta_st_dot = state.apply_fn(
            variables,
            I_s,
            v_ss,
            model_label,
            s,
            s_dot,
            t,
            t_dot,
            w,
            w_dot,
            t_min,
            t_min_dot,
            t_max,
            t_max_dot,
            strategy="padded",
            method=NFMR.delta_st_dual,
        )

        def velocity(delta_value, x_value, s_value):
            return state.apply_fn(
                variables, delta_value, x_value, s_value, method=NFMR.denoiser_velocity
            )

        v_st, dv_ds = jax.jvp(velocity, (delta_st, I_s, s), (delta_st_dot, v_ss, s_dot))
        student = -v_st + _expand_time(t - s, dv_ds) * jax.lax.stop_gradient(dv_ds)
        residual = jnp.square(jax.lax.stop_gradient(dot_Is_guided) + student)
        raw_mean_flow_loss = jnp.sum(residual, axis=tuple(range(1, residual.ndim)))
        mean_flow_loss = mf_adaptive_reweight(raw_mean_flow_loss, norm_eps, norm_p)
        loss = mean_flow_loss
        metrics = {
            "raw_mean_flow_loss": raw_mean_flow_loss.mean(),
            "x_1_noise_mag": current_x1_noise_mag,
        }
        if use_lpips or use_convnext:
            if auxloss_fn is None:
                raise ValueError("Perceptual loss enabled without auxloss_fn")
            predicted_x1 = I_s + _expand_time(1.0 - s, v_st) * v_st
            raw_lpips_loss, raw_convnext_loss = auxloss_fn(
                predicted_x1, x1, rng_perceptual
            )
            mask = s > perceptual_min_s
            raw_lpips_loss = jnp.where(mask, raw_lpips_loss, 0.0)
            raw_convnext_loss = jnp.where(mask, raw_convnext_loss, 0.0)
            lpips_loss = mf_adaptive_reweight(raw_lpips_loss, norm_eps, norm_p)
            convnext_loss = mf_adaptive_reweight(raw_convnext_loss, norm_eps, norm_p)
            loss += lpips_lambda * lpips_loss + convnext_lambda * convnext_loss
            metrics["raw_lpips_loss"] = raw_lpips_loss.mean()
            metrics["raw_convnext_loss"] = raw_convnext_loss.mean()
        loss = loss.mean()
        metrics["loss"] = loss
        return (loss, metrics)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_loss, dict_losses), grads = grad_fn(state.params)
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
