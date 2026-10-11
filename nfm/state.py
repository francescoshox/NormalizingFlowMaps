"""Adjusted Muon, paper learning-rate schedules and EMA training state."""

from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from nfm.models.nfmd import NFMD
from nfm.models.nfmr import NFMR


class TrainState(train_state.TrainState):
    ema_params: dict[float, Any]


def make_model(config):
    kwargs = dict(config["model"])
    precision = kwargs.pop("precision", "bf16")
    if precision not in ("bf16", "fp32"):
        raise ValueError("precision must be bf16 or fp32")
    kwargs["dtype"] = jnp.bfloat16 if precision == "bf16" else jnp.float32
    return (NFMD if config["method"] == "nfmd" else NFMR)(**kwargs)


def initialize(model, method, seed):
    x = jnp.ones((1, model.img_size, model.img_size, model.out_channels), jnp.float32)
    label = jnp.zeros((1,), jnp.int32)
    zero, one = jnp.zeros((1,), jnp.float32), jnp.ones((1,), jnp.float32)
    args = (x, label, zero, one)
    if method == "nfmr":
        args += (one, zero, one)
    return jax.jit(model.init)(jax.random.PRNGKey(seed), *args)["params"]


def learning_rate(training, steps_per_epoch):
    peak, floor = training["lr"], training.get("final_lr", training["lr"])
    warmup = training.get("warmup_steps", 0)
    start_lr = training.get("warmup_lr", 1e-6)
    start = training.get("cooldown_start", 0) * steps_per_epoch
    end = training.get("cooldown_end", 0) * steps_per_epoch
    if end and (start < warmup or end <= start):
        raise ValueError("Cooldown must follow warmup and have a positive duration")

    def schedule(step):
        step = jnp.asarray(step, jnp.float32)
        lr = jnp.asarray(peak, jnp.float32)
        if end:
            progress = jnp.clip((step - start) / (end - start), 0, 1)
            lr = floor + (peak - floor) * (1 + jnp.cos(jnp.pi * progress)) / 2
        if warmup:
            lr = jnp.where(
                step < warmup, start_lr + (peak - start_lr) * step / warmup, lr
            )
        return lr

    return schedule


def parameter_labels(params):
    def label(path, parameter):
        name = "/".join(str(getattr(entry, "key", entry)) for entry in path)
        if parameter.ndim >= 2 and "embedding" in name:
            return "adam_decay"
        if parameter.ndim > 2:
            raise ValueError(f"Muon requires matrices; {name}: {parameter.shape}")
        if parameter.ndim == 2:
            return "muon"
        return "adam_no_decay"

    return jax.tree.map_with_path(label, params)


def adjusted_muon(params, schedule):
    return optax.multi_transform(
        {
            "muon": optax.contrib.muon(
                learning_rate=schedule,
                ns_coeffs=(3.4445, -4.775, 2.0315),
                ns_steps=5,
                beta=0.95,
                eps=1e-7,
                nesterov=True,
                weight_decay=0.1,
                adam_weight_decay=0,
            ),
            "adam_decay": optax.adamw(schedule, b1=0.9, b2=0.95, weight_decay=1e-4),
            "adam_no_decay": optax.adamw(schedule, b1=0.9, b2=0.95, weight_decay=0),
        },
        parameter_labels(params),
    )


def create_state(model, config, schedule, params=None):
    if params is None:
        params = initialize(model, config["method"], config["seed"])
    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=adjusted_muon(params, schedule),
        ema_params={float(decay): params for decay in config["training"]["emas"]},
    )
