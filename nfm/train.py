"""Train NFMD or either NFMR stage, with exact optimizer/data-order resume."""

import argparse
from functools import partial
import json
import os
from pathlib import Path
import signal
import time

import jax
import numpy as np
from flax import jax_utils
from jax.experimental import multihost_utils

from nfm import checkpoint
from nfm.config import load, digest
from nfm.data import Dataset, epoch_indices, process_batch, prefetch_batches
from nfm.objectives.nfmd import nfmd_train_step
from nfm.objectives.nfmr import flow_matching_train_step, mean_flow_train_step
from nfm.state import make_model, learning_rate, create_state


def initialize_distributed():
    if jax.distributed.is_initialized():
        return
    address = os.environ.get("JAX_COORDINATOR_ADDRESS")
    count = os.environ.get("JAX_PROCESS_COUNT")
    rank = os.environ.get("JAX_PROCESS_INDEX")
    if not address and int(os.environ.get("SLURM_NTASKS", "1")) > 1:
        host = os.environ.get("MASTER_ADDR")
        if host:
            address = f"{host}:{os.environ.get('MASTER_PORT', '12355')}"
            count, rank = os.environ["SLURM_NTASKS"], os.environ["SLURM_PROCID"]
    if address:
        if count is None or rank is None:
            raise ValueError(
                "Set JAX_PROCESS_COUNT and JAX_PROCESS_INDEX with the coordinator"
            )
        kwargs = {}
        local_ids = os.environ.get("JAX_LOCAL_DEVICE_IDS")
        if local_ids:
            kwargs["local_device_ids"] = [int(value) for value in local_ids.split(",")]
        jax.distributed.initialize(address, int(count), int(rank), **kwargs)
    elif int(os.environ.get("SLURM_NTASKS", "1")) > 1:
        # JAX's Slurm discovery uses the allocation's coordinator and rank.
        jax.distributed.initialize()


def make_step(config, schedule, steps_per_epoch):
    processor = partial(process_batch, use_flip=config["dataset"]["flip"])
    args = dict(learning_rate_fn=schedule, process_batch_on_device_fn=processor)
    objective = config["objective"]
    if config["stage"] == "nfmd":
        function = nfmd_train_step
        args.update(s_max=objective["s_max"], dropout_rate=objective["label_dropout"])
    else:
        args.update(
            time_mu=objective["time_mu"],
            time_sigma=objective["time_sigma"],
            steps_per_epoch=steps_per_epoch,
            x_1_noise_mag=0.0,
            x_1_noise_anneal=None,
            x_1_noise_max_anneal_epoch=100,
            x_1_noise_min=0.0,
            num_classes=config["model"]["num_classes"],
            class_dropout_prob=objective["label_dropout"],
            clip_target=objective["clip_target"],
        )
        function = flow_matching_train_step
        if config["stage"] == "mean_flow":
            from nfm.auxiliary import init_auxloss

            function = mean_flow_train_step
            args.update(
                {
                    key: objective[key]
                    for key in (
                        "norm_eps",
                        "norm_p",
                        "diagonal_prop",
                        "st_uniform",
                        "use_lpips",
                        "lpips_lambda",
                        "perceptual_min_s",
                        "use_convnext",
                        "convnext_lambda",
                        "use_cfg",
                        "cfg_beta",
                        "cfg_scale_max",
                        "expand_batch_for_cfg",
                    )
                }
            )
            args["auxloss_fn"] = init_auxloss(
                use_lpips=objective["use_lpips"], use_convnext=objective["use_convnext"]
            )

    def step(state, batch, rng, stop_requested):
        state, metrics = function(state, batch, rng, **args)
        metrics["stop_requested"] = jax.lax.pmax(stop_requested, "batch")
        return state, metrics

    return jax.pmap(step, axis_name="batch", in_axes=(0, 0, None, None))


def run(
    config,
    data_path,
    workdir,
    init_from=None,
    init_ema=None,
    max_steps=None,
    wandb_project=None,
    max_seconds=None,
):
    initialize_distributed()
    global_batch = config["training"]["batch_size"]
    if global_batch % jax.device_count():
        raise ValueError(
            f"Global batch {global_batch} must divide by {jax.device_count()} devices"
        )
    dataset = Dataset(
        data_path,
        config["model"]["img_size"],
        require_couplings=config["method"] == "nfmr",
        num_classes=config["model"]["num_classes"],
    )
    epoch_samples = config["dataset"]["epoch_samples"] or dataset.length
    if epoch_samples > dataset.length:
        raise ValueError("Configured epoch size exceeds available dataset")
    steps_per_epoch = epoch_samples // global_batch
    if not steps_per_epoch:
        raise ValueError("Dataset is smaller than one global batch")
    total_steps = config["training"]["epochs"] * steps_per_epoch
    model = make_model(config)
    schedule = learning_rate(config["training"], steps_per_epoch)
    workdir = Path(workdir)
    recovery = checkpoint.latest(workdir)
    params = None
    if recovery is None and init_from:
        source = json.loads((Path(init_from) / "metadata.json").read_text())
        if (
            config["stage"] != "mean_flow"
            or source["config"]["stage"] != "flow_matching"
        ):
            raise ValueError("Stage handoff must be NFMR flow_matching → mean_flow")
        if not source["complete"]:
            raise ValueError("Flow-matching stage is unfinished")
        if source["config"]["model"] != config["model"]:
            raise ValueError("Model architecture/precision changed across stages")
        if source["dataset"] != dataset.identity():
            raise ValueError("NFMR stages must use the same coupling dataset")
        params = checkpoint.load_weights(init_from, init_ema or config["handoff_ema"])
    if recovery is None and config["stage"] == "mean_flow" and params is None:
        raise ValueError(
            "Mean-flow stage requires --init-from a completed flow-matching checkpoint"
        )
    state = create_state(model, config, schedule, params)
    if recovery:
        metadata = json.loads((recovery / "metadata.json").read_text())
        if metadata["config_digest"] != digest(config):
            raise ValueError(
                "Resume configuration changed; use a separate output directory"
            )
        if metadata["dataset"] != dataset.identity():
            raise ValueError("Resume dataset changed")
        state = checkpoint.restore(recovery, state)
    first_step = int(state.step)
    if first_step >= total_steps:
        if jax.process_index() == 0:
            print(f"{config['stage']} already complete: {recovery}")
        return recovery
    if max_steps is not None and max_steps < 1:
        raise ValueError("--max-steps must be positive")
    if max_seconds is not None and max_seconds <= 0:
        raise ValueError("--max-seconds must be positive")
    stop_step = min(total_steps, first_step + max_steps) if max_steps else total_steps
    p_step = make_step(config, schedule, steps_per_epoch)
    # JAX 0.10 uses explicit device meshes. An identity pmap establishes the
    # same layout as subsequent training calls, including scalar state.step.
    state = jax.pmap(lambda value: value)(jax_utils.replicate(state))
    rng = jax.random.PRNGKey(config["seed"])
    local_devices = jax.local_device_count()
    metadata = {
        "config": config,
        "config_digest": digest(config),
        "dataset": dataset.identity(),
        "steps_per_epoch": steps_per_epoch,
        "seed": config["seed"],
    }
    tracker = None
    if jax.process_index() == 0:
        workdir.mkdir(parents=True, exist_ok=True)
        (workdir / "resolved_config.json").write_text(
            json.dumps(config, indent=2) + "\n"
        )
        if wandb_project:
            import wandb

            tracker = wandb.init(
                project=wandb_project,
                entity=os.environ.get("WANDB_ENTITY"),
                name=config["name"],
                config=config,
                resume="allow",
                id=workdir.name + "-" + digest(config)[:12],
            )
        print(
            f"{config['stage']}: global batch {global_batch}, {steps_per_epoch} steps/epoch; "
            f"step {first_step} → {stop_step} ({jax.device_count()} devices)",
            flush=True,
        )
    last_checkpoint = recovery
    started = time.monotonic()
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    previous_handlers = {
        value: signal.signal(value, request_stop)
        for value in (signal.SIGTERM, signal.SIGUSR1)
    }
    for epoch in range(first_step // steps_per_epoch, config["training"]["epochs"]):
        order = epoch_indices(
            dataset.length,
            epoch_samples,
            config["seed"],
            epoch,
            global_batch,
            jax.process_count(),
            jax.process_index(),
        )
        offset = (
            first_step % steps_per_epoch
            if epoch == first_step // steps_per_epoch
            else 0
        )
        for index, raw_batch in enumerate(
            prefetch_batches(dataset, order[offset:]), offset
        ):
            batch = {
                key: array.reshape(local_devices, -1, *array.shape[1:])
                for key, array in raw_batch.items()
            }
            requested = stop_requested or (
                max_seconds is not None and time.monotonic() - started >= max_seconds
            )
            state, metrics = p_step(state, batch, rng, np.asarray(requested, np.int32))
            stopping = bool(
                np.asarray(jax.device_get(metrics.pop("stop_requested")))[0]
            )
            step = epoch * steps_per_epoch + index + 1
            if step % config["training"]["log_steps"] == 0 or step == stop_step:
                values = {
                    key: float(np.asarray(value)[0])
                    for key, value in jax.device_get(metrics).items()
                }
                if not all(np.isfinite(value) for value in values.values()):
                    raise FloatingPointError(
                        f"Nonfinite training metrics at step {step}: {values}"
                    )
                if jax.process_index() == 0:
                    record = dict(
                        step=step,
                        epoch=step / steps_per_epoch,
                        seconds=time.monotonic() - started,
                        **values,
                    )
                    with (workdir / "metrics.jsonl").open("a") as handle:
                        handle.write(json.dumps(record) + "\n")
                    print(json.dumps(record), flush=True)
                    if tracker:
                        tracker.log(values, step=step)
            save_epoch = (
                step % steps_per_epoch == 0
                and (epoch + 1) % config["training"]["checkpoint_epochs"] == 0
            )
            if save_epoch or step == stop_step or stopping:
                multihost_utils.sync_global_devices(f"save-{step}-before")
                if jax.process_index() == 0:
                    last_checkpoint = checkpoint.save(
                        workdir,
                        jax_utils.unreplicate(state),
                        dict(metadata, step=step, complete=step == total_steps),
                    )
                multihost_utils.sync_global_devices(f"save-{step}-after")
            if step == stop_step or stopping:
                for value, handler in previous_handlers.items():
                    signal.signal(value, handler)
                if tracker:
                    tracker.finish()
                if stopping and step < total_steps:
                    # A launcher must stop here, rather than evaluating an
                    # unfinished recipe or advancing to the next NFMR stage.
                    raise SystemExit(75)
                return checkpoint.latest(workdir)
    return last_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--init-from")
    parser.add_argument("--init-ema")
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Additional steps for a probe, never marks it complete",
    )
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT"))
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=os.environ.get("NFM_MAX_SECONDS"),
        help="Save and exit 75 before a job time limit; allow time for checkpoint writing",
    )
    args = parser.parse_args()
    run(
        load(args.config),
        args.data,
        args.workdir,
        args.init_from,
        args.init_ema,
        args.max_steps,
        args.wandb_project,
        args.max_seconds,
    )


if __name__ == "__main__":
    main()
