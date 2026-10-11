"""Fixed-config sampling/FID and unguided model likelihood."""

import argparse
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from nfm.checkpoint import load_weights
from nfm.config import load
from nfm.data import Dataset
from nfm.fid.metrics import FeatureExtractor, Moments, frechet, sha256
from nfm.models import nfmd, nfmr
from nfm.state import make_model


def to_pixels(samples):
    # Preserve the reference sampler's quantization convention.
    return np.clip(127.5 * np.asarray(samples) + 128.0, 0, 255).astype(np.uint8)


def sampler(config, params, batch_size, guidance):
    model = make_model(config)
    variables = {"params": params}
    evaluation = config["evaluation"]
    if config["method"] == "nfmd":
        function = lambda key, index: nfmd.generate(
            variables,
            model,
            key,
            batch_size,
            noise_level=0,
            guidance=guidance,
            image_idx=index,
        )
    elif config["stage"] == "flow_matching":
        function = lambda key, index: nfmr.flow_matching_generate(
            variables,
            model,
            key,
            batch_size,
            image_idx=index,
            n_steps=evaluation["fm_steps"],
            cfg_w=guidance,
            cfg_t_min=evaluation["interval"][0],
            cfg_t_max=evaluation["interval"][1],
        )
    else:
        function = lambda key, index: nfmr.mean_flow_generate(
            variables,
            model,
            key,
            batch_size,
            image_idx=index,
            cfg_w=guidance,
            cfg_t_min=evaluation["interval"][0],
            cfg_t_max=evaluation["interval"][1],
        )
    return jax.jit(function)


def sample_batches(config, params, count, batch_size, guidance, seed):
    if count < 1 or batch_size < 1:
        raise ValueError("Sample count and batch size must be positive")
    generate = sampler(config, params, batch_size, guidance)
    key = jax.random.PRNGKey(seed)
    for index, start in enumerate(range(0, count, batch_size)):
        samples = generate(
            jax.random.fold_in(key, index), jnp.asarray(index, jnp.int32)
        )
        yield to_pixels(samples)[: min(batch_size, count - start)]


def likelihood(config, params, images, labels, seed=0):
    """Uniform dequantization; return per-example bits/dimension at CFG off.

    NFMD encodes directly; NFMR requires a sequential inverse. NFMR's hard
    affine scales may make that inverse numerically unstable, so check finite
    values and round-trip errors separately before interpreting its density.
    """
    model = make_model(config)
    variables = {"params": params}
    noise = jax.random.uniform(jax.random.PRNGKey(seed), images.shape)
    x = (jnp.asarray(images, jnp.float32) + noise) / 128.0 - 1.0
    label = jnp.asarray(labels, jnp.int32)
    zero = jnp.zeros((len(images),), jnp.float32)
    one = jnp.ones_like(zero)
    if config["method"] == "nfmd":
        z, logdet = model.apply(variables, x, label, zero, one)
    else:
        z, logdet = nfmr.reverse(
            variables, model, model.patchify(x), label, zero, one, one, zero, one
        )
        z = model.patchify(z)
    nll = 0.5 * jnp.mean(jnp.square(z), axis=tuple(range(1, z.ndim)))
    nll += 0.5 * jnp.log(2 * jnp.pi) - logdet
    bpd = nll / jnp.log(2.0) + 7.0
    if not np.isfinite(np.asarray(bpd)).all():
        raise FloatingPointError("Nonfinite model likelihood")
    return np.asarray(bpd)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--reference",
        help="FID statistics; required unless --sample-only or --likelihood-data",
    )
    parser.add_argument(
        "--weights", help="Local Inception FID weights (otherwise download/cache)"
    )
    parser.add_argument("--sample-only", action="store_true")
    parser.add_argument("--likelihood-data")
    parser.add_argument("--samples", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ema", help="Select one configured EMA, or online")
    parser.add_argument(
        "--cfg", type=float, help="Select one configured guidance value"
    )
    args = parser.parse_args()
    config = load(args.config)
    saved = json.loads((Path(args.checkpoint) / "metadata.json").read_text())
    if (
        saved["config"]["model"] != config["model"]
        or saved["config"]["stage"] != config["stage"]
    ):
        raise ValueError("Checkpoint model/stage does not match evaluation config")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    count = args.samples or config["evaluation"]["samples"]
    batch_size = args.batch_size or config["evaluation"]["batch_size"]
    emas = [args.ema] if args.ema else config["evaluation"]["emas"]
    cfgs = [args.cfg] if args.cfg is not None else config["evaluation"]["cfg"]
    if not args.likelihood_data and not args.sample_only and not args.reference:
        parser.error("--reference is required for FID")
    if not args.likelihood_data and any(
        cfg not in config["evaluation"]["cfg"] for cfg in cfgs
    ):
        parser.error("--cfg must be a configured value; ImageNet NFMD uses 3.3")
    extractor = None
    reference_digest = None
    if args.reference and not args.likelihood_data and not args.sample_only:
        extractor = FeatureExtractor(args.weights)
        reference_digest = sha256(args.reference)
        with np.load(args.reference) as ref:
            ref_mu = ref["mu"] if "mu" in ref else ref["ref_mu"]
            ref_sigma = ref["sigma"] if "sigma" in ref else ref["ref_sigma"]
            if (
                "weights_sha256" in ref
                and str(ref["weights_sha256"]) != extractor.weights_digest
            ):
                raise ValueError(
                    "Reference statistics used different Inception weights"
                )
    for ema in emas:
        params = load_weights(args.checkpoint, ema)
        if args.likelihood_data:
            dataset = Dataset(
                args.likelihood_data,
                config["model"]["img_size"],
                num_classes=config["model"]["num_classes"],
            )
            count = min(count, dataset.length)
            values = []
            for start in range(0, count, batch_size):
                batch = dataset.batch(np.arange(start, min(start + batch_size, count)))
                values.extend(
                    likelihood(
                        config,
                        params,
                        batch["image"],
                        batch["label"],
                        args.seed + start,
                    )
                )
            atomic_json(
                output / f"ema_{ema}_likelihood.json",
                {
                    "bpd": float(np.mean(values)),
                    "samples": count,
                    "ema": str(ema),
                    "dequantization": "uniform",
                    "guidance": "off",
                },
            )
            continue
        for guidance in cfgs:
            name = f"ema_{ema}_cfg_{guidance}"
            identity = {
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "ema": str(ema),
                "cfg": guidance,
                "samples": count,
                "batch_size": batch_size,
                "seed": args.seed,
                "reference_sha256": reference_digest,
                "weights_sha256": extractor.weights_digest if extractor else None,
            }
            target = output / f"{name}.json"
            if not args.sample_only and target.exists():
                previous = json.loads(target.read_text())
                if previous.get("status") == "complete" and all(
                    previous.get(k) == v for k, v in identity.items()
                ):
                    print(f"Already evaluated: {target}")
                    continue
            moments = Moments()
            offset = 0
            for pixels in sample_batches(
                config, params, count, batch_size, guidance, args.seed
            ):
                if extractor:
                    moments.update(extractor(pixels))
                if args.sample_only:
                    folder = output / name
                    folder.mkdir(exist_ok=True)
                    for index, image in enumerate(pixels, offset):
                        Image.fromarray(image).save(folder / f"{index:06d}.png")
                offset += len(pixels)
                print(f"{name}: {offset}/{count}", flush=True)
            result = dict(identity, status="complete")
            if extractor:
                mu, sigma = moments.statistics()
                result["fid"] = frechet(mu, sigma, ref_mu, ref_sigma)
            atomic_json(target, result)


if __name__ == "__main__":
    main()
