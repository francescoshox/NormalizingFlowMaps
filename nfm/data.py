"""Memory-mapped pixels/couplings and deterministic global epoch sharding."""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import jax
import jax.numpy as jnp
import numpy as np
from safetensors import safe_open


class Dataset:
    def __init__(self, path, image_size, require_couplings=False, num_classes=None):
        self.path = Path(path)
        self.handle = None
        if self.path.is_dir():
            self.arrays = {
                key: np.load(self.path / f"{key}.npy", mmap_mode="r")
                for key in ("x1", "label", "x0")
                if (self.path / f"{key}.npy").exists()
            }
        else:
            self.handle = safe_open(str(self.path), framework="np")
            self.arrays = {
                key: self.handle.get_slice(key) for key in self.handle.keys()
            }
        if not {"x1", "label"} <= self.arrays.keys():
            raise ValueError("Dataset requires x1 (uint8 NHWC) and label (integer N)")
        shape = self.shape("x1")
        if len(shape) != 4 or shape[1:] != (image_size, image_size, 3):
            raise ValueError(f"Expected [N,{image_size},{image_size},3], got {shape}")
        self.length = shape[0]
        if self.shape("label") != (self.length,):
            raise ValueError("Label shape must match the number of images")
        if self.read("x1", [0]).dtype != np.uint8:
            raise ValueError("x1 storage must be uint8")
        labels = np.asarray(self.arrays["label"][:])
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("Labels must be integers")
        if np.any(labels < 0) or (
            num_classes is not None and np.any(labels >= num_classes)
        ):
            raise ValueError("Label outside the model's class range")
        if require_couplings and "x0" not in self.arrays:
            raise ValueError(
                "NFMR requires paired Gaussian x0; do not substitute fresh noise"
            )
        if "x0" in self.arrays:
            if self.shape("x0") != shape:
                raise ValueError("Paired x0 and x1 shapes must match")
            if not np.issubdtype(self.read("x0", [0]).dtype, np.floating):
                raise ValueError("Paired x0 must be floating point")

    def shape(self, key):
        array = self.arrays[key]
        return tuple(array.shape if hasattr(array, "shape") else array.get_shape())

    def read(self, key, indices):
        array = self.arrays[key]
        if isinstance(array, np.ndarray):
            return np.asarray(array[indices])
        # Safetensors' numpy slicing supports only basic slices, not fancy indexing.
        return np.stack([array[int(i)] for i in indices])

    def batch(self, indices):
        values = {
            "image": self.read("x1", indices),
            "label": self.read("label", indices).astype(np.int32),
        }
        if "x0" in self.arrays:
            values["x0"] = self.read("x0", indices).astype(np.float32)
        return values

    def identity(self):
        paths = sorted(self.path.glob("*.npy")) if self.path.is_dir() else [self.path]
        return {
            "length": self.length,
            "files": [
                {
                    "name": p.name,
                    "size": p.stat().st_size,
                    "mtime_ns": p.stat().st_mtime_ns,
                }
                for p in paths
            ],
        }


def epoch_indices(
    length, samples, seed, epoch, batch_size, process_count=1, process_index=0
):
    samples = samples or length
    if samples > length:
        raise ValueError(f"Requested {samples} epoch examples from a pool of {length}")
    if batch_size % process_count:
        raise ValueError("Global batch size must divide evenly over processes")
    count = samples // batch_size * batch_size  # Drop one incomplete global batch.
    if count == 0:
        raise ValueError("Dataset is smaller than a global batch")
    order = np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(
        length
    )[:count]
    order = order.reshape(-1, process_count, batch_size // process_count)
    return order[:, process_index]


def process_batch(batch, rng_key, use_flip=True):
    output = dict(batch)
    output["image"] = batch["image"].astype(jnp.float32) / 255.0 * 2 - 1
    if use_flip:
        flip = jax.random.bernoulli(rng_key, 0.5, (batch["image"].shape[0], 1, 1, 1))
        for key in ("image", "x0"):
            if key in output:
                output[key] = jnp.where(flip, output[key][:, :, ::-1, :], output[key])
    return output


def prefetch_batches(dataset, indices):
    """Overlap one host batch read with the current accelerator update."""
    iterator = iter(indices)
    with ThreadPoolExecutor(max_workers=1) as executor:
        first = next(iterator, None)
        if first is None:
            return
        pending = executor.submit(dataset.batch, first)
        for following in iterator:
            batch = pending.result()
            pending = executor.submit(dataset.batch, following)
            yield batch
        yield pending.result()
