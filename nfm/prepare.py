"""Prepare CIFAR-10/ImageNet-64 pixels or compute compatible FID references."""

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import tarfile
import tempfile

import numpy as np
from PIL import Image
import requests
from tqdm import tqdm

from nfm.data import Dataset
from nfm.fid.metrics import FeatureExtractor, Moments


def center_crop(image, size):
    # Same progressive BOX / BICUBIC center crop as the source pixel pipeline.
    image = image.convert("RGB")
    while min(image.size) >= 2 * size:
        image = image.resize(
            tuple(value // 2 for value in image.size), Image.Resampling.BOX
        )
    scale = size / min(image.size)
    image = image.resize(
        tuple(round(value * scale) for value in image.size), Image.Resampling.BICUBIC
    )
    array = np.asarray(image)
    top, left = (array.shape[0] - size) // 2, (array.shape[1] - size) // 2
    return array[top : top + size, left : left + size]


def cifar(destination, split, archive=None):
    if archive is None:
        temporary = tempfile.TemporaryDirectory(prefix="nfm-cifar-")
        archive = Path(temporary.name) / "cifar.tar.gz"
        url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        with requests.get(url, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            with archive.open("wb") as handle:
                for block in response.iter_content(1024 * 1024):
                    handle.write(block)
    else:
        temporary = None
        archive = Path(archive)
    hasher = hashlib.md5(usedforsecurity=False)
    with archive.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    if hasher.hexdigest() != "c58f30108f718f92721af3b95e74349a":
        raise ValueError("CIFAR archive checksum mismatch")
    names = (
        [f"data_batch_{index}" for index in range(1, 6)]
        if split == "train"
        else ["test_batch"]
    )
    images, labels = [], []
    with tarfile.open(archive) as bundle:
        for name in names:
            with bundle.extractfile(f"cifar-10-batches-py/{name}") as handle:
                batch = pickle.load(handle, encoding="bytes")
            images.append(batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1))
            labels.extend(batch[b"labels"])
    destination.mkdir(parents=True, exist_ok=False)
    np.save(destination / "x1.npy", np.concatenate(images))
    np.save(destination / "label.npy", np.asarray(labels, np.int32))
    if temporary:
        temporary.cleanup()


def imagenet(source, destination):
    classes = sorted(path.name for path in source.iterdir() if path.is_dir())
    if not classes:
        raise ValueError("Expected class subdirectories with RGB images")
    records = [
        (path, index)
        for index, name in enumerate(classes)
        for path in sorted((source / name).rglob("*"))
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    ]
    if not records:
        raise ValueError("No images found")
    destination.mkdir(parents=True, exist_ok=False)
    images = np.lib.format.open_memmap(
        destination / "x1.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(len(records), 64, 64, 3),
    )
    labels = np.lib.format.open_memmap(
        destination / "label.npy", mode="w+", dtype=np.int32, shape=(len(records),)
    )
    for index, (path, label) in enumerate(tqdm(records, desc="Preparing ImageNet-64")):
        with Image.open(path) as image:
            images[index] = center_crop(image, 64)
        labels[index] = label
    images.flush()
    labels.flush()
    (destination / "classes.json").write_text(json.dumps(classes, indent=2) + "\n")


def reference(data, destination, size, weights, batch_size, samples):
    dataset = Dataset(data, size)
    count = min(samples or dataset.length, dataset.length)
    extractor, moments = FeatureExtractor(weights), Moments()
    for start in tqdm(range(0, count, batch_size), desc="FID reference"):
        pixels = dataset.read("x1", np.arange(start, min(start + batch_size, count)))
        moments.update(extractor(pixels))
    mu, sigma = moments.statistics()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle:
        np.savez(
            handle,
            mu=mu,
            sigma=sigma,
            samples=count,
            image_size=size,
            weights_sha256=extractor.weights_digest,
            preprocessing="uint8 / legacy grid_sample / Inception FID",
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("cifar10")
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--split", choices=("train", "test"), default="train")
    command.add_argument("--archive", type=Path)
    command = commands.add_parser("imagenet64")
    command.add_argument("--input", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command = commands.add_parser("fid-reference")
    command.add_argument("--data", required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--image-size", type=int, choices=(32, 64), required=True)
    command.add_argument("--weights")
    command.add_argument("--batch-size", type=int, default=80)
    command.add_argument("--samples", type=int)
    args = parser.parse_args()
    if args.command == "cifar10":
        cifar(args.output, args.split, args.archive)
    elif args.command == "imagenet64":
        imagenet(args.input, args.output)
    else:
        reference(
            args.data,
            args.output,
            args.image_size,
            args.weights,
            args.batch_size,
            args.samples,
        )


if __name__ == "__main__":
    main()
