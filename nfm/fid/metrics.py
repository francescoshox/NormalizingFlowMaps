"""Streaming FP64 FID moments, with the inherited Inception preprocessing."""

import hashlib

import jax
import jax.numpy as jnp
import numpy as np
from scipy import linalg


class Moments:
    def __init__(self):
        self.count = 0
        self.mean = self.m2 = None

    def update(self, features):
        features = np.asarray(features, np.float64)
        if features.ndim != 2 or not np.isfinite(features).all():
            raise ValueError("Features must be a finite [N,D] array")
        count = len(features)
        if not count:
            return
        mean = features.mean(axis=0)
        centered = features - mean
        m2 = centered.T @ centered
        if self.count == 0:
            self.mean, self.m2, self.count = mean, m2, count
            return
        delta = mean - self.mean
        total = self.count + count
        self.m2 += m2 + np.outer(delta, delta) * self.count * count / total
        self.mean += delta * count / total
        self.count = total

    def statistics(self):
        if self.count < 2:
            raise ValueError("At least two samples are needed for FID covariance")
        return self.mean, self.m2 / (self.count - 1)


def frechet(mu1, sigma1, mu2, sigma2):
    if not all(np.isfinite(value).all() for value in (mu1, sigma1, mu2, sigma2)):
        raise ValueError("FID statistics must be finite")
    if mu1.shape != mu2.shape or sigma1.shape != sigma2.shape:
        raise ValueError("Reference and generated feature dimensions differ")
    root = linalg.sqrtm(sigma1 @ sigma2)
    if not np.isfinite(root).all():
        offset = np.eye(sigma1.shape[0]) * 1e-6
        root = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))
    if np.iscomplexobj(root):
        if np.max(np.abs(root.imag)) > 1e-3:
            raise ValueError("Complex FID covariance square root")
        root = root.real
    value = np.sum((mu1 - mu2) ** 2) + np.trace(sigma1 + sigma2 - 2 * root)
    if not np.isfinite(value):
        raise FloatingPointError("Nonfinite FID")
    return float(max(0, value))  # Tiny negative roundoff at identical statistics.


def sha256(path):
    value = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


class FeatureExtractor:
    def __init__(self, weights=None):
        from nfm.fid.inception import InceptionV3
        from nfm.fid.utils import download

        kwargs = {"ckpt_path": weights} if weights else {}
        self.model = InceptionV3(pretrained=True, **kwargs)
        self.weights_digest = sha256(download(self.model.ckpt_path))
        self.variables = self.model.init(
            jax.random.PRNGKey(0), jnp.zeros((1, 299, 299, 3)), train=False
        )
        self.apply = jax.jit(
            lambda images: self.model.apply(self.variables, images, train=False)[0]
        )

    def __call__(self, pixels):
        import torch
        from nfm.fid.resize import forward

        if pixels.dtype != np.uint8:
            raise ValueError("FID input must be uint8 NHWC")
        images = torch.from_numpy(
            np.array(pixels.transpose(0, 3, 1, 2), dtype=np.float32, copy=True)
        )
        images = forward(images).numpy().transpose(0, 2, 3, 1)
        return np.asarray(self.apply(images)).reshape(len(images), 2048)
