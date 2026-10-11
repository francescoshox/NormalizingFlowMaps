# Adapted from BiFlow; see LICENSE and NOTICE.
# Copyright (c) Meta Platforms, Inc. and affiliates.

import re
from functools import partial
from typing import Sequence

import jax
import jax.numpy as jnp
import torch
from flax import linen as nn

log_for_0 = print


class ConvNextLayerNorm(nn.Module):
    normalized_shape: int
    eps: float = 1e-6

    def setup(self):
        self.weight = self.param(
            "weight", nn.initializers.ones, (self.normalized_shape,)
        )
        self.bias = self.param("bias", nn.initializers.zeros, (self.normalized_shape,))

    def __call__(self, x):
        mean = jnp.mean(x, axis=-1, keepdims=True)
        variance = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
        return self.weight * (x - mean) / jnp.sqrt(variance + self.eps) + self.bias


class ConvNextGRN(nn.Module):
    dim: int
    eps: float = 1e-6

    def setup(self):
        shape = (1, 1, 1, self.dim)
        self.gamma = self.param("gamma", nn.initializers.zeros, shape)
        self.beta = self.param("beta", nn.initializers.zeros, shape)

    def __call__(self, x):
        response = jnp.sqrt(
            jnp.sum(jnp.square(x), axis=(1, 2), keepdims=True) + self.eps
        )
        response /= jnp.mean(response, axis=-1, keepdims=True) + self.eps
        return self.gamma * (x * response) + self.beta + x


class ConvNextBlock(nn.Module):
    dim: int

    def setup(self):
        self.dwconv = nn.Conv(
            features=self.dim,
            kernel_size=(7, 7),
            padding="SAME",
            feature_group_count=self.dim,
            name="dwconv",
        )
        self.norm = ConvNextLayerNorm(self.dim, eps=1e-6)
        self.pwconv1 = nn.Dense(features=4 * self.dim, name="pwconv1")
        self.grn = ConvNextGRN(4 * self.dim)
        self.pwconv2 = nn.Dense(features=self.dim, name="pwconv2")

    def __call__(self, x):
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = jax.nn.gelu(x, approximate=False)
        x = self.grn(x)
        return residual + self.pwconv2(x)


class ConvNextV2(nn.Module):
    depths: Sequence[int] = (3, 3, 9, 3)
    dims: Sequence[int] = (96, 192, 384, 768)

    def setup(self):
        downsample_layers = [
            nn.Sequential(
                [
                    nn.Conv(
                        features=self.dims[0],
                        kernel_size=(4, 4),
                        strides=(4, 4),
                    ),
                    ConvNextLayerNorm(self.dims[0]),
                ],
                name="downsample_layers_0",
            )
        ]
        for index in range(3):
            downsample_layers.append(
                nn.Sequential(
                    [
                        ConvNextLayerNorm(self.dims[index]),
                        nn.Conv(
                            features=self.dims[index + 1],
                            kernel_size=(2, 2),
                            strides=(2, 2),
                        ),
                    ],
                    name=f"downsample_layers_{index + 1}",
                )
            )
        self.downsample_layers = downsample_layers
        self.stages = [
            nn.Sequential(
                [ConvNextBlock(self.dims[index]) for _ in range(self.depths[index])],
                name=f"stages_{index}",
            )
            for index in range(4)
        ]
        self.norm = nn.LayerNorm(epsilon=1e-6)

    def __call__(self, x):
        for downsample, stage in zip(self.downsample_layers, self.stages):
            x = stage(downsample(x))
        return self.norm(jnp.mean(x, axis=(1, 2)))


ConvNextBase = partial(ConvNextV2, depths=(3, 3, 27, 3), dims=(128, 256, 512, 1024))


def _rename_hf_path(path):
    path = re.sub(r"classifier\.", "head.", path)
    path = re.sub(r"convnextv2\.encoder\.", "", path)
    path = re.sub(
        r"convnextv2\.embeddings\.patch_embeddings\.",
        "downsample_layers_0.layers_0.",
        path,
    )
    path = re.sub(
        r"convnextv2\.embeddings\.layernorm\.",
        "downsample_layers_0.layers_1.",
        path,
    )
    path = re.sub(
        r"stages\.([0-3])\.downsampling_layer\.(\d+)",
        lambda match: f"downsample_layers_{match.group(1)}.layers_{match.group(2)}",
        path,
    )
    path = re.sub(
        r"stages\.([0-3])\.layers\.(\d+)",
        lambda match: f"stages_{match.group(1)}.layers_{match.group(2)}",
        path,
    )
    path = re.sub(r"layernorm", "norm", path)
    path = re.sub(r"convnextv2\.", "", path)
    path = re.sub(r"grn\.weight", "grn.gamma", path)
    return re.sub(r"grn\.bias", "grn.beta", path)


def convert_hf_weights(jax_params, torch_params):
    log_for_0("Converting ConvNeXt weights to JAX...")
    renamed = {
        f"params.{_rename_hf_path(path)}": value for path, value in torch_params.items()
    }
    leaves_with_path, tree = jax.tree_util.tree_flatten_with_path(jax_params)
    converted = []
    missing = []

    for path, reference in leaves_with_path:
        name = ".".join(entry.key for entry in path)
        torch_name = re.sub(r"\.scale|\.kernel", ".weight", name)
        if torch_name not in renamed:
            missing.append(torch_name)
            converted.append(reference)
            continue

        value = renamed[torch_name]
        if "grn" in torch_name:
            value = value.reshape(reference.shape)
        elif reference.ndim == 4:
            value = torch.permute(value, (2, 3, 1, 0))
        elif reference.ndim > 1:
            value = torch.permute(value, tuple(reversed(range(reference.ndim))))
        value = jnp.asarray(value.detach().cpu().numpy())
        if value.shape != reference.shape:
            raise ValueError(
                f"ConvNeXt parameter {torch_name} has shape {value.shape}, "
                f"expected {reference.shape}"
            )
        converted.append(value)

    if missing:
        raise ValueError(f"Missing ConvNeXt parameters: {missing}")
    log_for_0("ConvNeXt conversion done.")
    return jax.tree_util.tree_unflatten(tree, converted)


def load_convnext_jax_model():
    model = ConvNextBase()
    params = model.init(jax.random.key(0), jnp.ones((1, 224, 224, 3)))

    from transformers import ConvNextV2ForImageClassification

    torch_model = ConvNextV2ForImageClassification.from_pretrained(
        "facebook/convnextv2-base-22k-224"
    )
    params = convert_hf_weights(params, torch_model.state_dict())
    return model, params
