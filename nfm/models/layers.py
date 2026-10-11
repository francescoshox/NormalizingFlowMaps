# Adapted from BiFlow; see LICENSE and NOTICE.
import math
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from flax import linen as nn


dense = partial(
    nn.Dense,
    param_dtype=jnp.float32,
    kernel_init=nn.initializers.xavier_uniform(),
    bias_init=nn.initializers.zeros,
)


def at_least_fp32(x: jnp.array, dtype: Any):
    return x.astype(jnp.promote_types(dtype, jnp.float32))


class RMSNorm(nn.Module):
    """Root Mean Square Normalization."""

    hidden_size: int
    eps: float = 1e-6

    def setup(self):  # fp32 weights by default
        self.weight = self.param("kernel", nn.initializers.ones, (self.hidden_size,))

    def _norm(self, x):
        return x * jnp.reciprocal(
            jnp.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + self.eps)
        )

    def __call__(self, x):
        orig_dtype = x.dtype
        x = at_least_fp32(x, orig_dtype)
        output = self._norm(x)
        return (output * self.weight).astype(orig_dtype)


########################################################################
# Shared model layers                                            #
########################################################################


# https://github.com/Lyy-iiis/imeanflow/blob/main/models/torch_models.py
class ScaledLinear(nn.Module):
    in_features: int
    out_features: int
    bias: bool = True
    init_constant: float = 1.0
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    def setup(self):
        std = self.init_constant / math.sqrt(self.in_features)
        weight_initialiser = nn.initializers.normal(stddev=std)

        self._flax_linear = nn.Dense(
            features=self.out_features,
            use_bias=self.bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=weight_initialiser,
            bias_init=nn.initializers.zeros,
        )

    def __call__(self, x):
        return self._flax_linear(x)


class ClassEmbedding(nn.Module):
    num_embeddings: int
    embedding_dim: int
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    def setup(self):
        std = 1 / math.sqrt(self.embedding_dim)
        self.embedding_table = nn.Embed(
            num_embeddings=self.num_embeddings,
            features=self.embedding_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            embedding_init=lambda key, shape, dtype: (
                jax.random.normal(key, shape, dtype) * std
            ),
        )

    def __call__(self, x):
        return self.embedding_table(x)


# Sinusoidal time embedding
class TimestepEmbedder(nn.Module):
    """Embeds a scalar timestep (or scalar conditioning) into a vector. Everything done in at least FP32 and returned as the promoted dtype."""

    hidden_size: int
    frequency_embedding_size: int = 256
    init_constant: float = 1.0
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    def setup(self):
        init_kwargs = {
            "out_features": self.hidden_size,
            "bias": True,
            "init_constant": self.init_constant,
            "param_dtype": self.param_dtype,
            "dtype": self.dtype,
        }
        self.mlp = nn.Sequential(
            [
                ScaledLinear(self.frequency_embedding_size, **init_kwargs),
                nn.silu,
                ScaledLinear(self.hidden_size, **init_kwargs),
            ]
        )

    @staticmethod
    def timestep_embedding(t, dim, param_dtype=jnp.float32, max_period=10000):
        """Create sinusoidal timestep embeddings."""
        half = dim // 2
        freqs = jnp.exp(
            -math.log(max_period)
            * jnp.arange(start=0, stop=half, dtype=param_dtype)
            / half
        )
        args = at_least_fp32(t[:, None], t.dtype) * freqs[None]
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        if dim % 2:
            embedding = jnp.concatenate(
                [embedding, jnp.zeros_like(embedding[:, :1])], axis=-1
            )
        return embedding

    def __call__(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class LabelEmbedder(nn.Module):
    num_classes: int
    channels: int
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    def setup(self):
        self.embedding_table = ClassEmbedding(
            self.num_classes + 1,  # dedicated null embedding
            self.channels,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

    def __call__(self, labels):
        return self.embedding_table(labels)


class SwiGLUMlp(nn.Module):
    in_features: int
    hidden_features: int
    weight_init_constant: float
    param_dtype: Any = jnp.float32
    dtype: Any = jnp.float32

    def setup(self):

        init_kwargs = {
            "bias": False,
            "init_constant": self.weight_init_constant,
            "param_dtype": self.param_dtype,
            "dtype": self.dtype,
        }
        self.w1 = ScaledLinear(
            self.in_features,
            self.hidden_features,
            **init_kwargs,
        )
        self.w3 = ScaledLinear(
            self.in_features,
            self.hidden_features,
            **init_kwargs,
        )
        self.w2 = ScaledLinear(
            self.hidden_features,
            self.in_features,
            **init_kwargs,
        )

    def __call__(self, x):
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))
