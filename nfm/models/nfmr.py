# Adapted from BiFlow; see LICENSE and NOTICE.
import math
from typing import Any

import jax
import jax.numpy as jnp
from flax import linen as nn

from nfm.models.attention import CausalRoPEAttention, precompute_rope_freqs
from nfm.models.permutation import PermutationFlip
from nfm.models.layers import (
    LabelEmbedder,
    RMSNorm,
    SwiGLUMlp,
    TimestepEmbedder,
    at_least_fp32,
)

INV_SOFTPLUS_ONE = math.log(math.expm1(1.0))


def _expand_time(time, value):
    return time.reshape((time.shape[0],) + (1,) * (value.ndim - 1))


########################################################################
# NFMR time-conditioned autoregressive model                                                 #
########################################################################
class NFMRInContextLayer(nn.Module):
    channels: int
    param_dtype: Any = jnp.float32
    dtype: Any = jnp.float32
    mlp_ratio: float = 8 / 3
    weight_init_constant: float = 0.32
    attn_impl: str = "causal_xla"

    def setup(self):
        assert self.channels % 64 == 0, (
            f"channels ({self.channels}) must be divisible by head_dim 64"
        )
        self.num_heads = self.channels // 64
        dtype_kwargs = {"param_dtype": self.param_dtype, "dtype": self.dtype}
        self.norm1 = RMSNorm(self.channels)
        self.attn = CausalRoPEAttention(
            channels=self.channels,
            weight_init_constant=self.weight_init_constant,
            attn_impl=self.attn_impl,
            **dtype_kwargs,
        )
        self.norm2 = RMSNorm(self.channels)
        self.mlp = SwiGLUMlp(
            self.channels,
            int(self.channels * self.mlp_ratio),
            weight_init_constant=self.weight_init_constant,
            **dtype_kwargs,
        )
        self.attn_scale = self.param(
            "attn_scale",
            nn.initializers.zeros,
            (self.channels,),
            self.param_dtype,
        )
        self.mlp_scale = self.param(
            "mlp_scale",
            nn.initializers.zeros,
            (self.channels,),
            self.param_dtype,
        )

    def _residual(self, x, attn):
        x = x + attn.astype(x.dtype) * self.attn_scale.astype(x.dtype)
        mlp = self.mlp(self.norm2(x))
        return x + mlp.astype(x.dtype) * self.mlp_scale.astype(x.dtype)

    def __call__(self, x, rope_freqs):
        attn = self.attn(self.norm1(x), rope_freqs)
        return self._residual(x, attn)

    def dual(self, x, x_dot, rope_freqs, *, strategy="padded"):
        normed, normed_dot = jax.jvp(self.norm1, (x,), (x_dot,))
        attn, attn_dot = self.attn.dual(
            normed, normed_dot, rope_freqs, strategy=strategy
        )
        scale = self.attn_scale.astype(x.dtype)
        x = x + attn.astype(x.dtype) * scale
        x_dot = x_dot + attn_dot.astype(x_dot.dtype) * scale

        def mlp_branch(value):
            return self.mlp(self.norm2(value))

        mlp, mlp_dot = jax.jvp(mlp_branch, (x,), (x_dot,))
        scale = self.mlp_scale.astype(x.dtype)
        x = x + mlp.astype(x.dtype) * scale
        x_dot = x_dot + mlp_dot.astype(x_dot.dtype) * scale
        return x, x_dot

    def prefill(self, x, rope_freqs, cache_length):
        attn, k_cache, v_cache = self.attn.prefill(
            self.norm1(x), rope_freqs, cache_length
        )
        return self._residual(x, attn), k_cache, v_cache

    def decode_step(self, token, rope_freq, position, k_cache, v_cache):
        attn, k_cache, v_cache = self.attn.decode_step(
            self.norm1(token), rope_freq, position, k_cache, v_cache
        )
        return self._residual(token, attn), k_cache, v_cache


class FinalLayer(nn.Module):
    hidden_dim: int
    out_channels: int
    param_dtype: Any = jnp.float32
    dtype: Any = jnp.float32

    def setup(self):
        self.norm = RMSNorm(self.hidden_dim)
        self.proj_out = nn.Dense(
            2 * self.out_channels,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )

    def __call__(self, x):
        return self.proj_out(self.norm(x))


class NFMRMetaBlock(nn.Module):
    in_channels: int
    channels: int
    num_patches: int
    num_layers: int
    num_classes: int
    permutation: PermutationFlip
    param_dtype: Any = jnp.float32
    dtype: Any = jnp.float32
    weight_init_constant: float = 0.32
    num_class_tokens: int = 8
    num_time_tokens: int = 4
    num_cfg_tokens: int = 4
    num_interval_tokens: int = 2
    mlp_ratio: float = 8 / 3
    clip_limit: float = 1.0
    affine_parametrisation: str = "hard_clip_exp"
    attn_impl: str = "causal_xla"

    def setup(self):
        assert self.channels % 64 == 0, (
            f"channels ({self.channels}) must be divisible by head_dim 64"
        )
        self.num_heads = self.channels // 64
        head_dim = 64
        if self.affine_parametrisation not in (
            "hard_clip_exp",
            "soft_clip_softplus",
        ):
            raise ValueError(
                f"Unrecognised affine parametrisation {self.affine_parametrisation!r}"
            )

        dtype_kwargs = {"param_dtype": self.param_dtype, "dtype": self.dtype}
        self.proj_in = nn.Dense(
            features=self.channels,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
        )

        ### Embedders ###
        self.class_embedder = LabelEmbedder(
            self.num_classes, self.channels, **dtype_kwargs
        )
        self.s_embedder = TimestepEmbedder(self.channels, **dtype_kwargs)
        self.t_embedder = TimestepEmbedder(self.channels, **dtype_kwargs)
        self.omega_embedder = TimestepEmbedder(self.channels, **dtype_kwargs)
        self.cfg_t_start_embedder = TimestepEmbedder(self.channels, **dtype_kwargs)
        self.cfg_t_end_embedder = TimestepEmbedder(self.channels, **dtype_kwargs)

        ##### Learned embedding offsets for repeated embeddings #####
        token_initializer = nn.initializers.normal(stddev=1 / math.sqrt(self.channels))
        self.s_time_tokens = self.param(
            "s_time_tokens",
            token_initializer,
            (self.num_time_tokens, self.channels),
            self.param_dtype,
        )
        self.t_time_tokens = self.param(
            "t_time_tokens",
            token_initializer,
            (self.num_time_tokens, self.channels),
            self.param_dtype,
        )
        self.omega_tokens = self.param(
            "omega_tokens",
            token_initializer,
            (self.num_cfg_tokens, self.channels),
            self.param_dtype,
        )
        self.t_min_tokens = self.param(
            "t_min_tokens",
            token_initializer,
            (self.num_interval_tokens, self.channels),
            self.param_dtype,
        )
        self.t_max_tokens = self.param(
            "t_max_tokens",
            token_initializer,
            (self.num_interval_tokens, self.channels),
            self.param_dtype,
        )
        self.class_tokens = self.param(
            "class_tokens",
            token_initializer,
            (self.num_class_tokens, self.channels),
            self.param_dtype,
        )

        self.prefix_tokens = (
            2 * self.num_time_tokens
            + self.num_cfg_tokens
            + 2 * self.num_interval_tokens
            + self.num_class_tokens
        )
        self.total_tokens = self.prefix_tokens + self.num_patches
        self.rope_freqs = precompute_rope_freqs(head_dim, self.total_tokens)
        self.blocks = [
            NFMRInContextLayer(
                channels=self.channels,
                weight_init_constant=self.weight_init_constant,
                mlp_ratio=self.mlp_ratio,
                attn_impl=self.attn_impl,
                **dtype_kwargs,
            )
            for _ in range(self.num_layers)
        ]
        self.proj_out = FinalLayer(self.channels, self.in_channels, **dtype_kwargs)

    # TODO: May be necessary to let this do fp64
    def _build_c_attn(self, s, t, label, w, t_min, t_max):
        s_embed = self.s_embedder(s)
        t_embed = self.t_embedder(t)
        w_work = w.astype(jnp.promote_types(w.dtype, jnp.float32))
        omega_embed = self.omega_embedder(1.0 - jnp.reciprocal(w_work))
        t_min_embed = self.cfg_t_start_embedder(t_min)
        t_max_embed = self.cfg_t_end_embedder(t_max)
        class_embed = self.class_embedder(label)

        condition_tokens = jnp.concatenate(
            [
                self.s_time_tokens[None] + s_embed[:, None],
                self.t_time_tokens[None] + t_embed[:, None],
                self.omega_tokens[None] + omega_embed[:, None],
                self.t_min_tokens[None] + t_min_embed[:, None],
                self.t_max_tokens[None] + t_max_embed[:, None],
                self.class_tokens[None] + class_embed[:, None],
            ],
            axis=1,
        )
        return condition_tokens.astype(jnp.float32)

    def _affine_params(self, raw):
        raw = at_least_fp32(raw, raw.dtype)
        if self.affine_parametrisation == "hard_clip_exp":
            raw = jnp.clip(raw, -self.clip_limit, self.clip_limit)
            return tuple(jnp.split(raw, 2, axis=-1))
        if self.affine_parametrisation == "soft_clip_softplus":
            raw = self.clip_limit * jnp.tanh(raw / self.clip_limit)
            raw_scale, mu = jnp.split(raw, 2, axis=-1)
            scale = jax.nn.softplus(raw_scale + INV_SOFTPLUS_ONE)
            return jnp.log(scale), mu
        raise ValueError(
            f"Unrecognised affine parametrisation {self.affine_parametrisation!r}"
        )

    def _compute_log_scale_mu(self, x, label, s, t, w, t_min, t_max):
        x_in = self.permutation(x)
        patch_tokens = self.proj_in(x_in)
        patch_tokens = at_least_fp32(patch_tokens, patch_tokens.dtype)
        condition_tokens = self._build_c_attn(s, t, label, w, t_min, t_max)
        seq = jnp.concatenate([condition_tokens, patch_tokens], axis=1)

        for layer in self.blocks:
            seq = layer(seq, self.rope_freqs)

        hidden = seq[:, self.prefix_tokens :]
        raw = self.proj_out(hidden).astype(jnp.float32)
        raw = jnp.concatenate([jnp.zeros_like(raw[:, :1]), raw[:, :-1]], axis=1)
        log_scale, mu = self._affine_params(raw)
        return x_in, log_scale, mu

    def __call__(self, x, label, s, t, w, t_min, t_max):
        x_in, log_scale, mu = self._compute_log_scale_mu(
            x, label, s, t, w, t_min, t_max
        )
        # once these outputs pass through the next linear/attn block they will be cast back to bf16
        z = (x_in.astype(log_scale.dtype) - mu) * jnp.exp(-log_scale)
        mean_logdet = -jnp.mean(log_scale, axis=(1, 2))
        return self.permutation(z), mean_logdet

    def dual(
        self,
        x,
        x_dot,
        label,
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
        *,
        strategy="padded",
    ):
        """Return one meta-block primal and combined tangent lane."""
        x_in = self.permutation(x)
        x_in_dot = self.permutation(x_dot)
        patch_tokens, patch_tokens_dot = jax.jvp(self.proj_in, (x_in,), (x_in_dot,))
        patch_tokens = patch_tokens.astype(jnp.float32)
        patch_tokens_dot = patch_tokens_dot.astype(jnp.float32)

        def build_condition(s_value, t_value, w_value, t_min_value, t_max_value):
            return self._build_c_attn(
                s_value,
                t_value,
                label,
                w_value,
                t_min_value,
                t_max_value,
            )

        condition_tokens, condition_tokens_dot = jax.jvp(
            build_condition,
            (s, t, w, t_min, t_max),
            (s_dot, t_dot, w_dot, t_min_dot, t_max_dot),
        )
        seq = jnp.concatenate([condition_tokens, patch_tokens], axis=1)
        seq_dot = jnp.concatenate([condition_tokens_dot, patch_tokens_dot], axis=1)

        for layer in self.blocks:
            seq, seq_dot = layer.dual(seq, seq_dot, self.rope_freqs, strategy=strategy)

        hidden = seq[:, self.prefix_tokens :]
        hidden_dot = seq_dot[:, self.prefix_tokens :]
        raw, raw_dot = jax.jvp(self.proj_out, (hidden,), (hidden_dot,))
        raw = raw.astype(jnp.float32)
        raw_dot = raw_dot.astype(jnp.float32)
        raw = jnp.concatenate([jnp.zeros_like(raw[:, :1]), raw[:, :-1]], axis=1)
        raw_dot = jnp.concatenate(
            [jnp.zeros_like(raw_dot[:, :1]), raw_dot[:, :-1]], axis=1
        )
        (log_scale, mu), (log_scale_dot, mu_dot) = jax.jvp(
            self._affine_params, (raw,), (raw_dot,)
        )
        inverse_scale = jnp.exp(-log_scale)
        z = (x_in.astype(log_scale.dtype) - mu) * inverse_scale
        z_dot = (
            x_in_dot.astype(log_scale.dtype) - mu_dot
        ) * inverse_scale - z * log_scale_dot
        mean_logdet = -jnp.mean(log_scale, axis=(1, 2))
        mean_logdet_dot = -jnp.mean(log_scale_dot, axis=(1, 2))
        return (
            self.permutation(z),
            mean_logdet,
        ), (
            self.permutation(z_dot),
            mean_logdet_dot,
        )

    def _reverse(self, z, label, s, t, w, t_min, t_max):
        reconstructed = self.permutation(z).astype(jnp.float32)
        condition_tokens = self._build_c_attn(s, t, label, w, t_min, t_max)
        condition_rope = self.rope_freqs[: self.prefix_tokens]

        k_caches = []
        v_caches = []
        for layer in self.blocks:
            condition_tokens, k_cache, v_cache = layer.prefill(
                condition_tokens, condition_rope, self.total_tokens
            )
            k_caches.append(k_cache)
            v_caches.append(v_cache)
        k_caches = jnp.stack(k_caches)
        v_caches = jnp.stack(v_caches)

        def decode(carry, patch_index):
            patches, k_cache_stack, v_cache_stack = carry
            token = jax.lax.dynamic_slice_in_dim(patches, patch_index, 1, axis=1)
            token = self.proj_in(token).astype(jnp.float32)
            position = self.prefix_tokens + patch_index
            rope_freq = jax.lax.dynamic_slice_in_dim(
                self.rope_freqs, position, 1, axis=0
            )

            next_k_caches = []
            next_v_caches = []
            for layer_index, layer in enumerate(self.blocks):
                token, layer_k_cache, layer_v_cache = layer.decode_step(
                    token,
                    rope_freq,
                    position,
                    k_cache_stack[layer_index],
                    v_cache_stack[layer_index],
                )
                next_k_caches.append(layer_k_cache)
                next_v_caches.append(layer_v_cache)

            raw = self.proj_out(token).astype(jnp.float32)
            log_scale, mu = self._affine_params(raw)
            z_next = jax.lax.dynamic_slice_in_dim(patches, patch_index + 1, 1, axis=1)
            x_next = z_next * jnp.exp(log_scale) + mu
            patches = jax.lax.dynamic_update_slice_in_dim(
                patches, x_next, patch_index + 1, axis=1
            )
            carry = (
                patches,
                jnp.stack(next_k_caches),
                jnp.stack(next_v_caches),
            )
            return carry, log_scale[:, 0]

        (reconstructed, _, _), step_log_scales = jax.lax.scan(
            decode,
            (reconstructed, k_caches, v_caches),
            jnp.arange(self.num_patches - 1, dtype=jnp.int32),
        )
        step_log_scales = jnp.swapaxes(step_log_scales, 0, 1)
        log_scales = jnp.concatenate(
            [
                jnp.zeros(
                    (reconstructed.shape[0], 1, self.in_channels),
                    dtype=jnp.float32,
                ),
                step_log_scales,
            ],
            axis=1,
        )
        mean_reverse_logdet = jnp.mean(log_scales, axis=(1, 2))
        return self.permutation(reconstructed), mean_reverse_logdet, log_scales


class NFMR(nn.Module):
    img_size: int = 64
    out_channels: int = 3
    channels: int = 384
    patch_size: int = 2
    num_layers: int = 8
    num_blocks: int = 8
    num_classes: int = 1000
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    num_class_tokens: int = 8
    num_time_tokens: int = 4
    num_cfg_tokens: int = 4
    num_interval_tokens: int = 2
    mlp_ratio: float = 8 / 3
    weight_init_constant: float = 0.32
    clip_limit: float = 1.0
    denoiser_min_clip: float = 0.05
    affine_parametrisation: str = "hard_clip_exp"
    attn_impl: str = "causal_xla"

    def setup(self):
        if self.img_size % self.patch_size != 0:
            raise ValueError("img_size must be divisible by patch_size")
        assert self.channels % 64 == 0, (
            f"channels ({self.channels}) must be divisible by head_dim 64"
        )
        self.num_heads = self.channels // 64
        num_patches = (self.img_size // self.patch_size) ** 2
        in_channels = self.out_channels * self.patch_size**2
        self.blocks = [
            NFMRMetaBlock(
                in_channels=in_channels,
                channels=self.channels,
                num_patches=num_patches,
                num_layers=self.num_layers,
                num_classes=self.num_classes,
                permutation=PermutationFlip(i % 2 == 1),
                param_dtype=self.param_dtype,
                dtype=self.dtype,
                weight_init_constant=self.weight_init_constant,
                num_class_tokens=self.num_class_tokens,
                num_time_tokens=self.num_time_tokens,
                num_cfg_tokens=self.num_cfg_tokens,
                num_interval_tokens=self.num_interval_tokens,
                mlp_ratio=self.mlp_ratio,
                clip_limit=self.clip_limit,
                affine_parametrisation=self.affine_parametrisation,
                attn_impl=self.attn_impl,
            )
            for i in range(self.num_blocks)
        ]

    @property
    def num_patches(self):
        return (self.img_size // self.patch_size) ** 2

    @property
    def in_channels(self):
        return self.out_channels * self.patch_size**2

    def patchify(self, x):
        batch, height, width, channels = x.shape
        if (height, width, channels) != (
            self.img_size,
            self.img_size,
            self.out_channels,
        ):
            raise ValueError(
                "Expected NHWC input shape "
                f"(*, {self.img_size}, {self.img_size}, {self.out_channels}), "
                f"got {x.shape}"
            )
        p = self.patch_size
        h = height // p
        w = width // p
        x = x.reshape(batch, h, p, w, p, channels)
        x = x.transpose(0, 1, 3, 5, 2, 4)
        return x.reshape(batch, h * w, channels * p * p)

    def unpatchify(self, x):
        batch = x.shape[0]
        p = self.patch_size
        h = self.img_size // p
        w = self.img_size // p
        expected = (batch, h * w, self.in_channels)
        if x.shape != expected:
            raise ValueError(f"Expected patch shape {expected}, got {x.shape}")
        x = x.reshape(batch, h, w, self.out_channels, p, p)
        x = x.transpose(0, 1, 4, 2, 5, 3)
        return x.reshape(batch, self.img_size, self.img_size, self.out_channels)

    def initialize(self, x, label, s, t, w, t_min, t_max):
        return self.__call__(x, label, s, t, w, t_min, t_max)

    def __call__(self, x, label, s, t, w, t_min, t_max):
        x = self.patchify(x).astype(jnp.float32)
        total_mean_logdet = jnp.zeros((x.shape[0],), dtype=jnp.float32)
        for block in self.blocks:
            x, block_mean_logdet = block(x, label, s, t, w, t_min, t_max)
            total_mean_logdet += block_mean_logdet.astype(jnp.float32)
        return x.astype(jnp.float32), total_mean_logdet

    def dual(
        self,
        x,
        x_dot,
        label,
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
        *,
        strategy="padded",
    ):
        """Return the model primal and one combined tangent lane."""
        x = self.patchify(x).astype(jnp.float32)
        x_dot = self.patchify(x_dot).astype(jnp.float32)
        total_mean_logdet = jnp.zeros((x.shape[0],), dtype=jnp.float32)
        total_mean_logdet_dot = jnp.zeros((x.shape[0],), dtype=jnp.float32)

        for block in self.blocks:
            (x, block_logdet), (x_dot, block_logdet_dot) = block.dual(
                x,
                x_dot,
                label,
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
                strategy=strategy,
            )
            total_mean_logdet += block_logdet.astype(jnp.float32)
            total_mean_logdet_dot += block_logdet_dot.astype(jnp.float32)

        return (
            x.astype(jnp.float32),
            total_mean_logdet,
        ), (
            x_dot.astype(jnp.float32),
            total_mean_logdet_dot,
        )

    def delta_st(self, x, label, s, t, w, t_min, t_max):
        patches, _mean_logdet = self(x, label, s, t, w, t_min, t_max)
        return self.unpatchify(patches)

    def delta_st_dual(
        self,
        x,
        x_dot,
        label,
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
        *,
        strategy="padded",
    ):
        (patches, _mean_logdet), (patches_dot, _mean_logdet_dot) = self.dual(
            x,
            x_dot,
            label,
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
            strategy=strategy,
        )
        return self.unpatchify(patches), self.unpatchify(patches_dot)

    def denoiser_velocity(self, delta, x, s):
        denominator = jnp.maximum(1.0 - _expand_time(s, x), self.denoiser_min_clip)
        return (delta - x) / denominator

    def v_st(self, x, label, s, t, w, t_min, t_max):
        delta = self.delta_st(x, label, s, t, w, t_min, t_max)
        return self.denoiser_velocity(delta, x, s)

    def v_t(self, x, label, t, w, t_min, t_max):
        return self.v_st(x, label, t, t, w, t_min, t_max)

    def cond_and_uncond_v_t(self, x, t, label, w, num_classes, expand_batch=True):
        batch_size = x.shape[0]
        null_label = jnp.full_like(label, num_classes)

        if expand_batch:
            x_both = jnp.concatenate((x, x), axis=0)
            label_both = jnp.concatenate((label, null_label), axis=0)
            t_both = jnp.concatenate((t, t), axis=0)
            w_both = jnp.concatenate((w, jnp.ones_like(w)), axis=0)
            t_min = jnp.zeros_like(t_both)
            t_max = jnp.ones_like(t_both)
            output = self.v_t(x_both, label_both, t_both, w_both, t_min, t_max)
            return tuple(jnp.split(output, 2, axis=0))

        t_min = jnp.zeros((batch_size,), dtype=t.dtype)
        t_max = jnp.ones((batch_size,), dtype=t.dtype)
        v_cond = self.v_t(x, label, t, w, t_min, t_max)
        v_uncond = self.v_t(x, null_label, t, jnp.ones_like(w), t_min, t_max)
        return v_cond, v_uncond


def _reference_block(model, block_index, inverse_dtype=jnp.float32):
    reference_attn = (
        model.attn_impl if model.attn_impl in ("causal_xla", "einsum") else "causal_xla"
    )
    return NFMRMetaBlock(
        in_channels=model.in_channels,
        channels=model.channels,
        num_patches=model.num_patches,
        num_layers=model.num_layers,
        num_classes=model.num_classes,
        permutation=PermutationFlip(block_index % 2 == 1),
        param_dtype=inverse_dtype,
        dtype=inverse_dtype,
        weight_init_constant=model.weight_init_constant,
        num_class_tokens=model.num_class_tokens,
        num_time_tokens=model.num_time_tokens,
        num_cfg_tokens=model.num_cfg_tokens,
        num_interval_tokens=model.num_interval_tokens,
        mlp_ratio=model.mlp_ratio,
        clip_limit=model.clip_limit,
        affine_parametrisation=model.affine_parametrisation,
        attn_impl=reference_attn,
    )


def reverse_block(params, block, z, label, s, t, w, t_min, t_max):
    """Invert one meta-block with prefix prefill and functional KV caches."""
    return block.apply(
        params,
        z,
        label,
        s,
        t,
        w,
        t_min,
        t_max,
        method=block._reverse,
    )


def reverse(
    params,
    model,
    z,
    label,
    s,
    t,
    w,
    t_min,
    t_max,
    inverse_dtype=jnp.float32,
):
    """Invert patch-space latents using FP32/FP64 reference XLA attention."""
    inverse_dtype = jnp.dtype(inverse_dtype)
    if inverse_dtype not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.float64)):
        raise ValueError(f"Reverse supports FP32 or FP64, got {inverse_dtype}")

    param_tree = params["params"] if "params" in params else params
    x = z.astype(inverse_dtype)
    total_mean_reverse_logdet = jnp.zeros((x.shape[0],), dtype=inverse_dtype)
    for block_index in reversed(range(model.num_blocks)):
        block = _reference_block(model, block_index, inverse_dtype)
        x, block_mean_reverse_logdet, _ = reverse_block(
            {"params": param_tree[f"blocks_{block_index}"]},
            block,
            x,
            label,
            s,
            t,
            w,
            t_min,
            t_max,
        )
        total_mean_reverse_logdet += block_mean_reverse_logdet.astype(inverse_dtype)
    return model.unpatchify(x), total_mean_reverse_logdet


def left_inverse_metrics(
    params,
    model,
    noise,
    label,
    s,
    t,
    w,
    t_min,
    t_max,
    inverse_dtype=jnp.float32,
):
    """Measure a reference forward-then-inverse round trip."""
    inverse_dtype = jnp.dtype(inverse_dtype)
    reference_model = model.clone(
        dtype=inverse_dtype,
        param_dtype=inverse_dtype,
        attn_impl="causal_xla",
    )
    reference_params = jax.tree.map(
        lambda value: (
            value.astype(inverse_dtype)
            if jnp.issubdtype(value.dtype, jnp.floating)
            else value
        ),
        params,
    )
    noise = noise.astype(inverse_dtype)
    patches, _ = reference_model.apply(
        reference_params, noise, label, s, t, w, t_min, t_max
    )
    generated = reference_model.unpatchify(patches)
    predicted_noise, _ = reverse(
        reference_params,
        reference_model,
        reference_model.patchify(generated),
        label,
        s,
        t,
        w,
        t_min,
        t_max,
        inverse_dtype=inverse_dtype,
    )
    error = predicted_noise - noise
    return (
        jnp.mean(jnp.square(error), dtype=inverse_dtype),
        jnp.max(jnp.abs(error)),
    )


def left_inverse_mse(
    params, model, noise, label, s, t, w, t_min, t_max, inverse_dtype=jnp.float32
):
    return left_inverse_metrics(
        params, model, noise, label, s, t, w, t_min, t_max, inverse_dtype
    )[0]


def fp32_left_inverse_mse(params, model, noise, label, s, t, w, t_min, t_max):
    return left_inverse_mse(
        params, model, noise, label, s, t, w, t_min, t_max, jnp.float32
    )


def mean_flow_generate(
    params,
    model,
    rng,
    n_sample,
    image_idx=None,
    cfg_w=1.0,
    cfg_t_min=0.0,
    cfg_t_max=1.0,
):
    rng_noise, rng_label = jax.random.split(rng)
    x0 = jax.random.normal(
        rng_noise,
        (n_sample, model.img_size, model.img_size, model.out_channels),
        dtype=jnp.float32,
    )
    if image_idx is None:
        label = jax.random.randint(rng_label, (n_sample,), 0, model.num_classes)
    else:
        label = (
            jnp.arange(n_sample, dtype=jnp.int32) + image_idx * n_sample
        ) % model.num_classes

    s = jnp.zeros((n_sample,), dtype=jnp.float32)
    t = jnp.ones((n_sample,), dtype=jnp.float32)
    w = jnp.full((n_sample,), cfg_w, dtype=jnp.float32)
    t_min = jnp.full((n_sample,), cfg_t_min, dtype=jnp.float32)
    t_max = jnp.full((n_sample,), cfg_t_max, dtype=jnp.float32)
    return model.apply(
        params,
        x0,
        label,
        s,
        t,
        w,
        t_min,
        t_max,
        method=model.delta_st,
    )


def flow_matching_generate(
    params,
    model,
    rng,
    n_sample,
    image_idx=None,
    n_steps=50,
    cfg_w=1.0,
    cfg_t_min=0.0,
    cfg_t_max=1.0,
    expand_batch_for_cfg=True,
):
    """Euler integration of the diagonal velocity, with external interval CFG.

    The model is conditioned at w=1, as during FM training. cfg_w may be a
    traced scalar; the step count, interval and batching strategy are static.
    """
    if n_steps < 1:
        raise ValueError("FM sampling requires n_steps >= 1")
    if not 0.0 <= cfg_t_min <= cfg_t_max <= 1.0:
        raise ValueError("FM sampling requires 0 <= cfg_t_min <= cfg_t_max <= 1")
    rng_noise, rng_label = jax.random.split(rng)
    x0 = jax.random.normal(
        rng_noise,
        (n_sample, model.img_size, model.img_size, model.out_channels),
        dtype=jnp.float32,
    )
    if image_idx is None:
        label = jax.random.randint(rng_label, (n_sample,), 0, model.num_classes)
    else:
        label = (
            jnp.arange(n_sample, dtype=jnp.int32) + image_idx * n_sample
        ) % model.num_classes
    times = jnp.linspace(0.0, 1.0, n_steps + 1, dtype=jnp.float32)
    ones = jnp.ones((n_sample,), dtype=jnp.float32)
    zeros = jnp.zeros_like(ones)

    def step(i, x):
        s = jnp.full((n_sample,), times[i], dtype=jnp.float32)

        def conditional_velocity(x):
            return model.apply(params, x, label, s, ones, zeros, ones, method=model.v_t)

        def guided_velocity(x):
            v_cond, v_uncond = model.apply(
                params,
                x,
                s,
                label,
                ones,
                model.num_classes,
                expand_batch_for_cfg,
                method=model.cond_and_uncond_v_t,
            )
            return v_uncond + cfg_w * (v_cond - v_uncond)

        use_guidance = (
            (cfg_w != 1.0) & (times[i] >= cfg_t_min) & (times[i] <= cfg_t_max)
        )
        velocity = jax.lax.cond(use_guidance, guided_velocity, conditional_velocity, x)
        return x + (times[i + 1] - times[i]) * velocity

    return jax.lax.fori_loop(0, n_steps, step, x0)


############################################################
# iMF/pMF helper funcs.                                    #
############################################################


def mf_adaptive_reweight(loss, norm_eps=0.01, norm_p=1.0):
    weight = jax.lax.stop_gradient(jnp.power(loss + norm_eps, norm_p))
    return loss / weight


def logit_normal_sample(key, batch_size, mean=-0.8, std=0.8):
    normal = jax.random.normal(key, (batch_size,), dtype=jnp.float32)
    return jax.nn.sigmoid(normal * std + mean)


# x1 is data and x0 is noise
def sample_pixel_mean_flow_time(
    key,
    batch_size,
    diagonal_prop=0.5,
    mu=-0.8,
    sigma=0.8,
    st_uniform=False,
):
    rng_s, rng_t, rng_mask, rng_uniform_s, rng_uniform_t = jax.random.split(key, 5)
    s = logit_normal_sample(rng_s, batch_size, mu, sigma)
    t = logit_normal_sample(rng_t, batch_size, mu, sigma)

    if st_uniform:
        uniform_mask = jax.random.bernoulli(rng_mask, 0.1, (batch_size,))
        s = jnp.where(
            uniform_mask,
            jax.random.uniform(rng_uniform_s, (batch_size,), dtype=jnp.float32),
            s,
        )
        t = jnp.where(
            uniform_mask,
            jax.random.uniform(rng_uniform_t, (batch_size,), dtype=jnp.float32),
            t,
        )

    diagonal_count = int(batch_size * diagonal_prop)
    diagonal_mask = jnp.arange(batch_size) < diagonal_count
    s = jnp.where(diagonal_mask, t, s)
    s, t = jnp.minimum(s, t), jnp.maximum(s, t)
    return s, t, diagonal_mask


def sample_pmf_cfg_scale(key, b, cfg_beta, s_max=7.0):
    """
    Sample CFG scale from power distribution.
    """
    u = jax.random.uniform(key, (b,))
    if cfg_beta == 1.0:
        scale = jnp.exp(u * jnp.log1p(jnp.array(s_max, dtype=jnp.float32)))
    else:
        smax = jnp.array(s_max, dtype=jnp.float32)
        beta = jnp.array(cfg_beta, dtype=jnp.float32)

        log_base = (1.0 - beta) * jnp.log1p(smax)
        log_inner = jnp.log1p(u * jnp.expm1(log_base))
        scale = jnp.exp(log_inner / (1.0 - beta))

    return scale


def sample_pmf_cfg_interval(key, b, fm_mask=None):
    """
    Sample CFG interval [t_min, t_max] from uniform distribution.
    """
    key1, key2 = jax.random.split(key)
    t_min = jax.random.uniform(key1, (b,)) * 0.5
    t_max = jax.random.uniform(key2, (b,)) * 0.5 + 0.5

    t_min = jnp.where(fm_mask, 0.0, t_min)
    t_max = jnp.where(fm_mask, 1.0, t_max)

    return t_min, t_max
