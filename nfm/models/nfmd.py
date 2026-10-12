# Adapted from BiFlow; see LICENSE and NOTICE.
"""NFMD: a time-conditioned autoregressive normalizing flow derived from BiFlow."""

import math
from typing import Any
import jax
import jax.numpy as jnp
from flax import linen as nn
from nfm.models.attention import MLP, ConditionalAttention, modulate
from nfm.models.permutation import PermutationFlip
from nfm.models.layers import dense


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding + 2-layer SiLU MLP. -> [B, channels]."""

    channels: int
    max_period: int = 10000
    dtype: Any = jnp.float32

    def setup(self):
        assert self.channels % 2 == 0, "channels must be even for sinusoidal embedding"
        self.linear1 = dense(features=self.channels, dtype=self.dtype)
        self.linear2 = dense(features=self.channels, dtype=self.dtype)

    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        half_dim = self.channels // 2
        freqs = jnp.exp(
            -math.log(self.max_period)
            * jnp.arange(half_dim, dtype=jnp.float32)
            / half_dim
        )
        args = t[:, None].astype(jnp.float32) * freqs[None, :]
        emb = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        emb = self.linear1(emb.astype(self.dtype))
        emb = nn.silu(emb)
        emb = self.linear2(emb)
        return emb


class NFMDInContextLayer(nn.Module):
    """DiT block with in-context conditioning that accepts multiple cond tokens.

    Identical to ``tarflow.InContextLayer`` except ``c`` is passed through with
    its token dimension intact (``[B, T_c, C]``) instead of being reshaped to a
    single token.
    """

    channels: int
    dtype: Any = jnp.float32
    attn_impl: str = "einsum"

    def setup(self):
        self.attn = ConditionalAttention(
            in_channels=self.channels, dtype=self.dtype, attn_impl=self.attn_impl
        )
        self.mlp = MLP(out_channels=self.channels, dtype=self.dtype)

    def __call__(
        self,
        x: jnp.ndarray,
        c: jnp.ndarray,
        k_cache: dict | None = None,
        v_cache: dict | None = None,
        mask: jnp.ndarray | None = None,
        which_cache: str = "cond",
        train: bool = True,
    ):
        attn, k_cache, v_cache = self.attn(
            x,
            mask,
            c,
            k_cache=k_cache,
            v_cache=v_cache,
            which_cache=which_cache,
            train=train,
        )
        x = x + attn.astype(jnp.float32)
        x = x + self.mlp(x).astype(jnp.float32)
        return (x, k_cache, v_cache)


class FinalLayer(nn.Module):
    """Final layer of the meta-block (adaLN + zero-init projection to 2*C)."""

    hidden_dim: int
    out_channels: int
    dtype: Any = jnp.float32

    def setup(self):
        self.norm_final = nn.LayerNorm(
            use_bias=False, use_scale=False, dtype=self.dtype
        )
        self.proj_out = dense(
            features=2 * self.out_channels,
            dtype=self.dtype,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )
        self.adaLN_modulation = nn.Sequential(
            [
                nn.silu,
                dense(
                    2 * self.hidden_dim,
                    dtype=self.dtype,
                    kernel_init=nn.initializers.zeros,
                    bias_init=nn.initializers.zeros,
                ),
            ]
        )

    def __call__(self, x, c):
        shift, scale = jnp.split(self.adaLN_modulation(c), 2, axis=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.proj_out(x)
        return x


class NFMDMetaBlock(nn.Module):
    """Time-conditioned meta block (one flow step)."""

    in_channels: int
    channels: int
    num_patches: int
    num_layers: int
    num_classes: int
    permutation: PermutationFlip
    num_condition_tokens: int = 1
    dtype: Any = jnp.float32
    clip_range: float = None
    attn_impl: str = "einsum"

    def setup(self):
        if self.num_condition_tokens < 1:
            raise ValueError("num_condition_tokens must be positive")
        self.proj_in = dense(features=self.channels, dtype=self.dtype)
        self.pos_embedding = self.param(
            "pos_embedding",
            nn.initializers.normal(stddev=0.02),
            (1, self.num_patches, self.channels),
            jnp.float32,
        )
        self.class_embedding = self.param(
            "class_embedding",
            nn.initializers.normal(stddev=0.02),
            (self.num_classes, self.channels),
            jnp.float32,
        )
        if self.num_condition_tokens > 1:
            token_initializer = nn.initializers.normal(
                stddev=1 / math.sqrt(self.channels)
            )
            self.class_tokens = self.param(
                "class_tokens",
                token_initializer,
                (self.num_condition_tokens, self.channels),
                jnp.float32,
            )
            self.s_time_tokens = self.param(
                "s_time_tokens",
                token_initializer,
                (self.num_condition_tokens, self.channels),
                jnp.float32,
            )
            self.t_time_tokens = self.param(
                "t_time_tokens",
                token_initializer,
                (self.num_condition_tokens, self.channels),
                jnp.float32,
            )
        self.blocks = [
            NFMDInContextLayer(
                channels=self.channels, dtype=self.dtype, attn_impl=self.attn_impl
            )
            for _ in range(self.num_layers)
        ]
        self.proj_out = FinalLayer(
            hidden_dim=self.channels, out_channels=self.in_channels, dtype=self.dtype
        )
        self.attn_mask = lambda: jnp.tril(
            jnp.ones((self.num_patches, self.num_patches), dtype=jnp.bool)
        )

    def _conditioning(self, y, c_s, c_t, B):
        """Build the 3 attention tokens and the summed adaLN vector."""
        if y is not None:
            mask = (y < 0).astype(jnp.float32).reshape(-1, 1)
            c_class = (1 - mask) * self.class_embedding[
                y
            ] + mask * self.class_embedding.mean(axis=0)
        else:
            c_class = self.class_embedding.mean(axis=0).reshape(1, -1).repeat(B, axis=0)
        if self.num_condition_tokens == 1:
            c_attn = jnp.stack([c_class, c_s, c_t], axis=1)
        else:
            c_attn = jnp.concatenate(
                [
                    c_class[:, None] + self.class_tokens[None],
                    c_s[:, None] + self.s_time_tokens[None],
                    c_t[:, None] + self.t_time_tokens[None],
                ],
                axis=1,
            )
        c_adaln = c_class + c_s + c_t
        return (c_attn, c_adaln)

    def __call__(self, x, y, c_s, c_t, train: bool = True, dt=None):
        """x: [B, T, C_in]; y: [B]; c_s, c_t: [B, channels]."""
        x_in = self.permutation(x)
        x = self.proj_in(x_in).astype(jnp.float32)
        x = x + self.permutation(self.pos_embedding)
        c_attn, c_adaln = self._conditioning(y, c_s, c_t, x.shape[0])
        attention_mask = self.attn_mask()
        for block in self.blocks:
            x, _, _ = block(x, c_attn, mask=attention_mask, train=train)
        x = self.proj_out(x, c_adaln)
        x = jnp.concatenate([jnp.zeros_like(x[:, :1]), x[:, :-1]], axis=1)
        x = jnp.clip(x.astype(jnp.float32), -self.clip_range, self.clip_range)
        alpha, mu = jnp.split(x, 2, axis=-1)
        if dt is not None:
            alpha = alpha * dt[:, None, None]
            mu = mu * dt[:, None, None]
        x_new = (x_in - mu) * jnp.exp(-alpha)
        log_jacob = -alpha.mean(axis=(1, 2))
        x_new = self.permutation(x_new)
        return (x_new, log_jacob, alpha, mu)

    def reverse_step(
        self,
        x: jnp.ndarray,
        i: int,
        c_s: jnp.ndarray,
        c_t: jnp.ndarray,
        y: jnp.ndarray | None = None,
        k_cache: dict | None = None,
        v_cache: dict | None = None,
        which_cache: str = "cond",
    ):
        B, T, C = x.shape
        x_in = jax.lax.dynamic_slice(x, (0, i, 0), (x.shape[0], 1, x.shape[2]))
        assert x_in.shape == (B, 1, C)
        x = self.proj_in(x_in).astype(jnp.float32)
        pos_embed = self.permutation(self.pos_embedding)
        x = x + jax.lax.dynamic_slice(
            pos_embed, (0, i, 0), (pos_embed.shape[0], 1, pos_embed.shape[2])
        )
        c_attn, c_adaln = self._conditioning(y, c_s, c_t, B)
        for bi, block in enumerate(self.blocks):
            mask = self.attn_mask()
            mask = jax.lax.dynamic_slice(mask, (i, 0), (1, mask.shape[1]))
            x, k_cache[bi], v_cache[bi] = block(
                x,
                c_attn,
                k_cache=k_cache[bi],
                v_cache=v_cache[bi],
                which_cache=which_cache,
                train=False,
                mask=mask,
            )
        x = self.proj_out(x, c_adaln)
        x = jnp.clip(x.astype(jnp.float32), -self.clip_range, self.clip_range)
        alpha, mu = jnp.split(x, 2, axis=-1)
        return (alpha, mu, k_cache, v_cache)


class NFMD(nn.Module):
    """(s, t)-time-conditioned normalizing flow. __call__ returns (z, logdet)."""

    img_size: int = 32
    out_channels: int = 4
    channels: int = 384
    patch_size: int = 2
    num_layers: int = 8
    num_blocks: int = 8
    num_classes: int = 1000
    num_condition_tokens: int = 1
    dtype: Any = jnp.float32
    clip_range: float = 1.0
    position_embed: str = "learned"
    attn_impl: str = "einsum"
    gated: bool = False

    def setup(self):
        patch_num = self.img_size // self.patch_size
        self.num_patches = patch_num**2
        in_channels = self.out_channels * self.patch_size * self.patch_size
        assert self.patch_size * patch_num == self.img_size, (
            "img_size must be divisible by num_patches"
        )
        self.time_embed_s = TimestepEmbedding(self.channels, dtype=self.dtype)
        self.time_embed_t = TimestepEmbedding(self.channels, dtype=self.dtype)
        self.blocks = [
            NFMDMetaBlock(
                in_channels=in_channels,
                channels=self.channels,
                num_patches=self.num_patches,
                num_layers=self.num_layers,
                num_classes=self.num_classes,
                permutation=PermutationFlip(i % 2 == 1),
                num_condition_tokens=self.num_condition_tokens,
                dtype=self.dtype,
                clip_range=self.clip_range,
                attn_impl=self.attn_impl,
            )
            for i in range(self.num_blocks)
        ]

    def patchify(self, x):
        B, H, W, C = x.shape
        assert C == self.out_channels, "input must be RGB image"
        x = x.reshape(
            B,
            H // self.patch_size,
            self.patch_size,
            W // self.patch_size,
            self.patch_size,
            C,
        )
        x = x.transpose(0, 1, 3, 2, 4, 5)
        x = x.reshape(
            B, (self.img_size // self.patch_size) ** 2,
            self.patch_size * self.patch_size * C,
        )
        return x

    def unpatchify(self, x):
        B, T, C = x.shape
        x = x.reshape(
            B,
            self.img_size // self.patch_size,
            self.img_size // self.patch_size,
            self.patch_size,
            self.patch_size,
            self.out_channels,
        )
        x = x.transpose(0, 1, 3, 2, 4, 5)
        x = x.reshape(B, self.img_size, self.img_size, self.out_channels)
        return x

    def embed_times(self, s, t):
        """Embed the scalar times; used by the module-level reverse path."""
        return (self.time_embed_s(s), self.time_embed_t(t))

    def initialize(self, x, y, s, t):
        self.__call__(x, y, s, t)

    def __call__(self, x, y, s, t):
        """x: [B, H, W, C]; y: [B]; s, t: [B]. Returns (z, total_logdet)."""
        x = self.patchify(x)
        c_s = self.time_embed_s(s)
        c_t = self.time_embed_t(t)
        total_logdet = jnp.zeros((x.shape[0],), dtype=jnp.float32)
        for block in self.blocks:
            x, logdet, _, _ = block(
                x, y, c_s, c_t, dt=(t - s) if self.gated else None
            )
            total_logdet += logdet
        return (x, total_logdet)


def reverse_block(params, block, x, y, c_s, c_t, guidance=0, dt=None):
    """Invert one NFMDMetaBlock autoregressively. c_s, c_t: [B, channels]."""
    x = block.permutation(x.astype(jnp.float32))
    B, T, C = x.shape
    num_heads = max(1, block.channels // 64)
    head_dim = block.channels // num_heads
    k_cache = [
        {
            "cond": jnp.zeros((B, T, num_heads, head_dim), dtype=block.dtype),
            "uncond": jnp.zeros((B, T, num_heads, head_dim), dtype=block.dtype),
            "idx": 0,
        }
        for _ in range(block.num_layers)
    ]
    v_cache = [
        {
            "cond": jnp.zeros((B, T, num_heads, head_dim), dtype=block.dtype),
            "uncond": jnp.zeros((B, T, num_heads, head_dim), dtype=block.dtype),
            "idx": 0,
        }
        for _ in range(block.num_layers)
    ]
    assert T == block.num_patches

    def step_fn(i, vals):
        x_step, k_cache_step, v_cache_step = vals
        for obj in k_cache_step + v_cache_step:
            obj["idx"] = i
        alpha_i, mu_i, k_cache_step, v_cache_step = block.apply(
            params,
            x=x_step,
            i=i,
            c_s=c_s,
            c_t=c_t,
            y=y,
            k_cache=k_cache_step,
            v_cache=v_cache_step,
            which_cache="cond",
            method=block.reverse_step,
        )
        alpha_i_u, mu_i_u, k_cache_step, v_cache_step = block.apply(
            params,
            x=x_step,
            i=i,
            c_s=c_s,
            c_t=c_t,
            y=None,
            k_cache=k_cache_step,
            v_cache=v_cache_step,
            which_cache="uncond",
            method=block.reverse_step,
        )
        w = guidance * (i + 1) / (T - 1)
        alpha_i = (1 + w) * alpha_i - w * alpha_i_u
        mu_i = (1 + w) * mu_i - w * mu_i_u
        if dt is not None:
            alpha_i = alpha_i * dt[:, None, None]
            mu_i = mu_i * dt[:, None, None]
        x_sliced = jax.lax.dynamic_slice(
            x_step, (0, i + 1, 0), (x_step.shape[0], 1, x_step.shape[2])
        )
        val = x_sliced * jnp.exp(alpha_i) + mu_i
        x_step = jax.lax.dynamic_update_slice(x_step, val, (0, i + 1, 0))
        return (x_step, k_cache_step, v_cache_step)

    x, _, _ = jax.lax.fori_loop(0, T - 1, step_fn, (x, k_cache, v_cache))
    return block.permutation(x)


def reverse(params, nf, x, y, s, t, guidance=0):
    """Invert the whole flow: z (patch space) -> x (image space)."""
    patch_num = nf.img_size // nf.patch_size
    num_patches = patch_num**2
    in_channels = nf.out_channels * nf.patch_size * nf.patch_size
    c_s, c_t = nf.apply(params, s, t, method=nf.embed_times)
    for i in range(nf.num_blocks - 1, -1, -1):
        block_param = params["params"][f"blocks_{i}"]
        x = reverse_block(
            {"params": block_param},
            NFMDMetaBlock(
                in_channels=in_channels,
                channels=nf.channels,
                num_patches=num_patches,
                num_layers=nf.num_layers,
                num_classes=nf.num_classes,
                permutation=PermutationFlip(i % 2 == 1),
                num_condition_tokens=nf.num_condition_tokens,
                dtype=nf.dtype,
                clip_range=nf.clip_range,
                attn_impl=nf.attn_impl,
            ),
            x,
            y,
            c_s,
            c_t,
            guidance=guidance,
            dt=(t - s) if nf.gated else None,
        )
    x = nf.unpatchify(x)
    return x


def generate(
    params, model, rng, n_sample, noise_level, guidance, image_idx=None,
    split=None, first_guidance=0.0, first_step_unconditional=False,
):
    """Invert f(0,1), or a two-interval partition with shared weights. No denoising.

    ``noise_level`` is accepted for signature parity with the MLE sampler but is
    unused (NFMD has no score-based denoise step).
    """
    patch_num = model.img_size // model.patch_size
    num_patches = patch_num**2
    in_channels = model.out_channels * model.patch_size * model.patch_size
    x_shape = (n_sample, num_patches, in_channels)
    rng_used, rng = jax.random.split(rng, 2)
    rng_used_2, rng = jax.random.split(rng, 2)
    z = jax.random.normal(rng_used, x_shape, dtype=jnp.float32)
    if image_idx is None:
        y = jax.random.randint(rng_used_2, (n_sample,), 0, model.num_classes)
    else:
        all_y = jnp.arange(n_sample, dtype=jnp.int32)
        y = (all_y + image_idx * n_sample) % model.num_classes
    s = jnp.zeros((n_sample,), dtype=jnp.float32)
    t = jnp.ones((n_sample,), dtype=jnp.float32)
    if split is not None:
        middle = jnp.full_like(s, split)
        first_label = -jnp.ones_like(y) if first_step_unconditional else y
        intermediate = reverse(
            params, model, z, first_label, middle, t, guidance=first_guidance
        )
        return reverse(
            params, model, model.patchify(intermediate), y, s, middle,
            guidance=guidance,
        )
    x = reverse(params, model, z, y, s, t, guidance=guidance)
    return x
