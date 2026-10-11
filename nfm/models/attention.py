# Adapted from BiFlow; see LICENSE and NOTICE.
from typing import Any
import jax
import jax.numpy as jnp
from flax import linen as nn
from nfm.models.layers import RMSNorm, ScaledLinear, dense

ATTN_IMPLS = ["cudnn", "causal_xla", "einsum", "naive_xla"]


def unsqueeze(t, dim):
    return jnp.expand_dims(t, axis=dim)


def modulate(x, shift, scale):
    """do an affine transformation on x"""
    return x * (1 + unsqueeze(scale, 1)) + unsqueeze(shift, 1)


def einsum_attention(q, k, v, mask, dtype=jnp.float32):
    """
    Compute the attention weights and output.
    sanity check with nn.dot_product_attention
    qkv shape: [B, T, num_head, head_dim]
    """
    head_dim = q.shape[-1]
    dot_product = jnp.einsum(
        "...qhd,...khd->...hqk", q, k, preferred_element_type=jnp.float32
    )
    dot_product /= jnp.sqrt(jnp.asarray(head_dim, dtype=jnp.float32))
    big_neg = jnp.finfo(jnp.float32).min
    dot_product = jnp.where(mask, dot_product, big_neg)
    attn_weights = nn.softmax(dot_product, axis=-1).astype(dtype)
    output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, v)
    return output


def naive_xla_attention(q, k, v, mask, dtype=jnp.bfloat16):
    """Fully materialised attention with QK^T computed in dtype (bf16 by default), cast to float32 for the softmax and the final atten_weights@V computed in dtype."""
    head_dim = q.shape[-1]
    q = q / jnp.sqrt(head_dim).astype(dtype)
    q = jnp.swapaxes(q, -3, -2)
    k = jnp.swapaxes(k, -3, -2)
    dot_product = jnp.matmul(q, jnp.swapaxes(k, -1, -2))
    dot_product = dot_product.astype(jnp.float32)
    dot_product = jnp.where(mask, dot_product, jnp.finfo(jnp.float32).min)
    attn_weights = nn.softmax(dot_product, axis=-1).astype(dtype)
    output = jnp.matmul(attn_weights, jnp.swapaxes(v, -3, -2))
    return jnp.swapaxes(output, -3, -2)


class ConditionalAttention(nn.Module):
    """Attention mechanism."""

    in_channels: int
    dtype: Any = jnp.float32
    attn_impl: str = "einsum"

    def setup(self):
        if self.attn_impl not in ATTN_IMPLS:
            raise ValueError(
                f"Unknown attn_impl {self.attn_impl!r}; expected one of {ATTN_IMPLS}"
            )
        self.num_heads = max(1, self.in_channels // 64)
        self.head_dim = self.in_channels // self.num_heads
        assert self.head_dim * self.num_heads == self.in_channels, (
            "in_channels must be divisible by num_heads"
        )
        self.qkv = dense(features=3 * self.in_channels, dtype=self.dtype)
        self.proj = dense(features=self.in_channels, dtype=self.dtype)
        self.norm = nn.LayerNorm(
            dtype=self.dtype, param_dtype=jnp.float32, force_float32_reductions=True
        )

    def __call__(
        self,
        x: jnp.ndarray,
        mask,
        c: jnp.ndarray,
        k_cache: dict | None = None,
        v_cache: dict | None = None,
        which_cache: str = "cond",
        train: bool = True,
    ):
        num_heads = self.num_heads
        assert c is not None, "c must be provided for ConditionalAttention"
        if mask is None:
            raise ValueError("this attention path requires an attention mask")
        B, T, C = x.shape
        B_c, T_c, C_c = c.shape
        assert C == C_c and B == B_c, (
            "x and c must have the same number of channels and batch size"
        )
        if train and self.attn_impl in ["cudnn", "causal_xla"]:
            if self.attn_impl in ("cudnn",) and self.dtype not in (jnp.bfloat16,):
                raise ValueError(
                    f"attn_impl='{self.attn_impl}' requires bf16 compute (training precision 'bf16-mixed'), got dtype={self.dtype}"
                )
            assert mask.shape == (T, T), (
                f"fused attention expects the (T, T) causal training mask, got {mask.shape}"
            )
            assert k_cache is None and v_cache is None, (
                "k_cache and v_cache must be None during training"
            )
            S = T_c + T - 1
            z = jnp.concatenate([c, x[:, :-1]], axis=1)
            z = self.norm(z)
            qkv = self.qkv(z)
            q, k, v = jnp.split(qkv, 3, axis=-1)
            q = q.reshape(B, S, num_heads, self.head_dim)
            k = k.reshape(B, S, num_heads, self.head_dim)
            v = v.reshape(B, S, num_heads, self.head_dim)
            if S % 2 == 1 and self.attn_impl in ("cudnn", "causal_xla"):
                pad = ((0, 0), (0, 1), (0, 0), (0, 0))
                q = jnp.pad(q, pad)
                k = jnp.pad(k, pad)
                v = jnp.pad(v, pad)
            out = jax.nn.dot_product_attention(
                q,
                k,
                v,
                is_causal=True,
                implementation="cudnn" if self.attn_impl == "cudnn" else "xla",
            )
            attn = out[:, T_c:S]
            attn = jnp.concatenate([attn, jnp.zeros_like(attn[:, :1])], axis=1)
            attn = attn.reshape(B, T, C)
            attn = self.proj(attn)
            return (attn, k_cache, v_cache)
        x = jnp.concatenate([x, c], axis=1)
        x = self.norm(x)
        qkv = self.qkv(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q = q.reshape(B, T + T_c, num_heads, self.head_dim)
        k = k.reshape(B, T + T_c, num_heads, self.head_dim)
        v = v.reshape(B, T + T_c, num_heads, self.head_dim)
        q = q[:, :-T_c]
        if train:
            assert k_cache is None and v_cache is None, (
                "k_cache and v_cache must be None during training"
            )
        else:
            k_c, v_c = (k[:, -T_c:], v[:, -T_c:])
            k, v = (k[:, :-T_c], v[:, :-T_c])
            k_idx = k_cache["idx"]
            v_idx = v_cache["idx"]
            k = jax.lax.dynamic_update_slice(k_cache[which_cache], k, (0, k_idx, 0, 0))
            v = jax.lax.dynamic_update_slice(v_cache[which_cache], v, (0, v_idx, 0, 0))
            k_cache[which_cache] = k
            v_cache[which_cache] = v
            k = jnp.concatenate([k, k_c], axis=1)
            v = jnp.concatenate([v, v_c], axis=1)
        mask = jnp.concatenate(
            [mask, jnp.ones((mask.shape[0], T_c), dtype=mask.dtype)], axis=1
        )
        attention_fn = (
            naive_xla_attention if self.attn_impl == "naive_xla" else einsum_attention
        )
        attn = attention_fn(q, k, v, mask=mask, dtype=self.dtype).reshape(B, T, C)
        attn = self.proj(attn)
        return (attn, k_cache, v_cache)


class MLP(nn.Module):
    """MLP."""

    out_channels: int
    dtype: Any = jnp.float32

    def setup(self):
        self.norm = nn.LayerNorm(
            dtype=self.dtype, param_dtype=jnp.float32, force_float32_reductions=True
        )
        self.net1 = dense(features=4 * self.out_channels, dtype=self.dtype)
        self.net2 = dense(features=self.out_channels, dtype=self.dtype)

    def __call__(self, x: jnp.ndarray):
        x = self.norm(x)
        x = self.net1(x)
        x = jax.nn.gelu(x, approximate=False)
        x = self.net2(x)
        return x


def precompute_rope_freqs(
    dim: int, seq_len: int, theta: float = 10000.0, real_dtype: Any = jnp.float32
):
    if dim % 2 != 0:
        raise ValueError(f"RoPE head dimension must be even, got {dim}")
    real_dtype = jnp.dtype(real_dtype)
    if real_dtype not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.float64)):
        raise ValueError(
            f"RoPE real dtype must be float32 or float64, got {real_dtype}"
        )
    freqs = 1.0 / theta ** (jnp.arange(0, dim, 2, dtype=real_dtype) / dim)
    positions = jnp.arange(seq_len, dtype=real_dtype)
    freqs_cis = jnp.outer(positions, freqs)
    real = jnp.cos(freqs_cis)
    imag = jnp.sin(freqs_cis)
    return jax.lax.complex(real, imag)


def apply_rotary_pos_emb(x, freqs_cis):
    """Apply RoPE in at least FP32 and return the input dtype."""
    orig_dtype = x.dtype
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"RoPE head dimension must be even, got {x.shape[-1]}")
    real_dtype = jnp.float64 if orig_dtype == jnp.float64 else jnp.float32
    pairs = x.astype(real_dtype).reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    x_complex = jax.lax.complex(pairs[..., 0], pairs[..., 1])
    freqs = freqs_cis.astype(x_complex.dtype)[None, :, None, :]
    rotated = x_complex * freqs
    out = jnp.stack([jnp.real(rotated), jnp.imag(rotated)], axis=-1).reshape(x.shape)
    return out.astype(orig_dtype)


class CausalRoPEAttention(nn.Module):
    """Causal self-attention with focused full, prefill, and decode APIs."""

    channels: int
    weight_init_constant: float = 1.0
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    attn_impl: str = "causal_xla"

    def setup(self):
        assert self.channels % 64 == 0, (
            f"channels ({self.channels}) must be divisible by head_dim 64"
        )
        self.head_dim = 64
        self.num_heads = self.channels // 64
        init_kwargs = {
            "in_features": self.channels,
            "out_features": self.channels,
            "bias": False,
            "init_constant": self.weight_init_constant,
            "dtype": self.dtype,
            "param_dtype": self.param_dtype,
        }
        self.q_proj = ScaledLinear(**init_kwargs)
        self.k_proj = ScaledLinear(**init_kwargs)
        self.v_proj = ScaledLinear(**init_kwargs)
        self.out_proj = ScaledLinear(**init_kwargs)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def _project_qkv(self, seq, rope_freqs):
        batch, seq_len, _ = seq.shape
        shape = (batch, seq_len, self.num_heads, self.head_dim)
        q = self.q_norm(self.q_proj(seq).reshape(shape))
        k = self.k_norm(self.k_proj(seq).reshape(shape))
        v = self.v_proj(seq).reshape(shape)
        q = apply_rotary_pos_emb(q, rope_freqs)
        k = apply_rotary_pos_emb(k, rope_freqs)
        return (q, k, v)

    def _reference_attention(self, q, k, v, *, is_causal, mask=None):
        if self.attn_impl == "einsum":
            if mask is None:
                if not is_causal:
                    raise ValueError("Non-causal einsum attention requires a mask")
                query_len, key_len = (q.shape[1], k.shape[1])
                mask = jnp.tril(jnp.ones((query_len, key_len), dtype=jnp.bool_))
            return einsum_attention(q, k, v, mask, dtype=self.dtype)
        return jax.nn.dot_product_attention(
            q, k, v, mask=mask, is_causal=is_causal, implementation="xla"
        )

    def __call__(self, seq, rope_freqs):
        batch, seq_len, _ = seq.shape
        q, k, v = self._project_qkv(seq, rope_freqs)
        out = self._reference_attention(q, k, v, is_causal=True)
        out = out.reshape(batch, seq_len, self.channels)
        return self.out_proj(out)

    def dual(self, seq, seq_dot, rope_freqs, *, strategy="padded"):
        """Portable directional derivative, including attention and projections."""
        return jax.jvp(lambda value: self(value, rope_freqs), (seq,), (seq_dot,))

    def prefill(self, seq, rope_freqs, cache_length):
        """Run a causal prefix and place its RoPE-applied K/V into full caches."""
        batch, seq_len, _ = seq.shape
        if cache_length < seq_len:
            raise ValueError(
                f"cache_length ({cache_length}) is shorter than prefix ({seq_len})"
            )
        q, k, v = self._project_qkv(seq, rope_freqs)
        out = self._reference_attention(q, k, v, is_causal=True)
        cache_shape = (batch, cache_length, self.num_heads, self.head_dim)
        k_cache = jnp.zeros(cache_shape, dtype=k.dtype).at[:, :seq_len].set(k)
        v_cache = jnp.zeros(cache_shape, dtype=v.dtype).at[:, :seq_len].set(v)
        out = self.out_proj(out.reshape(batch, seq_len, self.channels))
        return (out, k_cache, v_cache)

    def decode_step(self, token, rope_freq, position, k_cache, v_cache):
        """Decode one token at its global sequence position."""
        batch = token.shape[0]
        q, k, v = self._project_qkv(token, rope_freq)
        zero = jnp.zeros((), dtype=position.dtype)
        start = (zero, position, zero, zero)
        k_cache = jax.lax.dynamic_update_slice(k_cache, k, start)
        v_cache = jax.lax.dynamic_update_slice(v_cache, v, start)
        mask = (jnp.arange(k_cache.shape[1]) <= position)[None, None, None, :]
        out = self._reference_attention(q, k_cache, v_cache, mask=mask, is_causal=False)
        out = self.out_proj(out.reshape(batch, 1, self.channels))
        return (out, k_cache, v_cache)
