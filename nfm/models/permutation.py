import jax.numpy as jnp


class PermutationFlip:
    """Permutation and flip for images."""

    def __init__(self, flip: bool = True):
        self.flip = flip

    def __call__(self, x: jnp.ndarray):
        if self.flip:
            x = jnp.flip(x, axis=1)
        return x
