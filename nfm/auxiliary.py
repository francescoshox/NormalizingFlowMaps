from typing import Tuple
import jax
import jax.numpy as jnp

log_for_0 = print


def _resample_crop_to_square(image, top, left, crop_h, crop_w, out_size):
    output_size = jnp.asarray(out_size, jnp.float32)
    scale = output_size / jnp.asarray([crop_h, crop_w], jnp.float32)
    translation = -jnp.asarray([top, left], jnp.float32) * scale
    return jax.image.scale_and_translate(
        image,
        shape=(out_size, out_size, image.shape[-1]),
        spatial_dims=(0, 1),
        scale=scale,
        translation=translation,
        method="cubic",
        antialias=True,
    )


def paired_random_resized_crop(
    rng,
    first,
    second,
    out_size=224,
    scale: Tuple[float, float] = (0.08, 1.0),
    ratio: Tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
):
    if first.shape != second.shape:
        raise ValueError(
            f"Perceptual inputs have different shapes: {first.shape}, {second.shape}"
        )
    batch, height, width, _channels = first.shape
    keys = jax.random.split(rng, batch)

    def sample_params(key):
        area_key, ratio_key, top_key, left_key = jax.random.split(key, 4)
        target_area = (
            height
            * width
            * jax.random.uniform(area_key, (), minval=scale[0], maxval=scale[1])
        )
        log_ratio = jnp.log(jnp.asarray(ratio, jnp.float32))
        aspect = jnp.exp(
            jax.random.uniform(ratio_key, (), minval=log_ratio[0], maxval=log_ratio[1])
        )
        crop_w = jnp.clip(
            jnp.round(jnp.sqrt(target_area * aspect)).astype(jnp.int32), 1, width
        )
        crop_h = jnp.clip(
            jnp.round(jnp.sqrt(target_area / aspect)).astype(jnp.int32), 1, height
        )
        top = jax.random.randint(top_key, (), 0, height - crop_h + 1)
        left = jax.random.randint(left_key, (), 0, width - crop_w + 1)
        return (top, left, crop_h, crop_w)

    tops, lefts, crop_heights, crop_widths = jax.vmap(sample_params)(keys)
    resize = lambda image, top, left, crop_h, crop_w: _resample_crop_to_square(
        image, top, left, crop_h, crop_w, out_size
    )
    first = jax.vmap(resize)(first, tops, lefts, crop_heights, crop_widths)
    second = jax.vmap(resize)(second, tops, lefts, crop_heights, crop_widths)
    return (first, second)


def init_auxloss(*, use_lpips=False, use_convnext=False):
    if not use_lpips and (not use_convnext):
        return None
    lpips_model = lpips_params = None
    if use_lpips:
        from lpips_j.lpips import LPIPS

        log_for_0("Loading LPIPS model...")
        lpips_model = LPIPS()
        dummy = jnp.zeros((1, 224, 224, 3), dtype=jnp.float32)
        lpips_params = lpips_model.init(jax.random.key(0), dummy, dummy)
    convnext = convnext_params = None
    if use_convnext:
        from nfm.models.convnext import load_convnext_jax_model

        log_for_0("Loading ConvNeXt model...")
        convnext, convnext_params = load_convnext_jax_model()

    def decode(values):
        return values

    def auxloss_fn(prediction, target, rng):
        prediction = decode(prediction).astype(jnp.float32)
        target = jax.lax.stop_gradient(decode(target).astype(jnp.float32))
        if prediction.shape[-1] != 3:
            raise ValueError(
                f"Perceptual losses require RGB images, got {prediction.shape}"
            )
        prediction, target = paired_random_resized_crop(rng, prediction, target)
        if use_lpips:
            lpips_distance = lpips_model.apply(
                lpips_params, prediction, target
            ).reshape(-1)
        else:
            lpips_distance = jnp.zeros((prediction.shape[0],), jnp.float32)
        if use_convnext:
            prediction_features = convnext.apply(convnext_params, prediction)
            target_features = convnext.apply(convnext_params, target)
            convnext_distance = jnp.sum(
                jnp.square(prediction_features - target_features), axis=-1
            )
        else:
            convnext_distance = jnp.zeros((prediction.shape[0],), jnp.float32)
        return (lpips_distance, convnext_distance)

    return auxloss_fn
