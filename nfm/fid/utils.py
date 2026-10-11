"""Pretrained FID weights: local paths or an atomic cached download."""

import os
from pathlib import Path
import tempfile
from urllib.parse import urlparse

import requests


def download(url, ckpt_dir=None):
    if not urlparse(url).scheme:
        path = Path(url)
        if not path.is_file():
            raise FileNotFoundError(path)
        return str(path)
    cache = Path(
        ckpt_dir
        or os.environ.get("JAX_FID_CACHE_DIR", Path(tempfile.gettempdir()) / "jax_fid")
    )
    destination = cache / Path(urlparse(url).path).name
    if destination.is_file():
        return str(destination)
    if os.environ.get("JAX_FID_OFFLINE", "").lower() in {"1", "true", "yes"}:
        raise FileNotFoundError(f"Offline FID weights missing: {destination}")
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=cache, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            with requests.get(url, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                written = 0
                for block in response.iter_content(1024 * 1024):
                    handle.write(block)
                    written += len(block)
                expected = response.headers.get("content-length")
                if expected and written != int(expected):
                    raise IOError("Incomplete FID weights download")
            handle.flush()
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    return str(destination)


def get(dictionary, key):
    return None if dictionary is None else dictionary.get(key)
