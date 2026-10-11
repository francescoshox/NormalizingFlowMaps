"""Atomic complete checkpoints and parameter-only stage handoff."""

import json
import os
from pathlib import Path
import shutil
import tempfile

import jax
from flax import serialization


def latest(workdir):
    marker = Path(workdir) / "latest.json"
    if marker.exists():
        return Path(workdir) / json.loads(marker.read_text())["checkpoint"]
    return None


def save(workdir, state, metadata):
    root = Path(workdir)
    root.mkdir(parents=True, exist_ok=True)
    state = jax.device_get(state)
    name = f"checkpoint_{int(state.step):09d}"
    destination = root / name
    if destination.exists():
        # Saving the same finished step twice is harmless only for identical metadata.
        if json.loads((destination / "metadata.json").read_text()) != metadata:
            raise ValueError(f"Checkpoint metadata differs: {destination}")
        # Recover a crash between the directory rename and marker publication.
        marker = root / ".latest.tmp"
        marker.write_text(json.dumps({"checkpoint": name}) + "\n")
        os.replace(marker, root / "latest.json")
        return destination
    temporary = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=root))
    try:
        (temporary / "state.msgpack").write_bytes(serialization.to_bytes(state))
        weights = {"online": state.params}
        weights.update({str(decay): tree for decay, tree in state.ema_params.items()})
        (temporary / "weights.msgpack").write_bytes(
            serialization.msgpack_serialize(weights)
        )
        (temporary / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        os.replace(temporary, destination)
        marker = root / ".latest.tmp"
        marker.write_text(json.dumps({"checkpoint": name}) + "\n")
        os.replace(marker, root / "latest.json")
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination


def restore(path, template):
    path = Path(path)
    return serialization.from_bytes(template, (path / "state.msgpack").read_bytes())


def load_weights(path, ema):
    path = Path(path)
    if not (path / "metadata.json").is_file():
        raise ValueError(f"Incomplete checkpoint: {path}")
    weights = serialization.msgpack_restore((path / "weights.msgpack").read_bytes())
    key = str(float(ema)) if ema != "online" else "online"
    if key not in weights:
        raise ValueError(f"EMA {key} missing; available: {list(weights)}")
    # Direct embedding gathers in the models require JAX arrays when the
    # parameters are captured by a compiled sampling function.
    return jax.device_put(weights[key])
