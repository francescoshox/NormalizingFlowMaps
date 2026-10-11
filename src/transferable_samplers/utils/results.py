from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path
from typing import Any

import torch


def _json_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_value(item) for item in value]
    return value


def _atomic_write(path: Path, contents: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(contents, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def export_metrics(directory: Path, metrics: dict[str, Any], metadata: dict[str, Any]) -> None:
    """Write strict JSON with provenance and a CSV row for every scalar metric.

    Undefined numerical results become JSON null and empty CSV cells.
    Each evaluation epoch has its own directory, preserving validation history.
    """
    directory.mkdir(parents=True, exist_ok=True)
    values = _json_value(metrics)
    payload = {"metadata": _json_value(metadata), "metrics": values}
    _atomic_write(directory / "metrics.json", json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(["metric", "value"])
    writer.writerows(sorted(values.items()))
    _atomic_write(directory / "metrics.csv", output.getvalue())
