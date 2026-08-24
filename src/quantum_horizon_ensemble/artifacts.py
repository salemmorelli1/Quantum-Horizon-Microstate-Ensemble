"""Atomic and provenance-aware artifact helpers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch


def tensor_to_data(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if is_dataclass(value):
        return tensor_to_data(asdict(value))
    if isinstance(value, dict):
        return {str(key): tensor_to_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [tensor_to_data(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        json.dump(tensor_to_data(payload), stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def write_samples_atomic(path: Path, samples: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save({"samples": samples.detach().cpu()}, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_samples(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    samples = payload["samples"]
    if not isinstance(samples, torch.Tensor) or samples.ndim != 2:
        raise ValueError("Invalid sample artifact")
    return samples


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance(config_path: Path, seed: int) -> dict[str, Any]:
    return {
        "config_path": str(config_path),
        "config_sha256": sha256(config_path),
        "seed": int(seed),
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "platform": platform.platform(),
    }
