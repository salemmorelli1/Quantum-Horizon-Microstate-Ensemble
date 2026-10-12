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

from .validation import validate_samples


def tensor_to_data(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return tensor_to_data(value.detach().cpu().item())
        return tensor_to_data(value.detach().cpu().tolist())
    if is_dataclass(value):
        return tensor_to_data(asdict(value))
    if isinstance(value, dict):
        return {str(key): tensor_to_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [tensor_to_data(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite values cannot be written as scientific JSON")
    return value


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(tensor_to_data(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def tensor_sha256(samples: torch.Tensor) -> str:
    """Hash shape, dtype, and tensor bytes without depending on NumPy."""

    validate_samples(samples)
    tensor = samples.detach().cpu().contiguous()
    digest = hashlib.sha256(f"{tensor.dtype}:{list(tensor.shape)}:".encode())
    for chunk in tensor.view(torch.uint8).flatten().split(1024 * 1024):
        digest.update(bytes(chunk.tolist()))
    return digest.hexdigest()


def write_samples_atomic(
    path: Path,
    samples: torch.Tensor,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    validate_samples(samples)
    payload = {
        "schema_version": "2.0",
        "samples": samples.detach().cpu(),
        "samples_sha256": tensor_sha256(samples),
        "metadata": tensor_to_data(metadata or {}),
    }
    payload["metadata_sha256"] = _metadata_sha256(payload["metadata"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_samples(path: Path) -> torch.Tensor:
    samples, _ = read_sample_artifact(path)
    return samples


def read_sample_artifact(
    path: Path,
    *,
    expected_config_sha256: str | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") != "2.0":
        raise ValueError(
            "Unverified legacy sample artifact; regenerate it with the audited sampler"
        )
    samples = payload.get("samples")
    validate_samples(samples)
    if payload.get("samples_sha256") != tensor_sha256(samples):
        raise ValueError("Sample content SHA-256 mismatch")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Invalid sample metadata")
    if payload.get("metadata_sha256") != _metadata_sha256(metadata):
        raise ValueError("Sample metadata SHA-256 mismatch")
    if expected_config_sha256 is not None:
        if metadata.get("config_sha256") != expected_config_sha256:
            raise ValueError("Sample/configuration SHA-256 mismatch")
        if not isinstance(metadata.get("seed"), int) or isinstance(metadata["seed"], bool):
            raise ValueError("Sample artifact is missing its effective seed")
        if not metadata.get("run_id") or not metadata.get("source_sha256"):
            raise ValueError("Sample artifact is missing run/source provenance")
        if metadata.get("source_sha256") != source_sha256():
            raise ValueError("Sample/source SHA-256 mismatch; regenerate with the current source")
    return samples, metadata


def _metadata_sha256(metadata: dict[str, Any]) -> str:
    encoded = json.dumps(
        tensor_to_data(metadata), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_sha256() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def provenance(
    config_path: Path,
    seed: int,
    *,
    config_sha256: str | None = None,
) -> dict[str, Any]:
    return {
        "config_path": str(config_path),
        "config_sha256": config_sha256 or sha256(config_path),
        "source_sha256": source_sha256(),
        "seed": int(seed),
        "python": sys.version.split()[0],
        "pytorch": str(torch.__version__),
        "platform": platform.platform(),
    }
