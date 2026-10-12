"""Command-line interface for validated quantum-horizon experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import uuid
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from . import __version__
from .artifacts import (
    provenance,
    read_sample_snapshot,
    sha256,
    source_sha256,
    tensor_to_data,
    write_json_atomic,
    write_samples_atomic,
)
from .diagnostics import geometry_summary, hmc_diagnostic_summary, quantization_summary
from .hmc import HMCConfig, HMCResult, potential_and_grad, sample_hmc
from .model import EnsembleConfig, RelaxedHorizonEnsemble, area_matched_initial_position
from .validation import require_finite, require_integer, validate_samples


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Nonfinite JSON constant: {value}")


def _decode_config(
    content: bytes,
) -> tuple[EnsembleConfig, HMCConfig, dict[str, Any], dict[str, Any]]:
    raw = json.loads(content, parse_constant=_reject_json_constant)
    if not isinstance(raw, dict) or set(raw) - {"ensemble", "hmc", "runtime"}:
        raise ValueError("Configuration must contain only ensemble, hmc, and runtime sections")
    if not all(isinstance(raw.get(name), dict) for name in ("ensemble", "hmc")):
        raise ValueError("ensemble and hmc must be configuration objects")
    ensemble = EnsembleConfig.from_mapping(raw["ensemble"])
    hmc = HMCConfig.from_mapping(raw["hmc"])
    runtime = raw.get("runtime", {})
    if not isinstance(runtime, dict) or set(runtime) - {"seed", "device", "initial_noise_scale"}:
        raise ValueError("Invalid runtime controls")
    _validate_seed(runtime.get("seed", 0))
    require_finite("initial_noise_scale", runtime.get("initial_noise_scale", 0.02), minimum=0.0)
    if not isinstance(runtime.get("device", "cpu"), str):
        raise ValueError("runtime device must be a string")
    return ensemble, hmc, dict(runtime), raw


def _validate_seed(seed: int) -> None:
    require_integer("seed", seed)
    if seed > 2**63 - 1:
        raise ValueError("seed must be <= 2**63 - 1")


def load_experiment_config(
    path: Path,
) -> tuple[EnsembleConfig, HMCConfig, dict[str, Any], dict[str, Any]]:
    return _decode_config(path.read_bytes())


def _device(runtime: dict[str, Any]) -> torch.device:
    device = torch.device(runtime.get("device", "cpu"))
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Only CPU and CUDA generators are supported")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def validation_payload(config_path: Path) -> dict[str, Any]:
    ensemble_config, hmc_config, runtime, raw = load_experiment_config(config_path)
    device = _device(runtime)
    generator = torch.Generator(device=device).manual_seed(int(runtime.get("seed", 0)))
    model = RelaxedHorizonEnsemble(ensemble_config).to(device)
    initial = area_matched_initial_position(
        model,
        device=device,
        noise_scale=float(runtime.get("initial_noise_scale", 0.02)),
        generator=generator,
    )
    potential, autograd_gradient = potential_and_grad(model, initial)
    analytic_gradient = model.analytic_gradient(initial)
    gradient_error = torch.max(torch.abs(autograd_gradient - analytic_gradient))

    tail_reference = initial.clone()
    tail_far = initial.clone()
    tail_reference[0] = -10.0
    tail_far[0] = -40.0
    proper_tail_increment = model(tail_far) - model(tail_reference)
    terms = model.potential_terms(initial)

    status = "PASS"
    if gradient_error > 1.0e-8 or proper_tail_increment <= 0.0:
        status = "FAIL"

    return {
        "schema_version": "1.0",
        "package_version": __version__,
        "status": status,
        "config": raw,
        "derived": {
            "area_constant": ensemble_config.area_constant,
            "minimum_area": ensemble_config.minimum_area,
            "initial_area": terms["area"],
            "initial_potential": potential,
            "maximum_autograd_analytic_gradient_error": gradient_error,
            "proper_negative_tail_potential_increment": proper_tail_increment,
            "area_hessian_structure": "diagonal plus rank one",
        },
        "safeguards": [
            "The Softplus Jacobian is included; the latent target is proper.",
            "The factorized degeneracy is labeled as a proxy, not an intertwiner count.",
            "Rounded spins are diagnostics and do not constitute exact discrete inference.",
            "No entropy claim is made without a normalizing-constant estimator.",
        ],
        "hmc_controls": hmc_config.to_dict(),
    }


def run_experiment(config_path: Path, *, seed_override: int | None = None) -> HMCResult:
    content = config_path.read_bytes()
    ensemble_config, hmc_config, runtime, raw = _decode_config(content)
    device = _device(runtime)
    seed = runtime.get("seed", 0) if seed_override is None else seed_override
    _validate_seed(seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    model = RelaxedHorizonEnsemble(ensemble_config).to(device)
    initial = area_matched_initial_position(
        model,
        device=device,
        noise_scale=float(runtime.get("initial_noise_scale", 0.02)),
        generator=generator,
    )
    metadata = provenance(config_path, seed, config_sha256=hashlib.sha256(content).hexdigest())
    metadata["run_id"] = str(uuid.uuid4())
    metadata["config"] = raw
    result = sample_hmc(model, initial, hmc_config, generator=generator)
    return replace(result, metadata=metadata)


def analysis_payload(
    config_path: Path,
    result: HMCResult,
    *,
    seed: int,
) -> dict[str, Any]:
    _validate_seed(seed)
    if result.metadata is None:
        raise ValueError("Analysis requires a result with producer provenance")
    if sha256(config_path) != result.metadata.get("config_sha256"):
        raise ValueError("Configuration changed after sampling")
    if result.metadata.get("source_sha256") != source_sha256():
        raise ValueError("Source changed after sampling")
    if seed != result.metadata.get("seed"):
        raise ValueError("Analysis seed does not match the sampled run")
    raw = result.metadata["config"]
    ensemble_config, hmc_config, _, _ = _decode_config(json.dumps(raw).encode())
    model = RelaxedHorizonEnsemble(ensemble_config)
    samples = result.samples.detach().cpu()
    validate_samples(samples, n_punctures=ensemble_config.n_punctures)
    if samples.shape[0] != hmc_config.num_samples:
        raise ValueError("Sample count does not match the sampled configuration")
    hmc = hmc_diagnostic_summary(result.diagnostics)
    if hmc["retained_transitions"] != samples.shape[0]:
        raise ValueError("Retained diagnostic count does not match samples")
    quantization = quantization_summary(model, samples)
    warnings = []
    if hmc["divergences"]:
        warnings.append("Retained divergent transitions require investigation.")
    if hmc["nonfinite_energy_errors"]:
        warnings.append("Nonfinite proposal energy errors occurred.")
    if samples.shape[0] < 4:
        warnings.append("Too few draws to estimate scalar ESS.")
    if quantization["mixing"]["constant_coordinates"]:
        warnings.append("Some coordinates are constant in the retained chain.")
    if quantization["mixing"]["rounded_sector_transitions"] == 0:
        warnings.append("No rounded spin-sector transitions were observed.")
    if any(
        quantization[name]["ess"] is None or quantization[name]["ess"] < 50
        for name in ("area", "lattice_rms_distance", "comb_statistic", "potential")
    ):
        warnings.append(
            "At least one scalar ESS is unavailable or below 50; this run is exploratory."
        )
    failed = hmc["acceptance_rate"] == 0.0 or (
        samples.shape[0] >= 4
        and quantization["mixing"]["constant_coordinates"] == ensemble_config.n_punctures
    )
    return {
        "schema_version": "2.0",
        "package_version": __version__,
        "status": "FAIL" if failed else "WARN" if warnings else "DIAGNOSTICS_ONLY",
        "diagnostic_warnings": warnings,
        "convergence_status": "NOT_ASSESSED_SINGLE_CHAIN",
        "config": raw,
        "hmc": hmc,
        "quantization": quantization,
        "geometry": geometry_summary(model, samples),
        "final_step_size": result.final_step_size,
        "final_mass": {
            "minimum": float(result.final_mass_diag.min().item()),
            "median": float(result.final_mass_diag.median().item()),
            "maximum": float(result.final_mass_diag.max().item()),
        },
        "estimand_boundary": {
            "current": "relaxed normalized ensemble diagnostics",
            "not_yet_identified": "exact microcanonical state count or entropy",
            "required_next_stage": (
                "replica exchange plus thermodynamic integration and exact intertwiner correction"
            ),
        },
        "provenance": {key: value for key, value in result.metadata.items() if key != "config"},
        "hmc_config": hmc_config.to_dict(),
    }


def save_experiment(
    config_path: Path,
    *,
    seed: int,
    samples_path: Path,
    summary_path: Path,
) -> dict[str, Any]:
    if samples_path.resolve() == summary_path.resolve():
        raise ValueError("Sample and summary paths must be distinct")
    if config_path.resolve() in (samples_path.resolve(), summary_path.resolve()):
        raise ValueError("Output paths must not overwrite the experiment configuration")
    result = run_experiment(config_path, seed_override=seed)
    payload = analysis_payload(config_path, result, seed=seed)
    metadata = dict(result.metadata or {})
    metadata["sampling_status"] = payload["status"]
    metadata["diagnostic_warnings"] = payload["diagnostic_warnings"]
    written_digest = write_samples_atomic(samples_path, result.samples, metadata=metadata)
    payload["provenance"]["samples_file_sha256"] = written_digest
    write_json_atomic(summary_path, payload)
    return payload


def verified_analysis_input(
    config_path: Path,
    samples_path: Path,
) -> tuple[RelaxedHorizonEnsemble, torch.Tensor, dict[str, Any], dict[str, Any]]:
    content = config_path.read_bytes()
    ensemble, hmc, _, raw = _decode_config(content)
    snapshot = read_sample_snapshot(
        samples_path,
        expected_config_sha256=hashlib.sha256(content).hexdigest(),
    )
    samples, metadata = snapshot.samples, snapshot.metadata
    validate_samples(samples, n_punctures=ensemble.n_punctures)
    if samples.shape[0] != hmc.num_samples:
        raise ValueError("Sample artifact draw count does not match the configuration")
    if metadata.get("config") != raw:
        raise ValueError("Sample configuration snapshot does not match the input")
    if metadata.get("sampling_status") not in ("WARN", "DIAGNOSTICS_ONLY"):
        raise ValueError("Sample artifact lacks a usable producer sampling assessment")
    producer = {key: value for key, value in metadata.items() if key != "config"}
    producer["samples_file_sha256"] = snapshot.file_sha256
    return RelaxedHorizonEnsemble(ensemble), samples, raw, producer


def command_validate(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    payload = validation_payload(config_path)
    if args.output:
        write_json_atomic(Path(args.output), payload)
        print(f"Wrote {args.output}")
    print(json.dumps(tensor_to_data(payload), indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


def command_sample(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    _, _, runtime, _ = load_experiment_config(config_path)
    seed = int(runtime.get("seed", 0) if args.seed is None else args.seed)
    payload = save_experiment(
        config_path,
        seed=seed,
        samples_path=Path(args.samples_output),
        summary_path=Path(args.summary_output),
    )
    print(f"Wrote {args.samples_output}")
    print(f"Wrote {args.summary_output}")
    print(f"Sampling status: {payload['status']}; convergence: {payload['convergence_status']}")
    return 1 if payload["status"] == "FAIL" else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qhme",
        description="Quantum Horizon Microstate Ensemble research CLI",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate model and configuration")
    validate.add_argument("--config", default="configs/baseline.json")
    validate.add_argument("--output")
    validate.set_defaults(handler=command_validate)

    sample = subparsers.add_parser("sample", help="run adaptive HMC and write artifacts")
    sample.add_argument("--config", default="configs/baseline.json")
    sample.add_argument("--seed", type=int)
    sample.add_argument("--samples-output", default="artifacts/part1/samples.pt")
    sample.add_argument("--summary-output", default="artifacts/part1/hmc_summary.json")
    sample.set_defaults(handler=command_sample)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (ValueError, TypeError, OSError, RuntimeError, FloatingPointError) as exc:
        print(f"qhme: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
