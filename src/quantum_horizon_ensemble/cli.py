"""Command-line interface for validated quantum-horizon experiments."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from . import __version__
from .artifacts import provenance, tensor_to_data, write_json_atomic, write_samples_atomic
from .diagnostics import geometry_summary, hmc_diagnostic_summary, quantization_summary
from .hmc import HMCConfig, HMCResult, potential_and_grad, sample_hmc
from .model import EnsembleConfig, RelaxedHorizonEnsemble, area_matched_initial_position


def load_experiment_config(
    path: Path,
) -> tuple[EnsembleConfig, HMCConfig, dict[str, Any], dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        raw = json.load(stream)
    ensemble = EnsembleConfig.from_mapping(raw["ensemble"])
    hmc = HMCConfig.from_mapping(raw["hmc"])
    runtime = dict(raw.get("runtime", {}))
    return ensemble, hmc, runtime, raw


def _device(runtime: dict[str, Any]) -> torch.device:
    requested = str(runtime.get("device", "cpu"))
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


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
    ensemble_config, hmc_config, runtime, _ = load_experiment_config(config_path)
    device = _device(runtime)
    seed = int(runtime.get("seed", 0) if seed_override is None else seed_override)
    generator = torch.Generator(device=device).manual_seed(seed)
    model = RelaxedHorizonEnsemble(ensemble_config).to(device)
    initial = area_matched_initial_position(
        model,
        device=device,
        noise_scale=float(runtime.get("initial_noise_scale", 0.02)),
        generator=generator,
    )
    return sample_hmc(model, initial, hmc_config, generator=generator)


def analysis_payload(
    config_path: Path,
    result: HMCResult,
    *,
    seed: int,
) -> dict[str, Any]:
    ensemble_config, hmc_config, _, raw = load_experiment_config(config_path)
    model = RelaxedHorizonEnsemble(ensemble_config)
    samples = result.samples.detach().cpu()
    return {
        "schema_version": "1.0",
        "package_version": __version__,
        "status": "PASS",
        "config": raw,
        "hmc": hmc_diagnostic_summary(result.diagnostics),
        "quantization": quantization_summary(model, samples),
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
                "replica exchange plus thermodynamic integration and exact "
                "intertwiner correction"
            ),
        },
        "provenance": provenance(config_path, seed),
        "hmc_config": hmc_config.to_dict(),
    }


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
    result = run_experiment(config_path, seed_override=seed)
    write_samples_atomic(Path(args.samples_output), result.samples)
    payload = analysis_payload(config_path, result, seed=seed)
    write_json_atomic(Path(args.summary_output), payload)
    print(f"Wrote {args.samples_output}")
    print(f"Wrote {args.summary_output}")
    return 0


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
    torch.set_default_dtype(torch.float64)
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
