"""Diagnostics for relaxed quantum-horizon HMC experiments."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from typing import Any

import torch

from .hmc import HMCDiagnostic
from .model import RelaxedHorizonEnsemble, Tensor


def effective_sample_size(trace: Tensor) -> float:
    """Estimate scalar ESS using the initial-positive autocorrelation sequence."""

    values = trace.detach().double().flatten()
    n = values.numel()
    if n < 4:
        return float(n)
    centered = values - values.mean()
    variance = torch.mean(centered.square())
    if variance <= torch.finfo(values.dtype).eps:
        return float(n)

    autocorrelations: list[float] = []
    for lag in range(1, n):
        correlation = torch.mean(centered[:-lag] * centered[lag:]) / variance
        autocorrelations.append(float(correlation.item()))

    positive_sum = 0.0
    for start in range(0, len(autocorrelations) - 1, 2):
        pair_sum = autocorrelations[start] + autocorrelations[start + 1]
        if pair_sum <= 0.0:
            break
        positive_sum += pair_sum
    return min(float(n), float(n) / max(1.0 + 2.0 * positive_sum, 1.0))


def hmc_diagnostic_summary(
    diagnostics: Sequence[HMCDiagnostic],
) -> dict[str, Any]:
    retained = [item for item in diagnostics if not item.warmup]
    if not retained:
        raise ValueError("At least one retained HMC diagnostic is required")
    accepted = sum(item.accepted for item in retained)
    divergent = sum(item.divergent for item in retained)
    finite_errors = [
        abs(item.delta_energy) for item in retained if math.isfinite(item.delta_energy)
    ]
    return {
        "retained_transitions": len(retained),
        "acceptance_rate": accepted / len(retained),
        "mean_acceptance_probability": sum(
            item.acceptance_probability for item in retained
        )
        / len(retained),
        "divergences": divergent,
        "divergence_rate": divergent / len(retained),
        "mean_absolute_energy_error": (
            sum(finite_errors) / len(finite_errors) if finite_errors else None
        ),
        "maximum_absolute_energy_error": max(finite_errors) if finite_errors else None,
    }


def _stack_scalar_trace(
    samples: Tensor,
    function,
) -> Tensor:
    return torch.stack([function(sample).detach() for sample in samples])


def quantization_summary(
    model: RelaxedHorizonEnsemble,
    samples: Tensor,
) -> dict[str, Any]:
    """Summarize area-shell and half-integer-lattice behavior."""

    if samples.ndim != 2 or samples.shape[1] != model.config.n_punctures:
        raise ValueError("samples must have shape (draws, n_punctures)")

    area_trace = _stack_scalar_trace(samples, lambda z: model.area(model.proxy_spins(z)))
    lattice_trace = _stack_scalar_trace(samples, model.lattice_distance)
    comb_trace = _stack_scalar_trace(samples, model.comb_energy)
    potential_trace = _stack_scalar_trace(samples, model)

    rounded = torch.stack([model.rounded_spins(sample) for sample in samples])
    counts = Counter(float(value) for value in rounded.flatten().tolist())
    total_spins = rounded.numel()
    occupancy = {
        f"{spin:g}": count / total_spins for spin, count in sorted(counts.items())
    }

    local_log_degeneracy = torch.log(2.0 * rounded + 1.0).sum(dim=1)
    area_error = area_trace - model.config.area_target
    return {
        "draws": int(samples.shape[0]),
        "n_punctures": int(samples.shape[1]),
        "area": {
            "target": model.config.area_target,
            "mean": float(area_trace.mean().item()),
            "sd": float(area_trace.std(unbiased=True).item()),
            "rmse": float(torch.sqrt(torch.mean(area_error.square())).item()),
            "ess": effective_sample_size(area_trace),
        },
        "lattice_rms_distance": {
            "mean": float(lattice_trace.mean().item()),
            "sd": float(lattice_trace.std(unbiased=True).item()),
            "ess": effective_sample_size(lattice_trace),
        },
        "comb_statistic": {
            "mean": float(comb_trace.mean().item()),
            "sd": float(comb_trace.std(unbiased=True).item()),
            "thermodynamic_integration_integrand": float(comb_trace.mean().item()),
        },
        "potential": {
            "mean": float(potential_trace.mean().item()),
            "sd": float(potential_trace.std(unbiased=True).item()),
            "ess": effective_sample_size(potential_trace),
        },
        "rounded_spin_occupancy": occupancy,
        "factorized_log_degeneracy_proxy": {
            "mean": float(local_log_degeneracy.mean().item()),
            "sd": float(local_log_degeneracy.std(unbiased=True).item()),
            "warning": (
                "This is not the exact SU(2) invariant intertwiner dimension and "
                "must not be reported as black-hole entropy."
            ),
        },
    }


def geometry_summary(
    model: RelaxedHorizonEnsemble,
    samples: Tensor,
) -> dict[str, Any]:
    """Report scalable curvature proxies without materializing an N x N Hessian."""

    area_norms = _stack_scalar_trace(
        samples, lambda z: torch.linalg.vector_norm(model.area_gradient(z))
    )
    macro_eigenvalue = model.config.kappa_area * area_norms.square()

    x = model.proxy_spins(samples[-1])
    comb_curvature = 8.0 * math.pi**2 * model.config.kappa_quant * torch.cos(
        4.0 * math.pi * x
    )
    return {
        "hessian_structure": "diagonal plus rank one in proxy-spin coordinates",
        "area_gradient_norm": {
            "mean": float(area_norms.mean().item()),
            "maximum": float(area_norms.max().item()),
        },
        "macro_rank_one_eigenvalue_proxy": {
            "mean": float(macro_eigenvalue.mean().item()),
            "maximum": float(macro_eigenvalue.max().item()),
        },
        "final_comb_curvature": {
            "minimum": float(comb_curvature.min().item()),
            "median": float(comb_curvature.median().item()),
            "maximum": float(comb_curvature.max().item()),
        },
        "recommended_metric": (
            "Use a diagonal mass plus one low-rank correction aligned with the "
            "latent area-gradient direction."
        ),
        "mixing_warning": (
            "Finite comb barriers are nonabsorbing but may generate exponentially "
            "slow transitions among coordinated spin sectors."
        ),
    }
