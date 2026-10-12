"""Diagnostics for relaxed quantum-horizon HMC experiments."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from typing import Any

import torch

from .hmc import HMCDiagnostic
from .model import RelaxedHorizonEnsemble, Tensor
from .validation import require_float_tensor, validate_samples


def effective_sample_size(trace: Tensor) -> float | None:
    """Conservative single-chain ESS with Geyer's initial monotone sequence.

    This is a scalar autocorrelation diagnostic, not rank-normalized multichain
    bulk/tail ESS. Constant or very short traces do not identify an ESS.
    """

    require_float_tensor("trace", trace)
    if trace.ndim != 1 or trace.numel() < 1:
        raise ValueError("ESS requires a nonempty one-dimensional trace")
    values = trace.detach().double()
    n = values.numel()
    if n < 4:
        return None
    # Remove the location before scaling. Scaling large-offset, small-spread
    # values directly can round away variation that is present in the input.
    centered = values - values[0]
    if not torch.isfinite(centered).all():
        # Finite values of opposite signs can overflow on subtraction; only
        # that case needs scaling before the location is removed.
        values = values / values.abs().max()
        centered = values - values[0]
    scale = centered.abs().max()
    if scale == 0.0:
        return None
    centered = centered / scale
    centered = centered - centered.mean()
    fft_size = 1 << (2 * n - 1).bit_length()
    spectrum = torch.fft.rfft(centered, n=fft_size)
    autocovariance = torch.fft.irfft(spectrum * spectrum.conj(), n=fft_size)[:n]
    autocorrelations = (autocovariance / autocovariance[0]).cpu().tolist()

    positive_sum = 0.0
    previous_pair = float("inf")
    # Pair rho[0]+rho[1], rho[2]+rho[3], ...; then tau = -1 + 2*sum.
    for start in range(0, n - 1, 2):
        pair_sum = autocorrelations[start] + autocorrelations[start + 1]
        if pair_sum <= 0.0:
            break
        pair_sum = min(previous_pair, pair_sum)
        positive_sum += pair_sum
        previous_pair = pair_sum
    return float(n) / max(-1.0 + 2.0 * positive_sum, 1.0)


def _sd(trace: Tensor) -> float | None:
    return float(trace.std(unbiased=True).item()) if trace.numel() > 1 else None


def _mcse(trace: Tensor) -> float | None:
    ess = effective_sample_size(trace)
    sd = _sd(trace)
    return sd / math.sqrt(ess) if sd is not None and ess is not None else None


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
        "mean_acceptance_probability": sum(item.acceptance_probability for item in retained)
        / len(retained),
        "divergences": divergent,
        "divergence_rate": divergent / len(retained),
        "nonfinite_energy_errors": len(retained) - len(finite_errors),
        "energy_error_statistics_scope": "finite errors only; inspect nonfinite count",
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

    validate_samples(samples, n_punctures=model.config.n_punctures)
    samples = samples.detach().double()

    area_trace = _stack_scalar_trace(samples, lambda z: model.area(model.proxy_spins(z)))
    lattice_trace = _stack_scalar_trace(samples, model.lattice_distance)
    comb_trace = _stack_scalar_trace(samples, model.comb_energy)
    potential_trace = _stack_scalar_trace(samples, model)

    rounded = torch.stack([model.rounded_spins(sample) for sample in samples])
    rounded_area = _stack_scalar_trace(rounded, model.area)
    counts = Counter(float(value) for value in rounded.flatten().tolist())
    total_spins = rounded.numel()
    occupancy = {f"{spin:g}": count / total_spins for spin, count in sorted(counts.items())}

    local_log_degeneracy = torch.log(2.0 * rounded + 1.0).sum(dim=1)
    area_error = area_trace - model.config.area_target
    for name, trace in (
        ("area", area_trace),
        ("lattice distance", lattice_trace),
        ("comb statistic", comb_trace),
        ("potential", potential_trace),
    ):
        require_float_tensor(name, trace)
    return {
        "draws": int(samples.shape[0]),
        "n_punctures": int(samples.shape[1]),
        "area": {
            "target": model.config.area_target,
            "mean": float(area_trace.mean().item()),
            "sd": _sd(area_trace),
            "rmse": float(torch.sqrt(torch.mean(area_error.square())).item()),
            "ess": effective_sample_size(area_trace),
            "mcse_mean": _mcse(area_trace),
        },
        "lattice_rms_distance": {
            "mean": float(lattice_trace.mean().item()),
            "sd": _sd(lattice_trace),
            "ess": effective_sample_size(lattice_trace),
            "mcse_mean": _mcse(lattice_trace),
        },
        "comb_statistic": {
            "mean": float(comb_trace.mean().item()),
            "sd": _sd(comb_trace),
            "ess": effective_sample_size(comb_trace),
            "mcse_mean": _mcse(comb_trace),
            "thermodynamic_integration_integrand": -float(comb_trace.mean().item()),
            "thermodynamic_integration_identity": "d log Z / d kappa_Q = -E[Q]",
        },
        "potential": {
            "mean": float(potential_trace.mean().item()),
            "sd": _sd(potential_trace),
            "ess": effective_sample_size(potential_trace),
            "mcse_mean": _mcse(potential_trace),
        },
        "rounded_spin_occupancy": occupancy,
        "rounded_area": {
            "mean": float(rounded_area.mean()),
            "sd": _sd(rounded_area),
            "rmse": float(
                torch.sqrt(torch.mean((rounded_area - model.config.area_target).square()))
            ),
            "warning": "Rounding changes area and does not enforce an exact discrete area window.",
        },
        "factorized_log_degeneracy_proxy": {
            "mean": float(local_log_degeneracy.mean().item()),
            "sd": _sd(local_log_degeneracy),
            "warning": (
                "This is not the exact SU(2) invariant intertwiner dimension and "
                "must not be reported as black-hole entropy."
            ),
        },
        "mixing": {
            "constant_coordinates": int(
                (samples.max(dim=0).values == samples.min(dim=0).values).sum()
            ),
            "rounded_sector_transitions": int((rounded[1:] != rounded[:-1]).any(dim=1).sum()),
            "ess_method": "single-chain initial monotone autocorrelation sequence, capped at draws",
            "r_hat": None,
            "convergence_status": "NOT_ASSESSED_SINGLE_CHAIN",
            "warning": "Scalar ESS and acceptance do not establish mixing across comb wells.",
        },
    }


def geometry_summary(
    model: RelaxedHorizonEnsemble,
    samples: Tensor,
) -> dict[str, Any]:
    """Report scalable curvature proxies without materializing an N x N Hessian."""

    validate_samples(samples, n_punctures=model.config.n_punctures)
    samples = samples.detach().double()
    area_norms = _stack_scalar_trace(
        samples, lambda z: torch.linalg.vector_norm(model.area_gradient(z))
    )
    macro_eigenvalue = model.config.kappa_area * area_norms.square()

    x = model.proxy_spins(samples[-1])
    comb_curvature = 8.0 * math.pi**2 * model.config.kappa_quant * torch.cos(4.0 * math.pi * x)
    return {
        "hessian_structure": "diagonal plus rank one in proxy-spin coordinates",
        "coordinate_conventions": {
            "area_gradient_norm": "latent z coordinates",
            "macro_rank_one_eigenvalue_proxy": "latent z coordinates; rank-one term only",
            "final_comb_curvature": "proxy-spin x coordinates; comb term only",
        },
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
