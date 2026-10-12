"""Differentiable LQG-inspired quantum-horizon area ensemble.

The probability density is defined on proxy-spin coordinates ``x`` and then
pulled back to unconstrained coordinates ``z``.  Consequently, the Softplus
Jacobian is part of the latent potential.  Omitting it would leave an improper
tail as any ``z_i`` tends to negative infinity.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as functional

from .validation import require_finite, require_float_tensor, require_integer

Tensor = torch.Tensor


@dataclass(frozen=True)
class EnsembleConfig:
    """Physical and statistical configuration for a fixed-puncture ensemble."""

    n_punctures: int
    area_target: float
    gamma_bi: float = 0.274
    planck_length_sq: float = 1.0
    kappa_area: float = 1.0e-3
    kappa_quant: float = 4.0
    eta_degeneracy: float = 1.0

    @property
    def area_constant(self) -> float:
        """Return ``8 pi gamma l_p^2`` in the configured units."""

        return 8.0 * math.pi * self.gamma_bi * self.planck_length_sq

    @property
    def minimum_area(self) -> float:
        """Minimum relaxed area when every puncture approaches spin one-half."""

        return self.area_constant * self.n_punctures * math.sqrt(0.5 * 1.5)

    def validate(self) -> None:
        require_integer("n_punctures", self.n_punctures, minimum=1)
        for name in (
            "area_target",
            "gamma_bi",
            "planck_length_sq",
            "kappa_area",
            "kappa_quant",
            "eta_degeneracy",
        ):
            require_finite(name, getattr(self, name))
        if self.gamma_bi <= 0.0 or self.planck_length_sq <= 0.0:
            raise ValueError("gamma_bi and planck_length_sq must be positive")
        if self.kappa_area <= 0.0 or self.kappa_quant < 0.0:
            raise ValueError("kappa_area must be positive and kappa_quant nonnegative")
        if self.eta_degeneracy < 0.0:
            raise ValueError("eta_degeneracy must be nonnegative")
        if not math.isfinite(self.minimum_area) or self.minimum_area <= 0.0:
            raise ValueError("Derived area scale must be finite and positive")
        if self.area_target < self.minimum_area:
            raise ValueError(
                f"area_target={self.area_target:.8g} is below the fixed-N minimum "
                f"area {self.minimum_area:.8g}"
            )

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> EnsembleConfig:
        config = cls(**values)
        config.validate()
        return config

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RelaxedHorizonEnsemble(nn.Module):
    """Proper relaxed area ensemble with an autograd-compatible potential."""

    def __init__(self, config: EnsembleConfig):
        super().__init__()
        config.validate()
        self.config = config

    def _validate_position(self, z: Tensor) -> None:
        require_float_tensor("position", z)
        if z.ndim != 1:
            raise ValueError("A single Markov-chain position must be one-dimensional")
        if z.numel() != self.config.n_punctures:
            raise ValueError(
                f"Expected {self.config.n_punctures} coordinates; received {z.numel()}"
            )

    def proxy_spins(self, z: Tensor) -> Tensor:
        """Map ``R^N`` smoothly to the physical proxy domain ``(1/2, infinity)^N``."""

        self._validate_position(z)
        # logaddexp is stable without the piecewise linear threshold of
        # functional.softplus, so the analytic sigmoid gradient agrees in tails.
        return 0.5 + torch.logaddexp(z, torch.zeros_like(z))

    def area(self, x: Tensor) -> Tensor:
        """Evaluate the continuously relaxed LQG area operator."""

        require_float_tensor("proxy spins", x)
        if x.ndim != 1 or x.numel() != self.config.n_punctures or torch.any(x < 0.5):
            raise ValueError("proxy spins must have shape (n_punctures,) and be >= 1/2")
        return self.config.area_constant * (torch.sqrt(x) * torch.sqrt(x + 1.0)).sum()

    def potential_terms(self, z: Tensor) -> dict[str, Tensor]:
        """Return the physical terms and the required measure correction."""

        x = self.proxy_spins(z)
        relaxed_area = self.area(x)
        residual = relaxed_area - self.config.area_target

        macro = 0.5 * self.config.kappa_area * residual.square()
        quantization = self.config.kappa_quant * torch.sin(2.0 * math.pi * x).square().sum()
        degeneracy_proxy = -self.config.eta_degeneracy * torch.log(2.0 * x + 1.0).sum()

        # dx_i/dz_i = sigmoid(z_i).  This term is essential when the intended
        # target is a density with respect to dx rather than an arbitrary base
        # density with respect to dz.
        negative_log_jacobian = -functional.logsigmoid(z).sum()

        return {
            "macro": macro,
            "quantization": quantization,
            "degeneracy_proxy": degeneracy_proxy,
            "negative_log_jacobian": negative_log_jacobian,
            "area": relaxed_area,
            "area_residual": residual,
        }

    def forward(self, z: Tensor) -> Tensor:
        terms = self.potential_terms(z)
        return (
            terms["macro"]
            + terms["quantization"]
            + terms["degeneracy_proxy"]
            + terms["negative_log_jacobian"]
        )

    def physical_gradient_x(self, x: Tensor) -> Tensor:
        """Closed-form ``dF/dx`` for independent verification of autograd."""

        area = self.area(x)
        residual = area - self.config.area_target
        area_prime = (2.0 * x + 1.0) / (2.0 * torch.sqrt(x * (x + 1.0)))
        macro = self.config.kappa_area * self.config.area_constant * residual * area_prime
        quantization = 2.0 * math.pi * self.config.kappa_quant * torch.sin(4.0 * math.pi * x)
        degeneracy = -2.0 * self.config.eta_degeneracy / (2.0 * x + 1.0)
        return macro + quantization + degeneracy

    def analytic_gradient(self, z: Tensor) -> Tensor:
        """Closed-form gradient of the complete latent potential ``dU/dz``."""

        x = self.proxy_spins(z)
        q = torch.sigmoid(z)
        return q * self.physical_gradient_x(x) + q - 1.0

    def area_gradient(self, z: Tensor) -> Tensor:
        """Gradient of relaxed area, defining the stiff collective direction."""

        x = self.proxy_spins(z)
        area_prime = (2.0 * x + 1.0) / (2.0 * torch.sqrt(x * (x + 1.0)))
        return self.config.area_constant * torch.sigmoid(z) * area_prime

    def rounded_spins(self, z: Tensor) -> Tensor:
        """Project proxies to half-integers for diagnostics, not exact inference."""

        x = self.proxy_spins(z)
        return 0.5 * torch.round(2.0 * x).clamp_min(1.0)

    def lattice_distance(self, z: Tensor, *, root_mean_square: bool = True) -> Tensor:
        """Distance between proxy spins and their nearest half-integer vector."""

        residual = self.proxy_spins(z) - self.rounded_spins(z)
        if root_mean_square:
            return torch.sqrt(torch.mean(residual.square()))
        return torch.linalg.vector_norm(residual)

    def comb_energy(self, z: Tensor) -> Tensor:
        x = self.proxy_spins(z)
        return torch.sin(2.0 * math.pi * x).square().sum()


def softplus_inverse(y: Tensor) -> Tensor:
    """Stable inverse of ``log(1 + exp(z))`` for strictly positive ``y``."""

    require_float_tensor("softplus_inverse input", y)
    if torch.any(y <= 0.0):
        raise ValueError("softplus_inverse requires strictly positive inputs")
    return y + torch.log(-torch.expm1(-y))


def area_matched_initial_position(
    model: RelaxedHorizonEnsemble,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
    noise_scale: float = 0.02,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Construct a nearly homogeneous initial state on the target-area shell."""

    config = model.config
    require_finite("noise_scale", noise_scale, minimum=0.0)
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("Initialization requires float32 or float64")
    device = torch.device(device)
    mean_area_factor = config.area_target / (config.area_constant * config.n_punctures)
    factor = torch.tensor(mean_area_factor, dtype=dtype, device=device)
    x0 = torch.hypot(factor, torch.full_like(factor, 0.5)) - 0.5
    boundary_distance = (x0 - 0.5).clamp_min(torch.finfo(dtype).eps)
    z0 = softplus_inverse(boundary_distance)
    position = z0.expand(config.n_punctures).clone()
    if noise_scale > 0.0:
        position = position + noise_scale * torch.randn(
            config.n_punctures,
            dtype=dtype,
            device=device,
            generator=generator,
        )
    return position
