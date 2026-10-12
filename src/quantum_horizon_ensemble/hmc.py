"""Pure-PyTorch Hamiltonian Monte Carlo for the relaxed horizon ensemble."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch

from .model import RelaxedHorizonEnsemble, Tensor
from .validation import require_finite, require_float_tensor, require_integer


@dataclass(frozen=True)
class HMCConfig:
    """Numerical controls for adaptive warmup followed by fixed-kernel sampling."""

    num_samples: int = 200
    warmup_steps: int = 200
    step_size: float = 0.01
    min_leapfrog_steps: int = 5
    max_leapfrog_steps: int = 20
    step_jitter: float = 0.10
    target_accept: float = 0.85
    adapt_step_size: bool = True
    adapt_diagonal_mass: bool = True
    mass_update_interval: int = 50
    mass_ridge: float = 1.0e-3
    minimum_mass: float = 1.0e-3
    maximum_mass: float = 1.0e3
    divergence_threshold: float = 1000.0

    def validate(self) -> None:
        for name, minimum in (
            ("num_samples", 1),
            ("warmup_steps", 0),
            ("min_leapfrog_steps", 1),
            ("max_leapfrog_steps", 1),
            ("mass_update_interval", 5),
        ):
            require_integer(name, getattr(self, name), minimum=minimum)
        for name in (
            "step_size",
            "step_jitter",
            "target_accept",
            "mass_ridge",
            "minimum_mass",
            "maximum_mass",
            "divergence_threshold",
        ):
            require_finite(name, getattr(self, name))
        for name in ("adapt_step_size", "adapt_diagonal_mass"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        if self.step_size <= 0.0:
            raise ValueError("step_size must be positive")
        if not 1 <= self.min_leapfrog_steps <= self.max_leapfrog_steps:
            raise ValueError("Invalid leapfrog-step interval")
        if not 0.0 <= self.step_jitter < 1.0:
            raise ValueError("step_jitter must lie in [0, 1)")
        if not 0.0 < self.target_accept < 1.0:
            raise ValueError("target_accept must lie in (0, 1)")
        if self.mass_update_interval < 5:
            raise ValueError("mass_update_interval must be at least five")
        if self.mass_ridge <= 0.0:
            raise ValueError("mass_ridge must be positive")
        if not 0.0 < self.minimum_mass <= self.maximum_mass:
            raise ValueError("Invalid mass bounds")
        if self.divergence_threshold <= 0.0:
            raise ValueError("divergence_threshold must be positive")

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> HMCConfig:
        config = cls(**values)
        config.validate()
        return config

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DiagonalMetric:
    """Momentum covariance ``M`` represented by its diagonal."""

    mass_diag: Tensor

    def __post_init__(self) -> None:
        require_float_tensor("mass_diag", self.mass_diag)
        if self.mass_diag.ndim != 1 or self.mass_diag.numel() < 1:
            raise ValueError("mass_diag must be a nonempty one-dimensional tensor")
        if torch.any(self.mass_diag <= 0.0):
            raise ValueError("All mass entries must be positive")

    def sample_momentum(self, generator: torch.Generator | None = None) -> Tensor:
        noise = torch.randn(
            self.mass_diag.shape,
            dtype=self.mass_diag.dtype,
            device=self.mass_diag.device,
            generator=generator,
        )
        return torch.sqrt(self.mass_diag) * noise

    def velocity(self, momentum: Tensor) -> Tensor:
        self.validate_vector(momentum)
        return momentum / self.mass_diag

    def kinetic_energy(self, momentum: Tensor) -> Tensor:
        self.validate_vector(momentum)
        return 0.5 * torch.sum(momentum.square() / self.mass_diag)

    def validate_vector(self, vector: Tensor) -> None:
        if (
            vector.shape != self.mass_diag.shape
            or vector.dtype != self.mass_diag.dtype
            or vector.device != self.mass_diag.device
        ):
            raise ValueError("Metric and trajectory vectors must share shape, dtype, and device")


@dataclass(frozen=True)
class HMCDiagnostic:
    accepted: bool
    acceptance_probability: float
    delta_energy: float
    proposed_potential: float
    step_size: float
    n_steps: int
    divergent: bool
    warmup: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HMCResult:
    samples: Tensor
    diagnostics: tuple[HMCDiagnostic, ...]
    final_step_size: float
    final_mass_diag: Tensor
    metadata: dict[str, Any] | None = None


def potential_and_grad(
    model: RelaxedHorizonEnsemble,
    z: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return ``U(z)`` and its exact reverse-mode autograd gradient."""

    z_leaf = z.detach().requires_grad_(True)
    potential = model(z_leaf)
    if potential.ndim != 0 or not torch.isfinite(potential):
        raise FloatingPointError("Nonfinite or nonscalar potential energy")
    (gradient,) = torch.autograd.grad(
        outputs=potential,
        inputs=z_leaf,
        create_graph=False,
        retain_graph=False,
    )
    if not torch.isfinite(gradient).all():
        raise FloatingPointError("Nonfinite potential gradient")
    return potential.detach(), gradient.detach()


def leapfrog(
    model: RelaxedHorizonEnsemble,
    initial_z: Tensor,
    initial_momentum: Tensor,
    metric: DiagonalMetric,
    *,
    step_size: float,
    n_steps: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Apply a reversible, volume-preserving leapfrog trajectory."""

    require_finite("step_size", step_size)
    require_integer("n_steps", n_steps, minimum=1)
    if step_size <= 0.0:
        raise ValueError("step_size must be positive and n_steps at least one")
    metric.validate_vector(initial_z)
    metric.validate_vector(initial_momentum)
    require_float_tensor("initial momentum", initial_momentum)

    z = initial_z.detach().clone()
    momentum = initial_momentum.detach().clone()
    potential, gradient = potential_and_grad(model, z)
    momentum = momentum - 0.5 * step_size * gradient

    for step in range(n_steps):
        z = z + step_size * metric.velocity(momentum)
        if not torch.isfinite(z).all():
            raise FloatingPointError("Nonfinite trajectory position")
        potential, gradient = potential_and_grad(model, z)
        if step != n_steps - 1:
            momentum = momentum - step_size * gradient

    momentum = momentum - 0.5 * step_size * gradient

    # Momentum negation makes the complete proposal map an involution.
    return z.detach(), (-momentum).detach(), potential.detach()


def hmc_transition(
    model: RelaxedHorizonEnsemble,
    current_z: Tensor,
    metric: DiagonalMetric,
    *,
    step_size: float,
    n_steps: int,
    divergence_threshold: float,
    warmup: bool,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, HMCDiagnostic]:
    """Perform one Metropolis-corrected HMC transition."""

    require_finite("divergence_threshold", divergence_threshold)
    if divergence_threshold <= 0.0:
        raise ValueError("divergence_threshold must be positive")
    require_finite("step_size", step_size)
    if step_size <= 0.0:
        raise ValueError("step_size must be positive")
    require_integer("n_steps", n_steps, minimum=1)
    metric.validate_vector(current_z)
    current_z = current_z.detach()
    current_momentum = metric.sample_momentum(generator)
    current_potential, _ = potential_and_grad(model, current_z)
    current_hamiltonian = current_potential + metric.kinetic_energy(current_momentum)

    try:
        proposed_z, proposed_momentum, proposed_potential = leapfrog(
            model,
            current_z,
            current_momentum,
            metric,
            step_size=step_size,
            n_steps=n_steps,
        )
        proposed_hamiltonian = proposed_potential + metric.kinetic_energy(proposed_momentum)
        delta_energy_tensor = proposed_hamiltonian - current_hamiltonian
        if not torch.isfinite(delta_energy_tensor):
            raise FloatingPointError("Nonfinite Hamiltonian error")

        log_acceptance = torch.minimum(torch.zeros_like(delta_energy_tensor), -delta_energy_tensor)
        acceptance_probability = torch.exp(log_acceptance)
        log_uniform = torch.log(
            torch.rand(
                (),
                dtype=current_z.dtype,
                device=current_z.device,
                generator=generator,
            )
        )
        accepted_tensor = log_uniform < log_acceptance
        next_z = torch.where(accepted_tensor, proposed_z, current_z)
        accepted = bool(accepted_tensor.item())
        delta_energy = float(delta_energy_tensor.item())
        proposed_potential_value = float(proposed_potential.item())
        acceptance_value = float(acceptance_probability.item())
        divergent = abs(delta_energy) > divergence_threshold
    except FloatingPointError:
        next_z = current_z
        accepted = False
        delta_energy = float("inf")
        proposed_potential_value = float("inf")
        acceptance_value = 0.0
        divergent = True

    return next_z.detach(), HMCDiagnostic(
        accepted=accepted,
        acceptance_probability=acceptance_value,
        delta_energy=delta_energy,
        proposed_potential=proposed_potential_value,
        step_size=step_size,
        n_steps=n_steps,
        divergent=divergent,
        warmup=warmup,
    )


class DualAveraging:
    """Nesterov dual averaging for warmup-only step-size adaptation."""

    def __init__(self, initial_step_size: float, target: float):
        require_finite("target_accept", target)
        if not 0.0 < target < 1.0:
            raise ValueError("target_accept must lie in (0, 1)")
        self.target = target
        self.gamma = 0.05
        self.t0 = 10.0
        self.kappa = 0.75
        self.reset(initial_step_size)

    def reset(self, initial_step_size: float) -> None:
        require_finite("initial_step_size", initial_step_size)
        if initial_step_size <= 0.0:
            raise ValueError("initial_step_size must be positive")
        self.mu = math.log(initial_step_size) + math.log(10.0)
        self.log_step = math.log(initial_step_size)
        self.log_step_average = math.log(initial_step_size)
        self.h_average = 0.0
        self.iteration = 0

    def update(self, acceptance_probability: float) -> float:
        require_finite("acceptance_probability", acceptance_probability)
        if not 0.0 <= acceptance_probability <= 1.0:
            raise ValueError("acceptance_probability must lie in [0, 1]")
        self.iteration += 1
        t = float(self.iteration)
        eta = 1.0 / (t + self.t0)
        self.h_average = (1.0 - eta) * self.h_average + eta * (self.target - acceptance_probability)
        # Avoid overflow/underflow during extreme warmup. Adaptation is still
        # restricted to warmup, so this numerical bound does not alter sampling.
        self.log_step = max(-30.0, min(10.0, self.mu - math.sqrt(t) * self.h_average / self.gamma))
        weight = t ** (-self.kappa)
        self.log_step_average = weight * self.log_step + (1.0 - weight) * self.log_step_average
        return math.exp(self.log_step)

    def final_step_size(self) -> float:
        return math.exp(self.log_step_average)


class DiagonalWelford:
    """Online marginal-variance estimator used only during warmup."""

    def __init__(self, template: Tensor):
        self.count = 0
        self.mean = torch.zeros_like(template)
        self.m2 = torch.zeros_like(template)

    def update(self, value: Tensor) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean = self.mean + delta / self.count
        self.m2 = self.m2 + delta * (value - self.mean)

    def precision_mass(
        self,
        *,
        ridge: float,
        minimum: float,
        maximum: float,
    ) -> Tensor:
        if self.count < 2:
            return torch.ones_like(self.mean)
        variance = self.m2 / (self.count - 1)
        # In this implementation M is momentum covariance.  For a Gaussian
        # target, the frequency is stabilized by M approximately equal to the
        # target precision, hence the reciprocal marginal variance.
        return torch.reciprocal(variance + ridge).clamp(minimum, maximum)


def _randomized_controls(
    z: Tensor,
    config: HMCConfig,
    step_size: float,
    generator: torch.Generator | None,
) -> tuple[float, int]:
    jitter = torch.rand((), dtype=z.dtype, device=z.device, generator=generator)
    randomized_step = step_size * torch.exp(config.step_jitter * (2.0 * jitter - 1.0)).item()
    n_steps = int(
        torch.randint(
            config.min_leapfrog_steps,
            config.max_leapfrog_steps + 1,
            (),
            device=z.device,
            generator=generator,
        ).item()
    )
    return randomized_step, n_steps


def sample_hmc(
    model: RelaxedHorizonEnsemble,
    initial_z: Tensor,
    config: HMCConfig,
    *,
    generator: torch.Generator | None = None,
) -> HMCResult:
    """Run adaptive warmup and then a fixed-kernel HMC chain."""

    config.validate()
    model._validate_position(initial_z)
    z = initial_z.detach().clone()
    metric = DiagonalMetric(torch.ones_like(z))
    moments = DiagonalWelford(z)
    dual = DualAveraging(config.step_size, config.target_accept)
    step_size = config.step_size
    diagnostics: list[HMCDiagnostic] = []
    samples: list[Tensor] = []

    total = config.warmup_steps + config.num_samples
    for iteration in range(total):
        warmup = iteration < config.warmup_steps
        randomized_step, n_steps = _randomized_controls(z, config, step_size, generator)
        z, diagnostic = hmc_transition(
            model,
            z,
            metric,
            step_size=randomized_step,
            n_steps=n_steps,
            divergence_threshold=config.divergence_threshold,
            warmup=warmup,
            generator=generator,
        )
        diagnostics.append(diagnostic)

        if warmup:
            moments.update(z)
            if config.adapt_step_size:
                step_size = dual.update(diagnostic.acceptance_probability)

            completed = iteration + 1
            update_mass = (
                config.adapt_diagonal_mass
                and completed >= config.mass_update_interval
                and completed % config.mass_update_interval == 0
                and completed < config.warmup_steps
            )
            if update_mass:
                metric = DiagonalMetric(
                    moments.precision_mass(
                        ridge=config.mass_ridge,
                        minimum=config.minimum_mass,
                        maximum=config.maximum_mass,
                    )
                )
                if config.adapt_step_size:
                    dual.reset(step_size)

            if completed == config.warmup_steps:
                if config.adapt_step_size:
                    step_size = dual.final_step_size()
        else:
            samples.append(z.clone())

    return HMCResult(
        samples=torch.stack(samples),
        diagnostics=tuple(diagnostics),
        final_step_size=step_size,
        final_mass_diag=metric.mass_diag.detach().clone(),
    )
