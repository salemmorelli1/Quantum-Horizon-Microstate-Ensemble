"""Independent checks of target measure, MCMC uncertainty, and numerical controls."""

from __future__ import annotations

import math
import unittest
from dataclasses import replace

import torch

from quantum_horizon_ensemble.diagnostics import (
    effective_sample_size,
    geometry_summary,
    hmc_diagnostic_summary,
    quantization_summary,
)
from quantum_horizon_ensemble.hmc import (
    DiagonalMetric,
    DualAveraging,
    HMCConfig,
    hmc_transition,
    leapfrog,
    potential_and_grad,
    sample_hmc,
)
from quantum_horizon_ensemble.model import (
    EnsembleConfig,
    RelaxedHorizonEnsemble,
    area_matched_initial_position,
    softplus_inverse,
)


class InputIntegrityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = EnsembleConfig(4, 40.0)
        self.model = RelaxedHorizonEnsemble(self.config)
        self.initial = area_matched_initial_position(self.model, noise_scale=0.0)

    def test_all_physical_controls_reject_nonfinite_values(self) -> None:
        for name in (
            "area_target",
            "gamma_bi",
            "planck_length_sq",
            "kappa_area",
            "kappa_quant",
            "eta_degeneracy",
        ):
            for value in (float("nan"), float("inf"), -float("inf"), True, "1"):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    replace(self.config, **{name: value}).validate()

    def test_puncture_count_is_a_positive_integer(self) -> None:
        for value in (True, 4.5, 0, -1, "4"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                replace(self.config, n_punctures=value).validate()

    def test_all_hmc_numeric_controls_reject_nonfinite_values(self) -> None:
        for name in (
            "step_size",
            "step_jitter",
            "target_accept",
            "mass_ridge",
            "minimum_mass",
            "maximum_mass",
            "divergence_threshold",
        ):
            for value in (float("nan"), float("inf"), -float("inf"), True):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    replace(HMCConfig(), **{name: value}).validate()

    def test_hmc_counts_and_adaptation_flags_have_strict_types(self) -> None:
        for name in (
            "num_samples",
            "warmup_steps",
            "min_leapfrog_steps",
            "max_leapfrog_steps",
            "mass_update_interval",
        ):
            for value in (True, 3.5, "5"):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    replace(HMCConfig(), **{name: value}).validate()
        for name in ("adapt_step_size", "adapt_diagonal_mass"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                replace(HMCConfig(), **{name: "false"}).validate()
        for value in (0.0, -1.0):
            with self.subTest(value=value), self.assertRaises(ValueError):
                replace(HMCConfig(), divergence_threshold=value).validate()

    def test_positions_and_initialization_reject_invalid_numbers(self) -> None:
        for z in (
            torch.ones(4, dtype=torch.int64),
            torch.full((4,), float("nan")),
            torch.full((4,), float("inf")),
            torch.ones(1, 4),
            torch.ones(3),
        ):
            with self.subTest(z=z), self.assertRaises(ValueError):
                self.model.proxy_spins(z)
        for noise in (-1.0, float("nan"), float("inf"), True):
            with self.subTest(noise=noise), self.assertRaises(ValueError):
                area_matched_initial_position(self.model, noise_scale=noise)

    def test_inverse_softplus_is_finite_and_stable(self) -> None:
        y = torch.tensor([1e-12, 0.02, 1.0, 40.0, 1e6], dtype=torch.float64)
        inverse = softplus_inverse(y)
        torch.testing.assert_close(torch.logaddexp(inverse, torch.zeros_like(inverse)), y)
        for value in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                softplus_inverse(torch.tensor([value]))

    def test_analytic_gradient_agrees_in_softplus_tails(self) -> None:
        z = torch.tensor([-40.0, -2.0, 20.0001, 50.0], dtype=torch.float64)
        _, automatic = potential_and_grad(self.model, z)
        torch.testing.assert_close(
            automatic, self.model.analytic_gradient(z), atol=1e-10, rtol=1e-10
        )

    def test_metric_rejects_empty_and_integer_mass(self) -> None:
        for value in (torch.ones(4, dtype=torch.int64), torch.empty(0), torch.zeros(4)):
            with self.subTest(value=value), self.assertRaises(ValueError):
                DiagonalMetric(value)

    def test_leapfrog_rejects_broadcasting_and_invalid_controls(self) -> None:
        metric = DiagonalMetric(torch.ones_like(self.initial))
        with self.assertRaises(ValueError):
            leapfrog(self.model, self.initial, torch.ones(1), metric, step_size=0.01, n_steps=2)
        for controls in (
            {"step_size": float("nan"), "n_steps": 2},
            {"step_size": 0.01, "n_steps": True},
        ):
            with self.subTest(controls=controls), self.assertRaises(ValueError):
                leapfrog(self.model, self.initial, self.initial, metric, **controls)

    def test_extreme_proposal_is_rejected_and_counted(self) -> None:
        metric = DiagonalMetric(torch.ones_like(self.initial))
        next_z, diagnostic = hmc_transition(
            self.model,
            self.initial,
            metric,
            step_size=1e308,
            n_steps=2,
            divergence_threshold=1000.0,
            warmup=False,
            generator=torch.Generator().manual_seed(1),
        )
        torch.testing.assert_close(next_z, self.initial)
        self.assertFalse(diagnostic.accepted)
        self.assertTrue(diagnostic.divergent)
        summary = hmc_diagnostic_summary([diagnostic])
        self.assertEqual(summary["nonfinite_energy_errors"], 1)
        self.assertIsNone(summary["maximum_absolute_energy_error"])

    def test_dual_averaging_does_not_overflow(self) -> None:
        dual = DualAveraging(0.01, 0.85)
        for _ in range(10000):
            self.assertTrue(math.isfinite(dual.update(1.0)))
        self.assertGreater(dual.final_step_size(), 0.0)

    def test_nonunit_metric_leapfrog_is_involutive(self) -> None:
        metric = DiagonalMetric(torch.tensor([0.2, 0.5, 2.0, 10.0], dtype=torch.float64))
        momentum = torch.tensor([-0.2, 0.3, -0.5, 0.7], dtype=torch.float64)
        z1, p1, _ = leapfrog(
            self.model, self.initial, momentum, metric, step_size=0.001, n_steps=10
        )
        z2, p2, _ = leapfrog(self.model, z1, p1, metric, step_size=0.001, n_steps=10)
        torch.testing.assert_close(z2, self.initial, atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(p2, momentum, atol=1e-10, rtol=1e-10)


class StatisticalDiagnosticsTests(unittest.TestCase):
    def test_constant_and_short_traces_have_unidentified_ess(self) -> None:
        self.assertIsNone(effective_sample_size(torch.ones(80)))
        self.assertIsNone(effective_sample_size(torch.arange(3, dtype=torch.float64)))

    def test_ess_rejects_nonfinite_empty_and_multidimensional_traces(self) -> None:
        for trace in (torch.empty(0), torch.ones(2, 8), torch.tensor([float("nan")] * 8)):
            with self.subTest(trace=trace), self.assertRaises(ValueError):
                effective_sample_size(trace)

    def test_ess_is_invariant_to_units(self) -> None:
        trace = torch.sin(torch.arange(128, dtype=torch.float64) / 7.0)
        reference = effective_sample_size(trace)
        for scale in (1e-20, 1e200):
            self.assertAlmostEqual(effective_sample_size(scale * trace), reference, places=10)

    def test_ess_distinguishes_iid_and_correlated_draws(self) -> None:
        noise = torch.randn(4608, dtype=torch.float64, generator=torch.Generator().manual_seed(93))
        trace = torch.zeros_like(noise)
        for i in range(1, trace.numel()):
            trace[i] = 0.9 * trace[i - 1] + noise[i]
        correlated = effective_sample_size(trace[512:])
        iid = effective_sample_size(noise[512:])
        # AR(1) theoretical ESS is n*(1-rho)/(1+rho) = 215.6.
        self.assertTrue(100 < correlated < 400)
        self.assertGreater(iid, 2500)

    def test_one_draw_has_no_fabricated_sd_or_mcse(self) -> None:
        model = RelaxedHorizonEnsemble(EnsembleConfig(1, 10.0))
        summary = quantization_summary(model, torch.zeros(1, 1))
        self.assertIsNone(summary["area"]["sd"])
        self.assertIsNone(summary["area"]["mcse_mean"])
        self.assertIsNone(summary["area"]["ess"])
        self.assertEqual(summary["mixing"]["convergence_status"], "NOT_ASSESSED_SINGLE_CHAIN")

    def test_thermodynamic_integrand_matches_derivative_of_log_normalizer(self) -> None:
        config = EnsembleConfig(1, 10.0, kappa_area=0.3, kappa_quant=0.7, eta_degeneracy=0.5)
        x = torch.linspace(0.5, 8.0, 20001, dtype=torch.float64)
        area = config.area_constant * torch.sqrt(x * (x + 1.0))
        base = -0.5 * config.kappa_area * (area - config.area_target).square()
        base += config.eta_degeneracy * torch.log(2.0 * x + 1.0)
        comb = torch.sin(2.0 * math.pi * x).square()

        def log_normalizer(kappa: float) -> torch.Tensor:
            return torch.log(torch.trapezoid(torch.exp(base - kappa * comb), x))

        weights = torch.exp(base - config.kappa_quant * comb)
        expected_q = torch.trapezoid(weights * comb, x) / torch.trapezoid(weights, x)
        derivative = (
            log_normalizer(config.kappa_quant + 1e-5) - log_normalizer(config.kappa_quant - 1e-5)
        ) / 2e-5
        self.assertAlmostEqual(float(derivative), -float(expected_q), places=8)
        model = RelaxedHorizonEnsemble(config)
        summary = quantization_summary(
            model, torch.linspace(-2, 1, 20, dtype=torch.float64).reshape(-1, 1)
        )
        self.assertEqual(
            summary["comb_statistic"]["thermodynamic_integration_integrand"],
            -summary["comb_statistic"]["mean"],
        )

    def test_sample_summary_rejects_corrupt_and_empty_samples(self) -> None:
        model = RelaxedHorizonEnsemble(EnsembleConfig(1, 10.0))
        for samples in (torch.empty(0, 1), torch.ones(4, 2), torch.full((8, 1), float("inf"))):
            for function in (quantization_summary, geometry_summary):
                with (
                    self.subTest(function=function, samples=samples),
                    self.assertRaises(ValueError),
                ):
                    function(model, samples)

    def test_hmc_matches_independent_one_dimensional_quadrature(self) -> None:
        config = EnsembleConfig(1, 10.0, kappa_area=0.3, kappa_quant=0.25, eta_degeneracy=0.5)
        model = RelaxedHorizonEnsemble(config)
        x_grid = torch.linspace(0.5, 8.0, 30001, dtype=torch.float64)
        area = config.area_constant * torch.sqrt(x_grid * (x_grid + 1.0))
        log_density = (
            -0.5 * config.kappa_area * (area - config.area_target).square()
            - config.kappa_quant * torch.sin(2 * math.pi * x_grid).square()
            + config.eta_degeneracy * torch.log(2 * x_grid + 1.0)
        )
        density = torch.exp(log_density - log_density.max())
        normalizer = torch.trapezoid(density, x_grid)
        expected_mean = float(torch.trapezoid(x_grid * density, x_grid) / normalizer)
        expected_second = float(torch.trapezoid(x_grid.square() * density, x_grid) / normalizer)
        generator = torch.Generator().manual_seed(547)
        result = sample_hmc(
            model,
            area_matched_initial_position(model, noise_scale=0.0),
            HMCConfig(
                num_samples=1600,
                warmup_steps=150,
                step_size=0.12,
                min_leapfrog_steps=3,
                max_leapfrog_steps=8,
                adapt_diagonal_mass=False,
                adapt_step_size=False,
            ),
            generator=generator,
        )
        x = 0.5 + torch.logaddexp(result.samples[:, 0], torch.zeros_like(result.samples[:, 0]))
        for trace, expected in ((x, expected_mean), (x.square(), expected_second)):
            ess = effective_sample_size(trace)
            self.assertGreater(ess, 100)
            mcse = float(trace.std()) / math.sqrt(ess)
            self.assertLess(abs(float(trace.mean()) - expected), 6 * mcse + 1e-4)
        self.assertGreater(hmc_diagnostic_summary(result.diagnostics)["acceptance_rate"], 0.9)
        self.assertEqual(hmc_diagnostic_summary(result.diagnostics)["divergences"], 0)


if __name__ == "__main__":
    unittest.main()
