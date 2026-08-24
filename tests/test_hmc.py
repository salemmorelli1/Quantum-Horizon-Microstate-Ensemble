"""Tests for the custom leapfrog integrator and adaptive HMC chain."""

from __future__ import annotations

import unittest

import torch

from quantum_horizon_ensemble.diagnostics import (
    geometry_summary,
    hmc_diagnostic_summary,
    quantization_summary,
)
from quantum_horizon_ensemble.hmc import (
    DiagonalMetric,
    HMCConfig,
    leapfrog,
    sample_hmc,
)
from quantum_horizon_ensemble.model import (
    EnsembleConfig,
    RelaxedHorizonEnsemble,
    area_matched_initial_position,
)

torch.set_default_dtype(torch.float64)


class HMCTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = RelaxedHorizonEnsemble(
            EnsembleConfig(
                n_punctures=6,
                area_target=60.0,
                kappa_area=0.01,
                kappa_quant=1.5,
                eta_degeneracy=0.25,
            )
        )
        self.initial = area_matched_initial_position(self.model, noise_scale=0.0)
        self.metric = DiagonalMetric(torch.ones_like(self.initial))

    def test_leapfrog_proposal_is_numerically_involutive(self) -> None:
        momentum = torch.linspace(-0.3, 0.3, self.initial.numel())
        z1, p1, _ = leapfrog(
            self.model,
            self.initial,
            momentum,
            self.metric,
            step_size=1.0e-4,
            n_steps=5,
        )
        z2, p2, _ = leapfrog(
            self.model,
            z1,
            p1,
            self.metric,
            step_size=1.0e-4,
            n_steps=5,
        )
        torch.testing.assert_close(z2, self.initial, atol=1.0e-10, rtol=1.0e-10)
        torch.testing.assert_close(p2, momentum, atol=1.0e-10, rtol=1.0e-10)

    def test_short_hmc_chain_is_finite(self) -> None:
        generator = torch.Generator().manual_seed(20260823)
        config = HMCConfig(
            num_samples=16,
            warmup_steps=16,
            step_size=0.003,
            min_leapfrog_steps=2,
            max_leapfrog_steps=4,
            mass_update_interval=8,
        )
        result = sample_hmc(self.model, self.initial, config, generator=generator)
        self.assertEqual(tuple(result.samples.shape), (16, 6))
        self.assertTrue(torch.isfinite(result.samples).all())
        self.assertGreater(result.final_step_size, 0.0)
        self.assertTrue(torch.all(result.final_mass_diag > 0.0))

        hmc = hmc_diagnostic_summary(result.diagnostics)
        quantization = quantization_summary(self.model, result.samples)
        geometry = geometry_summary(self.model, result.samples)
        self.assertEqual(hmc["retained_transitions"], 16)
        self.assertTrue(0.0 <= hmc["acceptance_rate"] <= 1.0)
        self.assertIn("area", quantization)
        self.assertEqual(geometry["hessian_structure"], "diagonal plus rank one in proxy-spin coordinates")


if __name__ == "__main__":
    unittest.main()
