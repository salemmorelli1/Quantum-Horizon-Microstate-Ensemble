"""Tests for the relaxed probability measure and physical geometry."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import torch

from quantum_horizon_ensemble.artifacts import (
    read_samples,
    write_json_atomic,
    write_samples_atomic,
)
from quantum_horizon_ensemble.hmc import potential_and_grad
from quantum_horizon_ensemble.model import (
    EnsembleConfig,
    RelaxedHorizonEnsemble,
    area_matched_initial_position,
    softplus_inverse,
)

torch.set_default_dtype(torch.float64)


class RelaxedModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = EnsembleConfig(
            n_punctures=8,
            area_target=80.0,
            kappa_area=0.01,
            kappa_quant=2.0,
            eta_degeneracy=0.5,
        )
        self.model = RelaxedHorizonEnsemble(self.config)

    def test_configuration_rejects_impossible_area(self) -> None:
        with self.assertRaises(ValueError):
            EnsembleConfig(n_punctures=8, area_target=1.0).validate()

    def test_softplus_map_respects_lower_bound(self) -> None:
        z = torch.linspace(-50.0, 20.0, self.config.n_punctures)
        x = self.model.proxy_spins(z)
        # At very negative finite z, the strictly positive Softplus increment
        # may round to zero in IEEE float64, yielding the numerical boundary.
        self.assertTrue(torch.all(x >= 0.5))
        self.assertGreater(float(x[-1]), 0.5)
        self.assertTrue(torch.isfinite(x).all())

    def test_comb_vanishes_at_reachable_half_integers(self) -> None:
        x = torch.tensor([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])
        z = softplus_inverse(x - 0.5)
        self.assertLess(float(self.model.comb_energy(z)), 1.0e-26)

    def test_autograd_matches_closed_form_gradient(self) -> None:
        z = torch.linspace(-1.5, 1.5, self.config.n_punctures)
        _, automatic = potential_and_grad(self.model, z)
        analytic = self.model.analytic_gradient(z)
        torch.testing.assert_close(automatic, analytic, atol=1.0e-10, rtol=1.0e-10)

    def test_jacobian_makes_negative_tail_confining(self) -> None:
        reference = torch.zeros(self.config.n_punctures)
        far = reference.clone()
        reference[0] = -10.0
        far[0] = -40.0
        self.assertGreater(float(self.model(far)), float(self.model(reference)))

    def test_area_matched_initialization(self) -> None:
        z = area_matched_initial_position(self.model, noise_scale=0.0)
        area = self.model.area(self.model.proxy_spins(z))
        self.assertAlmostEqual(float(area), self.config.area_target, places=9)

    def test_rounding_returns_half_integer_lattice(self) -> None:
        z = torch.linspace(-2.0, 2.0, self.config.n_punctures)
        rounded = self.model.rounded_spins(z)
        self.assertTrue(torch.all(rounded >= 0.5))
        torch.testing.assert_close(2.0 * rounded, torch.round(2.0 * rounded))

    def test_area_constant_matches_definition(self) -> None:
        expected = 8.0 * math.pi * self.config.gamma_bi * self.config.planck_length_sq
        self.assertAlmostEqual(self.config.area_constant, expected, places=14)

    def test_atomic_artifacts_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json_atomic(root / "result.json", {"value": torch.tensor([1.0, 2.0])})
            samples = torch.randn(5, self.config.n_punctures)
            write_samples_atomic(root / "samples.pt", samples)
            torch.testing.assert_close(read_samples(root / "samples.pt"), samples)


if __name__ == "__main__":
    unittest.main()
