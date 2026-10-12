"""Exercise real subprocess entry points using disposable configurations/output."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from test_artifact_provenance import experiment_config

from quantum_horizon_ensemble.cli import load_experiment_config, run_experiment

ROOT = Path(__file__).resolve().parents[1]


class PipelineContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.config = self.root / "config.json"
        self.config.write_text(json.dumps(experiment_config()), encoding="utf-8")
        self.env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}

    def run_command(self, command: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, *command],
            cwd=self.root,
            env=self.env,
            text=True,
            capture_output=True,
            check=False,
            timeout=120,
        )

    def test_pipeline_from_external_directory_preserves_override_provenance(self) -> None:
        completed = self.run_command(
            [
                str(ROOT / "run_research_pipeline.py"),
                "--config",
                "config.json",
                "--output-root",
                "outputs",
                "--seed",
                "123",
                "--skip-tests",
            ]
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        reports = [
            json.loads((self.root / "outputs" / path).read_text())
            for path in (
                "part1/hmc_summary.json",
                "part2/quantization_summary.json",
                "part3/geometry_summary.json",
            )
        ]
        self.assertEqual(len({report["provenance"]["run_id"] for report in reports}), 1)
        self.assertEqual(
            len({report["provenance"]["samples_file_sha256"] for report in reports}), 1
        )
        for report in reports:
            self.assertEqual(report["provenance"]["seed"], 123)
            self.assertEqual(report["convergence_status"], "NOT_ASSESSED_SINGLE_CHAIN")
            self.assertIn(report["status"], ("WARN", "DIAGNOSTICS_ONLY"))
        self.assertEqual(reports[0]["quantization"], reports[1]["quantization"])
        self.assertEqual(reports[0]["geometry"], reports[2]["geometry"])

    def test_sampling_is_reproducible_without_global_dtype_side_effects(self) -> None:
        previous = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float32)
            first = run_experiment(self.config, seed_override=123)
            second = run_experiment(self.config, seed_override=123)
            self.assertEqual(torch.get_default_dtype(), torch.float32)
        finally:
            torch.set_default_dtype(previous)
        torch.testing.assert_close(first.samples, second.samples, rtol=0, atol=0)
        self.assertEqual(first.samples.dtype, torch.float64)
        self.assertNotEqual(first.metadata["run_id"], second.metadata["run_id"])

    def test_invalid_runtime_controls_are_rejected(self) -> None:
        for controls in (
            {"seed": True},
            {"seed": 1.5},
            {"seed": -1},
            {"seed": 2**63},
            {"initial_noise_scale": -1},
            {"initial_noise_scale": float("nan")},
            {"device": False},
            {"unused_typo": 1},
        ):
            raw = experiment_config()
            raw["runtime"] = controls
            self.config.write_text(json.dumps(raw), encoding="utf-8")
            with self.subTest(controls=controls), self.assertRaises(ValueError):
                load_experiment_config(self.config)

    def test_cli_failure_cannot_write_a_passing_report(self) -> None:
        raw = experiment_config()
        raw["ensemble"]["kappa_area"] = float("nan")
        self.config.write_text(json.dumps(raw), encoding="utf-8")
        completed = self.run_command(
            [
                "-m",
                "quantum_horizon_ensemble.cli",
                "validate",
                "--config",
                "config.json",
                "--output",
                "invalid.json",
            ]
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("Nonfinite JSON", completed.stderr)
        self.assertFalse((self.root / "invalid.json").exists())

    def test_pipeline_stops_before_sampling_on_invalid_config(self) -> None:
        raw = experiment_config()
        raw["hmc"]["divergence_threshold"] = -1
        self.config.write_text(json.dumps(raw), encoding="utf-8")
        completed = self.run_command(
            [
                str(ROOT / "run_research_pipeline.py"),
                "--config",
                "config.json",
                "--output-root",
                "outputs",
                "--skip-tests",
            ]
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertNotIn("[Part 1]", completed.stdout)
        self.assertFalse((self.root / "outputs/part1/samples.pt").exists())


if __name__ == "__main__":
    unittest.main()
