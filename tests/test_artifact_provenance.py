"""Regressions for stale, swapped, corrupt, and unidentified scientific artifacts."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch

from quantum_horizon_ensemble.artifacts import (
    read_sample_artifact,
    read_samples,
    sha256,
    write_json_atomic,
    write_samples_atomic,
)
from quantum_horizon_ensemble.cli import (
    analysis_payload,
    load_experiment_config,
    run_experiment,
    save_experiment,
    verified_analysis_input,
)
from quantum_horizon_ensemble.hmc import HMCDiagnostic


def experiment_config() -> dict:
    return {
        "ensemble": {
            "n_punctures": 3,
            "area_target": 30.0,
            "kappa_area": 0.05,
            "kappa_quant": 0.3,
            "eta_degeneracy": 0.25,
        },
        "hmc": {
            "num_samples": 24,
            "warmup_steps": 24,
            "step_size": 0.01,
            "min_leapfrog_steps": 2,
            "max_leapfrog_steps": 3,
            "mass_update_interval": 8,
        },
        "runtime": {"seed": 88, "device": "cpu", "initial_noise_scale": 0.02},
    }


class ArtifactProvenanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.config = self.root / "config.json"
        self.config.write_text(json.dumps(experiment_config()), encoding="utf-8")
        self.samples = self.root / "samples.pt"
        self.summary = self.root / "summary.json"
        self.payload = save_experiment(
            self.config, seed=123, samples_path=self.samples, summary_path=self.summary
        )

    def test_artifact_retains_effective_seed_run_and_file_digest(self) -> None:
        model, samples, raw, producer = verified_analysis_input(self.config, self.samples)
        self.assertEqual(producer["seed"], 123)
        self.assertEqual(raw["runtime"]["seed"], 88)
        self.assertEqual(producer["run_id"], self.payload["provenance"]["run_id"])
        self.assertEqual(producer["samples_file_sha256"], sha256(self.samples))
        self.assertEqual(tuple(samples.shape), (24, model.config.n_punctures))
        self.assertNotEqual(self.payload["status"], "PASS")
        self.assertEqual(self.payload["convergence_status"], "NOT_ASSESSED_SINGLE_CHAIN")

    def test_same_dimension_different_configuration_is_rejected(self) -> None:
        raw = experiment_config()
        raw["ensemble"]["area_target"] = 31.0
        self.config.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "configuration SHA-256 mismatch"):
            verified_analysis_input(self.config, self.samples)

    def test_changed_sample_content_is_rejected(self) -> None:
        payload = torch.load(self.samples, weights_only=True)
        payload["samples"][0, 0] += 0.01
        torch.save(payload, self.samples)
        with self.assertRaisesRegex(ValueError, "content SHA-256 mismatch"):
            read_samples(self.samples)

    def test_changed_seed_metadata_is_rejected(self) -> None:
        payload = torch.load(self.samples, weights_only=True)
        payload["metadata"]["seed"] = 999
        torch.save(payload, self.samples)
        with self.assertRaisesRegex(ValueError, "metadata SHA-256 mismatch"):
            read_samples(self.samples)

    def test_changed_producer_source_is_rejected(self) -> None:
        with patch("quantum_horizon_ensemble.artifacts.source_sha256", return_value="0" * 64):
            with self.assertRaisesRegex(ValueError, "source SHA-256 mismatch"):
                verified_analysis_input(self.config, self.samples)

    def test_legacy_samples_cannot_receive_new_provenance(self) -> None:
        torch.save({"samples": torch.zeros(24, 3)}, self.samples)
        with self.assertRaisesRegex(ValueError, "legacy sample artifact"):
            verified_analysis_input(self.config, self.samples)

    def test_draw_count_mismatch_is_rejected(self) -> None:
        samples, metadata = read_sample_artifact(self.samples)
        write_samples_atomic(self.samples, samples[:-1], metadata=metadata)
        with self.assertRaisesRegex(ValueError, "draw count"):
            verified_analysis_input(self.config, self.samples)

    def test_failed_chain_cannot_be_reused_for_downstream_claims(self) -> None:
        samples, metadata = read_sample_artifact(self.samples)
        metadata["sampling_status"] = "FAIL"
        write_samples_atomic(self.samples, samples, metadata=metadata)
        with self.assertRaisesRegex(ValueError, "usable producer sampling assessment"):
            verified_analysis_input(self.config, self.samples)

    def test_analysis_rejects_relabeling_after_run(self) -> None:
        result = run_experiment(self.config, seed_override=123)
        with self.assertRaisesRegex(ValueError, "seed does not match"):
            analysis_payload(self.config, result, seed=88)
        raw = experiment_config()
        raw["ensemble"]["area_target"] = 31.0
        self.config.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "changed after sampling"):
            analysis_payload(self.config, result, seed=123)

    def test_fully_rejected_chain_is_failed_not_passed(self) -> None:
        result = run_experiment(self.config, seed_override=123)
        diagnostics = tuple(
            HMCDiagnostic(False, 0.0, 1.0, 1.0, 0.01, 2, False, False) for _ in range(24)
        )
        failed = replace(result, samples=torch.zeros_like(result.samples), diagnostics=diagnostics)
        payload = analysis_payload(self.config, failed, seed=123)
        self.assertEqual(payload["status"], "FAIL")
        self.assertIsNone(payload["quantization"]["area"]["ess"])

    def test_nonfinite_json_never_silently_becomes_null_or_invalid_json(self) -> None:
        output = self.root / "finite.json"
        write_json_atomic(output, {"ok": 1.0})
        original = output.read_bytes()
        for value in (float("nan"), torch.tensor(float("inf")), torch.tensor([float("nan")])):
            with self.subTest(value=value), self.assertRaises(ValueError):
                write_json_atomic(output, {"bad": value})
            self.assertEqual(output.read_bytes(), original)

    def test_failed_json_replacement_cleans_up_and_preserves_prior_file(self) -> None:
        output = self.root / "atomic.json"
        write_json_atomic(output, {"before": 1})
        entries = set(self.root.iterdir())
        with patch("quantum_horizon_ensemble.artifacts.os.replace", side_effect=OSError("test")):
            with self.assertRaises(OSError):
                write_json_atomic(output, {"after": 2})
        self.assertEqual(json.loads(output.read_text()), {"before": 1})
        self.assertEqual(set(self.root.iterdir()), entries)

    def test_sample_writer_rejects_empty_nonfinite_and_integer_tensors(self) -> None:
        for samples in (
            torch.empty(0, 3),
            torch.ones(4, 3, dtype=torch.int64),
            torch.full((4, 3), float("nan")),
        ):
            with self.subTest(samples=samples), self.assertRaises(ValueError):
                write_samples_atomic(self.root / "bad.pt", samples)
        self.assertFalse((self.root / "bad.pt").exists())

    def test_output_paths_cannot_destroy_config_or_each_other(self) -> None:
        for samples, summary in (
            (self.config, self.summary),
            (self.samples, self.config),
            (self.samples, self.samples),
        ):
            with self.subTest(samples=samples, summary=summary), self.assertRaises(ValueError):
                save_experiment(self.config, seed=123, samples_path=samples, summary_path=summary)
        load_experiment_config(self.config)


if __name__ == "__main__":
    unittest.main()
