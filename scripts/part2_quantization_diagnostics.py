#!/usr/bin/env python3
"""Part 2: summarize area-shell and half-integer-lattice behavior."""

from __future__ import annotations

import argparse
from pathlib import Path

from quantum_horizon_ensemble.artifacts import provenance, read_samples, write_json_atomic
from quantum_horizon_ensemble.cli import load_experiment_config
from quantum_horizon_ensemble.diagnostics import quantization_summary
from quantum_horizon_ensemble.model import RelaxedHorizonEnsemble


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.json")
    parser.add_argument("--samples", default="artifacts/part1/samples.pt")
    parser.add_argument("--output", default="artifacts/part2/quantization_summary.json")
    args = parser.parse_args()
    config_path = Path(args.config)
    ensemble, _, runtime, raw = load_experiment_config(config_path)
    model = RelaxedHorizonEnsemble(ensemble)
    payload = {
        "schema_version": "1.0",
        "status": "PASS",
        "config": raw,
        "quantization": quantization_summary(model, read_samples(Path(args.samples))),
        "provenance": provenance(config_path, int(runtime.get("seed", 0))),
    }
    write_json_atomic(Path(args.output), payload)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
