#!/usr/bin/env python3
"""Part 2: summarize area-shell and half-integer-lattice behavior."""

from __future__ import annotations

import argparse
from pathlib import Path

from quantum_horizon_ensemble.artifacts import write_json_atomic
from quantum_horizon_ensemble.cli import verified_analysis_input
from quantum_horizon_ensemble.diagnostics import quantization_summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.json")
    parser.add_argument("--samples", default="artifacts/part1/samples.pt")
    parser.add_argument("--output", default="artifacts/part2/quantization_summary.json")
    args = parser.parse_args()
    config_path = Path(args.config)
    model, samples, raw, producer = verified_analysis_input(config_path, Path(args.samples))
    payload = {
        "schema_version": "2.0",
        "status": producer["sampling_status"],
        "convergence_status": "NOT_ASSESSED_SINGLE_CHAIN",
        "config": raw,
        "quantization": quantization_summary(model, samples),
        "provenance": producer,
    }
    write_json_atomic(Path(args.output), payload)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
