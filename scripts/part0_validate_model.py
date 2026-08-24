#!/usr/bin/env python3
"""Part 0: validate the probability measure, gradient, and configuration."""

from __future__ import annotations

import argparse
from pathlib import Path

from quantum_horizon_ensemble.artifacts import provenance, write_json_atomic
from quantum_horizon_ensemble.cli import load_experiment_config, validation_payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.json")
    parser.add_argument("--output", default="artifacts/part0/model_validation.json")
    args = parser.parse_args()
    config_path = Path(args.config)
    payload = validation_payload(config_path)
    _, _, runtime, _ = load_experiment_config(config_path)
    payload["provenance"] = provenance(config_path, int(runtime.get("seed", 0)))
    write_json_atomic(Path(args.output), payload)
    print(f"Wrote {args.output}")
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
