#!/usr/bin/env python3
"""Part 1: run adaptive HMC and retain the relaxed proxy-spin chain."""

from __future__ import annotations

import argparse
from pathlib import Path

from quantum_horizon_ensemble.artifacts import write_json_atomic, write_samples_atomic
from quantum_horizon_ensemble.cli import analysis_payload, load_experiment_config, run_experiment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.json")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--samples-output", default="artifacts/part1/samples.pt")
    parser.add_argument("--output", default="artifacts/part1/hmc_summary.json")
    args = parser.parse_args()
    config_path = Path(args.config)
    _, _, runtime, _ = load_experiment_config(config_path)
    seed = int(runtime.get("seed", 0) if args.seed is None else args.seed)
    result = run_experiment(config_path, seed_override=seed)
    write_samples_atomic(Path(args.samples_output), result.samples)
    payload = analysis_payload(config_path, result, seed=seed)
    write_json_atomic(Path(args.output), payload)
    print(f"Wrote {args.samples_output}")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
