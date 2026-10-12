#!/usr/bin/env python3
"""Part 1: run adaptive HMC and retain the relaxed proxy-spin chain."""

from __future__ import annotations

import argparse
from pathlib import Path

from quantum_horizon_ensemble.cli import load_experiment_config, save_experiment


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
    payload = save_experiment(
        config_path,
        seed=seed,
        samples_path=Path(args.samples_output),
        summary_path=Path(args.output),
    )
    print(f"Wrote {args.samples_output}")
    print(f"Wrote {args.output}")
    print(f"Sampling status: {payload['status']}; convergence: {payload['convergence_status']}")
    return 1 if payload["status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
