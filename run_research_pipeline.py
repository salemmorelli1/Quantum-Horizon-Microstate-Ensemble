#!/usr/bin/env python3
"""Canonical Part 0 -> Part 4 quantum-horizon research runner."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/baseline.json"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--output-root", default=str(ROOT / "artifacts"))
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    output_root = Path(args.output_root).resolve()
    shared = ["--config", str(config_path)]
    samples_path = str(output_root / "part1/samples.pt")
    seed = [] if args.seed is None else ["--seed", str(args.seed)]
    stages = [
        (
            "Part 0",
            [
                sys.executable,
                "scripts/part0_validate_model.py",
                *shared,
                "--output",
                str(output_root / "part0/model_validation.json"),
            ],
        ),
        (
            "Part 1",
            [
                sys.executable,
                "scripts/part1_sample_relaxed_ensemble.py",
                *shared,
                *seed,
                "--samples-output",
                samples_path,
                "--output",
                str(output_root / "part1/hmc_summary.json"),
            ],
        ),
        (
            "Part 2",
            [
                sys.executable,
                "scripts/part2_quantization_diagnostics.py",
                *shared,
                "--samples",
                samples_path,
                "--output",
                str(output_root / "part2/quantization_summary.json"),
            ],
        ),
        (
            "Part 3",
            [
                sys.executable,
                "scripts/part3_entropy_geometry.py",
                *shared,
                "--samples",
                samples_path,
                "--output",
                str(output_root / "part3/geometry_summary.json"),
            ],
        ),
    ]
    if not args.skip_tests:
        stages.append(
            (
                "Part 4",
                [
                    sys.executable,
                    "scripts/part4_validate_repository.py",
                    "--output",
                    str(output_root / "part4/validation_summary.json"),
                ],
            )
        )

    for label, command in stages:
        print(f"\n[{label}] {' '.join(command)}", flush=True)
        completed = subprocess.run(command, cwd=ROOT, check=False)
        if completed.returncode != 0:
            print(f"[{label}] FAILED with exit code {completed.returncode}", file=sys.stderr)
            return completed.returncode
        print(f"[{label}] completed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
