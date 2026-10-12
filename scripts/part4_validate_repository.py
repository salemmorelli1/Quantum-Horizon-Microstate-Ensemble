#!/usr/bin/env python3
"""Part 4: run the complete numerical unit-test suite."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from quantum_horizon_ensemble.artifacts import write_json_atomic

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/part4/validation_summary.json")
    args = parser.parse_args()
    command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]
    completed = subprocess.run(command, cwd=ROOT, check=False)
    output = Path(args.output)
    write_json_atomic(
        output,
        {
            "schema_version": "1.0",
            "status": "PASS" if completed.returncode == 0 else "FAIL",
            "command": command,
            "returncode": completed.returncode,
        },
    )
    print(f"Wrote {output}")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
