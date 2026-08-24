#!/usr/bin/env python3
"""Part 4: run the complete numerical unit-test suite."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]
    completed = subprocess.run(command, check=False)
    output = Path("artifacts/part4/validation_summary.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "status": "PASS" if completed.returncode == 0 else "FAIL",
                "command": command,
                "returncode": completed.returncode,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output}")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
