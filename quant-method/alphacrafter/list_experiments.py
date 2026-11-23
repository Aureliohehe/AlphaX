#!/usr/bin/env python3
"""List archived AlphaX experiment runs as JSON."""

import argparse
import json
from pathlib import Path

from alphax.experiment import list_runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="A run directory or the directory containing runs")
    args = parser.parse_args()
    try:
        runs = list_runs(args.root)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Cannot list experiments: {exc}\n")
    print(json.dumps(runs, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
