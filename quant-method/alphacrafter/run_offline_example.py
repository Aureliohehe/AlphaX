#!/usr/bin/env python3
"""Persist a deterministic synthetic pipeline run; its returns are not market evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd


def run(output: Path) -> None:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    dates = pd.bdate_range("2021-11-01", periods=100)
    rng = np.random.default_rng(24)
    rows = []
    for index, date in enumerate(dates):
        for asset in range(16):
            value = asset - 7.5
            rows.append({
                "date": date, "ticker": f"S{asset:02d}", "feature_x": value,
                "return_1d": rng.normal(0, .01),
                "target_open_close": .003 * value + rng.normal(0, .01),
                "target_1d": .003 * value + rng.normal(0, .01),
                "target_5d": .005 * value + rng.normal(0, .01),
                "target_open_close_5d": .005 * value + rng.normal(0, .01),
                "label_end_1d": dates[index + 1] if index + 1 < len(dates) else pd.NaT,
                "label_end_5d": dates[index + 5] if index + 5 < len(dates) else pd.NaT,
            })
    pd.DataFrame(rows).to_csv(output / "features.csv", index=False)
    pd.DataFrame({"date": dates}).to_csv(output / "calendar.csv", index=False)
    (output / "candidates.json").write_text(json.dumps([{
        "candidate_id": "synthetic_x", "op": "feature", "features": ["feature_x"],
        "hypothesis": "Artificial signal for testing pipeline execution only",
    }], indent=2) + "\n")
    (output / "README.txt").write_text(
        "SYNTHETIC DATA ONLY. Labels intentionally depend on feature_x.\n"
        "Returns validate software execution, not investment performance.\n"
        "Inspect pipeline/status.json, manifest.json, logs/ and trader/selection.json.\n"
    )
    day = lambda index: str(dates[index].date())
    subprocess.run([
        sys.executable, str(Path(__file__).with_name("run_pipeline.py")),
        "--input", str(output / "features.csv"),
        "--calendar", str(output / "calendar.csv"), "--require-explicit-label-dates",
        "--candidate-file", str(output / "candidates.json"),
        "--output-dir", str(output / "pipeline"),
        "--initial-as-of", day(44), "--train-start", day(0),
        "--selection-start", day(45), "--selection-end", day(69),
        "--holdout-start", day(70), "--holdout-end", day(89),
        "--admission-sessions", "20", "--min-train-sessions", "20",
        "--maintenance-days", "14", "--configs", ".8,0,2,2;.6,0,2,2",
    ], check=True)
    print(f"Synthetic example artifacts: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; never overwritten")
    args = parser.parse_args()
    run(args.output_dir)
