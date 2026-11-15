#!/usr/bin/env python3
"""Spot-check curated local pandas fields against Qlib Alpha158 expressions.

This only compares local fields to a Qlib provider built from the SAME raw
Yahoo quotes. It does not certify price vintages, a PIT universe or all 158
expressions. Default sample: six tickers in 2022; caller may expand it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_alpha158_model_diagnostic import candidate_features, fingerprint


def compare(local: pd.DataFrame, official: pd.DataFrame, fields: list[str], atol: float) -> dict:
    local, official = local.align(official, join="inner", axis=0)
    if local.empty or len(local) != len(official):
        raise ValueError("No overlapping Qlib and local prices")
    summary = {}
    for field in fields:
        a, b = local[field], official[field]
        mismatched_null = int((a.isna() != b.isna()).sum())
        valid = a.notna() & b.notna()
        delta = np.abs(a[valid].to_numpy(dtype=float) - b[valid].to_numpy(dtype=float))
        if not len(delta):
            raise ValueError(f"No comparable samples for {field}")
        summary[field] = {
            "samples": len(delta), "missing_mismatch": mismatched_null,
            "median_absolute_error": float(np.median(delta)),
            "p99_absolute_error": float(np.quantile(delta, 0.99)),
            "max_absolute_error": float(np.max(delta)),
            "rows_exceeding_tolerance": int((delta > atol).sum()),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--provider", type=Path, required=True)
    parser.add_argument("--candidate-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "JPM", "XOM", "NVDA", "TSLA"])
    parser.add_argument("--start", default="2022-01-03")
    parser.add_argument("--end", default="2022-12-30")
    parser.add_argument("--absolute-tolerance", type=float, default=1e-4)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; use a fresh path")
    if not np.isfinite(args.absolute_tolerance) or args.absolute_tolerance <= 0:
        parser.error("absolute-tolerance must be positive and finite")
    if len(set(args.tickers)) != len(args.tickers) or not args.tickers:
        parser.error("Specify distinct ticker symbols")
    args.input, args.candidate_file = args.input.resolve(strict=True), args.candidate_file.resolve(strict=True)
    args.provider = args.provider.resolve(strict=True)
    fields = candidate_features(args.candidate_file)
    try:
        import qlib
        from qlib.contrib.data.handler import Alpha158
        from qlib.data import D
    except ImportError as exc:
        parser.error(f"Qlib is required for a real expression comparison: {exc}")
    expressions, names = Alpha158.get_feature_config(Alpha158.__new__(Alpha158))
    official_map = dict(zip(names, expressions))
    try:
        selected_expr = [official_map[field[len("alpha158_"):].upper()] for field in fields]
    except KeyError as exc:
        parser.error(f"Selected feature has no official Qlib Alpha158 name: {exc}")
    raw = pd.read_csv(args.input, usecols=["date", "ticker", *fields], parse_dates=["date"])
    raw = raw.loc[raw["date"].between(args.start, args.end) & raw["ticker"].isin(args.tickers)]
    local = raw.set_index(["ticker", "date"]).sort_index()
    qlib.init(provider_uri=str(args.provider), region="us")
    reference = D.features(args.tickers, selected_expr, start_time=args.start, end_time=args.end)
    reference.columns = fields
    reference.index.names = ["ticker", "date"]
    same = compare(local, reference, fields, args.absolute_tolerance)
    missing = set(local.index).symmetric_difference(set(reference.index))
    if missing:
        raise ValueError(f"Qlib/local row keys differ: first={list(missing)[:3]}")
    provider_sources = [args.provider / "calendars/day.txt", args.provider / "instruments/all.txt"]
    for ticker in args.tickers:
        provider_sources.extend((args.provider / "features" / ticker.lower() / f"{field}.day.bin")
                                for field in ("open", "high", "low", "close", "volume"))
    report = {
        "status": "sample_matched" if all(row["missing_mismatch"] == row["rows_exceeding_tolerance"] == 0 for row in same.values()) else "sample_mismatch",
        "scope": "same-source Qlib provider vs local pandas, not independent PIT or full Alpha158 certification",
        "sample_tickers": args.tickers, "start": args.start, "end": args.end,
        "absolute_tolerance": args.absolute_tolerance, "qlib_version": qlib.__version__,
        "rows": len(local), "factor_results": same,
        "official_expressions": dict(zip(fields, selected_expr)),
        "inputs": {"local_features": fingerprint(args.input), "candidate_file": fingerprint(args.candidate_file),
                   "provider_files": [fingerprint(path) for path in provider_sources],
                   "code": fingerprint(Path(__file__))},
        "limitations": "VWAP/other proxy fields excluded; provider and local features derive from the same late Yahoo extract; historical membership and adjusted price vintages remain unresolved",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "rows": len(local),
                      "factors": len(same), "max_absolute_error": max(row["max_absolute_error"] for row in same.values())}, indent=2))
    if report["status"] != "sample_matched":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
