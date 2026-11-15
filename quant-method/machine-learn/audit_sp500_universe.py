#!/usr/bin/env python3
"""Audit a current S&P 500 snapshot against historical feature rows.

A snapshot with Date added is NOT point-in-time membership: it cannot recover
constituents subsequently removed or certify full historical coverage. This
script reports observable contradictions and never outputs a PIT approval.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Optional

import pandas as pd

from data_contract import calendar_audit, parse_session_series, read_calendar


def fingerprint(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def snapshot_entries(path: Path) -> pd.Series:
    snapshot = pd.read_csv(path, dtype={"Symbol": "string", "Date added": "string"})
    if not {"Symbol", "Date added"}.issubset(snapshot.columns):
        raise ValueError("Snapshot must contain Symbol and Date added")
    symbols = snapshot["Symbol"].str.strip()
    if symbols.isna().any() or (symbols == "").any() or symbols.duplicated().any():
        raise ValueError("Snapshot tickers must be nonempty and unique")
    dates = parse_session_series(snapshot["Date added"], "snapshot Date added")
    return pd.Series(dates.to_numpy(), index=symbols.to_numpy())


def count_preentry(frame: pd.DataFrame, entries: pd.Series) -> tuple[int, dict, set]:
    """Count rows strictly before their *current snapshot* Date added; never label them PIT."""
    if not {"date", "ticker"}.issubset(frame):
        raise ValueError("Feature chunk requires date/ticker")
    dates = parse_session_series(frame["date"], "feature dates")
    added = frame["ticker"].map(entries)
    unknown = set(frame.loc[added.isna(), "ticker"].dropna().astype(str))
    before = dates < added
    years = dates[before].dt.year.value_counts().to_dict()
    return int(before.sum()), {str(int(year)): int(count) for year, count in years.items()}, unknown


def audit(features: Path, snapshot: Path, calendar: Path, source_manifest: Optional[Path] = None) -> dict:
    entries = snapshot_entries(snapshot)
    reference = read_calendar(calendar)
    preentry, by_year, unknown, count, observed, tickers = 0, {}, set(), 0, set(), set()
    for chunk in pd.read_csv(features, usecols=["date", "ticker"], chunksize=100_000, parse_dates=["date"]):
        if chunk[["date", "ticker"]].isna().any().any():
            raise ValueError("Feature keys must not be missing")
        n, annual, missing = count_preentry(chunk, entries)
        preentry += n
        count += len(chunk)
        observed.update(chunk["date"].unique())
        tickers.update(chunk["ticker"].astype(str).unique())
        unknown.update(missing)
        for year, value in annual.items():
            by_year[year] = by_year.get(year, 0) + value
    if not observed:
        raise ValueError("Empty feature panel")
    first, last = min(observed), max(observed)
    coverage = calendar_audit(pd.Series(sorted(observed)), reference, first, last)
    current_after = {str(year): int((entries > pd.Timestamp(str(year) + "-12-31")).sum())
                     for year in (2020, 2021, 2022, 2023, 2024)}
    provenance = None
    if source_manifest:
        provenance = json.loads(source_manifest.read_text())
        provenance = {key: provenance.get(key) for key in ("source", "constituents_source", "downloaded_at_utc", "start", "end")}
    return {
        "status": "not_point_in_time_certified", "historical_membership_complete": False,
        "feature_rows": count, "feature_tickers": len(tickers), "snapshot_tickers": len(entries),
        "feature_rows_before_current_snapshot_date_added": preentry,
        "preentry_fraction": preentry / count, "preentry_rows_by_year": dict(sorted(by_year.items())),
        "current_snapshot_members_added_after_year_end": current_after,
        "feature_tickers_missing_from_snapshot": sorted(unknown),
        "calendar_audit_after_feature_warmup": coverage,
        "provenance": provenance,
        "inputs": {"features": fingerprint(features), "snapshot": fingerprint(snapshot),
                   "calendar": fingerprint(calendar),
                   **({"source_manifest": fingerprint(source_manifest)} if source_manifest else {})},
        "limitation": "Date added for *current* members is only a lower bound on entry; departed constituents, ticker histories, corporate-action timestamps and adjusted price vintages are unavailable. Do not promote this audit to a PIT universe.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--calendar", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refuse to overwrite existing audit")
    paths = [args.features, args.snapshot, args.calendar, args.source_manifest]
    paths = [path.resolve(strict=True) if path else None for path in paths]
    result = audit(*paths)
    result["code"] = fingerprint(Path(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "feature_rows", "feature_rows_before_current_snapshot_date_added", "preentry_fraction", "feature_tickers_missing_from_snapshot")}, indent=2))


if __name__ == "__main__":
    main()
