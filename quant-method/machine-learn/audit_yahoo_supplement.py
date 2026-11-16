#!/usr/bin/env python3
"""Detect impossible Yahoo quote/membership overlap before using supplementary CSVs.

Positive overlap does not prove corporate identity, correct ticker mapping, or PIT prices.
Zero overlap makes a file ineligible for this membership/price reconstruction.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

from download_historical_member_prices import fingerprint


def audit(membership: Path, calendar: Path, quotes_dir: Path, start: date, end: date) -> dict:
    quote_days: dict[str, set[date]] = {}
    for path in sorted(quotes_dir.glob("*.csv")):
        days: set[date] = set()
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            if not {"date", "ticker", "close"}.issubset(reader.fieldnames or []):
                raise ValueError(f"Invalid quote schema: {path}")
            for row in reader:
                if row["ticker"].replace(".", "-") != path.stem:
                    raise ValueError(f"Ticker/file mismatch: {path}")
                day = date.fromisoformat(row["date"])
                if day in days:
                    raise ValueError(f"Duplicate quote date: {path}: {day}")
                days.add(day)
        if not days:
            raise ValueError(f"Empty quote file: {path}")
        quote_days[path.stem] = days
    with membership.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["date", "tickers"]:
            raise ValueError("Expected date,tickers membership CSV")
        snapshots = [(date.fromisoformat(row["date"]), set(row["tickers"].split(","))) for row in reader]
    if not snapshots or any(a[0] >= b[0] for a, b in zip(snapshots, snapshots[1:])):
        raise ValueError("Missing or unsorted membership snapshots")
    with calendar.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["date"]:
            raise ValueError("Expected date calendar")
        sessions = [date.fromisoformat(row["date"]) for row in reader]
    sessions = [d for d in sessions if start <= d <= end]
    if not sessions or any(a >= b for a, b in zip(sessions, sessions[1:])):
        raise ValueError("Missing or unsorted calendar sessions")
    if snapshots[0][0] > sessions[0]:
        raise ValueError("No membership snapshot covering first session")
    member_sessions = defaultdict(int)
    overlap_sessions = defaultdict(int)
    index = 0
    current: set[str] = set()
    for session in sessions:
        while index < len(snapshots) and snapshots[index][0] <= session:
            current = snapshots[index][1]
            index += 1
        for ticker in current:
            stem = ticker.replace(".", "-")
            if stem in quote_days:
                member_sessions[stem] += 1
                if session in quote_days[stem]:
                    overlap_sessions[stem] += 1
    rows = []
    for stem, days in sorted(quote_days.items()):
        overlap = overlap_sessions[stem]
        member_count = member_sessions[stem]
        rows.append({"symbol_file": stem, "quote_rows": len(days), "first_quote": min(days).isoformat(),
                     "last_quote": max(days).isoformat(), "member_sessions": member_count,
                     "quoted_member_sessions": overlap,
                     "member_session_quote_fraction": overlap / member_count if member_count else None,
                     "status": "no_membership_quote_overlap_suspected_reused_ticker" if overlap == 0
                               else "overlap_not_identity_verified"})
    return {"status": "diagnostic_not_pit_certified", "membership": fingerprint(membership),
            "calendar": fingerprint(calendar), "start": start.isoformat(), "end": end.isoformat(),
            "files_checked": len(rows),
            "zero_overlap_files": [row["symbol_file"] for row in rows if row["quoted_member_sessions"] == 0],
            "limitations": "Nonzero overlap is not corporate identity or PIT certification; dates may be missing,"
                           " ticker symbols can be reused, and current adjusted prices are retrospective.",
            "files": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--membership", type=Path, required=True)
    parser.add_argument("--calendar", type=Path, required=True)
    parser.add_argument("--quotes-dir", type=Path, required=True)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2016, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 2, 27))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refuse to overwrite existing audit")
    result = audit(args.membership.resolve(strict=True), args.calendar.resolve(strict=True),
                   args.quotes_dir.resolve(strict=True), args.start, args.end)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"files_checked": result["files_checked"],
                      "zero_overlap_files": result["zero_overlap_files"]}, indent=2))


if __name__ == "__main__":
    main()
