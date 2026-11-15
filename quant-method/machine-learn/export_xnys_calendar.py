#!/usr/bin/env python3
"""Export an independent, version-pinned XNYS schedule; not an exchange-issued certification."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from data_contract import calendar_audit, read_calendar

EXPECTED_VERSION = "4.13.3"
PROVIDER_WHEEL_SHA256 = "02b95a738b3848ab9c52529330f84a15f76c765edeb7790cfbbf0d45d92cc2b5"


def fingerprint(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observed-calendar", type=Path, required=True, help="Raw quote-union calendar used only for comparison")
    parser.add_argument("--start", default="2016-01-04")
    parser.add_argument("--end", default="2026-02-27")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.observed_calendar = args.observed_calendar.resolve(strict=True)
    manifest_path = args.output.with_name(args.output.name + ".manifest.json")
    if args.output.exists() or manifest_path.exists():
        parser.error("Refuse to overwrite existing calendar or manifest")
    try:
        import exchange_calendars as xc
    except ImportError as exc:
        parser.error(f"Install the pinned exchange-calendars=={EXPECTED_VERSION} to export an independent calendar: {exc}")
    if xc.__version__ != EXPECTED_VERSION:
        parser.error(f"Expected exchange-calendars=={EXPECTED_VERSION}, got {xc.__version__}")
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    if start >= end or start.tzinfo or end.tzinfo or start != start.normalize() or end != end.normalize():
        parser.error("Expected ordered, timezone-naive market session bounds")
    calendar = xc.get_calendar("XNYS", start=str((start - pd.Timedelta(days=10)).date()),
                               end=str((end + pd.Timedelta(days=10)).date()))
    sessions = calendar.sessions_in_range(start, end).tz_localize(None)
    if len(sessions) == 0 or sessions[0] != start or sessions[-1] != end:
        parser.error("Calendar does not cover exact requested start/end sessions")
    observed = read_calendar(args.observed_calendar)
    audit = calendar_audit(pd.Series(observed), sessions, start, end)
    if not audit["matches_reference"]:
        parser.error(f"Raw quotes do not match independent XNYS schedule: missing={audit['missing_sessions'][:10]}, unexpected={audit['unexpected_sessions'][:10]}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": sessions}).to_csv(args.output, index=False)
    manifest = {
        "status": "matches_quote_union", "authority": "third_party_schedule_not_exchange_issued",
        "provider": "exchange-calendars", "version": EXPECTED_VERSION,
        "package_wheel_sha256": PROVIDER_WHEEL_SHA256,
        "market": "XNYS", "start": args.start, "end": args.end, "sessions": len(sessions),
        "observed": fingerprint(args.observed_calendar), "output": fingerprint(args.output),
        "audit": audit, "code": fingerprint(Path(__file__)),
        "limitation": "Matching session dates does not establish point-in-time membership, quotes, or adjusted-price availability",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(sessions)} XNYS sessions to {args.output}")


if __name__ == "__main__":
    main()
