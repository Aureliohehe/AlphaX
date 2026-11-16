#!/usr/bin/env python3
"""Best-effort Yahoo quotes for historical index members missing from local files.

This cannot provide survivorship-free or point-in-time prices: delisted tickers
may have disappeared from Yahoo and today's adjusted prices are retrospective.
All results go into a separate directory; original raw quotes are read-only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from download_market_data import USER_AGENT, YAHOO_CHART_URL, yahoo_symbol


def fingerprint(path: Path) -> dict:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return {"path": str(path.resolve()), "sha256": h.hexdigest(), "bytes": path.stat().st_size}


def historical_symbols(path: Path, start: date, end: date) -> set[str]:
    """Union of snapshots effective within [start, end], including the preceding snapshot."""
    result: set[str] = set()
    previous: set[str] | None = None
    covered_start = False
    last: date | None = None
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["date", "tickers"]:
            raise ValueError("Expected date,tickers membership CSV")
        for row in reader:
            day = date.fromisoformat(row["date"])
            if last is not None and day <= last:
                raise ValueError("Membership dates must be strictly increasing")
            last = day
            symbols = row["tickers"].split(",")
            if not symbols or any(not s or s.strip() != s for s in symbols) or len(symbols) != len(set(symbols)):
                raise ValueError(f"Invalid or duplicate ticker on {day}")
            if day < start:
                previous = set(symbols)
                covered_start = True
            elif day <= end:
                if day == start:
                    covered_start = True
                if not result and previous is not None:
                    result.update(previous)
                result.update(symbols)
            else:
                break
    if previous is not None:
        result.update(previous)
    if not result or not covered_start:
        raise ValueError("No membership snapshot covering start of interval")
    return result


def get_quotes(symbol: str, start: date, end: date) -> tuple[str, list[dict], str | None]:
    period1 = int(datetime.combine(start, datetime.min.time(), timezone.utc).timestamp())
    period2 = int(datetime.combine(end + timedelta(days=1), datetime.min.time(), timezone.utc).timestamp())
    url = YAHOO_CHART_URL.format(symbol=quote(yahoo_symbol(symbol)))
    url += f"?period1={period1}&period2={period2}&interval=1d&events=history&includeAdjustedClose=true"
    for attempt in range(3):
        try:
            with urlopen(Request(url, headers={"User-Agent": USER_AGENT}), timeout=12) as response:
                payload = json.load(response)
            chart = payload.get("chart") or {}
            result = chart.get("result")
            if not result:
                return "unavailable", [], str(chart.get("error") or "no result")
            data = result[0]
            timestamps = data.get("timestamp") or []
            if not timestamps:
                return "unavailable", [], "empty price history"
            quoted = (data.get("indicators") or {}).get("quote", [{}])[0]
            adjusted = ((data.get("indicators") or {}).get("adjclose") or [{}])[0].get("adjclose") or []
            rows = []
            for i, timestamp in enumerate(timestamps):
                row = {"date": datetime.fromtimestamp(timestamp, timezone.utc).strftime("%Y-%m-%d"),
                       "ticker": symbol}
                for field in ("open", "high", "low", "close", "volume"):
                    values = quoted.get(field) or []
                    row[field] = values[i] if i < len(values) else ""
                row["adj_close"] = adjusted[i] if i < len(adjusted) else ""
                rows.append(row)
            return "downloaded", rows, None
        except HTTPError as exc:
            if exc.code in (404, 410):
                return "unavailable", [], f"HTTP {exc.code}"
            if exc.code not in (429, 500, 502, 503, 504):
                return "failed", [], f"HTTP {exc.code}"
            error = f"HTTP {exc.code}"
        except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            error = f"{type(exc).__name__}: {str(exc)[:120]}"
        if attempt < 2:
            time.sleep(2 ** attempt)
    return ("rate_limited" if error == "HTTP 429" else "failed"), [], error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--membership", type=Path, required=True)
    parser.add_argument("--existing-quotes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2016, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 2, 27))
    parser.add_argument("--max-symbols", type=int, default=0, help="Limit this run; 0 means all remaining")
    parser.add_argument("--delay", type=float, default=0.3, help="Minimum seconds between requests")
    args = parser.parse_args()
    if args.start > args.end or args.max_symbols < 0 or args.delay < 0:
        parser.error("Invalid date range, max-symbols, or delay")
    membership = args.membership.resolve(strict=True)
    existing = args.existing_quotes.resolve(strict=True)
    output = args.output_dir.resolve()
    if not existing.is_dir() or output == existing or existing in output.parents or output in existing.parents:
        parser.error("Output must be separate from existing raw quotes")
    output.mkdir(parents=True, exist_ok=True)
    members = historical_symbols(membership, args.start, args.end)
    pending = sorted(s for s in members if not (existing / (yahoo_symbol(s) + ".csv")).exists()
                     and not (output / (yahoo_symbol(s) + ".csv")).exists())
    # A second vintage or alias must never silently write over the first ticker's file.
    mapped = [yahoo_symbol(s) for s in members]
    if len(mapped) != len(set(mapped)):
        parser.error("Membership has ticker alias collision after Yahoo symbol mapping")
    # Resume without retrying terminal no-data responses recorded for the same
    # source vintage/date window. Transient failures and rate limits are retried.
    membership_id = fingerprint(membership)
    terminal = set()
    for manifest in output.glob("run_*.json"):
        previous = json.loads(manifest.read_text())
        if (previous.get("membership", {}).get("sha256") != membership_id["sha256"]
                or previous.get("start") != args.start.isoformat()
                or previous.get("end") != args.end.isoformat()):
            continue
        terminal.update(r["symbol"] for r in previous.get("results", [])
                        if r.get("status") == "unavailable"
                        or (r.get("status") == "failed" and r.get("reason") == "HTTP 400"))
    pending = [s for s in pending if s not in terminal]
    pending_total = len(pending)
    if args.max_symbols:
        pending = pending[:args.max_symbols]
    results = []
    for index, symbol in enumerate(pending, 1):
        status, rows, reason = get_quotes(symbol, args.start, args.end)
        record = {"symbol": symbol, "status": status}
        if reason:
            record["reason"] = reason
        if status == "downloaded":
            path = output / (yahoo_symbol(symbol) + ".csv")
            if path.exists():
                raise RuntimeError(f"Refuse to overwrite {path}")
            temp = path.with_suffix(".csv.tmp")
            with temp.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["date", "ticker", "open", "high", "low",
                                                            "close", "adj_close", "volume"])
                writer.writeheader()
                writer.writerows(rows)
            os.replace(temp, path)
            record.update({"rows": len(rows), "file": fingerprint(path)})
        results.append(record)
        print(f"[{index}/{len(pending)}] {symbol}: {status}" + (f" ({reason})" if reason else ""), flush=True)
        if status == "rate_limited":
            break
        time.sleep(args.delay)
    report = {"status": "partial_exploratory_not_pit_certified", "source": "Yahoo Finance chart API",
              "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
              "membership": membership_id, "existing_quotes_dir": str(existing),
              "start": args.start.isoformat(), "end": args.end.isoformat(),
              "historical_tickers": len(members), "missing_locally_before_run": pending_total,
              "attempted": len(results), "results": results,
              "limitation": "Yahoo omits some delisted symbols; remaining price gaps, ticker changes,"
                            " retrospective adjusted prices and missing delisting returns prevent PIT backtests."}
    manifest = output / f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.json"
    with manifest.open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"Manifest: {manifest}")
    if any(r["status"] == "rate_limited" for r in results):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
