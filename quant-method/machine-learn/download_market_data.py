#!/usr/bin/env python3
"""Download the price data needed for the AlphaCrafter comparison.

The paper does not publish a ready-made dataset.  This script obtains the S&P 500
price-volume data from Yahoo Finance's chart endpoint and stores one CSV per
symbol, so an interrupted download can be resumed safely.

The default cutoff is the last date used for the paper's reported backtest.
No data after that date is requested.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

# Pin the universe to a revision before the 2026-02-27 cutoff.  Using the
# repository's moving ``master`` file would introduce survivorship/look-ahead
# bias when the script is run after the experiment.
CONSTITUENTS_URL = (
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/"
    "b61fb6f07286/data/constituents.csv"
)
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
USER_AGENT = "alpha-comparison-data/0.1 (research; contact unavailable)"


def get_bytes(url: str, retries: int = 5) -> bytes:
    last_error: Optional[Exception] = None
    for attempt in range(retries):
        try:
            request = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(request, timeout=30) as response:
                return response.read()
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_error = exc
            # Yahoo occasionally responds with a transient 429/5xx.
            time.sleep(min(2 ** attempt, 16))
    raise RuntimeError("failed to download {}: {}".format(url, last_error))


def yahoo_symbol(symbol: str) -> str:
    # Yahoo uses '-' in place of '.' for share classes (BRK.B -> BRK-B).
    return symbol.replace(".", "-")


def download_symbol(
    symbol: str, start: str, end: str, output_dir: Path, overwrite: bool = False
) -> Dict[str, object]:
    output_path = output_dir / (yahoo_symbol(symbol) + ".csv")
    if output_path.exists() and not overwrite:
        return {"symbol": symbol, "status": "exists", "rows": None}

    start_ts = int(datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
    # Yahoo's period2 is exclusive; adding one day includes the requested end date.
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    end_ts = int((end_dt.replace(tzinfo=timezone.utc)).timestamp()) + 86400
    url = YAHOO_CHART_URL.format(symbol=quote(yahoo_symbol(symbol)))
    url += "?period1={}&period2={}&interval=1d&events=history&includeAdjustedClose=true".format(
        start_ts, end_ts
    )

    try:
        payload = json.loads(get_bytes(url).decode("utf-8"))
        result = (payload.get("chart") or {}).get("result")
        if not result:
            error = (payload.get("chart") or {}).get("error")
            raise RuntimeError(error or "Yahoo returned no result")
        result = result[0]
        timestamps = result.get("timestamp") or []
        quote_data = (result.get("indicators") or {}).get("quote", [{}])[0]
        adjusted = ((result.get("indicators") or {}).get("adjclose") or [{}])[0]
        adj_closes = adjusted.get("adjclose") or []
        fields = ["open", "high", "low", "close", "volume"]

        rows: List[Dict[str, object]] = []
        for i, timestamp in enumerate(timestamps):
            row: Dict[str, object] = {
                "date": datetime.fromtimestamp(timestamp, timezone.utc).strftime("%Y-%m-%d"),
                "ticker": symbol,
            }
            for field in fields:
                values = quote_data.get(field) or []
                row[field] = values[i] if i < len(values) else ""
            row["adj_close"] = adj_closes[i] if i < len(adj_closes) else ""
            rows.append(row)

        if not rows:
            raise RuntimeError("Yahoo returned zero rows")
        tmp_path = output_path.with_suffix(".csv.tmp")
        with tmp_path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"],
            )
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp_path, output_path)
        return {"symbol": symbol, "status": "downloaded", "rows": len(rows)}
    except Exception as exc:  # keep other symbols downloading
        return {"symbol": symbol, "status": "failed", "error": str(exc)}


def load_symbols(path: Path) -> List[str]:
    with path.open(newline="") as handle:
        return [row["Symbol"].strip() for row in csv.DictReader(handle) if row.get("Symbol")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", choices=["sp500"], default="sp500")
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--end", default="2026-02-27")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "raw" / "sp500",
    )
    args = parser.parse_args()

    args.root.mkdir(parents=True, exist_ok=True)
    constituents_path = args.root.parent / "sp500_constituents.csv"
    if not constituents_path.exists() or args.overwrite:
        constituents_path.write_bytes(get_bytes(CONSTITUENTS_URL))
    symbols = load_symbols(constituents_path)

    results: List[Dict[str, object]] = []
    print("Downloading {} symbols to {}".format(len(symbols), args.root))
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(download_symbol, symbol, args.start, args.end, args.root, args.overwrite): symbol
            for symbol in symbols
        }
        for n, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            if n % 25 == 0 or result["status"] == "failed":
                print("[{}/{}] {}: {}".format(n, len(symbols), result["symbol"], result["status"]), flush=True)

    results.sort(key=lambda item: str(item["symbol"]))
    manifest = {
        "source": "Yahoo Finance chart API",
        "constituents_source": CONSTITUENTS_URL,
        "symbols": len(symbols),
        "start": args.start,
        "end": args.end,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    (args.root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    failed = [item for item in results if item["status"] == "failed"]
    print("Finished: {}, failed: {}, manifest: {}".format(len(results), len(failed), args.root / "manifest.json"))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
