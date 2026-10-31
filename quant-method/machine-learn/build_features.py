#!/usr/bin/env python3
"""Build leakage-safe technical features and a next-day return label.

Every indicator is calculated independently within one ticker.  A row dated t
uses data available through t and its target is the adjusted-close return from
t to t+1.  The output is a panel suitable for the XGBoost baseline.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


SPLITS = [
    ("train", "2016-01-04", "2022-12-30"),
    ("validation", "2023-01-03", "2023-12-29"),
    ("backtest", "2024-01-02", "2026-02-27"),
]


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    value = 100 - 100 / (1 + rs)
    # A monotonically rising window should have RSI 100 rather than NA.
    return value.where(loss.ne(0), 100.0)


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values("date").drop_duplicates("date").copy()
    numeric = ["open", "high", "low", "close", "adj_close", "volume"]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    # Adjust OHLC by the same factor as adjusted close, so splits/dividends do
    # not create artificial jumps in range-based indicators.
    adjustment = frame["adj_close"] / frame["close"].replace(0, np.nan)
    close = frame["adj_close"]
    high = frame["high"] * adjustment
    low = frame["low"] * adjustment
    open_ = frame["open"] * adjustment
    log_return = np.log(close).diff()

    features = pd.DataFrame(index=frame.index)
    for window in (1, 5, 10, 20, 60):
        features[f"return_{window}d"] = close.pct_change(window)
    for window in (5, 10, 20, 60, 120):
        sma = close.rolling(window).mean()
        features[f"sma_{window}_ratio"] = close / sma - 1
    for window in (12, 26):
        ema = close.ewm(span=window, adjust=False, min_periods=window).mean()
        features[f"ema_{window}_ratio"] = close / ema - 1

    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    features["macd"] = macd / close
    features["macd_signal"] = macd_signal / close
    features["macd_hist"] = (macd - macd_signal) / close

    features["rsi_14"] = rsi(close)
    middle = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    features["bollinger_z"] = (close - middle) / std20.replace(0, np.nan)
    features["bollinger_width"] = 4 * std20 / middle.replace(0, np.nan)

    previous_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()], axis=1
    ).max(axis=1)
    features["atr_14_ratio"] = true_range.rolling(14).mean() / close
    features["intraday_range"] = (high - low) / close
    features["open_close_return"] = close / open_ - 1
    features["close_location"] = (close - low) / (high - low).replace(0, np.nan)

    for window in (5, 20, 60):
        features[f"volatility_{window}d"] = log_return.rolling(window).std()
    for window in (5, 20, 60):
        rolling_high = high.rolling(window).max()
        rolling_low = low.rolling(window).min()
        features[f"drawdown_from_high_{window}d"] = close / rolling_high - 1
        features[f"distance_from_low_{window}d"] = close / rolling_low - 1

    volume_mean20 = frame["volume"].rolling(20).mean()
    features["volume_ratio_20d"] = frame["volume"] / volume_mean20
    features["volume_change_1d"] = frame["volume"].pct_change()
    features["volume_volatility_20d"] = np.log1p(frame["volume"]).diff().rolling(20).std()

    # Stochastic oscillator.
    low14 = low.rolling(14).min()
    high14 = high.rolling(14).max()
    features["stochastic_14"] = (close - low14) / (high14 - low14).replace(0, np.nan)

    for column in features:
        features[column] = features[column].replace([np.inf, -np.inf], np.nan)
    result = pd.concat([frame[["date", "ticker"]], features], axis=1)
    result["target_1d"] = close.shift(-1) / close - 1
    return result


def split_name(date: pd.Series) -> pd.Series:
    output = pd.Series("excluded", index=date.index, dtype="object")
    parsed = pd.to_datetime(date)
    for name, start, end in SPLITS:
        output.loc[(parsed >= start) & (parsed <= end)] = name
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "raw" / "sp500",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "processed" / "sp500_features.csv.gz",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    input_files = sorted(args.input.glob("*.csv"))
    if not input_files:
        raise SystemExit("No per-ticker CSV files found in {}".format(args.input))

    # Compute once to discover the feature schema, then stream each ticker into
    # one compressed CSV to keep peak memory bounded.
    feature_columns: List[str] = []
    wrote_header = False
    total_rows = 0
    with args.output.open("wb") as raw_output:
        import gzip

        with gzip.GzipFile(fileobj=raw_output, mode="wb") as compressed:
            text_output = __import__("io").TextIOWrapper(compressed, encoding="utf-8", newline="")
            for n, path in enumerate(input_files, 1):
                frame = pd.read_csv(path, parse_dates=["date"])
                result = add_features(frame)
                if not feature_columns:
                    feature_columns = [c for c in result.columns if c not in ("date", "ticker", "target_1d")]
                result = result.dropna(subset=feature_columns + ["target_1d"]).copy()
                result["split"] = split_name(result["date"])
                result = result[result["split"] != "excluded"]
                result.to_csv(text_output, index=False, header=not wrote_header)
                wrote_header = True
                total_rows += len(result)
                if n % 25 == 0 or n == len(input_files):
                    print("[{}/{}] {} rows".format(n, len(input_files), total_rows), flush=True)
            text_output.flush()

    print("Wrote {} rows to {}".format(total_rows, args.output))
    print("Features ({}): {}".format(len(feature_columns), ", ".join(feature_columns)))


if __name__ == "__main__":
    main()
