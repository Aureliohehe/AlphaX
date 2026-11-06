#!/usr/bin/env python3
"""Build leakage-safe technical features and a next-day return label.

Every indicator is calculated independently within one ticker.  A row dated t
uses data available through t and its targets are forward adjusted-close returns
from t to t+1 and t+5.  The output is a panel suitable for the XGBoost baseline.
It also stores an open-to-close target for signals executed at the next day's open.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Optional

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


def _rolling_slope(values: np.ndarray) -> float:
    x = np.arange(len(values), dtype=float)
    centered = x - x.mean()
    denominator = np.dot(centered, centered)
    return float(np.dot(centered, values - values.mean()) / denominator) if denominator else np.nan


def _rolling_rsquare(values: np.ndarray) -> float:
    x = np.arange(len(values), dtype=float)
    slope = _rolling_slope(values)
    intercept = values.mean() - slope * x.mean()
    fitted = intercept + slope * x
    total = np.sum((values - values.mean()) ** 2)
    return float(1.0 - np.sum((values - fitted) ** 2) / total) if total else 0.0


def _rolling_residual(values: np.ndarray) -> float:
    slope = _rolling_slope(values)
    intercept = values.mean() - slope * (len(values) - 1) / 2.0
    return float(values[-1] - (intercept + slope * (len(values) - 1)))


def _alpha158_features(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.DataFrame:
    """Build a pandas implementation of Qlib Alpha158-style OHLCV factors.

    The names are prefixed to distinguish this reproducible implementation
    from Qlib's provider fields.  All rolling operations are backward-looking.
    """
    eps = 1e-12
    output = {}
    max_oc = pd.concat([open_, close], axis=1).max(axis=1)
    min_oc = pd.concat([open_, close], axis=1).min(axis=1)
    intraday = high - low
    output.update(
        {
            "alpha158_kmid": (close - open_) / (open_ + eps),
            "alpha158_klen": intraday / (open_ + eps),
            "alpha158_kmid2": (close - open_) / (intraday + eps),
            "alpha158_kup": (high - max_oc) / (open_ + eps),
            "alpha158_kup2": (high - max_oc) / (intraday + eps),
            "alpha158_klow": (min_oc - low) / (open_ + eps),
            "alpha158_klow2": (min_oc - low) / (intraday + eps),
            "alpha158_ksft": (2 * close - high - low) / (open_ + eps),
            "alpha158_ksft2": (2 * close - high - low) / (intraday + eps),
            "alpha158_open0": open_ / (close + eps),
            "alpha158_high0": high / (close + eps),
            "alpha158_low0": low / (close + eps),
            "alpha158_vwap0": ((high + low + close) / 3.0) / (close + eps),
        }
    )
    for lag in range(1, 5):
        output[f"alpha158_open{lag}"] = open_.shift(lag) / (close + eps)
        output[f"alpha158_high{lag}"] = high.shift(lag) / (close + eps)
        output[f"alpha158_low{lag}"] = low.shift(lag) / (close + eps)
        output[f"alpha158_vwap{lag}"] = ((high + low + close) / 3.0).shift(lag) / (close + eps)
        output[f"alpha158_volume{lag}"] = volume.shift(lag) / (volume + eps)
    output["alpha158_volume0"] = volume / (volume + eps)

    delta = close.diff()
    volume_change = np.log1p(volume).diff()
    for window in (5, 10, 20, 30, 60):
        close_roll = close.rolling(window, min_periods=window)
        high_roll = high.rolling(window, min_periods=window)
        low_roll = low.rolling(window, min_periods=window)
        output[f"alpha158_roc{window}"] = close.shift(window) / (close + eps)
        output[f"alpha158_ma{window}"] = close_roll.mean() / (close + eps)
        output[f"alpha158_std{window}"] = close_roll.std() / (close + eps)
        # Vectorized, leakage-safe approximations of the regression family.
        # They preserve Alpha158's scale and trend intent without thousands of
        # slow Python rolling callbacks during multi-ticker feature generation.
        output[f"alpha158_beta{window}"] = close.pct_change(window) / float(window)
        output[f"alpha158_rsqr{window}"] = close_roll.mean() / (close_roll.std() + eps)
        output[f"alpha158_resi{window}"] = (close - close_roll.mean()) / (close + eps)
        output[f"alpha158_max{window}"] = high_roll.max() / (close + eps)
        output[f"alpha158_min{window}"] = low_roll.min() / (close + eps)
        output[f"alpha158_qtlu{window}"] = close_roll.quantile(0.8) / (close + eps)
        output[f"alpha158_qtld{window}"] = close_roll.quantile(0.2) / (close + eps)
        output[f"alpha158_rank{window}"] = (close - close_roll.min()) / (close_roll.max() - close_roll.min() + eps)
        output[f"alpha158_rsv{window}"] = (close - low_roll.min()) / (high_roll.max() - low_roll.min() + eps)
        output[f"alpha158_imax{window}"] = high / (high_roll.max() + eps)
        output[f"alpha158_imin{window}"] = low / (low_roll.min() + eps)
        output[f"alpha158_imxd{window}"] = output[f"alpha158_imax{window}"] - output[f"alpha158_imin{window}"]
        output[f"alpha158_corr{window}"] = close_roll.corr(np.log1p(volume))
        output[f"alpha158_cord{window}"] = delta.rolling(window, min_periods=window).corr(volume_change)
        up = (delta > 0).astype(float)
        down = (delta < 0).astype(float)
        output[f"alpha158_cntp{window}"] = up.rolling(window, min_periods=window).mean()
        output[f"alpha158_cntn{window}"] = down.rolling(window, min_periods=window).mean()
        output[f"alpha158_cntd{window}"] = output[f"alpha158_cntp{window}"] - output[f"alpha158_cntn{window}"]
        positive = delta.clip(lower=0).rolling(window, min_periods=window).sum()
        negative = (-delta.clip(upper=0)).rolling(window, min_periods=window).sum()
        absolute = delta.abs().rolling(window, min_periods=window).sum()
        output[f"alpha158_sump{window}"] = positive / (absolute + eps)
        output[f"alpha158_sumn{window}"] = negative / (absolute + eps)
        output[f"alpha158_sumd{window}"] = (positive - negative) / (absolute + eps)
    return pd.DataFrame(output, index=close.index)


def _prepare_point_in_time_panel(
    panel: pd.DataFrame, prefix: str, aggregate_news: bool = False
) -> pd.DataFrame:
    """Normalize an optional point-in-time panel for leakage-safe as-of joins.

    The panel must contain ``ticker`` and either ``available_date`` or ``date``.
    ``available_date`` is preferred because report/fact dates are not necessarily
    the date on which the information became public.
    """
    required = {"ticker"}
    if not required.issubset(panel.columns):
        raise ValueError(f"{prefix} panel must contain ticker")
    date_column = "available_date" if "available_date" in panel.columns else "date"
    if date_column not in panel.columns:
        raise ValueError(f"{prefix} panel must contain available_date or date")
    prepared = panel.copy()
    prepared["ticker"] = prepared["ticker"].astype(str)
    prepared["_available_date"] = pd.to_datetime(prepared[date_column], errors="coerce")
    prepared = prepared.dropna(subset=["_available_date"])
    value_columns = []
    for column in prepared.columns:
        if column in {"ticker", "date", "available_date", "_available_date"}:
            continue
        numeric = pd.to_numeric(prepared[column], errors="coerce")
        if numeric.notna().any():
            prepared[column] = numeric
            value_columns.append(column)
    if aggregate_news:
        sentiment = next(
            (column for column in ("sentiment", "sentiment_score", "polarity") if column in prepared),
            None,
        )
        group_columns = ["ticker", "_available_date"]
        aggregates = {"news_count": (prepared["ticker"].groupby([prepared["ticker"], prepared["_available_date"]]).size())}
        if sentiment:
            aggregates[f"{prefix}sentiment"] = prepared.groupby(group_columns)[sentiment].mean()
        output = pd.DataFrame(aggregates).reset_index()
        output = output.rename(columns={"news_count": f"{prefix}count"})
        return output
    output = prepared[["ticker", "_available_date"] + value_columns].copy()
    output = output.rename(columns={column: f"{prefix}{column}" for column in value_columns})
    return output.groupby(["ticker", "_available_date"], as_index=False).last()


def _merge_asof_panel(
    frame: pd.DataFrame, panel: Optional[pd.DataFrame], prefix: str, aggregate_news: bool = False
) -> pd.DataFrame:
    """Merge external data using only observations available on or before date."""
    if panel is None or panel.empty:
        return frame
    prepared = _prepare_point_in_time_panel(panel, prefix, aggregate_news)
    value_columns = [column for column in prepared.columns if column not in {"ticker", "_available_date"}]
    output = pd.DataFrame(index=frame.index, columns=value_columns, dtype=float)
    for ticker, indices in frame.groupby("ticker", sort=False).groups.items():
        left = frame.loc[indices, ["date"]].copy()
        left["_row_id"] = list(indices)
        right = prepared[prepared["ticker"] == str(ticker)][["_available_date"] + value_columns]
        if right.empty:
            continue
        merged = pd.merge_asof(
            left.sort_values("date"),
            right.sort_values("_available_date"),
            left_on="date",
            right_on="_available_date",
            direction="backward",
        )
        merged = merged.set_index("_row_id")
        output.loc[merged.index, value_columns] = merged[value_columns].to_numpy()
    return pd.concat([frame, output], axis=1)


def add_features(
    frame: pd.DataFrame,
    fundamentals: Optional[pd.DataFrame] = None,
    news: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    frame = frame.sort_values("date").drop_duplicates("date").copy()
    frame = _merge_asof_panel(frame, fundamentals, "fundamental_")
    frame = _merge_asof_panel(frame, news, "news_", aggregate_news=True)
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

    # Qlib Alpha158-inspired K-bar, price/volume lag, and rolling operators.
    features = pd.concat(
        [features, _alpha158_features(open_, high, low, close, frame["volume"])], axis=1
    )

    for column in features:
        features[column] = features[column].replace([np.inf, -np.inf], np.nan)
    external_columns = [
        column
        for column in frame.columns
        if column.startswith("fundamental_") or column.startswith("news_")
    ]
    result = pd.concat(
        [
            frame[["date", "ticker"] + external_columns],
            features,
            pd.DataFrame(
                {
                    "target_1d": close.shift(-1) / close - 1,
                    "target_5d": close.shift(-5) / close - 1,
                    # Signal at T close, execution at T+1 open.
                    "target_open_close": close.shift(-1) / open_.shift(-1) - 1,
                    # Execution-aligned five-session label: enter at T+1 open,
                    # exit at T+5 close.
                    "target_open_close_5d": close.shift(-5) / open_.shift(-1) - 1,
                },
                index=frame.index,
            ),
        ],
        axis=1,
    )
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
    parser.add_argument(
        "--fundamentals",
        type=Path,
        default=None,
        help="Optional CSV with ticker, available_date/date, and numeric fundamental fields",
    )
    parser.add_argument(
        "--news",
        type=Path,
        default=None,
        help="Optional CSV with ticker, available_date/date, and sentiment fields",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    fundamentals = pd.read_csv(args.fundamentals) if args.fundamentals else None
    news = pd.read_csv(args.news) if args.news else None
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
                result = add_features(frame, fundamentals=fundamentals, news=news)
                if not feature_columns:
                    feature_columns = [
                        c for c in result.columns
                        if c not in (
                            "date",
                            "ticker",
                            "target_1d",
                            "target_5d",
                            "target_open_close",
                            "target_open_close_5d",
                        )
                    ]
                # Keep the final rows with unavailable forward labels so the
                # 1-day signal is not truncated.  Optional fundamentals/news
                # remain NA when unavailable; Miner reports their coverage
                # instead of silently deleting those observations.
                technical_columns = [
                    column
                    for column in feature_columns
                    if not column.startswith(("fundamental_", "news_"))
                ]
                result = result.dropna(
                    subset=technical_columns + ["target_1d", "target_open_close"]
                ).copy()
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
