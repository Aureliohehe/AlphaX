#!/usr/bin/env python3
"""Run a Qlib daily portfolio backtest on the frozen XGBoost predictions.

The feature/model pipeline stores predictions in a CSV, while Qlib's exchange
needs OHLCV data in its binary provider format.  This script converts the
Yahoo-style adjusted OHLCV files into a local Qlib provider, then uses Qlib's
Exchange, SimulatorExecutor and TopkDropoutStrategy for the actual portfolio
simulation.

The prediction on date T is traded on the next trading day's open.  This avoids
using the same day's closing price after the signal was calculated.  The
existing model label is close-to-next-close, so the execution convention is
recorded in the output metadata and should be followed by a future
open-to-close target experiment.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


DEFAULT_RAW_DIR = Path(__file__).resolve().parent / "data" / "raw" / "sp500"
DEFAULT_PREDICTIONS = (
    Path(__file__).resolve().parent / "data" / "qlib" / "xgboost_baseline" / "predictions.csv"
)
DEFAULT_PROVIDER = Path(__file__).resolve().parent / "data" / "qlib" / "sp500_provider"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "data" / "qlib" / "portfolio_backtest"


def build_adjusted_source(raw_dir: Path, source_dir: Path) -> int:
    """Create Qlib-compatible per-instrument files from Yahoo-style CSVs."""
    source_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for path in sorted(raw_dir.glob("*.csv")):
        frame = pd.read_csv(path, parse_dates=["date"])
        required = {"date", "ticker", "open", "high", "low", "close", "adj_close", "volume"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing fields: {sorted(missing)}")

        # Use adjusted OHLC prices so the valuation convention matches the
        # adjusted-close target used by build_features.py.  factor=1 disables
        # Qlib's additional price adjustment for this already-adjusted source.
        factor = frame["adj_close"] / frame["close"]
        adjusted_close = frame["adj_close"].astype(float)
        output = pd.DataFrame(
            {
                "date": frame["date"],
                "symbol": frame["ticker"].astype(str),
                "open": frame["open"].astype(float) * factor,
                "high": frame["high"].astype(float) * factor,
                "low": frame["low"].astype(float) * factor,
                "close": adjusted_close,
                "volume": frame["volume"].astype(float),
                "factor": 1.0,
                "change": adjusted_close.pct_change(),
            }
        ).replace([np.inf, -np.inf], np.nan)
        output = output.dropna(subset=["date", "open", "high", "low", "close", "volume"])
        output.to_csv(source_dir / path.name, index=False)
        count += 1
    if count == 0:
        raise FileNotFoundError(f"No CSV files found under {raw_dir}")
    return count


def provider_is_ready(provider_dir: Path) -> bool:
    return (provider_dir / "calendars" / "day.txt").exists() and (
        provider_dir / "features" 
    ).exists()


def ensure_qlib_provider(raw_dir: Path, provider_dir: Path, workers: int, rebuild: bool) -> None:
    """Build the local Qlib binary provider once, unless it already exists."""
    if provider_is_ready(provider_dir) and not rebuild:
        return
    if provider_dir.exists():
        shutil.rmtree(provider_dir)
    provider_dir.parent.mkdir(parents=True, exist_ok=True)

    # Keep the transformed CSV staging data temporary; only Qlib .bin files
    # remain in the project data directory.
    with tempfile.TemporaryDirectory(prefix="alphacrafter_qlib_source_") as temp:
        source_dir = Path(temp) / "source"
        count = build_adjusted_source(raw_dir, source_dir)
        qlib_module = Path(__import__("qlib").__file__).resolve()
        dump_bin = qlib_module.parents[1] / "scripts" / "dump_bin.py"
        if not dump_bin.exists():
            raise FileNotFoundError(f"Cannot locate Qlib dump_bin.py near {qlib_module}")
        command = [
            sys.executable,
            str(dump_bin),
            "dump_all",
            f"--data_path={source_dir}",
            f"--qlib_dir={provider_dir}",
            "--freq=day",
            "--date_field_name=date",
            "--symbol_field_name=symbol",
            "--include_fields=open,high,low,close,volume,factor,change",
            "--file_suffix=.csv",
            f"--max_workers={workers}",
        ]
        print(f"Preparing Qlib provider from {count} instruments...")
        subprocess.run(command, check=True)


def load_signal(path: Path, start: str, end: str, signal_column: str) -> pd.Series:
    frame = pd.read_csv(path, parse_dates=["datetime"])
    required = {"datetime", "instrument", signal_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction file is missing fields: {sorted(missing)}")
    # Keep the previous trading day's signal because Qlib's TopkDropoutStrategy
    # deliberately reads shift=1 and trades at the next day's open.
    frame = frame[(frame["datetime"] >= pd.Timestamp(start) - pd.DateOffset(days=10)) &
                  (frame["datetime"] <= pd.Timestamp(end))]
    frame = frame.dropna(subset=["datetime", "instrument", signal_column])
    signal = frame.set_index(["datetime", "instrument"])[signal_column].astype(float).sort_index()
    if signal.empty:
        raise ValueError("No predictions overlap the requested backtest interval")
    return signal


def build_equal_weight_benchmark(universe: list[str], start: str, end: str) -> pd.Series:
    """Build a benchmark with the same entry convention as the portfolio.

    Qlib's list benchmark is close-to-close even on the first backtest day,
    while this portfolio enters at that day's open.  Use open-to-close on the
    first day and close-to-close afterwards, equal-weighted across the local
    S&P 500 universe.
    """
    from qlib.data import D

    first_day = pd.Timestamp(start)
    quotes = D.features(
        universe,
        ["$open", "$close"],
        first_day - pd.DateOffset(days=10),
        end,
        freq="day",
    ).reset_index()
    quotes = quotes.sort_values(["instrument", "datetime"])
    quotes["previous_close"] = quotes.groupby("instrument")["$close"].shift(1)
    quotes["benchmark_return"] = np.where(
        quotes["datetime"].eq(first_day),
        quotes["$close"] / quotes["$open"] - 1.0,
        quotes["$close"] / quotes["previous_close"] - 1.0,
    )
    benchmark = (
        quotes.loc[quotes["datetime"].between(first_day, pd.Timestamp(end))]
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["benchmark_return"])
        .groupby("datetime")["benchmark_return"]
        .mean()
    )
    if benchmark.empty:
        raise ValueError("Unable to build an equal-weight benchmark")
    return benchmark


def run_backtest(args: argparse.Namespace) -> tuple[pd.DataFrame, Any]:
    import qlib
    from qlib.constant import REG_US
    from qlib.contrib.evaluate import backtest_daily
    from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
    from qlib.backtest.executor import SimulatorExecutor
    from qlib.data import D

    qlib.init(
        provider_uri=str(args.provider_dir),
        region=REG_US,
        expression_cache=None,
        dataset_cache=None,
    )
    signal = load_signal(args.predictions, args.start, args.end, args.signal_column)
    strategy = TopkDropoutStrategy(
        signal=signal,
        topk=args.topk,
        n_drop=args.n_drop,
        hold_thresh=1,
        risk_degree=args.risk_degree,
        only_tradable=True,
        forbid_all_trade_at_limit=False,
    )
    executor = SimulatorExecutor(
        time_per_step="day",
        generate_portfolio_metrics=True,
        verbose=False,
        track_data=True,
    )
    # Use an equal-weighted local S&P 500 benchmark.  Its first return is
    # open-to-close so it has the same entry convention as the portfolio.
    universe = D.instruments("all")
    benchmark = build_equal_weight_benchmark(universe, args.start, args.end)
    report, positions = backtest_daily(
        start_time=args.start,
        end_time=args.end,
        strategy=strategy,
        executor=executor,
        account=args.initial_cash,
        benchmark=benchmark,
        exchange_kwargs={
            "deal_price": "$open",
            "open_cost": args.open_cost,
            "close_cost": args.close_cost,
            "min_cost": args.min_cost,
            "impact_cost": args.impact_cost,
            "trade_unit": None,
            "limit_threshold": None,
        },
    )
    return report, positions


def summarize(report: pd.DataFrame, initial_cash: float) -> Dict[str, Any]:
    """Summarize portfolio and equal-weight universe benchmark performance.

    The first Qlib account value is the end-of-first-session value after the
    opening trades.  Include that initial-to-close return in risk statistics;
    dropping it would report 539 observations for a 540-session backtest.
    """
    account = report["account"].astype(float)
    if account.empty:
        raise ValueError("Qlib returned an empty portfolio report")
    daily_return = account.pct_change()
    daily_return.iloc[0] = account.iloc[0] / initial_cash - 1.0
    daily_return = daily_return.replace([np.inf, -np.inf], np.nan).dropna()
    annual = 252.0
    trading_days = len(report)
    cumulative = float(account.iloc[-1] / initial_cash - 1.0)
    annualized = float((account.iloc[-1] / initial_cash) ** (annual / trading_days) - 1.0)
    volatility = float(daily_return.std(ddof=1) * np.sqrt(annual))
    sharpe = float(daily_return.mean() / daily_return.std(ddof=1) * np.sqrt(annual)) if daily_return.std(ddof=1) else np.nan
    wealth = pd.concat(
        [pd.Series([initial_cash], dtype=float), account.reset_index(drop=True)],
        ignore_index=True,
    )
    drawdown = wealth / wealth.cummax() - 1.0

    benchmark_daily = report["bench"].astype(float).fillna(0.0)
    benchmark_cumulative = float((1.0 + benchmark_daily).prod() - 1.0)
    benchmark_annualized = float((1.0 + benchmark_cumulative) ** (annual / trading_days) - 1.0)
    return {
        "start": str(report.index.min().date()),
        "end": str(report.index.max().date()),
        "trading_days": int(trading_days),
        "initial_cash": float(initial_cash),
        "final_account": float(account.iloc[-1]),
        "cumulative_return": cumulative,
        "annualized_return": annualized,
        "annualized_volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
        "mean_daily_turnover": float(report["turnover"].mean()),
        "total_cost": float(report["total_cost"].iloc[-1]),
        "benchmark_cumulative_return": benchmark_cumulative,
        "benchmark_annualized_return": benchmark_annualized,
        "excess_cumulative_return": cumulative - benchmark_cumulative,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Qlib daily portfolio backtest")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--signal-column", default="prediction", help="Column used as the ranking signal")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--provider-dir", type=Path, default=DEFAULT_PROVIDER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start", default="2024-01-02")
    parser.add_argument("--end", default="2026-02-26")
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--n-drop", type=int, default=50)
    parser.add_argument("--risk-degree", type=float, default=0.95)
    parser.add_argument("--initial-cash", type=float, default=1_000_000.0)
    parser.add_argument("--open-cost", type=float, default=0.0005)
    parser.add_argument("--close-cost", type=float, default=0.0005)
    parser.add_argument("--impact-cost", type=float, default=0.0005)
    parser.add_argument("--min-cost", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--rebuild-provider", action="store_true")
    args = parser.parse_args()

    ensure_qlib_provider(args.raw_dir, args.provider_dir, args.workers, args.rebuild_provider)
    report, positions = run_backtest(args)
    summary = summarize(report, args.initial_cash)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output_dir / "portfolio_report.csv")
    pd.to_pickle(positions, args.output_dir / "positions.pkl")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "config.json").write_text(
        json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
