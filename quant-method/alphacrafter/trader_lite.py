#!/usr/bin/env python3
"""Auditable AlphaCrafter Trader proxy (Algorithms 3/4, Appendix B.2).

Signals at T close trade T+1 open and liquidate T+1 close. This intraday
proxy is NOT the paper's persistent-holdings exchange simulator. Charge both
legs, select only feasible validation trials, and otherwise stay in cash.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from temporal import label_end_dates, label_end_source, read_calendar, session_date


@dataclass(frozen=True)
class TraderConfig:
    beta: float
    gamma: float
    nlong: int
    nshort: int

    def __post_init__(self) -> None:
        if not np.isfinite(self.beta) or not 0 < self.beta <= 1:
            raise ValueError("beta must be in (0, 1]")
        if not np.isfinite(self.gamma) or not -1 <= self.gamma <= 1:
            raise ValueError("gamma must be in [-1, 1]")
        for name, count, exposure in (
            ("nlong", self.nlong, self.long_exposure),
            ("nshort", self.nshort, self.short_exposure),
        ):
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError(f"{name} must be a non-negative integer")
            if exposure > 0 and count == 0:
                raise ValueError(f"{name} must be positive for an active side")

    @property
    def long_exposure(self) -> float:
        return self.beta * (1.0 + self.gamma) / 2.0

    @property
    def short_exposure(self) -> float:
        return self.beta * (1.0 - self.gamma) / 2.0

    @property
    def name(self) -> str:
        return f"beta{self.beta:g}_gamma{self.gamma:g}_long{self.nlong}_short{self.nshort}"


def parse_configs(value: str) -> List[TraderConfig]:
    configs = []
    for item in value.split(";"):
        values = [part.strip() for part in item.split(",")]
        if len(values) != 4:
            raise ValueError("Each trader config must be beta,gamma,nlong,nshort")
        configs.append(TraderConfig(float(values[0]), float(values[1]), int(values[2]), int(values[3])))
    if len(configs) > 3:
        raise ValueError("Algorithm 3 allows at most three validation trials")
    if len(set(configs)) != len(configs):
        raise ValueError("Duplicate trader configurations")
    return configs


def validate_intervals(selection_start: str, selection_end: str, backtest_start: str, backtest_end: str) -> None:
    dates = [pd.Timestamp(value) for value in (selection_start, selection_end, backtest_start, backtest_end)]
    if any(pd.isna(date) for date in dates) or not dates[0] <= dates[1] < dates[2] <= dates[3]:
        raise ValueError("Require selection_start <= selection_end < backtest_start <= backtest_end")


def load_inputs(
    predictions_path: Path, feature_path: Path, as_of: Optional[str] = None,
    calendar_path: Optional[Path] = None, require_explicit_label_dates: bool = False,
) -> pd.DataFrame:
    predictions = pd.read_csv(predictions_path)
    required = {"datetime", "instrument", "prediction"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Predictions are missing fields: {sorted(missing)}")
    predictions = predictions.rename(columns={"datetime": "date", "instrument": "ticker"})
    predictions["date"] = pd.to_datetime(predictions["date"])
    # Header-only signals are a valid empty-library result. Keep a numeric
    # dtype after merging, otherwise np.isfinite fails on an all-object column.
    predictions["prediction"] = pd.to_numeric(predictions["prediction"], errors="raise").astype(float)
    predictions = predictions[["date", "ticker", "prediction"]]
    wanted = {"date", "ticker", "target_open_close", "label_end_1d"}
    labels = pd.read_csv(feature_path, compression="infer", usecols=lambda column: column in wanted, parse_dates=["date"])
    if "target_open_close" not in labels:
        raise ValueError("Trader requires target_open_close")
    for name, panel in (("predictions", predictions), ("features", labels)):
        if panel[["date", "ticker"]].isna().any().any() or panel.duplicated(["date", "ticker"]).any():
            raise ValueError(f"{name} must have unique, non-null date/ticker keys")
    if as_of is not None:
        cutoff = session_date(as_of)
        labels = labels.loc[labels["date"] <= cutoff].copy()
        predictions = predictions.loc[predictions["date"] <= cutoff].copy()
    # Use the visible panel calendar for execution, never a stock's delayed
    # next quote. A per-stock label ending later is not next-session P&L.
    reference = read_calendar(calendar_path) if calendar_path else None
    labels["label_end_date"] = label_end_dates(labels, 1, require_explicit_label_dates, reference)
    if calendar_path:
        if not labels["date"].isin(reference).all():
            raise ValueError("Feature signal dates outside the supplied execution calendar")
        calendar = pd.Series(reference)
    else:
        calendar = pd.Series(sorted(labels["date"].unique()))
    next_session = dict(zip(calendar, calendar.shift(-1)))
    frame = labels.merge(predictions, on=["date", "ticker"], how="left", validate="one_to_one")
    frame["execution_date"] = frame["date"].map(next_session)
    invalid = frame["label_end_date"].isna() | (frame["label_end_date"] != frame["execution_date"])
    if as_of is not None:
        invalid |= frame["label_end_date"] > session_date(as_of)
    frame.loc[invalid, "target_open_close"] = np.nan
    return frame.sort_values(["date", "ticker"]).reset_index(drop=True)


def target_weights(group: pd.DataFrame, config: TraderConfig) -> Dict[str, float]:
    """Choose names solely from observable signals, with deterministic tie breaks."""
    if group["ticker"].isna().any() or group["ticker"].duplicated().any():
        raise ValueError("Each signal cross-section needs unique, non-null tickers")
    ranked = group.loc[np.isfinite(group["prediction"])].sort_values(["prediction", "ticker"])
    nlong = config.nlong if config.long_exposure > 0 else 0
    nshort = config.nshort if config.short_exposure > 0 else 0
    if len(ranked) < nlong + nshort:
        return {}
    weights: Dict[str, float] = {}
    if nlong:
        weights.update({ticker: config.long_exposure / nlong for ticker in ranked.tail(nlong)["ticker"]})
    if nshort:
        weights.update({ticker: -config.short_exposure / nshort for ticker in ranked.head(nshort)["ticker"]})
    return weights


def simulate(
    frame: pd.DataFrame,
    config: Optional[TraderConfig],
    start: str,
    end: str,
    cost_rate: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Flat overnight; missing held-asset returns fail rather than imply zero P&L.

    Intervals refer to signal dates with an additional execution-end purge.
    If execution_date is absent (unit fixtures), labels are assumed pre-purged.
    Cash earns zero; short borrow, margin, slippage and liquidity are unmodelled.
    """
    if not np.isfinite(cost_rate) or cost_rate < 0:
        raise ValueError("cost_rate must be finite and non-negative")
    mask = frame["date"].between(start, end)
    if "execution_date" in frame:
        mask &= frame["execution_date"].notna() & (frame["execution_date"] <= pd.Timestamp(end))
    selected = frame.loc[mask]
    if selected.empty:
        raise ValueError(f"No settled rows in simulation interval {start} to {end}")
    returns, positions = [], []
    for date, group in selected.groupby("date", sort=True):
        weights = target_weights(group, config) if config is not None else {}
        quotes = group.set_index("ticker")["target_open_close"]
        held_returns = quotes.reindex(list(weights))
        if not np.isfinite(held_returns).all() or (held_returns < -1).any():
            raise ValueError(f"Missing or invalid execution return for selected asset on {date.date()}")
        gross_return = float(sum(weight * quotes[ticker] for ticker, weight in weights.items()))
        # Each day opens from cash and closes to cash. Closing notional changes
        # with price, for both long sales and short covers.
        entry_notional = sum(abs(weight) for weight in weights.values())
        exit_notional = sum(abs(weight) * (1.0 + quotes[ticker]) for ticker, weight in weights.items())
        traded_notional = float(entry_notional + exit_notional)
        transaction_cost = cost_rate * traded_notional
        benchmark_quotes = quotes[np.isfinite(quotes)]
        returns.append({
            "date": date,
            "execution_date": group["execution_date"].iloc[0] if "execution_date" in group else date,
            "gross_return": gross_return,
            "net_return": gross_return - transaction_cost,
            "benchmark_return": float(benchmark_quotes.mean()) if len(benchmark_quotes) else 0.0,
            "benchmark_coverage": float(len(benchmark_quotes) / len(group)),
            "turnover": traded_notional / 2.0,
            "traded_notional": traded_notional,
            "transaction_cost": transaction_cost,
            "gross_exposure": entry_notional,
            "selected_assets": len(weights),
        })
        positions.extend({"date": date, "ticker": ticker, "weight": weight} for ticker, weight in weights.items())
    return pd.DataFrame(returns).set_index("date"), pd.DataFrame(positions, columns=["date", "ticker", "weight"])


def summarize(report: pd.DataFrame, initial_cash: float, annual_risk_free_rate: float = 0.0381) -> Dict[str, object]:
    if not np.isfinite(initial_cash) or initial_cash <= 0:
        raise ValueError("initial_cash must be finite and positive")
    if not np.isfinite(annual_risk_free_rate) or annual_risk_free_rate <= -1:
        raise ValueError("annual_risk_free_rate must be finite and greater than -1")
    net = report["net_return"].astype(float)
    if net.empty or not np.isfinite(net).all() or (net <= -1).any():
        raise ValueError("Portfolio returns must be finite, non-empty and greater than -100%")
    wealth = initial_cash * (1.0 + net).cumprod()
    annual = 252.0
    std = float(net.std(ddof=1)) if len(net) > 1 else 0.0
    # Explicit geometric conversion; the paper specifies annual rf, not conversion.
    daily_rf = (1.0 + annual_risk_free_rate) ** (1.0 / annual) - 1.0
    sharpe = float((net.mean() - daily_rf) / std * np.sqrt(annual)) if std > 1e-12 else None
    drawdown = wealth / wealth.cummax().clip(lower=initial_cash) - 1.0
    cumulative = float(wealth.iloc[-1] / initial_cash - 1.0)
    benchmark = float((1.0 + report["benchmark_return"]).prod() - 1.0)
    return {
        "start": str(report.index.min().date()),
        "end": str(report.index.max().date()),
        "trading_days": int(len(net)),
        "cumulative_return": cumulative,
        "annualized_return": float((wealth.iloc[-1] / initial_cash) ** (annual / len(net)) - 1.0),
        "annualized_volatility": std * np.sqrt(annual),
        "annual_risk_free_rate": annual_risk_free_rate,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
        "mean_daily_turnover": float(report["turnover"].mean()),
        "total_cost": float((report["gross_return"] - net).sum()),
        "benchmark_cumulative_return": benchmark,
        "excess_cumulative_return": cumulative - benchmark,
    }


def is_feasible(result: Dict[str, object]) -> bool:
    values = [result.get(key) for key in ("annualized_return", "sharpe", "max_drawdown")]
    if any(value is None or not np.isfinite(value) for value in values):
        return False
    return values[0] > 0.08 and values[1] > 0.6 and values[2] > -0.08


def select_configuration(results: List[Dict[str, object]]) -> Optional[Dict[str, object]]:
    if not 1 <= len(results) <= 3:
        raise ValueError("Algorithm 3 requires one to three trials")
    return max((result for result in results if is_feasible(result)), key=lambda result: result["sharpe"], default=None)


def file_fingerprint(path: Path) -> Dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def main() -> None:
    base = Path(__file__).resolve().parents[1] / "machine-learn"
    parser = argparse.ArgumentParser(description="Run the strict-gated AlphaCrafter intraday Trader proxy")
    parser.add_argument("--predictions", type=Path, default=base / "data/qlib/alphacrafter_screener_lite/predictions.csv")
    parser.add_argument("--input", type=Path, default=base / "data/processed/sp500_features.csv.gz")
    parser.add_argument("--output-dir", type=Path, default=base / "data/qlib/alphacrafter_trader_lite")
    parser.add_argument("--selection-start", default="2023-01-03")
    parser.add_argument("--selection-end", default="2023-12-29")
    parser.add_argument("--backtest-start", default="2024-01-02")
    parser.add_argument("--backtest-end", default="2026-02-27")
    parser.add_argument("--configs", default="0.8,0.5,50,25;0.8,0.0,50,50;0.8,1.0,50,0", help="At most three beta,gamma,nlong,nshort trials")
    parser.add_argument("--cost-rate", type=float, default=0.0001, help="Cost per traded notional, charged at entry AND exit (S&P 500: 1bp)")
    parser.add_argument("--annual-risk-free-rate", type=float, default=0.0381)
    parser.add_argument("--initial-cash", type=float, default=1_000_000.0)
    parser.add_argument("--calendar", type=Path, help="Reference market-session CSV; otherwise use observed feature dates")
    parser.add_argument("--require-explicit-label-dates", action="store_true")
    args = parser.parse_args()
    try:
        validate_intervals(args.selection_start, args.selection_end, args.backtest_start, args.backtest_end)
        configs = parse_configs(args.configs)
    except ValueError as exc:
        parser.error(str(exc))
    frame = load_inputs(args.predictions, args.input, calendar_path=args.calendar,
                        require_explicit_label_dates=args.require_explicit_label_dates)
    validation_results = []
    for config in configs:
        report, _ = simulate(frame, config, args.selection_start, args.selection_end, args.cost_rate)
        metrics = summarize(report, args.initial_cash, args.annual_risk_free_rate)
        validation_results.append({"config": asdict(config), "name": config.name, **metrics})
    selected = select_configuration(validation_results)
    config = TraderConfig(**selected["config"]) if selected is not None else None
    report, positions = simulate(frame, config, args.backtest_start, args.backtest_end, args.cost_rate)
    metrics = summarize(report, args.initial_cash, args.annual_risk_free_rate)
    summary = {
        "trader_version": "trader-lite-v2",
        "inputs": {"features": file_fingerprint(args.input), "predictions": file_fingerprint(args.predictions)},
        "runtime": {"numpy": np.__version__, "pandas": pd.__version__},
        "label_end_source": label_end_source(frame, 1),
        "execution_calendar": file_fingerprint(args.calendar) if args.calendar else "observed_feature_dates",
        "require_explicit_label_dates": args.require_explicit_label_dates,
        "status": "selected" if selected else "no_feasible_strategy",
        "selected_config": asdict(config) if config else None,
        "selected_config_name": config.name if config else None,
        "selection_constraints": {"annualized_return": ">0.08", "sharpe": ">0.6", "max_drawdown": ">-0.08"},
        "feasible_config_count": sum(is_feasible(result) for result in validation_results),
        "validation_results": validation_results,
        "backtest": metrics,
        "execution": "T close signal; T+1 open entry, T+1 close liquidation; flat overnight; zero interest on cash",
        "cost_convention": "cost_rate * (entry_notional + exit_notional), relative to opening NAV",
        "limitations": [
            "Intraday proxy, not the paper's persistent-holdings simulator; no margin, borrow, liquidity or slippage model",
            "Input predictions must independently satisfy point-in-time factor admission; this tool cannot certify their provenance",
            "Benchmark is gross available-asset equal-weight intraday return, not the S&P 500 index",
            "No LLM-generated strategy code or regime-conditioned parameter search",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output_dir / "portfolio_returns.csv")
    positions.to_csv(args.output_dir / "positions.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (args.output_dir / "config.json").write_text(json.dumps({key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, indent=2) + "\n")
    sharpe_text = f"{metrics['sharpe']:.3f}" if metrics["sharpe"] is not None else "undefined (zero volatility)"
    lines = [
        "# AlphaCrafter Trader-lite 报告", "",
        f"- 状态：`{summary['status']}`；配置：`{summary['selected_config_name']}`",
        f"- 验证期可行配置数：`{summary['feasible_config_count']}`；无可行配置时保持现金，不回退。",
        f"- 回测累计收益：`{metrics['cumulative_return']:.2%}`",
        f"- 回测年化收益：`{metrics['annualized_return']:.2%}`",
        f"- 回测 Sharpe（年无风险收益 {args.annual_risk_free_rate:.2%}）：`{sharpe_text}`",
        f"- 回测最大回撤：`{metrics['max_drawdown']:.2%}`", "",
        "> 本地日内代理：次日开盘建仓、收盘平仓，双边按成交额计费。不是论文完整交易所模拟。",
        "> 因子库准入时间必须由上游保证；不能用在 2023 年筛选出的因子再证明 2023 年策略选择无泄漏。",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2, allow_nan=False))
    print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
