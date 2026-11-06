#!/usr/bin/env python3
"""AlphaCrafter Trader-lite.

This module implements the paper's Trader idea without executing generated
Python code.  It searches a small, explicit set of long/short portfolio
configurations, selects one on a validation interval, and evaluates it once on
the later holdout interval.

The signal is generated at T and uses ``target_open_close`` for T+1 execution,
matching the existing Qlib convention.  All portfolio weights are produced
locally and are safe to audit.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TraderConfig:
    beta: float
    gamma: float
    nlong: int
    nshort: int

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
        config = TraderConfig(float(values[0]), float(values[1]), int(values[2]), int(values[3]))
        if not 0 < config.beta <= 1:
            raise ValueError("beta must be in (0, 1]")
        if not -1 <= config.gamma <= 1:
            raise ValueError("gamma must be in [-1, 1]")
        if config.nlong < 1 or config.nshort < 1:
            raise ValueError("nlong and nshort must be positive")
        configs.append(config)
    if not configs:
        raise ValueError("At least one trader config is required")
    return configs


def load_inputs(predictions_path: Path, feature_path: Path) -> pd.DataFrame:
    predictions = pd.read_csv(predictions_path, parse_dates=["datetime"])
    required = {"datetime", "instrument", "prediction"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Predictions are missing fields: {sorted(missing)}")
    predictions = predictions.rename(columns={"datetime": "date", "instrument": "ticker"})
    predictions = predictions[["date", "ticker", "prediction"]]

    labels = pd.read_csv(
        feature_path,
        compression="infer",
        usecols=["date", "ticker", "target_open_close"],
        parse_dates=["date"],
    )
    frame = labels.merge(predictions, on=["date", "ticker"], how="left")
    return frame.sort_values(["date", "ticker"]).reset_index(drop=True)


def target_weights(group: pd.DataFrame, config: TraderConfig) -> Dict[str, float]:
    ranked = group.dropna(subset=["prediction"]).sort_values("prediction")
    if ranked.empty:
        return {}
    long_names = list(ranked.tail(min(config.nlong, len(ranked)))['ticker'])
    short_pool = ranked[~ranked["ticker"].isin(long_names)]
    short_names = list(short_pool.head(min(config.nshort, len(short_pool)))['ticker'])
    weights: Dict[str, float] = {}
    if long_names:
        long_weight = config.long_exposure / len(long_names)
        weights.update({ticker: long_weight for ticker in long_names})
    if short_names:
        short_weight = config.short_exposure / len(short_names)
        weights.update({ticker: -short_weight for ticker in short_names})
    return weights


def simulate(
    frame: pd.DataFrame,
    config: TraderConfig,
    start: str,
    end: str,
    cost_rate: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dates = sorted(frame.loc[frame["date"].between(start, end), "date"].drop_duplicates())
    if not dates:
        raise ValueError(f"No rows in simulation interval {start} to {end}")
    previous_weights: Dict[str, float] = {}
    returns: List[Dict[str, float]] = []
    positions: List[Dict[str, object]] = []
    for date in dates:
        group = frame[frame["date"] == date].dropna(subset=["target_open_close"])
        predictions_available = group["prediction"].notna().sum()
        if predictions_available >= config.nlong + config.nshort:
            weights = target_weights(group, config)
        else:
            weights = previous_weights
        weight_series = pd.Series(weights, dtype=float)
        previous_series = pd.Series(previous_weights, dtype=float)
        all_names = sorted(set(weight_series.index) | set(previous_series.index))
        current_vector = weight_series.reindex(all_names, fill_value=0.0)
        previous_vector = previous_series.reindex(all_names, fill_value=0.0)
        turnover = float((current_vector - previous_vector).abs().sum() / 2.0)
        quote_returns = group.set_index("ticker")["target_open_close"]
        gross_return = float(sum(weight * quote_returns.get(ticker, 0.0) for ticker, weight in weights.items()))
        net_return = gross_return - cost_rate * turnover
        benchmark = float(quote_returns.mean()) if not quote_returns.empty else 0.0
        returns.append(
            {
                "date": date,
                "gross_return": gross_return,
                "net_return": net_return,
                "benchmark_return": benchmark,
                "turnover": turnover,
                "selected_assets": float(len(weights)),
            }
        )
        for ticker, weight in weights.items():
            positions.append({"date": date, "ticker": ticker, "weight": weight})
        previous_weights = weights

    return pd.DataFrame(returns).set_index("date"), pd.DataFrame(positions)


def summarize(report: pd.DataFrame, initial_cash: float) -> Dict[str, float]:
    net = report["net_return"].astype(float)
    wealth = initial_cash * (1.0 + net).cumprod()
    annual = 252.0
    volatility = float(net.std(ddof=1) * np.sqrt(annual)) if len(net) > 1 else float("nan")
    sharpe = float(net.mean() / net.std(ddof=1) * np.sqrt(annual)) if net.std(ddof=1) else float("nan")
    drawdown = wealth / wealth.cummax() - 1.0
    cumulative = float(wealth.iloc[-1] / initial_cash - 1.0)
    annualized = float((wealth.iloc[-1] / initial_cash) ** (annual / len(net)) - 1.0)
    benchmark = float((1.0 + report["benchmark_return"]).prod() - 1.0)
    return {
        "start": str(report.index.min().date()),
        "end": str(report.index.max().date()),
        "trading_days": int(len(report)),
        "cumulative_return": cumulative,
        "annualized_return": annualized,
        "annualized_volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
        "mean_daily_turnover": float(report["turnover"].mean()),
        "total_cost": float((report["gross_return"] - report["net_return"]).sum()),
        "benchmark_cumulative_return": benchmark,
        "excess_cumulative_return": cumulative - benchmark,
    }


def main() -> None:
    base = Path(__file__).resolve().parents[1] / "machine-learn"
    parser = argparse.ArgumentParser(description="Run the AlphaCrafter Trader-lite")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=base / "data" / "qlib" / "alphacrafter_screener_lite" / "predictions.csv",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=base / "data" / "processed" / "sp500_features.csv.gz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base / "data" / "qlib" / "alphacrafter_trader_lite",
    )
    parser.add_argument("--selection-start", default="2023-01-03")
    parser.add_argument("--selection-end", default="2023-12-29")
    parser.add_argument("--backtest-start", default="2024-01-02")
    parser.add_argument("--backtest-end", default="2026-02-26")
    parser.add_argument(
        "--configs",
        default="0.95,0.0,50,50;0.95,0.25,50,35;0.8,0.5,50,25",
        help="Semicolon-separated beta,gamma,nlong,nshort candidates",
    )
    parser.add_argument(
        "--cost-rate",
        type=float,
        default=0.0001,
        help="Per-turnover transaction cost; paper S&P 500 setting is 1bp",
    )
    parser.add_argument("--initial-cash", type=float, default=1_000_000.0)
    args = parser.parse_args()
    configs = parse_configs(args.configs)
    frame = load_inputs(args.predictions, args.input)

    validation_results = []
    for config in configs:
        validation_report, _ = simulate(
            frame, config, args.selection_start, args.selection_end, args.cost_rate
        )
        metrics = summarize(validation_report, args.initial_cash)
        validation_results.append({"config": asdict(config), "name": config.name, **metrics})

    feasible = [
        result
        for result in validation_results
        if result["annualized_return"] > 0.08
        and result["sharpe"] > 0.6
        and result["max_drawdown"] > -0.08
    ]
    selection_pool = feasible or validation_results
    selected_result = max(selection_pool, key=lambda result: result["sharpe"])
    selected_config = TraderConfig(**selected_result["config"])
    backtest_report, positions = simulate(
        frame, selected_config, args.backtest_start, args.backtest_end, args.cost_rate
    )
    backtest_summary = summarize(backtest_report, args.initial_cash)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    backtest_report.to_csv(args.output_dir / "portfolio_returns.csv")
    positions.to_csv(args.output_dir / "positions.csv", index=False)
    summary = {
        "trader_version": "trader-lite-v1",
        "selected_config": asdict(selected_config),
        "selected_config_name": selected_config.name,
        "selection_constraints": {"annualized_return": ">0.08", "sharpe": ">0.6", "max_drawdown": ">-0.08"},
        "feasible_config_count": len(feasible),
        "validation_results": validation_results,
        "backtest": backtest_summary,
        "execution": "T close signal, T+1 open-to-close return; long/short weights are locally generated",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "config.json").write_text(
        json.dumps({key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, indent=2)
        + "\n"
    )
    report_lines = [
        "# AlphaCrafter Trader-lite 报告",
        "",
        f"- 选中配置：`{selected_config.name}`",
        f"- 验证期可行配置数：`{len(feasible)}`；若无可行配置则按验证期 Sharpe 回退选择。",
        f"- 回测累计收益：`{backtest_summary['cumulative_return']:.2%}`",
        f"- 回测年化收益：`{backtest_summary['annualized_return']:.2%}`",
        f"- 回测 Sharpe：`{backtest_summary['sharpe']:.3f}`",
        f"- 回测最大回撤：`{backtest_summary['max_drawdown']:.2%}`",
        "",
        "> 这是 Trader-lite 的本地长短组合模拟，不执行 LLM 生成代码，也不连接真实券商账户。",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report_lines) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
