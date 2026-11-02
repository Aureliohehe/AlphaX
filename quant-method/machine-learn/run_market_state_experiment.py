#!/usr/bin/env python3
"""Test aggregate, price-volume-only market-state features."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import train_xgboost_qlib as baseline
from run_xgb_experiments import ROUND_GRID, STANDARD_SEGMENTS, run_one


def add_market_state_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Add date-level features without applying CSRankNorm to them.

    These values use only the cross-section observed at date t.  They are
    deliberately kept in raw scale: rank-normalizing a date-constant feature
    would erase its market-regime information.
    """
    stock = raw["feature"]
    state = pd.DataFrame(index=raw.index)
    grouped = stock.groupby(level="datetime", sort=False)
    state["market_mean_return_1d"] = grouped["return_1d"].transform("mean")
    state["market_median_return_1d"] = grouped["return_1d"].transform("median")
    state["market_breadth_up_1d"] = grouped["return_1d"].transform(
        lambda values: (values > 0).mean()
    )
    state["market_dispersion_1d"] = grouped["return_1d"].transform("std")
    state["market_mean_return_5d"] = grouped["return_5d"].transform("mean")
    state["market_breadth_up_5d"] = grouped["return_5d"].transform(
        lambda values: (values > 0).mean()
    )
    state["market_mean_volatility_20d"] = grouped["volatility_20d"].transform("mean")
    state["market_mean_volume_ratio_20d"] = grouped["volume_ratio_20d"].transform("mean")
    state.columns = pd.MultiIndex.from_product([["feature"], state.columns])
    return pd.concat([raw, state], axis=1)


def make_transformed(raw: pd.DataFrame, with_market_state: bool) -> pd.DataFrame:
    transformed = baseline.apply_feature_transform(raw, "cs_rank")
    if with_market_state:
        state = add_market_state_features(raw).loc[:, "feature"].iloc[:, -8:]
        state.columns = pd.MultiIndex.from_tuples(
            [("feature", column) for column in state.columns]
        )
        transformed = pd.concat([transformed, state], axis=1)
    return baseline.apply_label_transform(transformed)


def main() -> None:
    root = Path(__file__).resolve().parent
    input_path = root / "data" / "processed" / "sp500_features.csv.gz"
    output_dir = root / "data" / "qlib" / "market_state_experiment"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw, features = baseline.load_frame(input_path)

    results = {}
    rows = []
    for name, with_state in (("technical_only", False), ("with_market_state", True)):
        transformed = make_transformed(raw, with_state)
        result = run_one(
            raw,
            transformed,
            STANDARD_SEGMENTS,
            "reg:pseudohubererror",
            ROUND_GRID,
        )
        results[name] = result
        rows.append(
            {
                "variant": name,
                "best_round": result["best_round"],
                "valid_rank_ic": result["valid"]["mean_daily_rank_ic"],
                "valid_rank_icir": result["valid"]["rank_icir"],
                "backtest_rank_ic": result["backtest"]["mean_daily_rank_ic"],
                "backtest_rank_icir": result["backtest"]["rank_icir"],
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "summary.csv", index=False)
    (output_dir / "results.json").write_text(
        json.dumps({"features": features, "results": results}, indent=2, default=float) + "\n"
    )
    report = f"""# 市场状态特征实验

- 数据：仅使用价格量数据，不接入基本面和新闻
- 新增状态特征：市场平均收益、中位数收益、上涨比例、横截面离散度、平均波动率、平均成交量比率
- 状态特征使用原始逐日横截面统计，未做 CSRankNorm；否则逐日常数会被归一化为无信息值
- 模型：`reg:pseudohubererror`，树轮数仅根据 2023 验证集 Rank IC 选择

| 方案 | 最佳轮数 | 验证 Rank IC | 回测 Rank IC | 验证 Rank ICIR | 回测 Rank ICIR |
| --- | ---: | ---: | ---: | ---: | ---: |
"""
    for row in rows:
        report += (
            f"| {row['variant']} | {row['best_round']} | {row['valid_rank_ic']:.6f} | "
            f"{row['backtest_rank_ic']:.6f} | {row['valid_rank_icir']:.6f} | "
            f"{row['backtest_rank_icir']:.6f} |\n"
        )
    report += "\n详细结果：`data/qlib/market_state_experiment/`。\n"
    (output_dir / "experiment_report.md").write_text(report)
    (root / "MARKET_STATE_EXPERIMENT.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
