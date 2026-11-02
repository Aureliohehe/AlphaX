#!/usr/bin/env python3
"""Select a validation-only blend of the XGBoost score and one-day reversal."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import train_xgboost_qlib as baseline


ALPHAS = np.linspace(0.0, 1.0, 21)


def cross_sectional_rank_score(series: pd.Series) -> pd.Series:
    ranks = series.groupby(level="datetime", sort=False).rank(pct=True)
    return (ranks - 0.5) * 3.46


def main() -> None:
    root = Path(__file__).resolve().parent
    input_path = root / "data" / "qlib" / "xgboost_baseline" / "predictions.csv"
    output_dir = root / "data" / "qlib" / "xgb_blend_experiment"
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(input_path, parse_dates=["datetime"])
    frame = frame.set_index(["datetime", "instrument"]).sort_index()
    frame["model_score_rank"] = cross_sectional_rank_score(frame["prediction"])
    frame["reversal_score_rank"] = cross_sectional_rank_score(frame["one_day_reversal"])

    rows = []
    for alpha in ALPHAS:
        for split, group in frame.groupby("split", sort=False):
            score = alpha * group["model_score_rank"] + (1 - alpha) * group[
                "reversal_score_rank"
            ]
            metrics = baseline.evaluate(group["label"], score)
            rows.append({"alpha_model": float(alpha), "split": split, **metrics})
    results = pd.DataFrame(rows)

    valid = results[results["split"] == "validation"].sort_values(
        ["mean_daily_rank_ic", "rank_icir"], ascending=False
    )
    best_alpha = float(valid.iloc[0]["alpha_model"])
    selected = results[results["alpha_model"] == best_alpha].copy()

    payload = {
        "input": str(input_path),
        "selection": "validation mean daily Rank IC, then Rank ICIR",
        "best_alpha_model": best_alpha,
        "alpha_reversal": 1.0 - best_alpha,
        "selected_metrics": selected.to_dict(orient="records"),
    }
    results.to_csv(output_dir / "alpha_search.csv", index=False)
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2, default=float) + "\n")

    report = f"""# XGBoost 与一日反转因子融合实验

- 输入模型：当前验证集选择出的 `reg:pseudohubererror` XGBoost（25 棵树）
- 融合方式：先对模型分数和一日反转因子分别做逐日横截面 RankNorm，再进行线性融合
- 融合公式：`score = {best_alpha:.2f} * model + {1.0 - best_alpha:.2f} * one_day_reversal`
- 选择标准：只使用验证集平均 Rank IC，Rank IC 相同时比较 Rank ICIR

## Alpha 搜索结果

| alpha_model | 验证 Rank IC | 回测 Rank IC | 验证 Rank ICIR | 回测 Rank ICIR |
| ---: | ---: | ---: | ---: | ---: |
"""
    for alpha in sorted(results["alpha_model"].unique()):
        valid_row = results[(results.alpha_model == alpha) & (results.split == "validation")].iloc[0]
        back_row = results[(results.alpha_model == alpha) & (results.split == "backtest")].iloc[0]
        marker = " **<-- selected**" if alpha == best_alpha else ""
        report += (
            f"| {alpha:.2f} | {valid_row.mean_daily_rank_ic:.6f} | "
            f"{back_row.mean_daily_rank_ic:.6f} | {valid_row.rank_icir:.6f} | "
            f"{back_row.rank_icir:.6f} |{marker}\n"
        )
    report += "\n详细结果：`data/qlib/xgb_blend_experiment/alpha_search.csv`。\n"
    (output_dir / "experiment_report.md").write_text(report)
    (root / "XGB_BLEND_EXPERIMENT.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
