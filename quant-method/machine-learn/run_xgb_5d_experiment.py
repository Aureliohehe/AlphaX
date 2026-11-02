#!/usr/bin/env python3
"""Evaluate a five-trading-day forward-return target with the same feature pool."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from qlib.data.dataset.handler import DataHandlerLP

import train_xgboost_qlib as baseline
from run_xgb_experiments import BASE_CONFIG, ROUND_GRID


SEGMENTS = {
    "train": ("2016-06-23", "2022-12-23"),
    "valid": ("2023-01-03", "2023-12-22"),
    "backtest": ("2024-01-02", "2026-02-20"),
}


def date_slice(frame: pd.DataFrame, bounds):
    dates = frame.index.get_level_values("datetime")
    return frame[(dates >= pd.Timestamp(bounds[0])) & (dates <= pd.Timestamp(bounds[1]))]


def build_five_day_frame(path: Path):
    source = pd.read_csv(path, compression="infer", parse_dates=["date"])
    source = source.sort_values(["ticker", "date"]).copy()

    def forward_return(series: pd.Series) -> pd.Series:
        return (1 + series).rolling(5, min_periods=5).apply(np.prod, raw=True).shift(-4) - 1

    source["target_5d"] = source.groupby("ticker", sort=False)["target_1d"].transform(
        forward_return
    )
    raw, features = baseline.load_frame(path)
    labels = source.set_index(["date", "ticker"])["target_5d"]
    keys = pd.MultiIndex.from_arrays(
        [raw.index.get_level_values("datetime"), raw.index.get_level_values("instrument")],
        names=["date", "ticker"],
    )
    raw[("label", "target_1d")] = labels.reindex(keys).to_numpy()
    raw = raw.dropna(subset=[("label", "target_1d")])
    return raw, features


def main() -> None:
    root = Path(__file__).resolve().parent
    input_path = root / "data" / "processed" / "sp500_features.csv.gz"
    output_dir = root / "data" / "qlib" / "xgb_5d_experiment"
    output_dir.mkdir(parents=True, exist_ok=True)

    raw, features = build_five_day_frame(input_path)
    transformed = baseline.apply_feature_transform(raw, "cs_rank")
    transformed = baseline.apply_label_transform(transformed)
    dataset = baseline.prepare_dataset(transformed, segments=SEGMENTS)

    train = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
    valid = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
    dtrain = xgb.DMatrix(train["feature"].values, label=train["label"].values.ravel())
    dvalid = xgb.DMatrix(valid["feature"].values, label=valid["label"].values.ravel())
    params = dict(BASE_CONFIG)
    params.update(objective="reg:pseudohubererror", eval_metric="rmse")
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=max(ROUND_GRID),
        evals=[(dtrain, "train"), (dvalid, "valid")],
        verbose_eval=False,
    )

    raw_valid = baseline.as_series(
        date_slice(raw, SEGMENTS["valid"])[[("label", "target_1d")]], "label"
    )
    search_rows = []
    for rounds in ROUND_GRID:
        prediction = pd.Series(
            booster.predict(dvalid, iteration_range=(0, rounds)), index=valid.index
        )
        search_rows.append({"rounds": rounds, **baseline.evaluate(raw_valid, prediction)})
    search = pd.DataFrame(search_rows)
    best = search.sort_values("mean_daily_rank_ic", ascending=False).iloc[0]
    best_round = int(best["rounds"])

    final = {}
    for split in ("valid", "backtest"):
        part = dataset.prepare(split, col_set="feature", data_key=DataHandlerLP.DK_I)
        prediction = pd.Series(
            booster.predict(xgb.DMatrix(part.values), iteration_range=(0, best_round)),
            index=part.index,
        )
        label = baseline.as_series(
            date_slice(raw, SEGMENTS[split])[[('label', 'target_1d')]], "label"
        )
        final[split] = baseline.evaluate(label, prediction)

    reversal_rows = {}
    for split in ("valid", "backtest"):
        part = date_slice(raw, SEGMENTS[split])
        # The existing five-day feature is the contemporaneous five-day return.
        score = -part[("feature", "return_5d")]
        label = baseline.as_series(part[[('label', 'target_1d')]], "label")
        reversal_rows[split] = baseline.evaluate(label, score)

    payload = {
        "target": "target_5d = close[t+5] / close[t] - 1",
        "best_round": best_round,
        "model": final,
        "five_day_reversal_baseline": reversal_rows,
        "features": features,
        "params": params,
    }
    search.to_csv(output_dir / "round_search.csv", index=False)
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2, default=float) + "\n")

    report = f"""# 五交易日预测目标实验

- 标签：`target_5d = close[t+5] / close[t] - 1`
- 特征：36 个技术特征，逐日 `CSRankNorm`
- 损失函数：`reg:pseudohubererror`
- 树轮数：验证集 Rank IC 选择为 `{best_round}`
- 时间切分：训练至 2022-12-23，验证 2023，回测 2024-01-02 至 2026-02-20

## 结果

| 方法 | 验证 Rank IC | 回测 Rank IC |
| --- | ---: | ---: |
| XGBoost 5 日目标 | {final['valid']['mean_daily_rank_ic']:.6f} | {final['backtest']['mean_daily_rank_ic']:.6f} |
| `-return_5d` 基线 | {reversal_rows['valid']['mean_daily_rank_ic']:.6f} | {reversal_rows['backtest']['mean_daily_rank_ic']:.6f} |

详细结果：`data/qlib/xgb_5d_experiment/results.json`。
"""
    (output_dir / "experiment_report.md").write_text(report)
    (root / "XGB_5D_EXPERIMENT.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
