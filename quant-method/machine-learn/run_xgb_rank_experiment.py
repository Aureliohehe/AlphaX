#!/usr/bin/env python3
"""Evaluate a date-grouped XGBoost pairwise-ranking baseline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from qlib.data.dataset.handler import DataHandlerLP

import train_xgboost_qlib as baseline


ROUNDS = [25, 50, 75, 100, 150, 200, 300, 400]
SEGMENTS = {
    "train": ("2016-06-23", "2022-12-30"),
    "valid": ("2023-01-03", "2023-12-29"),
    "backtest": ("2024-01-02", "2026-02-27"),
}
PARAMS = {
    "objective": "rank:pairwise",
    "eval_metric": "ndcg@10",
    "max_depth": 4,
    "eta": 0.03,
    "min_child_weight": 20,
    "subsample": 0.7,
    "colsample_bytree": 0.75,
    "gamma": 0.1,
    "reg_lambda": 10,
    "nthread": 4,
    "seed": 42,
}


def select(frame: pd.DataFrame, bounds: tuple[str, str]) -> pd.DataFrame:
    dates = frame.index.get_level_values("datetime")
    return frame[(dates >= pd.Timestamp(bounds[0])) & (dates <= pd.Timestamp(bounds[1]))]


def group_sizes(frame: pd.DataFrame) -> np.ndarray:
    return frame.groupby(level="datetime", sort=False).size().to_numpy(dtype=np.uint32)


def relevance_labels(frame: pd.DataFrame) -> np.ndarray:
    """Convert each date's label ordering to integer relevance levels."""
    label = frame["label"].iloc[:, 0]
    percentile = label.groupby(level="datetime", sort=False).rank(method="first", pct=True)
    # Ten relevance levels keep XGBoost's exponential NDCG gain bounded.
    return np.minimum((percentile * 10).astype(np.int32), 9).to_numpy()


def main() -> None:
    root = Path(__file__).resolve().parent
    output_dir = root / "data" / "qlib" / "xgb_rank_experiment"
    output_dir.mkdir(parents=True, exist_ok=True)

    raw, features = baseline.load_frame(root / "data" / "processed" / "sp500_features.csv.gz")
    transformed = baseline.apply_feature_transform(raw, "cs_rank")
    transformed = baseline.apply_label_transform(transformed)
    dataset = baseline.prepare_dataset(transformed, segments=SEGMENTS)

    train = dataset.prepare(
        "train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L
    )
    valid = dataset.prepare(
        "valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L
    )
    dtrain = xgb.DMatrix(train["feature"].values, label=relevance_labels(train))
    dvalid = xgb.DMatrix(valid["feature"].values, label=relevance_labels(valid))
    dtrain.set_group(group_sizes(train))
    dvalid.set_group(group_sizes(valid))

    booster = xgb.train(
        PARAMS,
        dtrain,
        num_boost_round=max(ROUNDS),
        evals=[(dtrain, "train"), (dvalid, "valid")],
        verbose_eval=False,
    )

    rows = []
    raw_valid = baseline.as_series(
        select(raw, SEGMENTS["valid"])[[("label", "target_1d")]], "label"
    )
    for rounds in ROUNDS:
        prediction = pd.Series(
            booster.predict(dvalid, iteration_range=(0, rounds)), index=valid.index
        )
        metrics = baseline.evaluate(raw_valid, prediction)
        rows.append({"rounds": rounds, **metrics})
    result = pd.DataFrame(rows)
    best = result.sort_values("mean_daily_rank_ic", ascending=False).iloc[0]
    best_round = int(best["rounds"])

    output = {}
    for split in ("valid", "backtest"):
        frame = dataset.prepare(split, col_set="feature", data_key=DataHandlerLP.DK_I)
        dmatrix = xgb.DMatrix(frame.values)
        dmatrix.set_group(group_sizes(frame)) if split == "backtest" else None
        prediction = pd.Series(
            booster.predict(dmatrix, iteration_range=(0, best_round)), index=frame.index
        )
        label = baseline.as_series(
            select(raw, SEGMENTS[split])[[('label', 'target_1d')]], "label"
        )
        output[split] = baseline.evaluate(label, prediction)

    result.to_csv(output_dir / "round_search.csv", index=False)
    payload = {
        "objective": PARAMS["objective"],
        "best_round": best_round,
        "valid": output["valid"],
        "backtest": output["backtest"],
        "features": features,
        "params": PARAMS,
    }
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2, default=float) + "\n")
    report = f"""# XGBoost Pairwise Ranking 实验

- 目标：`rank:pairwise`
- 分组：每个交易日一个 group
- 标签：将 Qlib `CSRankNorm` 标签转换为每个交易日内的整数相关性等级，满足 XGBoost ranking 的标签要求
- 特征：逐日 `CSRankNorm`
- 最佳树轮数：`{best_round}`，仅依据 2023 验证集 Rank IC 选择

## 验证集候选轮数

{result[['rounds', 'mean_daily_rank_ic', 'rank_icir', 'rank_ic_hit_ratio', 'mean_top_bottom_decile_spread']].to_string(index=False)}

## 最终结果

- 验证集：`{output['valid']}`
- 回测集：`{output['backtest']}`

该实验使用 Qlib `DatasetH` 准备数据，并通过 XGBoost 原生 ranking API 设置每日 group。
"""
    (output_dir / "experiment_report.md").write_text(report)
    (root / "XGB_RANK_EXPERIMENT.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
