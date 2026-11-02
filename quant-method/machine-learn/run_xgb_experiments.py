#!/usr/bin/env python3
"""Run controlled experiments for the Qlib XGBoost baseline.

Experiments are selected only by validation Rank IC.  The backtest segment is
reported after selection and is never used to choose an objective, round count,
or feature set.
"""

from __future__ import annotations

import argparse
import gc
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

import train_xgboost_qlib as baseline


BASE_CONFIG = {
    "max_depth": 4,
    "eta": 0.03,
    "min_child_weight": 20.0,
    "subsample": 0.70,
    "colsample_bytree": 0.75,
    "gamma": 0.10,
    "reg_alpha": 0.0,
    "reg_lambda": 10.0,
    "nthread": 4,
    "seed": 42,
}
ROUND_GRID = [25, 50, 75, 100, 150, 200, 300, 400]
STANDARD_SEGMENTS = {
    "train": ("2016-06-23", "2022-12-30"),
    "valid": ("2023-01-03", "2023-12-29"),
    "backtest": ("2024-01-02", "2026-02-27"),
}
ROLLING_FOLDS = [
    ("2021", ("2016-06-23", "2020-12-31"), ("2021-01-04", "2021-12-31")),
    ("2022", ("2016-06-23", "2021-12-31"), ("2022-01-03", "2022-12-30")),
    ("2023", ("2016-06-23", "2022-12-30"), ("2023-01-03", "2023-12-29")),
]


def date_slice(frame: pd.DataFrame, bounds: Tuple[str, str]) -> pd.DataFrame:
    dates = frame.index.get_level_values("datetime")
    return frame[(dates >= pd.Timestamp(bounds[0])) & (dates <= pd.Timestamp(bounds[1]))]


def make_features(raw: pd.DataFrame, dropped: Iterable[str] = ()) -> pd.DataFrame:
    frame = baseline.apply_feature_transform(raw, "cs_rank")
    frame = baseline.apply_label_transform(frame)
    dropped = set(dropped)
    if dropped:
        level0 = frame.columns.get_level_values(0)
        level1 = frame.columns.get_level_values(1)
        keep = ~((level0 == "feature") & level1.isin(dropped))
        frame = frame.loc[:, keep]
    return frame


def load_xgb():
    try:
        import xgboost as xgb
        from qlib.data.dataset.handler import DataHandlerLP
    except ImportError as exc:
        raise SystemExit(
            "Qlib and XGBoost are required. Use the tradingagents environment."
        ) from exc
    return xgb, DataHandlerLP


def prepare_matrices(dataset: Any, xgb: Any, data_handler_lp: Any):
    train = dataset.prepare(
        "train", col_set=["feature", "label"], data_key=data_handler_lp.DK_L
    )
    valid = dataset.prepare(
        "valid", col_set=["feature", "label"], data_key=data_handler_lp.DK_L
    )
    dtrain = xgb.DMatrix(train["feature"].values, label=train["label"].values.ravel())
    dvalid = xgb.DMatrix(valid["feature"].values, label=valid["label"].values.ravel())
    return train, valid, dtrain, dvalid


def score_prediction(raw: pd.DataFrame, prediction: np.ndarray, index: pd.Index, bounds):
    predicted = pd.Series(prediction, index=index, name="prediction")
    label = baseline.as_series(
        date_slice(raw, bounds)[[("label", "target_1d")]], "label"
    )
    return baseline.evaluate(label, predicted)


def run_one(
    raw: pd.DataFrame,
    transformed: pd.DataFrame,
    segments: Dict[str, Tuple[str, str]],
    objective: str,
    round_grid: List[int],
) -> Dict[str, Any]:
    xgb, data_handler_lp = load_xgb()
    dataset = baseline.prepare_dataset(transformed, segments=segments)
    _, valid, dtrain, dvalid = prepare_matrices(dataset, xgb, data_handler_lp)
    max_round = max(round_grid)
    params = dict(BASE_CONFIG)
    params.update(objective=objective, eval_metric="rmse")
    booster = xgb.train(
        params,
        dtrain=dtrain,
        num_boost_round=max_round,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        verbose_eval=False,
    )

    valid_bounds = segments["valid"]
    checkpoint_rows = []
    for rounds in round_grid:
        prediction = booster.predict(dvalid, iteration_range=(0, rounds))
        metrics = score_prediction(raw, prediction, valid.index, valid_bounds)
        checkpoint_rows.append(
            {
                "objective": objective,
                "rounds": rounds,
                "valid_rank_ic": metrics["mean_daily_rank_ic"],
                "valid_rank_icir": metrics["rank_icir"],
                "valid_top_bottom_spread": metrics["mean_top_bottom_decile_spread"],
            }
        )
    checkpoints = pd.DataFrame(checkpoint_rows)
    best_row = checkpoints.sort_values(
        ["valid_rank_ic", "valid_rank_icir"], ascending=False
    ).iloc[0]
    best_round = int(best_row["rounds"])
    valid_prediction = booster.predict(dvalid, iteration_range=(0, best_round))
    valid_metrics = score_prediction(raw, valid_prediction, valid.index, valid_bounds)

    result: Dict[str, Any] = {
        "objective": objective,
        "best_round": best_round,
        "valid": valid_metrics,
        "checkpoints": checkpoint_rows,
    }
    if "backtest" in segments:
        test = dataset.prepare("backtest", col_set="feature", data_key=data_handler_lp.DK_I)
        dtest = xgb.DMatrix(test.values)
        test_prediction = booster.predict(dtest, iteration_range=(0, best_round))
        result["backtest"] = score_prediction(
            raw, test_prediction, test.index, segments["backtest"]
        )
        del test, dtest
    del booster, dtrain, dvalid, valid, dataset
    gc.collect()
    return result


def drop_variant(name: str) -> List[str]:
    if name == "without_return_1d":
        return ["return_1d"]
    if name == "without_one_day_reversal":
        return ["one_day_reversal"]
    return []


def compact_result(result: Dict[str, Any], experiment: str, name: str) -> Dict[str, Any]:
    row = {
        "experiment": experiment,
        "name": name,
        "objective": result["objective"],
        "best_round": result["best_round"],
    }
    for split in ("valid", "backtest"):
        if split in result:
            row[f"{split}_rank_ic"] = result[split]["mean_daily_rank_ic"]
            row[f"{split}_rank_icir"] = result[split]["rank_icir"]
            row[f"{split}_hit_ratio"] = result[split]["rank_ic_hit_ratio"]
            row[f"{split}_top_bottom_spread"] = result[split][
                "mean_top_bottom_decile_spread"
            ]
    return row


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(无结果)"
    columns = [str(column) for column in frame.columns]
    rows = [columns, ["---"] * len(columns)]
    for values in frame.itertuples(index=False, name=None):
        rendered = []
        for value in values:
            if isinstance(value, (float, np.floating)):
                rendered.append(f"{value:.6f}")
            else:
                rendered.append(str(value))
        rows.append(rendered)
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "processed" / "sp500_features.csv.gz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "qlib" / "xgb_experiments",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw, feature_columns = baseline.load_frame(args.input)
    full_features = make_features(raw)

    all_checkpoint_rows = []
    summary_rows = []
    detailed: Dict[str, Any] = {}

    # 1/2. Select rounds by validation Rank IC and compare loss functions.
    loss_results = {}
    for objective in ("reg:squarederror", "reg:pseudohubererror", "reg:absoluteerror"):
        result = run_one(raw, full_features, STANDARD_SEGMENTS, objective, ROUND_GRID)
        loss_results[objective] = result
        all_checkpoint_rows.extend(
            [{"experiment": "loss", "name": objective, **row} for row in result["checkpoints"]]
        )
        summary_rows.append(compact_result(result, "loss", objective))
    chosen_objective = max(
        loss_results,
        key=lambda key: loss_results[key]["valid"]["mean_daily_rank_ic"],
    )
    detailed["loss"] = loss_results

    # 3. Ablate the two one-day return representations using the selected loss.
    ablation_results = {}
    for name in ("full", "without_return_1d", "without_one_day_reversal"):
        transformed = make_features(raw, drop_variant(name))
        result = run_one(raw, transformed, STANDARD_SEGMENTS, chosen_objective, ROUND_GRID)
        ablation_results[name] = result
        all_checkpoint_rows.extend(
            [{"experiment": "ablation", "name": name, **row} for row in result["checkpoints"]]
        )
        summary_rows.append(compact_result(result, "ablation", name))
        del transformed
        gc.collect()
    detailed["ablation"] = ablation_results

    # 4. Rolling validation with the objective selected only on the standard
    # validation period.  Each fold selects its own round count by Rank IC.
    rolling_rows = []
    rolling_detail = {}
    for fold_name, train_bounds, valid_bounds in ROLLING_FOLDS:
        segments = {"train": train_bounds, "valid": valid_bounds}
        result = run_one(raw, full_features, segments, chosen_objective, ROUND_GRID)
        rolling_detail[fold_name] = result
        row = compact_result(result, "rolling", fold_name)
        rolling_rows.append(row)
        all_checkpoint_rows.extend(
            [{"experiment": "rolling", "name": fold_name, **checkpoint} for checkpoint in result["checkpoints"]]
        )
    detailed["rolling"] = rolling_detail

    summary = pd.DataFrame(summary_rows + rolling_rows)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    pd.DataFrame(all_checkpoint_rows).to_csv(args.output_dir / "round_search.csv", index=False)
    (args.output_dir / "results.json").write_text(
        __import__("json").dumps(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "input": str(args.input),
                "rows": int(len(raw)),
                "features": feature_columns,
                "chosen_objective": chosen_objective,
                "base_config": BASE_CONFIG,
                "round_grid": ROUND_GRID,
                "results": detailed,
            },
            indent=2,
            ensure_ascii=False,
            default=float,
        )
        + "\n"
    )

    loss_table = summary[summary.experiment == "loss"].drop(columns="experiment")
    ablation_table = summary[summary.experiment == "ablation"].drop(columns="experiment")
    rolling_table = pd.DataFrame(rolling_rows).drop(columns="experiment")
    report = f"""# XGBoost 基线实验记录

- 实验时间（UTC）：{datetime.now(timezone.utc).isoformat()}
- 输入文件：`{args.input}`
- 样本数：{len(raw):,}
- 特征数：{len(feature_columns)}
- 标签：`target_1d`，训练时使用 Qlib `CSRankNorm`
- 特征：默认使用 Qlib `CSRankNorm` 逐日横截面归一化
- 评价：验证集只使用每日 Rank IC 选择树轮数、损失函数和特征方案；回测集仅用于最终报告
- 基础超参：`{BASE_CONFIG}`
- 树轮数候选：`{ROUND_GRID}`

## 1. 损失函数比较

候选损失函数为 `reg:squarederror`、`reg:pseudohubererror` 和 `reg:absoluteerror`。每个损失函数均在验证集 Rank IC 最高的树轮数处评估。

验证集选择出的损失函数：`{chosen_objective}`。

{markdown_table(loss_table)}

## 2. return_1d 与 one_day_reversal 消融

- `full`：同时保留两个特征
- `without_return_1d`：移除 `return_1d`
- `without_one_day_reversal`：移除 `one_day_reversal`

{markdown_table(ablation_table)}

## 3. 滚动验证

滚动验证的训练集只使用验证年份之前的数据，每个窗口独立按 Rank IC 选择树轮数。使用在标准 2023 验证集上选出的损失函数 `{chosen_objective}`。

{markdown_table(rolling_table)}

## 4. 当前结论

1. 所有模型选择均以验证集 Rank IC 为准，未使用回测集选择超参。
2. 需要重点观察不同滚动窗口中的 Rank IC 是否保持为正；单个年份的提升不能视为稳定 alpha。
3. `summary.csv` 保存汇总结果，`round_search.csv` 保存各候选树轮数结果，`results.json` 保存完整实验细节。

## 5. 复现实验

```bash
conda run -n tradingagents python run_xgb_experiments.py
```
"""
    (args.output_dir / "experiment_report.md").write_text(report)
    # Keep a tracked research note as well as the ignored machine-readable artifacts.
    (Path(__file__).resolve().parent / "XGB_EXPERIMENTS.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
