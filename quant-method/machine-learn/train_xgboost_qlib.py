#!/usr/bin/env python3
"""Train the technical-feature XGBoost baseline through Qlib.

The input is the panel produced by ``build_features.py``.  Qlib's model and
DatasetH interfaces are used, while the small in-memory handler keeps the
experiment independent of Qlib's binary provider format.  The label is the
next-trading-day adjusted-close return (``target_1d``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd


# Qlib's XGBModel expects the second segment to be named ``valid``.
SPLITS = {
    "train": ("2016-01-04", "2022-12-30"),
    "valid": ("2023-01-03", "2023-12-29"),
    "backtest": ("2024-01-02", "2026-02-27"),
}


class FeatureFrameHandler:
    """Minimal Qlib handler for a prepared, leakage-safe feature panel.

    DatasetH only requires a handler exposing ``fetch``.  Keeping the source
    frame in memory is sufficient here and avoids converting the already
    prepared panel into Qlib's binary data format before every experiment.
    """

    def __init__(self, frame: pd.DataFrame):
        self.frame = frame

    def fetch(
        self,
        selector: Any = None,
        level: Optional[int] = None,
        col_set: Any = "all",
        data_key: Any = None,
        **_: Any,
    ) -> pd.DataFrame:
        frame = self.frame
        if isinstance(selector, tuple):
            selector = {"start_time": selector[0], "end_time": selector[1]}
        if isinstance(selector, dict):
            start = selector.get("start_time")
            end = selector.get("end_time")
            dates = frame.index.get_level_values("datetime")
            if start is not None:
                frame = frame[dates >= pd.Timestamp(start)]
                dates = frame.index.get_level_values("datetime")
            if end is not None:
                frame = frame[dates <= pd.Timestamp(end)]

        if col_set != "all":
            wanted = {col_set} if isinstance(col_set, str) else set(col_set)
            columns = frame.columns.get_level_values(0).isin(wanted)
            frame = frame.loc[:, columns]
        return frame.copy()


def load_frame(path: Path) -> tuple[pd.DataFrame, list[str]]:
    frame = pd.read_csv(path, compression="infer", parse_dates=["date"])
    required = {"date", "ticker", "target_1d", "split"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Make the simple one-day reversal signal explicit in the feature pool.
    # It is mathematically equivalent to -return_1d and uses no future data.
    frame["one_day_reversal"] = -frame["return_1d"]
    feature_columns = [
        column
        for column in frame.columns
        if column not in {"date", "ticker", "target_1d", "split"}
    ]
    frame = frame.dropna(subset=feature_columns + ["target_1d"]).copy()
    frame["datetime"] = frame["date"]
    frame["instrument"] = frame["ticker"]
    frame = frame.set_index(["datetime", "instrument"]).sort_index()

    # Qlib DatasetH expects column groups named feature and label.
    grouped = pd.concat(
        [
            frame[feature_columns].rename(columns={c: ("feature", c) for c in feature_columns}),
            frame[["target_1d"]].rename(columns={"target_1d": ("label", "target_1d")}),
        ],
        axis=1,
    )
    grouped.columns = pd.MultiIndex.from_tuples(grouped.columns)
    return grouped, feature_columns


def as_series(value: Any, name: str) -> pd.Series:
    if isinstance(value, pd.DataFrame):
        value = value.iloc[:, 0]
    if not isinstance(value, pd.Series):
        value = pd.Series(np.asarray(value).reshape(-1))
    result = value.astype(float).rename(name)
    return result


def evaluate(
    label: pd.Series, prediction: pd.Series, include_regression_metrics: bool = False
) -> Dict[str, float]:
    values = pd.concat([label.rename("label"), prediction.rename("prediction")], axis=1).dropna()
    if values.empty:
        raise ValueError("No overlapping labels and predictions to evaluate")

    def correlation(group: pd.DataFrame, method: str) -> float:
        return group["label"].corr(group["prediction"], method=method) if len(group) > 1 else np.nan

    daily_ic = values.groupby(level="datetime", sort=False).apply(
        lambda group: correlation(group, "pearson"), include_groups=False
    ).dropna()
    daily_rank_ic = values.groupby(level="datetime", sort=False).apply(
        lambda group: correlation(group, "spearman"), include_groups=False
    ).dropna()

    def top_bottom_spread(group: pd.DataFrame) -> float:
        count = max(1, len(group) // 10)
        ordered = group.sort_values("prediction")
        return float(ordered["label"].iloc[-count:].mean() - ordered["label"].iloc[:count].mean())

    spreads = values.groupby(level="datetime", sort=False).apply(
        top_bottom_spread, include_groups=False
    )
    result = {
        "rows": int(len(values)),
        "mean_daily_ic": float(daily_ic.mean()),
        "daily_ic_std": float(daily_ic.std()),
        "mean_daily_rank_ic": float(daily_rank_ic.mean()),
        "daily_rank_ic_std": float(daily_rank_ic.std()),
        "rank_icir": float(daily_rank_ic.mean() / daily_rank_ic.std()),
        "rank_ic_hit_ratio": float((daily_rank_ic > 0).mean()),
        "mean_top_bottom_decile_spread": float(spreads.mean()),
    }
    # Raw RMSE/MAE are valid for the raw-return factor baseline, but not for a
    # model whose output is a CSRankNorm score.
    if include_regression_metrics:
        error = values["prediction"] - values["label"]
        result.update(
            {
                "rmse": float(np.sqrt(np.mean(error**2))),
                "mae": float(np.mean(np.abs(error))),
            }
        )
    return result


def apply_feature_transform(frame: pd.DataFrame, transform: str) -> pd.DataFrame:
    """Apply an optional per-date cross-sectional transform to all features."""
    if transform == "none":
        return frame.copy()
    try:
        from qlib.data.dataset.processor import CSRankNorm
    except ImportError as exc:
        raise SystemExit(
            "Qlib is required. Install dependencies with: "
            "python3 -m pip install -r requirements.txt"
        ) from exc
    if transform != "cs_rank":
        raise ValueError(f"Unsupported feature transform: {transform}")
    return CSRankNorm(fields_group="feature")(frame.copy())


def apply_label_transform(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply Qlib's cross-sectional rank normalization to the training label."""
    try:
        from qlib.data.dataset.processor import CSRankNorm
    except ImportError as exc:
        raise SystemExit(
            "Qlib is required. Install dependencies with: "
            "python3 -m pip install -r requirements.txt"
        ) from exc

    # Rank normalization is done within each date and only on the label.  The
    # raw label is retained separately for unbiased return/IC evaluation.
    return CSRankNorm(fields_group="label")(frame.copy())


def select_segment(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    start, end = SPLITS[split]
    dates = frame.index.get_level_values("datetime")
    return frame[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]


def feature_importance_table(model: Any, feature_columns: list[str]) -> pd.DataFrame:
    """Return XGBoost split, gain and coverage importance by feature."""
    booster = model.model
    scores = {}
    for importance_type in ("weight", "gain", "cover", "total_gain", "total_cover"):
        scores[importance_type] = booster.get_score(importance_type=importance_type)

    rows = []
    for index, feature in enumerate(feature_columns):
        key = f"f{index}"
        rows.append(
            {
                "feature": feature,
                **{importance_type: float(values.get(key, 0.0)) for importance_type, values in scores.items()},
            }
        )
    return pd.DataFrame(rows).sort_values("gain", ascending=False)


def prepare_dataset(frame: pd.DataFrame, segments: Dict[str, tuple] = SPLITS):
    # Imports are delayed so data preparation can still be inspected without
    # installing Qlib; training gives a clear dependency error instead.
    try:
        from qlib.data.dataset import DatasetH
        from qlib.data.dataset.handler import DataHandler
    except ImportError as exc:
        raise SystemExit(
            "Qlib is required. Install dependencies with: "
            "python3 -m pip install -r requirements.txt"
        ) from exc

    # DatasetH validates that its handler is a Qlib DataHandler.  The actual
    # fetch implementation remains in FeatureFrameHandler; this adapter only
    # supplies Qlib's required marker/base type without invoking DataHandler's
    # normal binary-provider initialization.
    class QlibFeatureFrameHandler(FeatureFrameHandler, DataHandler):
        pass

    handler = QlibFeatureFrameHandler(frame)
    return DatasetH(handler=handler, segments=segments)


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
        default=Path(__file__).resolve().parent / "data" / "qlib" / "xgboost_baseline",
    )
    parser.add_argument(
        "--objective",
        choices=["reg:squarederror", "reg:pseudohubererror", "reg:absoluteerror"],
        default="reg:pseudohubererror",
    )
    # Frozen from validation Rank IC selection in run_xgb_experiments.py.
    parser.add_argument("--n-estimators", type=int, default=25)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--min-child-weight", type=float, default=20.0)
    parser.add_argument("--subsample", type=float, default=0.70)
    parser.add_argument("--colsample-bytree", type=float, default=0.75)
    parser.add_argument("--gamma", type=float, default=0.10)
    parser.add_argument("--reg-alpha", type=float, default=0.0)
    parser.add_argument("--reg-lambda", type=float, default=10.0)
    parser.add_argument("--early-stopping-rounds", type=int, default=30)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--feature-transform", choices=["cs_rank", "none"], default="cs_rank"
    )
    args = parser.parse_args()

    raw_frame, feature_columns = load_frame(args.input)
    transformed_frame = apply_feature_transform(raw_frame, args.feature_transform)
    train_frame = apply_label_transform(transformed_frame)
    dataset = prepare_dataset(train_frame)

    try:
        from qlib.contrib.model.xgboost import XGBModel
    except ImportError as exc:
        raise SystemExit(
            "Qlib is required. Install dependencies with: "
            "python3 -m pip install -r requirements.txt"
        ) from exc

    model = XGBModel(
        max_depth=args.max_depth,
        eta=args.learning_rate,
        min_child_weight=args.min_child_weight,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        gamma=args.gamma,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        objective=args.objective,
        eval_metric="rmse",
        nthread=args.workers,
        seed=42,
    )
    model.fit(
        dataset,
        num_boost_round=args.n_estimators,
        early_stopping_rounds=args.early_stopping_rounds,
        verbose_eval=50,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics: Dict[str, Any] = {
        "input": str(args.input),
        "features": feature_columns,
        "label": "target_1d",
        "feature_transform": args.feature_transform,
        "training_label_transform": "Qlib CSRankNorm (cross-sectional, per date)",
        "evaluation_label": "raw target_1d return",
        "splits": SPLITS,
        "model": {
            "objective": args.objective,
            "n_estimators": args.n_estimators,
            "round_selection": "validation Rank IC",
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
            "min_child_weight": args.min_child_weight,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
            "gamma": args.gamma,
            "reg_alpha": args.reg_alpha,
            "reg_lambda": args.reg_lambda,
            "early_stopping_rounds": args.early_stopping_rounds,
            "random_state": 42,
        },
        "evaluation": {},
    }
    # Evaluate model scores against the original, unnormalized next-day return.
    # The one-day reversal factor is a deliberately simple, transparent baseline.
    baseline_scores = raw_frame[("feature", "one_day_reversal")].rename("one_day_reversal")
    prediction_frames = []
    metrics["baselines"] = {}
    for split in ("train", "valid", "backtest"):
        prediction = as_series(model.predict(dataset, segment=split), "prediction")
        raw_segment = select_segment(raw_frame, split)
        label = as_series(raw_segment[[("label", "target_1d")]], "label")
        baseline = baseline_scores.loc[label.index]
        result_name = "validation" if split == "valid" else split
        joined = pd.concat([label, prediction, baseline], axis=1).dropna()
        metrics["evaluation"][result_name] = evaluate(label, prediction)
        metrics["baselines"].setdefault("one_day_reversal", {})[result_name] = evaluate(
            label, baseline, include_regression_metrics=True
        )
        output = joined.reset_index()
        output["split"] = result_name
        prediction_frames.append(output)

    importance = feature_importance_table(model, feature_columns)
    importance.to_csv(args.output_dir / "feature_importance.csv", index=False)
    metrics["feature_importance"] = {
        "file": "feature_importance.csv",
        "top_by_gain": importance.head(15).to_dict(orient="records"),
    }

    pd.concat(prediction_frames, ignore_index=True).to_csv(
        args.output_dir / "predictions.csv", index=False
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(metrics["evaluation"], indent=2))
    print(f"Predictions and metrics written to {args.output_dir}")


if __name__ == "__main__":
    main()
